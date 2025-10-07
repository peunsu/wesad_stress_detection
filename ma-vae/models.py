import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
# 1. KL Annealing 로직 (Callback 대신 헬퍼 함수로 구현)
# --------------------------------------------------------------------------------

class KLAnnealingHelper:
    """TensorFlow VAE_kl_annealing 콜백의 로직을 구현하는 헬퍼 클래스"""
    def __init__(self, annealing_epochs=30, type="normal", grace_period=20, start=0.0001, end=0.1, lower_initial_betas=False):
        self.annealing_epochs = annealing_epochs
        self.type = type
        self.grace_period = grace_period
        self.grace_period_idx = max(0, grace_period - 1)  # 0부터 시작
        self.start = start
        self.end = end
        
        self.beta_values = None
        if type in ["cyclical", "monotonic"]:
            self.beta_values = np.linspace(start, end, annealing_epochs)
            if lower_initial_betas:
                # np.divmod 대신 정수 나누기
                self.beta_values[:annealing_epochs // 2] /= 2
                
    def get_beta(self, epoch):
        if epoch < self.grace_period_idx or self.type == "normal":
            # TF 코드의 로직을 따라 grace_period 동안 start 값까지 선형 증가
            step_size = (self.start / self.grace_period)
            new_value = step_size * (epoch % self.grace_period)
        elif self.type == "monotonic":
            # min(epoch, self.annealing_epochs - 1) -> 0부터 시작하는 에폭을 인덱스로 사용
            index = min(epoch, self.annealing_epochs - 1)
            new_value = self.beta_values[index]
        elif self.type == "cyclical":
            # shifted_epochs = max(0.0, epoch - self.grace_period_idx)
            shifted_epochs = max(0.0, epoch - self.grace_period_idx)
            # int(shifted_epochs % self.annealing_epochs)
            index = int(shifted_epochs % self.annealing_epochs)
            new_value = self.beta_values[index]
        else:
            new_value = self.end # 정의되지 않은 경우 최대값으로 설정 (혹은 오류)

        return new_value
    
    def print_status(self, epoch):
        shifted_epochs = max(0.0, epoch - self.grace_period_idx)
        beta_value = self.get_beta(epoch)
        print(f"Beta value: {beta_value:.10f}, cycle epoch {int(shifted_epochs) % self.annealing_epochs}")

# --------------------------------------------------------------------------------
# 2. 모델 구성 요소 (Encoder, Decoder, MA)
# --------------------------------------------------------------------------------

class VAE_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']

        # Keras의 GaussianNoise(0.01)는 훈련 중에만 적용됨. PyTorch에서는 forward에서 직접 처리하거나,
        # Keras와 유사하게 nn.Dropout을 훈련 시퀀스에 넣어서 입력에 노이즈를 주입하는 효과를 낼 수도 있음.
        # 여기서는 Keras의 `tfkl.GaussianNoise` 대신 `forward` 메서드에서 직접 노이즈를 추가.
        
        # BiLSTM 레이어 (512, 256)
        self.bilstm1 = nn.LSTM(self.features, 512, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(512 * 2, 256, bidirectional=True, batch_first=True) # 512 * 2 (양방향)

        # TimeDistributed(Dense) -> PyTorch에서는 nn.Linear를 사용하고, TimeDistributed 효과를 위해
        # LSTM의 `return_sequences=True`와 동일하게 시퀀스 차원을 보존하는 출력을 사용.
        # LSTM 출력은 (Batch, Seq_len, Hidden_size * 2)
        self.z_mean_layer = nn.Linear(256 * 2, self.latent_dim)
        self.z_log_var_layer = nn.Linear(256 * 2, self.latent_dim)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        
        # 1. Gaussian Noise 추가 (훈련 중일 때만)
        if self.training:
            # torch.randn_like(x)를 사용하여 입력과 동일한 shape의 노이즈 생성
            x = x + torch.randn_like(x) * 0.01

        # 2. BiLSTM
        # (h_n, c_n)을 사용하지 않으므로 초기 상태는 기본값 (0)
        bilstm_out, (h_n, c_n) = self.bilstm1(x)
        bilstm_out, (h_n, c_n) = self.bilstm2(bilstm_out)
        # bilstm_out shape: (Batch, Seq_len, 256 * 2)
        
        # 3. 분포 파라미터 계산
        z_mean = self.z_mean_layer(bilstm_out)
        z_log_var = self.z_log_var_layer(bilstm_out)
        # z_mean, z_log_var shape: (Batch, Seq_len, Latent_dim)

        # 4. Reparameterization Trick
        # tf.math.exp(z_log_var) -> torch.exp(z_log_var)
        # tf.sqrt(...) -> torch.sqrt(...)
        std = torch.exp(0.5 * z_log_var)
        # z_mean과 같은 shape의 표준정규분포(Normal(0, 1))에서 epsilon 샘플링
        eps = torch.randn_like(z_mean) 
        z = z_mean + std * eps
        
        # states는 Keras 코드에서 BiLSTM의 최종 출력을 의미 (256*2)
        states = bilstm_out

        return z_mean, z_log_var, z, states

class VAE_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']

        # BiLSTM 레이어 (256, 512)
        # Input shape: (Batch, Seq_len, Latent_dim)
        self.bilstm1 = nn.LSTM(self.latent_dim, 256, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(256 * 2, 512, bidirectional=True, batch_first=True) # 256 * 2 (양방향)

        # TimeDistributed(Dense) -> nn.Linear
        # Output shape: (Batch, Seq_len, Features)
        self.Xhat_mean_layer = nn.Linear(512 * 2, self.features)
        self.Xhat_log_var_layer = nn.Linear(512 * 2, self.features)

    def forward(self, attention_input):
        # attention_input shape: (Batch, Seq_len, Latent_dim)
        
        # 1. BiLSTM
        bilstm_out, _ = self.bilstm1(attention_input)
        bilstm_out, _ = self.bilstm2(bilstm_out)
        # bilstm_out shape: (Batch, Seq_len, 512 * 2)

        # 2. 분포 파라미터 계산
        Xhat_mean = self.Xhat_mean_layer(bilstm_out)
        Xhat_log_var = self.Xhat_log_var_layer(bilstm_out)
        # Xhat_mean, Xhat_log_var shape: (Batch, Seq_len, Features)

        # 3. Reparameterization Trick (재구성 샘플)
        std = torch.exp(0.5 * Xhat_log_var)
        eps = torch.randn_like(Xhat_mean)
        Xhat = Xhat_mean + std * eps

        return Xhat_mean, Xhat_log_var, Xhat

class MA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']

        # 1. Q, K Projection (features -> keras_key_dim=64)
        # Keras key_dim=64 의도 반영. Q와 K를 64차원으로 투영
        self.q_k_projection = nn.Linear(self.features, 64) 

        # 2. PyTorch MultiheadAttention 설정
        # Q와 K는 64차원으로 들어옴. V는 latent_dim 차원으로 들어옴.
        # embed_dim=64: Q의 차원 & 어텐션 결과 차원.
        self.attention_module = nn.MultiheadAttention(
            embed_dim=64,  # 64. Q의 차원 (Q_K_projected의 차원)
            num_heads=8,
            batch_first=True,
            kdim=64,       # 64. K의 차원
            vdim=self.latent_dim,               # latent_dim. V의 입력 차원
        )
        
        # 3. Output Projection (keras_key_dim=64 -> latent_dim)
        # Keras output_shape=latent_dim 의도 반영. 64차원 어텐션 출력을 latent_dim으로 변환
        self.output_projection = nn.Linear(64, self.latent_dim)


    def forward(self, inputs):
        # inputs: [ma_input (X: features), latent_input (z: latent_dim)]
        ma_input, latent_input = inputs
        
        # 1. Q, K Projection: features -> 64
        Q_K_projected = self.q_k_projection(ma_input) # (B, S, 64)

        # 2. Attention 계산 (Q=64, K=64, V=latent_dim -> Output=64)
        attn_output_64, _ = self.attention_module(
            query=Q_K_projected,
            key=Q_K_projected,
            value=latent_input
        )
        # attn_output_64 shape: (B, S, 64)
        
        # 3. Output Projection: 64 -> latent_dim
        A = self.output_projection(attn_output_64) # (B, S, latent_dim)

        return A

# --------------------------------------------------------------------------------
# 3. MA_VAE 모델 (훈련/테스트 스텝 포함)
# --------------------------------------------------------------------------------

class MA_VAE(nn.Module):
    def __init__(self, encoder, decoder, ma, beta=1e-8):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ma = ma
        # beta는 훈련 루프에서 외부적으로 관리
        self.beta = beta 
        
    def loss_fn(self, X, Xhat, z_mean, z_log_var):     
        batch_size = X.size(0) 
        recon_loss = F.mse_loss(Xhat, X, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        return recon_loss, kl_loss

    # Custom Training Step
    # PyTorch에서는 nn.Module의 메소드를 직접 오버라이드하는 대신, 훈련 루프 내에서 직접 구현
    def training_step(self, X, optimizer, beta):
        self.train() # 훈련 모드
        optimizer.zero_grad() # 그래디언트 초기화

        # Forward Pass
        # X shape: (Batch, Seq_len, Features)
        z_mean, z_log_var, z, states = self.encoder(X)
        A = self.ma([X, z]) # z 사용
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A)
        
        # Loss 계산
        recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
        # Total Loss
        total_loss_batch = recon_loss_batch + beta * KL_loss_batch
        
        # Backward Pass 및 최적화
        total_loss_batch.backward()
        optimizer.step()

        # 결과 반환 (Batch 평균)
        return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()
    
    # Custom Test/Validation Step
    def validation_step(self, X, beta):
        self.eval() # 평가 모드
        
        with torch.no_grad(): # 그래디언트 계산 비활성화
            # Forward Pass (z_mean 사용)
            z_mean, z_log_var, z, states = self.encoder(X)
            A = self.ma([X, z_mean]) # z_mean 사용 (test_step 참조)
            Xhat_mean, Xhat_log_var, Xhat = self.decoder(A)
            
            # Loss 계산
            recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
            
            # Keras의 validation_data에서는 beta가 고정된 값 (예: 1e-8)을 사용하거나
            # 훈련 중 적용된 beta를 사용할 수 있지만, 여기서는 NLL만 모니터링하므로,
            # KL loss는 단순히 기록용으로 남겨둘게.
            # Keras의 EarlyStopping이 `val_log_probs_loss`를 모니터링하므로, NLL_loss만 반환
            
            # Total Loss
            total_loss_batch = recon_loss_batch + beta * KL_loss_batch

            return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()

    # Keras의 call 메서드와 동일한 역할
    def forward(self, X):
        # Window-level Within-channel Normalization
        # X shape: (Batch, Seq_len, Features)
        # 1e-6은 0으로 나누는 것을 방지하기 위한 작은 상수
        #X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)
        
        # Encoder is fed with input window
        z_mean, z_log_var, z, states = self.encoder(X)
        # Mean matrix of latent distribution is passed to MA mechanism
        A = self.ma([X, z_mean])
        # Decoder is fed with the attention matrix from MA mechanism
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A)
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z, A

# --------------------------------------------------------------------------------
# 4. 데이터셋 및 훈련 루프
# --------------------------------------------------------------------------------

# EarlyStopping 구현
class EarlyStopping:
    def __init__(self, monitor='val_log_probs_loss', mode='min', patience=250, restore_best_weights=True, verbose=1):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_loss = float('inf') if mode == 'min' else float('-inf')
        self.epochs_no_improve = 0
        self.stop_training = False
        self.best_weights = None
        
    def __call__(self, current_loss, model, epoch):
        if self.mode == 'min':
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.epochs_no_improve = 0
                if self.restore_best_weights:
                    self.best_weights = model.state_dict()
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} improved to {self.best_loss:.6f}. Saving best weights.")
            else:
                self.epochs_no_improve += 1
                if self.verbose:
                    print(f"Epoch {epoch+1}: {self.monitor} did not improve. No. of epochs since last improvement: {self.epochs_no_improve}")
                if self.epochs_no_improve >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}.")
                    if self.restore_best_weights and self.best_weights is not None:
                        model.load_state_dict(self.best_weights)
                        if self.verbose:
                            print("Restoring best model weights.")