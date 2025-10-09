import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAE_Encoder(nn.Module):
    def __init__(self, config):
        super(VAE_Encoder, self).__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']
        
        # GRU 레이어
        # batch_first=True로 설정하여 (batch, seq, feature) 형태로 처리
        self.gru = nn.GRU(
            input_size=self.features,
            hidden_size=200,
            batch_first=True
        )
        
        # TimeDistributed(Dense)에 해당하는 부분은 PyTorch에선 일반적인 Linear 레이어를 사용하고
        # GRU가 return_sequences=True이므로 모든 시점에 대해 출력을 만들어 줘.
        self.z_mean_layer = nn.Linear(200, self.latent_dim)
        self.z_log_var_layer = nn.Linear(200, self.latent_dim)
        
    # Reparametrization Trick
    def reparameterize(self, z_mean, z_log_var):
        # z_log_var를 표준 편차(std dev)로 변환: scale = exp(0.5 * log_var)
        std = torch.exp(0.5 * z_log_var)
        # N(0, 1)에서 epsilon 샘플링 (z_mean과 동일한 shape)
        eps = torch.randn_like(std)
        # Reparametrization: z = mu + std * epsilon
        z = z_mean + std * eps
        return z

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, features)
        
        # GRU 포워드 (h_n은 최종 hidden state)
        gru_output, _ = self.gru(inputs) # gru_output shape: (batch_size, seq_len, 200)
        
        # z_mean과 z_log_var 계산
        z_mean = self.z_mean_layer(gru_output) # shape: (batch_size, seq_len, latent_dim)
        z_log_var = self.z_log_var_layer(gru_output) # shape: (batch_size, seq_len, latent_dim)
        
        # Reparametrization Trick을 이용해 z 샘플링
        z = self.reparameterize(z_mean, z_log_var)
        
        return z_mean, z_log_var, z

class VAE_Decoder(nn.Module):
    def __init__(self, config):
        super(VAE_Decoder, self).__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']
        
        # GRU 레이어 (입력은 Latent vector z)
        self.gru = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=200,
            batch_first=True
        )
        
        # 재구성된 X의 평균과 로그 분산 계산
        self.Xhat_mean_layer = nn.Linear(200, self.features)
        self.Xhat_log_var_layer = nn.Linear(200, self.features)

    # Reparametrization Trick (인코더와 동일)
    def reparameterize(self, Xhat_mean, Xhat_log_var):
        std = torch.exp(0.5 * Xhat_log_var)
        eps = torch.randn_like(std)
        Xhat = Xhat_mean + std * eps
        return Xhat

    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, latent_dim)
        
        # GRU 포워드
        gru_output, _ = self.gru(inputs) # gru_output shape: (batch_size, seq_len, 200)
        
        # Xhat_mean과 Xhat_log_var 계산
        Xhat_mean = self.Xhat_mean_layer(gru_output) # shape: (batch_size, seq_len, features)
        Xhat_log_var = self.Xhat_log_var_layer(gru_output) # shape: (batch_size, seq_len, features)
        
        # Reparametrization Trick을 이용해 Xhat 샘플링
        Xhat = self.reparameterize(Xhat_mean, Xhat_log_var)
        
        return Xhat_mean, Xhat_log_var, Xhat

# SISVAE 클래스
class SISVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(SISVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # 손실 함수 정의
    def loss_fn(self, X, Xhat, z_mean, z_log_var):
        recon_loss = torch.mean(
            torch.sum((X - Xhat).pow(2), dim=[1, 2])
        )
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        )
        
        # 3. Smoothing Loss (시점 t와 t-1 사이의 차이) - 기존 로직 유지
        smooth_loss_list = []
        for time_step in range(1, z_mean.shape[1]):
            # t-1 시점의 Latent 분포
            q_z_prev = Normal(loc=z_mean[:, time_step - 1], scale=torch.exp(0.5 * z_log_var[:, time_step - 1]))
            # t 시점의 Latent 분포
            q_z_curr = Normal(loc=z_mean[:, time_step], scale=torch.exp(0.5 * z_log_var[:, time_step]))
            
            # KL(Q(z_t-1) || Q(z_t))
            kl_smooth = torch.distributions.kl_divergence(q_z_prev, q_z_curr)
            smooth_loss_list.append(kl_smooth)
        
        if smooth_loss_list:
            smooth_loss_stacked = torch.stack(smooth_loss_list, dim=1) # (batch_size, seq_len-1, latent_dim)
            # 배치, 시퀀스, latent 차원에 걸친 평균
            smooth_loss = torch.mean(torch.sum(smooth_loss_stacked, dim=[1, 2]))
        else: # seq_len이 1일 경우
            smooth_loss = torch.tensor(0.0, device=X.device)
            
        # 재구성 손실을 'log_probs_loss' 대신 'recon_loss'로 반환
        return recon_loss, kl_loss, smooth_loss
    
    # Custom Training Step
    # PyTorch에서는 nn.Module의 메소드를 직접 오버라이드하는 대신, 훈련 루프 내에서 직접 구현
    def training_step(self, X, optimizer):
        self.train() # 훈련 모드
        optimizer.zero_grad() # 그래디언트 초기화

        # Forward Pass
        # X shape: (Batch, Seq_len, Features)
        z_mean, z_log_var, z = self.encoder(X)
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(z)

        # Loss 계산
        recon_loss_batch, KL_loss_batch, smooth_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
        # Total Loss
        total_loss_batch = recon_loss_batch + KL_loss_batch + 0.5 * smooth_loss_batch

        # Backward Pass 및 최적화
        total_loss_batch.backward()
        optimizer.step()

        # 결과 반환 (Batch 평균)
        return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item(), smooth_loss_batch.item()

    # Custom Test/Validation Step
    def validation_step(self, X):
        self.eval() # 평가 모드
        
        with torch.no_grad(): # 그래디언트 계산 비활성화
            # Forward Pass
            z_mean, z_log_var, z = self.encoder(X)
            Xhat_mean, Xhat_log_var, Xhat = self.decoder(z)
            
            # Loss 계산
            recon_loss_batch, KL_loss_batch, smooth_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
            
            # Total Loss
            total_loss_batch = recon_loss_batch + KL_loss_batch + 0.5 * smooth_loss_batch

            return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item(), smooth_loss_batch.item()

    # 학습 및 테스트 시 공통적으로 사용되는 forward pass
    def forward(self, X):
        # 인코더 포워드 패스
        z_mean, z_log_var, z = self.encoder(X)
        
        # 디코더 포워드 패스
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(z)
        
        # NOTE: loss_fn을 수정함에 따라, forward가 반환하는 Xhat을 사용하도록 
        # 학습 루프에서도 수정이 필요합니다.
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z

# EarlyStopping 구현
class EarlyStopping:
    def __init__(self, monitor='val_recon_loss', mode='min', patience=250, restore_best_weights=True, verbose=1):
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
                    print(f"Epoch {epoch+1}: {self.monitor} did not improve. Best loss: {self.best_loss:.6f}. No. of epochs since last improvement: {self.epochs_no_improve}")
                if self.epochs_no_improve >= self.patience:
                    self.stop_training = True
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}.")
                    if self.restore_best_weights and self.best_weights is not None:
                        model.load_state_dict(self.best_weights)
                        if self.verbose:
                            print("Restoring best model weights.")