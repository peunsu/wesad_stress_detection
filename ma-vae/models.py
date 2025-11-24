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
        print(f"KL Annealing Type: {self.type}, Beta value: {beta_value:.10f}, cycle epoch {int(shifted_epochs) % self.annealing_epochs}")

class VAE_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']
        
        self.bilstm1 = nn.LSTM(self.features, 512, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(512 * 2, 256, bidirectional=True, batch_first=True) # 512 * 2 (양방향)

        self.z_mean_layer = nn.Linear(256 * 2, self.latent_dim)
        self.z_log_var_layer = nn.Linear(256 * 2, self.latent_dim)

    def forward(self, x):
        # x shape: (Batch, Seq_len, Features)
        
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        bilstm_out, (h_n, c_n) = self.bilstm1(x)
        bilstm_out, (h_n, c_n) = self.bilstm2(bilstm_out)
        # bilstm_out shape: (Batch, Seq_len, 256 * 2)
        
        z_mean = self.z_mean_layer(bilstm_out)
        z_log_var = self.z_log_var_layer(bilstm_out)
        # z_mean, z_log_var shape: (Batch, Seq_len, Latent_dim)

        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(z_mean) 
        z = z_mean + std * eps
        
        states = bilstm_out

        return z_mean, z_log_var, z, states

class VAE_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.features = config['features']
        self.latent_dim = config['latent_dim']

        self.bilstm1 = nn.LSTM(self.latent_dim, 256, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(256 * 2, 512, bidirectional=True, batch_first=True) # 256 * 2 (양방향)

        self.Xhat_mean_layer = nn.Linear(512 * 2, self.features)
        self.Xhat_log_var_layer = nn.Linear(512 * 2, self.features)

    def forward(self, attention_input):
        # attention_input shape: (Batch, Seq_len, Latent_dim)
        
        bilstm_out, _ = self.bilstm1(attention_input)
        bilstm_out, _ = self.bilstm2(bilstm_out)
        # bilstm_out shape: (Batch, Seq_len, 512 * 2)

        Xhat_mean = self.Xhat_mean_layer(bilstm_out)
        Xhat_log_var = self.Xhat_log_var_layer(bilstm_out)
        # Xhat_mean, Xhat_log_var shape: (Batch, Seq_len, Features)

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

        self.q_k_projection = nn.Linear(self.features, 64) 

        self.attention_module = nn.MultiheadAttention(
            embed_dim=64,  # 64. Q의 차원 (Q_K_projected의 차원)
            num_heads=8,
            batch_first=True,
            kdim=64,       # 64. K의 차원
            vdim=self.latent_dim,               # latent_dim. V의 입력 차원
        )
        
        self.output_projection = nn.Linear(64, self.latent_dim)


    def forward(self, inputs):
        # inputs: [ma_input (X: features), latent_input (z: latent_dim)]
        ma_input, latent_input = inputs
        
        Q_K_projected = self.q_k_projection(ma_input) # (B, S, 64)

        attn_output_64, _ = self.attention_module(
            query=Q_K_projected,
            key=Q_K_projected,
            value=latent_input
        )
        # attn_output_64 shape: (B, S, 64)
        
        A = self.output_projection(attn_output_64) # (B, S, latent_dim)

        return A

class MA_VAE(nn.Module):
    def __init__(self, config, beta=1e-8):
        super().__init__()
        self.encoder = VAE_Encoder(config)
        self.decoder = VAE_Decoder(config)
        self.ma = MA(config)
        
        self.beta = beta 
        
    def loss_fn(self, X, Xhat, z_mean, z_log_var):
        recon_loss = torch.mean(
            torch.sum((X - Xhat).pow(2), dim=[1, 2])
        )
        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        )
        return recon_loss, kl_loss

    def training_step(self, X, optimizer, beta):
        self.train()
        optimizer.zero_grad()

        # X shape: (Batch, Seq_len, Features)
        z_mean, z_log_var, z, states = self.encoder(X)
        A = self.ma([X, z]) # z 사용
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A)
        
        recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
        total_loss_batch = recon_loss_batch + beta * KL_loss_batch
        
        total_loss_batch.backward()
        optimizer.step()

        return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()
    
    def validation_step(self, X, beta):
        self.eval()
        
        with torch.no_grad():
            z_mean, z_log_var, z, states = self.encoder(X)
            A = self.ma([X, z_mean]) # z_mean 사용
            Xhat_mean, Xhat_log_var, Xhat = self.decoder(A)
            
            recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
            total_loss_batch = recon_loss_batch + beta * KL_loss_batch

            return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()

    def forward(self, X):
        z_mean, z_log_var, z, states = self.encoder(X)
        A = self.ma([X, z_mean])
        Xhat_mean, Xhat_log_var, Xhat = self.decoder(A)
        return Xhat_mean, Xhat_log_var, Xhat, z_mean, z_log_var, z, A
    
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