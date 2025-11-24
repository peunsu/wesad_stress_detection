import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class KLAnnealingHelper:
    """KL Annealing Implmementation"""
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

class EarlyStopping:
    """Early Stopping Implementation"""
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

def depth_to_space_2d(x, block_size):
    """Rearranges data from depth into blocks of spatial data."""
    b, c, h, w = x.shape
    new_c = c // (block_size ** 2)
    x = x.view(b, new_c, block_size, block_size, h, w)
    # Rearrange to (b, new_c, h * block_size, w * block_size)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, new_c, h * block_size, w * block_size)    

class LocalModule(nn.Module):
    """Dual-branch VAE Local Module"""
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.small_seq_len = config['small_window_size']
        self.features = config['features']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        
        if self.small_seq_len == 24:
            self._encoder_symmetric_pad = True
            self.enc_conv1 = nn.Sequential(
                nn.Conv2d(self.features, self.hidden_dim // 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # (Batch, hidden_dim // 8, Seq_len / 2, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.enc_conv2 = nn.Sequential(
                nn.Conv2d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # (Batch, hidden_dim // 4, Seq_len / 4, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.enc_conv3 = nn.Sequential(
                nn.Conv2d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # (Batch, hidden_dim // 2, Seq_len / 8, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.enc_conv4 = nn.Sequential(
                nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, kernel_size=(4, 1), stride=(1, 1), padding=0), # (Batch, hidden_dim, Seq_len / 24, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
        elif self.small_seq_len == 48:
            self._encoder_symmetric_pad = False
            self.enc_conv1 = nn.Sequential(
                nn.Conv2d(self.features, self.hidden_dim // 8, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # (Batch, hidden_dim // 8, Seq_len / 2, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.enc_conv2 = nn.Sequential(
                nn.Conv2d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # (Batch, hidden_dim // 4, Seq_len / 4, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.enc_conv3 = nn.Sequential(
                nn.Conv2d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), # (Batch, hidden_dim // 2, Seq_len / 8, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.enc_conv4 = nn.Sequential(
                nn.Conv2d(self.hidden_dim // 2, self.hidden_dim, kernel_size=(6, 1), stride=(1, 1), padding=0), # (Batch, hidden_dim, Seq_len / 48, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            raise ValueError("Unsupported small_seq_len value")
        
        self.enc_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  # (Batch, features, Seq_len, Num_windows)
        
        if getattr(self, '_encoder_symmetric_pad', False):
            x = F.pad(x, (0, 0, 4, 4), mode='reflect')
        
        x = self.enc_conv1(x) # (Batch, hidden_dim // 8, 24, Num_windows)
        x = self.enc_conv2(x) # (Batch, hidden_dim // 4, 12, Num_windows)
        x = self.enc_conv3(x) # (Batch, hidden_dim // 2, 6, Num_windows)
        x = self.enc_conv4(x) # (Batch, hidden_dim, 1, Num_windows)
        
        x = x.permute(0, 3, 2, 1)  # (Batch, Num_windows, 1, hidden_dim)
        x = torch.flatten(x, start_dim=2)  # (Batch, Num_windows, hidden_dim)
        
        x = self.enc_fc(x)  # (Batch, Num_windows, Latent_dim * 4)

        return x

class GlobalModule(nn.Module):
    """Dual-branch VAE Global Module"""
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.small_seq_len = config['small_window_size']
        self.features = config['features']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        
        if self.seq_len == 72:
            self.enc_conv1 = nn.Sequential(
                nn.Conv1d(self.features, self.hidden_dim // 8, kernel_size=7, stride=3, padding=3), # (Batch, hidden_dim // 8, Seq_len / 3)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 8, Seq_len / 6)
                #nn.BatchNorm1d(self.hidden_dim // 8)
            )
            self.enc_conv2 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=5, stride=3, padding=2), # (Batch, hidden_dim // 4, Seq_len / 18)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 4, Seq_len / 36)
                #nn.BatchNorm1d(self.hidden_dim // 4)
            )
            self.enc_conv3 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=3, stride=1, padding=1), # (Batch, hidden_dim // 2, Seq_len / 72)
                nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, stride=1, padding=1), # (Batch, hidden_dim, Seq_len / 144)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 2, Seq_len / 144)
                #nn.BatchNorm1d(self.hidden_dim // 2)
            )
        elif self.seq_len == 144:
            self.enc_conv1 = nn.Sequential(
                nn.Conv1d(self.features, self.hidden_dim // 8, kernel_size=7, stride=3, padding=3), # (Batch, hidden_dim // 8, Seq_len / 3)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 8, Seq_len / 6)
                #nn.BatchNorm1d(self.hidden_dim // 8)
            )
            self.enc_conv2 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=5, stride=3, padding=2), # (Batch, hidden_dim // 4, Seq_len / 18)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 4, Seq_len / 36)
                #nn.BatchNorm1d(self.hidden_dim // 4)
            )
            self.enc_conv3 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=3, stride=2, padding=1), # (Batch, hidden_dim // 2, Seq_len / 72)
                nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, stride=1, padding=1), # (Batch, hidden_dim, Seq_len / 144)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 2, Seq_len / 144)
                #nn.BatchNorm1d(self.hidden_dim // 2)
            )
        elif self.seq_len == 288:
            self.enc_conv1 = nn.Sequential(
                nn.Conv1d(self.features, self.hidden_dim // 8, kernel_size=7, stride=3, padding=3), # (Batch, hidden_dim // 8, Seq_len / 3)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 8, Seq_len / 6)
                #nn.BatchNorm1d(self.hidden_dim // 8)
            )
            self.enc_conv2 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=5, stride=3, padding=2), # (Batch, hidden_dim // 4, Seq_len / 18)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 4, Seq_len / 36)
                #nn.BatchNorm1d(self.hidden_dim // 4)
            )
            self.enc_conv3 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=3, stride=2, padding=1), # (Batch, hidden_dim // 2, Seq_len / 72)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 2, Seq_len / 144)
                #nn.BatchNorm1d(self.hidden_dim // 2)
            )
            self.enc_conv4 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, stride=1, padding=1), # (Batch, hidden_dim, Seq_len / 144)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim, Seq_len / 288)
                #nn.BatchNorm1d(self.hidden_dim)
            )
        elif self.seq_len == 576:
            self.enc_conv1 = nn.Sequential(
                nn.Conv1d(self.features, self.hidden_dim // 8, kernel_size=7, stride=3, padding=3), # (Batch, hidden_dim // 8, Seq_len / 3)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 8, Seq_len / 6)
                #nn.BatchNorm1d(self.hidden_dim // 8)
            )
            self.enc_conv2 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 8, self.hidden_dim // 4, kernel_size=5, stride=3, padding=2), # (Batch, hidden_dim // 4, Seq_len / 18)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 4, Seq_len / 36)
                #nn.BatchNorm1d(self.hidden_dim // 4)
            )
            self.enc_conv3 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 4, self.hidden_dim // 2, kernel_size=3, stride=2, padding=1), # (Batch, hidden_dim // 2, Seq_len / 72)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim // 2, Seq_len / 144)
                #nn.BatchNorm1d(self.hidden_dim // 2)
            )
            self.enc_conv4 = nn.Sequential(
                nn.Conv1d(self.hidden_dim // 2, self.hidden_dim, kernel_size=3, stride=2, padding=1), # (Batch, hidden_dim, Seq_len / 288)
                nn.LeakyReLU(negative_slope=0.2),
                nn.MaxPool1d(kernel_size=2, stride=2),  # (Batch, hidden_dim, Seq_len / 576)
                #nn.BatchNorm1d(self.hidden_dim)
            )
        else:
            raise ValueError("Unsupported seq_len value")
        
        self.enc_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.latent_dim * 4),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):        
        x = x.permute(0, 2, 1)  # (Batch, Features, Seq_len) for Conv1d
        
        x = self.enc_conv1(x) # (Batch, hidden_dim // 8, 192)
        x = self.enc_conv2(x) # (Batch, hidden_dim // 4, 32)
        x = self.enc_conv3(x) # (Batch, hidden_dim // 2, 8)
        if hasattr(self, 'enc_conv4'):
            x = self.enc_conv4(x) # (Batch, hidden_dim, 2)
        
        x = x.permute(0, 2, 1)  # (Batch, 1, hidden_dim)
        
        x = self.enc_fc(x)  # (Batch, 1, Latent_dim * 4)

        return x

class MA(nn.Module):
    """Dual-branch VAE Multi-Head Attention Module"""
    def __init__(self, config):
        super().__init__()
        self.latent_dim = config['latent_dim']
        
        self.ma_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim * 4,
            num_heads=self.latent_dim,
            batch_first=True
        )
        self.ma_layernorm = nn.LayerNorm(self.latent_dim * 4)
    
    def forward(self, local_z, global_z):
        A, _ = self.ma_attn(local_z, global_z, global_z)  # A shape: (Batch, Num_windows, Latent_dim)
        A = self.ma_layernorm(A + local_z)  # Residual connection and layer norm
        
        return A

class VAE_Encoder(nn.Module):
    """Dual-branch VAE Encoder with Multi-Head Attention"""
    def __init__(self, config):
        super().__init__()
        self.local_module = LocalModule(config)
        self.global_module = GlobalModule(config)
        self.ma = MA(config)
        
        self.small_seq_len = config['small_window_size']
        self.latent_dim = config['latent_dim']
        
        #self.fc = nn.Linear(self.latent_dim * 8, self.latent_dim * 4)
        
        self.fc_mean = nn.Linear(self.latent_dim * 4, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim * 4, self.latent_dim)
    
    def cut_window(self, x):
        windows = []
        for start in range(0, x.size(1), self.small_seq_len):
            end = start + self.small_seq_len
            if end <= x.size(1):
                windows.append(x[:, start:end, :])
        windows = torch.stack(windows, dim=1)  # (Batch, Num_windows, small_window_size, Features)
        return windows

    def forward(self, x):
        x_wins = self.cut_window(x)  # (Batch, Num_windows, small_window_size, Features)
        
        local_z = self.local_module(x_wins)  # (Batch, Num_windows, Latent_dim * 4)
        global_z = self.global_module(x)  # (Batch, 1, Latent_dim * 4)
        
        combined_z = self.ma(local_z, global_z)  # (Batch, Num_windows, Latent_dim * 4)
        
        #combined_z = torch.cat([local_z, global_z.repeat(1, local_z.size(1), 1)], dim=-1)  # (Batch, Num_windows, Latent_dim * 8)
        #combined_z = F.leaky_relu(self.fc(combined_z), negative_slope=0.2)  # (Batch, Num_windows, Latent_dim * 4)
        
        #combined_z = local_z  # (Batch, Num_windows, Latent_dim * 4)
        
        z_mean = self.fc_mean(combined_z)  # (Batch, Num_windows, Latent_dim)
        z_log_var = F.relu(self.fc_log_var(combined_z))  # (Batch, Num_windows, Latent_dim)
        #z_log_var = torch.clamp(F.relu(self.fc_log_var(combined_z)), min=-10, max=10)  # Use it if KL divergence does not converge
        
        # Reparameterization trick
        std = torch.exp(0.5 * z_log_var) + 1e-2  # (Batch, Num_windows, Latent_dim)
        eps = torch.randn_like(std)
        z = z_mean + eps * std  # (Batch, Num_windows, Latent_dim)

        return z_mean, z_log_var, z
    
class VAE_Encoder_Linear(nn.Module):
    """Dual-branch VAE Encoder with Linear Combination"""
    def __init__(self, config):
        super().__init__()
        self.local_module = LocalModule(config)
        self.global_module = GlobalModule(config)
        
        self.small_seq_len = config['small_window_size']
        self.latent_dim = config['latent_dim']
        
        self.fc = nn.Linear(self.latent_dim * 8, self.latent_dim * 4)
        
        self.fc_mean = nn.Linear(self.latent_dim * 4, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim * 4, self.latent_dim)
    
    def cut_window(self, x):
        windows = []
        for start in range(0, x.size(1), self.small_seq_len):
            end = start + self.small_seq_len
            if end <= x.size(1):
                windows.append(x[:, start:end, :])
        windows = torch.stack(windows, dim=1)  # (Batch, Num_windows, small_window_size, Features)
        return windows

    def forward(self, x):
        x_wins = self.cut_window(x)  # (Batch, Num_windows, small_window_size, Features)
        
        local_z = self.local_module(x_wins)  # (Batch, Num_windows, Latent_dim * 4)
        global_z = self.global_module(x)  # (Batch, 1, Latent_dim * 4)
        
        combined_z = torch.cat([local_z, global_z.repeat(1, local_z.size(1), 1)], dim=-1)  # (Batch, Num_windows, Latent_dim * 8)
        combined_z = F.leaky_relu(self.fc(combined_z), negative_slope=0.2)  # (Batch, Num_windows, Latent_dim * 4)
        
        z_mean = self.fc_mean(combined_z)  # (Batch, Num_windows, Latent_dim)
        z_log_var = torch.clamp(F.relu(self.fc_log_var(combined_z)), min=-10, max=10)  # Use it if KL divergence does not converge
        
        # Reparameterization trick
        std = torch.exp(0.5 * z_log_var) + 1e-2  # (Batch, Num_windows, Latent_dim)
        eps = torch.randn_like(std)
        z = z_mean + eps * std  # (Batch, Num_windows, Latent_dim)

        return z_mean, z_log_var, z

class VAE_Encoder_Local(nn.Module):
    """VAE Encoder with Local Module Only"""
    def __init__(self, config):
        super().__init__()
        self.local_module = LocalModule(config)
        
        self.small_seq_len = config['small_window_size']
        self.latent_dim = config['latent_dim']
        
        self.fc_mean = nn.Linear(self.latent_dim * 4, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim * 4, self.latent_dim)
    
    def cut_window(self, x):
        windows = []
        for start in range(0, x.size(1), self.small_seq_len):
            end = start + self.small_seq_len
            if end <= x.size(1):
                windows.append(x[:, start:end, :])
        windows = torch.stack(windows, dim=1)  # (Batch, Num_windows, small_window_size, Features)
        return windows

    def forward(self, x):
        x_wins = self.cut_window(x)  # (Batch, Num_windows, small_window_size, Features)
        
        local_z = self.local_module(x_wins)  # (Batch, Num_windows, Latent_dim * 4)
        
        z_mean = self.fc_mean(local_z)  # (Batch, Num_windows, Latent_dim)
        z_log_var = torch.clamp(F.relu(self.fc_log_var(local_z)), min=-10, max=10)  # Use it if KL divergence does not converge
        
        # Reparameterization trick
        std = torch.exp(0.5 * z_log_var) + 1e-2  # (Batch, Num_windows, Latent_dim)
        eps = torch.randn_like(std)
        z = z_mean + eps * std  # (Batch, Num_windows, Latent_dim)

        return z_mean, z_log_var, z

class VAE_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['window_size']
        self.small_seq_len = config['small_window_size']
        self.features = config['features']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        
        self.dec_fc = nn.Linear(self.latent_dim, self.hidden_dim) # (Batch, Num_windows, hidden_dim)
        
        if self.small_seq_len == 24:
            self.dec_conv1 = nn.Sequential(
                nn.Conv2d(self.hidden_dim, 128 * 3, kernel_size=(1, 1), padding=(0, 0)),  # (Batch, hidden_dim, 1, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.dec_conv2 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)), # (Batch, hidden_dim // 4, 3, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.dec_conv3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)), # (Batch, hidden_dim // 8, 6, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.dec_conv4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)), # (Batch, hidden_dim // 16, 12, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.dec_out = nn.Conv2d(16, self.features, kernel_size=(5, 1), padding=(2, 0)) # (Batch, Features, 24, Num_windows)
        elif self.small_seq_len == 48:
            self.dec_conv1 = nn.Sequential(
                nn.Conv2d(self.hidden_dim, 256 * 3, kernel_size=(1, 1), padding=(0, 0)),  # (Batch, 256 * 3, 1, Num_windows)
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.dec_conv2 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 1), padding=(1, 0)),  # (Batch, 256, 3, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.dec_conv3 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0)),  # (Batch, 128, 6, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )
            self.dec_conv4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1, 0)),  # (Batch, 32, 24, Num_windows)
                nn.LeakyReLU(negative_slope=0.2),
            )         
            self.dec_out = nn.Conv2d(16, self.features, kernel_size=(5, 1), padding=(2, 0)) # (Batch, Features, 48, Num_windows)
        else:
            raise ValueError("Unsupported seq_len or small_seq_len value")
        
    def forward(self, z):
        num_windows = z.size(1)
        x = F.leaky_relu(self.dec_fc(z), negative_slope=0.2) # (B, Num_windows, hidden_dim)
        x = x.permute(0, 2, 1) # (B, hidden_dim, Num_windows)
        x = x.view(-1, self.hidden_dim, 1, num_windows)  # (B, hidden_dim, 1, Num_windows)
        
        if self.small_seq_len == 24:
            x = self.dec_conv1(x)
            b = x.size(0)
            x = x.view(b, 128, 3, num_windows)

            x = self.dec_conv2(x)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 64, 6, num_windows)

            x = self.dec_conv3(x)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 32, 12, num_windows)
            
            x = self.dec_conv4(x)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 16, -1, num_windows)
            
            x = self.dec_out(x)
        elif self.small_seq_len == 48:            
            x = self.dec_conv1(x)
            b = x.size(0)
            x = x.view(b, 256, 3, num_windows)
            
            x = self.dec_conv2(x)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 128, 6, num_windows)
            
            x = self.dec_conv3(x)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 32, 24, num_windows)
            
            x = self.dec_conv4(x)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 16, -1, num_windows)

            x = self.dec_out(x)
        else:
            raise ValueError("Unsupported seq_len or small_seq_len value")
        
        x = x.permute(0, 3, 2, 1)  # (Batch, Num_windows, seq_len, Features)
        return x

class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config
        self.latent_dim = config['latent_dim']
        self.hidden_dim_lstm = config['hidden_dim_lstm']

        # lstm layer 3개
        self.lstm1 = nn.LSTM(self.latent_dim, self.hidden_dim_lstm, batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_dim_lstm, self.hidden_dim_lstm, batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_dim_lstm, self.latent_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return x

class VAE_LSTM_MA(nn.Module):
    """VAE-LSTM model with Multi-Head Attention Encoder"""
    def __init__(self, config, beta=1e-8):
        super().__init__()
        self.encoder = VAE_Encoder(config)
        self.decoder = VAE_Decoder(config)
        self.lstm = LSTMModel(config)
        
        self.config = config
        self.beta = beta
        
    def loss_fn(self, X, Xhat, z_mean, z_log_var):
        recon_loss = torch.mean(
            torch.sum((X - Xhat).pow(2), dim=[2, 3])
        )
        kl_loss = 0.5 * torch.mean(
            torch.sum(z_mean.pow(2), dim=1) +
            torch.sum(z_log_var.exp(), dim=1) -
            torch.sum(z_log_var, dim=1) -
            self.config['latent_dim']
        )
        return recon_loss, kl_loss

    def training_step(self, X, optimizer, beta):
        self.train()
        optimizer.zero_grad()

        # X shape: (Batch, Seq_len, Features)
        z_mean, z_log_var, z = self.encoder(X)
        z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
        Xhat = self.decoder(z_pred)
        
        X = X.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
        X = X[:, 1:, :, :]  # Align X with the number of windows
        
        recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
        total_loss_batch = recon_loss_batch + beta * KL_loss_batch
        
        total_loss_batch.backward()
        optimizer.step()

        return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()
    
    def validation_step(self, X, beta):
        self.eval()
        
        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X)
            z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
            Xhat = self.decoder(z_pred)
            
            X = X.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
            X = X[:, 1:, :, :]  # Align X with the number of windows
            
            recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
            total_loss_batch = recon_loss_batch + beta * KL_loss_batch

            return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
        Xhat = self.decoder(z_pred)
        
        return Xhat, z_mean, z_log_var, z

class VAE_LSTM_Linear(nn.Module):
    """VAE-LSTM model with Linear Combination Encoder"""
    def __init__(self, config, beta=1e-8):
        super().__init__()
        self.encoder = VAE_Encoder_Linear(config)
        self.decoder = VAE_Decoder(config)
        self.lstm = LSTMModel(config)
        
        self.config = config
        self.beta = beta
        
    def loss_fn(self, X, Xhat, z_mean, z_log_var):
        recon_loss = torch.mean(
            torch.sum((X - Xhat).pow(2), dim=[2, 3])
        )
        kl_loss = 0.5 * torch.mean(
            torch.sum(z_mean.pow(2), dim=1) +
            torch.sum(z_log_var.exp(), dim=1) -
            torch.sum(z_log_var, dim=1) -
            self.config['latent_dim']
        )
        return recon_loss, kl_loss

    def training_step(self, X, optimizer, beta):
        self.train()
        optimizer.zero_grad()

        # X shape: (Batch, Seq_len, Features)
        z_mean, z_log_var, z = self.encoder(X)
        z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
        Xhat = self.decoder(z_pred)
        
        X = X.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
        X = X[:, 1:, :, :]  # Align X with the number of windows
        
        recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
        total_loss_batch = recon_loss_batch + beta * KL_loss_batch
        
        total_loss_batch.backward()
        optimizer.step()

        return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()
    
    def validation_step(self, X, beta):
        self.eval()
        
        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X)
            z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
            Xhat = self.decoder(z_pred)
            
            X = X.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
            X = X[:, 1:, :, :]  # Align X with the number of windows
            
            recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
            total_loss_batch = recon_loss_batch + beta * KL_loss_batch

            return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
        Xhat = self.decoder(z_pred)
        
        return Xhat, z_mean, z_log_var, z

class VAE_LSTM_Local(nn.Module):
    """VAE-LSTM model with Local Module Only Encoder"""
    def __init__(self, config, beta=1e-8):
        super().__init__()
        self.encoder = VAE_Encoder_Local(config)
        self.decoder = VAE_Decoder(config)
        self.lstm = LSTMModel(config)
        
        self.config = config
        self.beta = beta
        
    def loss_fn(self, X, Xhat, z_mean, z_log_var):
        recon_loss = torch.mean(
            torch.sum((X - Xhat).pow(2), dim=[2, 3])
        )
        kl_loss = 0.5 * torch.mean(
            torch.sum(z_mean.pow(2), dim=1) +
            torch.sum(z_log_var.exp(), dim=1) -
            torch.sum(z_log_var, dim=1) -
            self.config['latent_dim']
        )
        return recon_loss, kl_loss

    def training_step(self, X, optimizer, beta):
        self.train()
        optimizer.zero_grad()

        # X shape: (Batch, Seq_len, Features)
        z_mean, z_log_var, z = self.encoder(X)
        z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
        Xhat = self.decoder(z_pred)
        
        X = X.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
        X = X[:, 1:, :, :]  # Align X with the number of windows
        
        recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
        total_loss_batch = recon_loss_batch + beta * KL_loss_batch
        
        total_loss_batch.backward()
        optimizer.step()

        return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()
    
    def validation_step(self, X, beta):
        self.eval()
        
        with torch.no_grad():
            z_mean, z_log_var, z = self.encoder(X)
            z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
            Xhat = self.decoder(z_pred)
            
            X = X.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
            X = X[:, 1:, :, :]  # Align X with the number of windows
            
            recon_loss_batch, KL_loss_batch = self.loss_fn(X, Xhat, z_mean, z_log_var)
            total_loss_batch = recon_loss_batch + beta * KL_loss_batch

            return total_loss_batch.item(), recon_loss_batch.item(), KL_loss_batch.item()

    def forward(self, X):
        z_mean, z_log_var, z = self.encoder(X)
        z_pred = self.lstm(z[:, :-1, :])  # Predict next latent vector
        Xhat = self.decoder(z_pred)
        
        return Xhat, z_mean, z_log_var, z

class MA_VAE_Encoder(nn.Module):
    """MA-VAE Encoder"""
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

class MA_VAE_Decoder(nn.Module):
    """MA-VAE Decoder"""
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

class MA_VAE_MA(nn.Module):
    """MA-VAE Multi-Head Attention Module"""
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
    """MA-VAE model"""
    def __init__(self, config, beta=1e-8):
        super().__init__()
        self.encoder = MA_VAE_Encoder(config)
        self.decoder = MA_VAE_Decoder(config)
        self.ma = MA_VAE_MA(config)
        
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