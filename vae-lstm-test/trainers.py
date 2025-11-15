from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm

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

        
class VAELSTMTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.base_lr = config['learning_rate']
        self.beta = config.get('vae_beta', 0.0001)
        self.optimizer = optim.Adam(model.parameters(), lr=self.base_lr, betas=(0.9, 0.95))
        self.annealing = KLAnnealingHelper(
            annealing_epochs=20,
            type="cyclical",
            grace_period=20,
            start=0.0001,
            end=0.1,
            lower_initial_betas=False,
        )
        self.es = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.kl_losses = []
        self.recon_losses = []
        
        # Create directories
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['result_dir']).mkdir(parents=True, exist_ok=True)
    
    def vae_loss(self, recon_x, x, mu, std_dev):
        """Calculate VAE loss"""
        kl_loss = 0.5 * torch.mean(
            torch.sum(mu.pow(2), dim=1) +
            torch.sum(std_dev.pow(2), dim=1) -
            torch.sum(torch.log(std_dev.pow(2)), dim=1) -
            self.config['code_size']
        )
        
        recon_loss = torch.mean(
            torch.sum((x[:, 1:, :] - recon_x).pow(2), dim=[1, 2])
        )
        
        loss = recon_loss + self.beta * kl_loss
        
        return loss, recon_loss, kl_loss
    
    def train_epoch(self, train_loader, epoch, epochs):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]

            batch_data = batch_data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, std_dev = self.model(batch_data)

            # Calculate loss
            loss, recon_loss, kl_loss = self.vae_loss(recon_batch, batch_data, mu, std_dev)

            # Backward pass
            loss.backward()
            clip_grad_value_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss

    def validate(self, val_loader, epoch, epochs):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]

                batch_data = batch_data.to(self.device)
                
                # Forward pass
                recon_batch, mu, std_dev = self.model(batch_data)
                
                # Calculate loss
                loss, recon_loss, kl_loss = self.vae_loss(recon_batch, batch_data, mu, std_dev)
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Train the VAE-LSTM model"""
        print("Starting VAE-LSTM training...")

        for epoch in range(epochs):
            if self.es.stop_training:
                break
                
            current_lr = self.base_lr * (0.98 ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            self.beta = self.annealing.get_beta(epoch)
            self.annealing.print_status(epoch)

            # Training
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch, epochs)

            # Validation
            val_loss, val_recon, val_kl = self.validate(val_loader, epoch, epochs)

            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.recon_losses.append(train_recon)
            self.kl_losses.append(train_kl)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - train loss: {train_loss:.6f} - recon_loss: {train_recon:.6f} - kl_loss: {train_kl:.6f}")
            print(f"Epoch {epoch+1}/{epochs} - val loss: {val_loss:.6f} - recon_loss: {val_recon:.6f} - kl_loss: {val_kl:.6f}")
            print(f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.6f}")
            
            # Early Stopping 체크
            self.es(val_loss, self.model, epoch)
            
            # Save model checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(epoch)
        
        # Save final model
        self.save_model(epochs - 1)
        self.plot_training_curves()
    
    def save_model(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses
        }
        
        torch.save(checkpoint, Path(self.config['checkpoint_dir']) / f"vae_checkpoint_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")
    
    def load_model(self, checkpoint_path):
        """Load model checkpoint"""
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.recon_losses = checkpoint.get('recon_losses', [])
            self.kl_losses = checkpoint.get('kl_losses', [])
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train')
        axes[0, 0].plot(self.val_losses, label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(self.recon_losses, label='Train')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL loss
        axes[1, 0].plot(self.kl_losses, label='Train')
        axes[1, 0].set_title('KL Divergence Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.config['result_dir']) / "training_curves.pdf")
        plt.close()