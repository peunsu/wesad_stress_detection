import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm


class VAETrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.base_lr = config['learning_rate_vae']
        self.optimizer = optim.Adam(model.parameters(), lr=self.base_lr, betas=(0.9, 0.95))
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.kl_losses = []
        self.recon_losses = []
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['result_dir'], exist_ok=True)
    
    def vae_loss(self, recon_x, x, mu, std_dev, sigma2):
        """Calculate VAE loss (ELBO) - exactly matching TensorFlow version"""
        # KL divergence loss - analytical result (matching TensorFlow)
        # KL divergence loss = 0.5 * (mean^2 + std^2 - log(std^2) - code_size) => 수업 시간 코드에 나와있는 코드와 동일
        kl_loss = 0.5 * torch.mean(
            torch.sum(mu.pow(2), dim=1) +
            torch.sum(std_dev.pow(2), dim=1) -
            torch.sum(torch.log(std_dev.pow(2)), dim=1) -
            self.config['code_size']
        )
        
        # Weighted reconstruction error (matching TensorFlow)
        # MSE = (x - recon_x)^2 => 수업 시간 코드에 나와있는 코드와 동일
        weighted_reconstruction_error = torch.mean(
            torch.sum((x - recon_x).pow(2), dim=[1, 2])
        ) / (2 * sigma2)
        
        # Sigma regularizer (matching TensorFlow)
        # -D/2 * log(sigma^2*2*pi)
        sigma_regularizer = self.model.input_dims / 2 * torch.log(sigma2) #  정규분포에서의 분산값을 정규화하기 위한 항
        two_pi = self.model.input_dims / 2 * torch.tensor(2 * np.pi, device=sigma2.device)
        
        # ELBO loss (matching TensorFlow exactly)
        # 일반적으로 VAE의 ELBO는 reconstruction error / (2*sigma^2) + KL loss 형태 
        # 여기서는 TensorFlow 코드와 맞추기 위해 two_pi와 sigma_regularizer 항을 추가했음.
        # 하지만 논문이나 일반적인 구현에서는 two_pi, sigma_regularizer 항은 보통 포함하지 않음.
        elbo_loss = two_pi + sigma_regularizer + 0.5 * weighted_reconstruction_error + kl_loss
        
        return elbo_loss, weighted_reconstruction_error, kl_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_data in tqdm(train_loader, desc="Training VAE"):
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]

            batch_data = batch_data.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, std_dev = self.model(batch_data)

            # Calculate loss
            sigma2 = self.model.get_sigma2()
            loss, recon_loss, kl_loss = self.vae_loss(recon_batch, batch_data, mu, std_dev, sigma2)

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
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                recon_batch, mu, std_dev = self.model(batch_data)
                
                # Calculate loss
                sigma2 = self.model.get_sigma2()
                loss, recon_loss, kl_loss = self.vae_loss(recon_batch, batch_data, mu, std_dev, sigma2)
                
                # Accumulate losses
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_kl_loss = total_kl_loss / len(val_loader)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def train(self, train_loader, val_loader, epochs):
        """Train the VAE model"""
        print("Starting VAE training...")

        for epoch in range(epochs):
            current_lr = self.base_lr * (0.98 ** epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

            # Training
            train_loss, train_recon, train_kl = self.train_epoch(train_loader)

            # Validation
            val_loss, val_recon, val_kl = self.validate(val_loader)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.recon_losses.append(train_recon)
            self.kl_losses.append(train_kl)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
            print(f"  Sigma2: {self.model.get_sigma2().item():.4f}")
            print("-" * 50)
            
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
        
        torch.save(checkpoint, f"{self.config['checkpoint_dir']}/vae_checkpoint_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")
    
    def load_model(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
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
        
        # Sigma2 over time
        sigma2_values = [self.model.get_sigma2().item()] * len(self.train_losses)
        axes[1, 1].plot(sigma2_values)
        axes[1, 1].set_title('Sigma2 Parameter')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Sigma2')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['result_dir']}/training_curves.pdf")
        plt.close()
    
    def generate_embeddings(self, data_loader):
        """Generate embeddings for LSTM training"""
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                
                batch_data = batch_data.to(self.device)
                mu, _ = self.model.encode(batch_data)
                embeddings.append(mu.cpu().numpy())
        
        return np.concatenate(embeddings, axis=0)
    
    def reconstruct_sequence(self, embeddings):
        """Reconstruct sequence from embeddings"""
        self.model.eval()
        reconstructions = []
        
        with torch.no_grad():
            for embedding in embeddings:
                embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
                recon = self.model.decode(embedding_tensor)
                reconstructions.append(recon.cpu().numpy())
        
        return np.concatenate(reconstructions, axis=0)
