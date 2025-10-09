import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
import torch.optim as optim

from data_loader import DataGenerator
from models import VAE_Encoder, VAE_Decoder, SISVAE, EarlyStopping
from utils import process_config, create_dirs, get_args, save_config

# 시드 설정
def set_random_seeds(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# 학습된 VAE 모델의 pth 파일을 trainer에 로드하는 역할
def load_latest_vae_checkpoint(model, optimizer, device, checkpoint_dir):
    if not Path(checkpoint_dir).is_dir():
        return False

    checkpoint_files = [f for f in Path(checkpoint_dir).iterdir() if f.name.startswith('vae_checkpoint') and f.name.endswith('.pth')]
    if not checkpoint_files:
        return False

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded VAE checkpoint: {latest_checkpoint}")
    return True

def save_model(model, optimizer, epoch, history, config):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': history['loss'],
        'val_losses': history['val_recon_loss'],
        'recon_losses': history['recon_loss'],
        'kl_losses': history['kl_loss'],
        'smooth_losses': history['smooth_loss']
    }

    torch.save(checkpoint, f"{config['checkpoint_dir']}/vae_checkpoint_epoch_{epoch+1}.pth")
    print(f"Model saved at epoch {epoch+1}")

def plot_training_curves(history, config):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Total loss
    axes[0, 0].plot(history['loss'], label='Train')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Reconstruction loss
    axes[0, 1].plot(history['recon_loss'], label='Train')
    axes[0, 1].plot(history['val_recon_loss'], label='Validation')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # KL loss
    axes[1, 0].plot(history['kl_loss'], label='Train')
    axes[1, 0].set_title('KL Divergence Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{config['result_dir']}/training_curves.pdf")
    plt.close()

def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    
    create_dirs([config['result_dir'], config['checkpoint_dir'], config['checkpoint_dir_lstm']])
    save_config(config)
    
    seed = config.get('seed', 42)
    set_random_seeds(seed)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 더미 데이터셋 및 DataLoader (LOAD_PATH 대체)
    # 원본 코드는 unbatch().batch(512)를 사용하지만, 여기서는 직접 DataLoader 사용
    data = DataGenerator(config)
    train_loader, val_loader = data.get_vae_dataloaders()

    # 모델 인스턴스화
    encoder = VAE_Encoder(config).to(device)
    decoder = VAE_Decoder(config).to(device)
    model = SISVAE(encoder, decoder).to(device)

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), amsgrad=True)

    # 콜백 설정 (PyTorch 버전)
    es = EarlyStopping(
        monitor='val_recon_loss', # Keras의 val_log_probs_loss는 NLL_loss와 동일
        mode='min',
        patience=config['patience'],
        restore_best_weights=True,
        verbose=1
    )
    
    # config 확인해서 VAE 모델 학습 or 불러오기
    if config['TRAIN_VAE'] and config['epochs'] > 0:
        pass
    else:
        if not load_latest_vae_checkpoint(model, optimizer, device, config['checkpoint_dir']):
            raise RuntimeError("VAE training disabled and no checkpoint found to load.")

    # --------------------------------------------------------------------------------
    # 5. 훈련 실행
    # --------------------------------------------------------------------------------

    print(f"Starting training on device: {device}")
    history = {'loss': [], 'recon_loss': [], 'kl_loss': [], 'val_recon_loss': [], 'smooth_loss': []}

    for epoch in range(config['epochs']):
        if es.stop_training:
            break
        
        # 훈련 단계
        train_losses = []

        for X in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} (Train)"):
            if isinstance(X, (list, tuple)):
                X = X[0]
            X = X.to(device)
            total_loss, recon_loss, kl_loss, smooth_loss = model.training_step(X, optimizer)
            train_losses.append({
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'smooth_loss': smooth_loss
            })
        
        # 훈련 손실 평균
        avg_total_loss = np.mean([l['total_loss'] for l in train_losses])
        avg_recon_loss = np.mean([l['recon_loss'] for l in train_losses])
        avg_kl_loss = np.mean([l['kl_loss'] for l in train_losses])
        avg_smooth_loss = np.mean([l['smooth_loss'] for l in train_losses])
        
        # 검증 단계
        val_losses = []
        for X_val in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} (Val)"):
            if isinstance(X_val, (list, tuple)):
                X_val = X_val[0]
            X_val = X_val.to(device)
            total_loss, recon_loss, kl_loss, smooth_loss = model.validation_step(X_val)
            val_losses.append({
                'total_loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'smooth_loss': smooth_loss
            })
            
        # 검증 손실 평균
        avg_val_recon_loss = np.mean([l['recon_loss'] for l in val_losses])
        
        # 기록
        history['loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['smooth_loss'].append(avg_smooth_loss)
        history['val_recon_loss'].append(avg_val_recon_loss)

        # 출력
        print(f"Epoch {epoch+1}/{config['epochs']} - loss: {avg_total_loss:.6f} - recon_loss: {avg_recon_loss:.6f} - kl_loss: {avg_kl_loss:.6f} - smooth_loss: {avg_smooth_loss:.6f} - val_recon_loss: {avg_val_recon_loss:.6f}")

        # Early Stopping 체크
        es(avg_val_recon_loss, model, epoch)

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            save_model(model, optimizer, epoch, history, config)

    # Save final model
    save_model(model, optimizer, epoch - 1, history, config)
    plot_training_curves(history, config)
    
    print("=" * 50)
    print("Training completed successfully!")
    print("=" * 50)
    
if __name__ == "__main__":
    main()