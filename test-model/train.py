import random
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset


from models import VAEmodel, LSTMModel, VAE_LSTM_Model
from data_loader import DataGenerator
from trainers import VAELSTMTrainer
from utils import process_config, create_dirs, get_args, save_config


def set_random_seeds(seed_value): #random seed 고정 => 매번 같은 값
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# 학습된 VAE 모델의 pth 파일을 trainer에 로드하는 역할
def load_latest_vae_checkpoint(trainer, checkpoint_dir):
    if not Path(checkpoint_dir).is_dir():
        return False

    checkpoint_files = [f for f in Path(checkpoint_dir).iterdir() if f.name.startswith('vae_checkpoint') and f.name.endswith('.pth')]
    if not checkpoint_files:
        return False

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
    trainer.load_model(Path(checkpoint_dir) / latest_checkpoint)
    print(f"Loaded VAE checkpoint: {latest_checkpoint}")
    return True

def load_latest_lstm_checkpoint(trainer, checkpoint_dir):
    if not Path(checkpoint_dir).is_dir():
        return False

    checkpoint_files = [f for f in Path(checkpoint_dir).iterdir() if f.name.startswith('lstm_checkpoint') and f.name.endswith('.pth')]
    if not checkpoint_files:
        return False

    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
    trainer.load_model(Path(checkpoint_dir) / latest_checkpoint)
    print(f"Loaded LSTM checkpoint: {latest_checkpoint}")
    return True

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
    set_random_seeds(seed) # seed 고정 => 같은 실험값 나오도록
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data = DataGenerator(config)

    vae_model = VAE_LSTM_Model(config).to(device)
    vae_trainer = VAELSTMTrainer(vae_model, data, config)
    train_loader, val_loader = data.get_lstm_dataloaders()

    # config 확인해서 VAE 모델 학습 or 불러오기
    if config['TRAIN_VAE'] and config['num_epochs_vae'] > 0:
        vae_trainer.train(train_loader, val_loader, config['num_epochs_vae'])
    else:
        if not load_latest_vae_checkpoint(vae_trainer, config['checkpoint_dir']):
            raise RuntimeError("VAE training disabled and no checkpoint found to load.")

    vae_model.eval()
    with torch.no_grad():
        lstm_predictions = []
        for batch_x in train_loader:
            if isinstance(batch_x, (list, tuple)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(device)
            preds, _, _ = vae_model(batch_x)
            lstm_predictions.append(preds.cpu().numpy())
        lstm_predictions = np.concatenate(lstm_predictions, axis=0)

    print("=" * 50)
    print("Training completed successfully!")
    print(f"LSTM predictions shape: {lstm_predictions.shape}")
    print("=" * 50)


if __name__ == '__main__':
    main()
