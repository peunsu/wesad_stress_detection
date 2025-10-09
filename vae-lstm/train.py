import random
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from models import VAEmodel, LSTMModel
from data_loader import DataGenerator
from trainers import VAETrainer, LSTMTrainer
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


# VAE 모델을 사용해서 입력된 sequence를 latent space embedding(mu vector)로 변환
def generate_lstm_embeddings(model, sequences, device):
    model.eval()
    l_seq = sequences.shape[1] # sequence 길이
    code_size = model.code_size # latent space embedding의 차원
    embeddings = np.zeros((len(sequences), l_seq, code_size), dtype=np.float32) # 결과 저장할 numpy 배열

    with torch.no_grad():
        for idx in range(len(sequences)):
            batch = torch.from_numpy(sequences[idx]).float().to(device)  # (l_seq, l_win, n_channel) => sequece 데이터 tensor로 변환
            #batch = batch.squeeze(-1)  # (l_seq, l_win)
            recon, mu, _ = model(batch)
            embeddings[idx] = mu.cpu().numpy()  # mu 벡터를 numpy 배열로 변환하여 저장

    return embeddings


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

    vae_model = VAEmodel(config).to(device)
    vae_trainer = VAETrainer(vae_model, data, config)
    train_loader, val_loader = data.get_vae_dataloaders(config['batch_size'])

    # config 확인해서 VAE 모델 학습 or 불러오기
    if config['TRAIN_VAE'] and config['num_epochs_vae'] > 0:
        vae_trainer.train(train_loader, val_loader, config['num_epochs_vae'])
    else:
        if not load_latest_vae_checkpoint(vae_trainer, config['checkpoint_dir']):
            raise RuntimeError("VAE training disabled and no checkpoint found to load.")

    # config 확인해서 LSTM 모델 학습 or 불러오기
    if config['TRAIN_LSTM'] and config['num_epochs_lstm'] > 0:
        if config['TRAIN_VAE']:
            # ensure we reuse trained weights without reinitialising
            load_latest_vae_checkpoint(vae_trainer, config['checkpoint_dir'])
        vae_model.eval()

        print("Generating embeddings for LSTM training...")
        train_sequences = data.train_set_lstm['data']
        val_sequences = data.val_set_lstm['data']

        train_embeddings = generate_lstm_embeddings(vae_model, train_sequences, device)
        val_embeddings = generate_lstm_embeddings(vae_model, val_sequences, device)

        x_train = train_embeddings[:, :-1]
        y_train = train_embeddings[:, 1:]
        x_val = val_embeddings[:, :-1]
        y_val = val_embeddings[:, 1:]

        # data를 tensor로 변환
        train_dataset = TensorDataset(torch.from_numpy(x_train).float(),
                                       torch.from_numpy(y_train).float())
        val_dataset = TensorDataset(torch.from_numpy(x_val).float(),
                                     torch.from_numpy(y_val).float())

        # tensor data를 batch단위로 학습을 가능하게 준비
        train_lstm_loader = DataLoader(train_dataset, batch_size=config['batch_size_lstm'], shuffle=True)
        val_lstm_loader = DataLoader(val_dataset, batch_size=config['batch_size_lstm'], shuffle=False)

        # LSTM 모델 생성
        lstm_model = LSTMModel(config).to(device)
        lstm_trainer = LSTMTrainer(lstm_model, config)

        # LSTM 모델 있으면 불러오기
        lstm_checkpoint_path = Path(config['checkpoint_dir_lstm']) / 'lstm_model.pth'
        if lstm_checkpoint_path.is_file():
            lstm_model.load_state_dict(torch.load(lstm_checkpoint_path, map_location=device))
            print("Loaded existing LSTM checkpoint.")

        # LSTM 모델 학습
        lstm_trainer.train(train_lstm_loader, val_lstm_loader, config['num_epochs_lstm'])
        torch.save(lstm_model.state_dict(), lstm_checkpoint_path)
        print("Saved LSTM checkpoint.")

        lstm_model.eval()
        with torch.no_grad():
            lstm_predictions = []
            for batch_x, _ in train_lstm_loader:
                batch_x = batch_x.to(device)
                preds = lstm_model(batch_x)
                lstm_predictions.append(preds.cpu().numpy())
            lstm_predictions = np.concatenate(lstm_predictions, axis=0)

        print("=" * 50)
        print("Training completed successfully!")
        print(f"LSTM predictions shape: {lstm_predictions.shape}")
        print("=" * 50)


if __name__ == '__main__':
    main()
