import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.load_dataset()

    # 데이터 셋 정규화 후, VAE, LSTM 학습에 사용할 dataset 생성
    def load_dataset(self):
        data_dir = Path('../data')
        train_df = pd.read_csv(data_dir / 'train.csv').drop(columns=['subject_id', 'label'])
        val_df = pd.read_csv(data_dir / 'val.csv').drop(columns=['subject_id', 'label'])
        test_df = pd.read_csv(data_dir / 'test.csv').drop(columns=['subject_id', 'label'])
        
        # 평균, 표준편차 계산하여 정규화
        train_m = train_df.mean()
        train_std = train_df.std()
        train_df_normalized = (train_df - train_m) / train_std
        val_df_normalized = (val_df - train_m) / train_std
        test_df_normalized = (test_df - train_m) / train_std
        
        data = {
            'training': train_df_normalized.to_numpy(),
            'validation': val_df_normalized.to_numpy(),
            'test': test_df_normalized.to_numpy()
        }

        self._create_vae_sets(data)
        self._create_lstm_sets(data)

    def _create_vae_sets(self, data):
        rolling_windows_dict = {}
        for mode in ['training', 'validation', 'test']:
            n_sample = len(data[mode])
            n_vae = n_sample - self.config['l_win'] + 1
            # 연속된 구간별로 데이터를 잘라 VAE 입력을 위한 빈 배열 생성 (l_win: 각 window의 timestep 개수 => hyperparameter)
            rolling_windows = np.zeros((n_vae, self.config['l_win'], data[mode].shape[1]))

            # overlap 존재! => 윈도우가 한칸씩 이동하면서 생성
            for i in range(n_vae):
                rolling_windows[i] = data[mode][i:i + self.config['l_win']]

            rolling_windows_dict[mode] = rolling_windows

        self.train_set_vae = {'data': rolling_windows_dict['training']}
        self.val_set_vae = {'data': rolling_windows_dict['validation']}
        self.test_set_vae = {'data': rolling_windows_dict['test']}

    def _create_lstm_sets(self, data):
        l_win = self.config['l_win']
        l_seq = self.config['l_seq']

        lstm_sequences_dict = {}
        for mode in ['training', 'validation', 'test']:
            n_sample = len(data[mode])
            lstm_sequences = None

            for k in range(l_win):
                # VAE는 겹치도록 sliding window 생성, LSTM은 겹치지 않는 sliding sequence 생성
                n_not_overlap_wins = (n_sample - k) // l_win
                n_lstm = n_not_overlap_wins - l_seq + 1
                if n_lstm <= 0:
                    continue

                cur_seq = np.zeros((n_lstm, l_seq, l_win, data[mode].shape[1]))
                for i in range(n_lstm):
                    for j in range(l_seq):
                        start = k + l_win * (j + i)
                        end = start + l_win
                        cur_seq[i, j] = data[mode][start:end]

                if lstm_sequences is None:
                    lstm_sequences = cur_seq
                else:
                    lstm_sequences = np.concatenate((lstm_sequences, cur_seq), axis=0)

            if lstm_sequences is None:
                lstm_sequences = np.zeros((1, l_seq, l_win, data[mode].shape[1]))

            lstm_sequences_dict[mode] = lstm_sequences

        self.train_set_lstm = {'data': lstm_sequences_dict['training']}
        self.val_set_lstm = {'data': lstm_sequences_dict['validation']}
        self.test_set_lstm = {'data': lstm_sequences_dict['test']}

    def get_vae_datasets(self):
        return self.train_set_vae['data'], self.val_set_vae['data']

    def get_vae_dataloaders(self, batch_size):
        train_data = torch.from_numpy(self.train_set_vae['data']).float()
        val_data = torch.from_numpy(self.val_set_vae['data']).float()

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
