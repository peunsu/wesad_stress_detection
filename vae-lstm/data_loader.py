import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.load_dataset()
    
    def separate_train_and_val_set(self, n_win):
        n_train = int(np.floor((n_win * 0.9)))
        n_val = n_win - n_train
        idx_train = random.sample(range(n_win), n_train)
        idx_val = list(set(idx_train) ^ set(range(n_win)))
        return idx_train, idx_val, n_train, n_val

    # 데이터 셋 정규화 후, VAE, LSTM 학습에 사용할 dataset 생성
    def load_dataset(self):
        data_dir = Path('../data')
        train_df = pd.read_csv(data_dir / 'train.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        
        # subject_id, label 분리
        train_subjects = train_df.pop('subject_id')
        test_subjects = test_df.pop('subject_id')
        
        train_df = train_df.drop(columns=['label'])
        test_df = test_df.drop(columns=['label'])
        
        # float 컬럼만 선택
        float_cols = train_df.select_dtypes(include='float').columns

        # 평균과 표준편차 계산
        train_m = train_df[float_cols].mean()
        train_std = train_df[float_cols].std()

        # float 컬럼만 정규화
        train_df_normalized = train_df.copy()
        train_df_normalized[float_cols] = (train_df[float_cols] - train_m) / train_std

        test_df_normalized = test_df.copy()
        test_df_normalized[float_cols] = (test_df[float_cols] - train_m) / train_std
        
        # subject_id별로 그룹화해서 딕셔너리 형태로 저장
        data = {
            'training': {
                sid: group.to_numpy()
                for sid, group in train_df_normalized.groupby(train_subjects)
            },
            'test': {
                sid: group.to_numpy()
                for sid, group in test_df_normalized.groupby(test_subjects)
            }
        }

        self._create_vae_sets(data)
        self._create_lstm_sets(data)

    def _create_vae_sets(self, data):
        rolling_windows_dict = {}
        for mode in ['training', 'test']:
            subject_windows = []
            
            if mode == 'training':
                for sid, subject_data in data[mode].items():
                    n_sample = len(subject_data)
                    n_vae = n_sample - self.config['l_win'] + 1
                    if n_vae <= 0:
                        continue
                    windows = np.zeros((n_vae, self.config['l_win'], subject_data.shape[1]), dtype=np.float16)
                    for i in range(n_vae):
                        windows[i] = subject_data[i:i + self.config['l_win']]
                    subject_windows.append(windows)
            elif mode == 'test':
                data_all = np.concatenate(list(data[mode].values()), axis=0)
                n_sample = len(data_all)
                n_vae = n_sample - self.config['l_win'] + 1
                if n_vae <= 0:
                    continue
                windows = np.zeros((n_vae, self.config['l_win'], data_all.shape[1]), dtype=np.float16)
                for i in range(n_vae):
                    windows[i] = data_all[i:i + self.config['l_win']]
                subject_windows.append(windows)

            if subject_windows:
                rolling_windows_dict[mode] = np.concatenate(subject_windows, axis=0, dtype=np.float16)
            else:
                rolling_windows_dict[mode] = np.zeros((1, self.config['l_win'], list(data[mode].values())[0].shape[1]), dtype=np.float16)
        

        self.idx_train, self.idx_val, n_train, n_val = self.separate_train_and_val_set(rolling_windows_dict['training'].shape[0])

        self.train_set_vae = {'data': rolling_windows_dict['training'][self.idx_train]}
        self.val_set_vae = {'data': rolling_windows_dict['training'][self.idx_val]}
        self.test_set_vae = {'data': rolling_windows_dict['test']}

    def _create_lstm_sets(self, data):
        l_win = self.config['l_win']
        l_seq = self.config['l_seq']

        lstm_sequences_dict = {}
        for mode in ['training', 'test']:
            subject_sequences = []
            if mode == 'training':
                for sid, subject_data in data[mode].items():
                    n_sample = len(subject_data)
                    lstm_sequences = []

                    for k in range(l_win):
                        n_not_overlap_wins = (n_sample - k) // l_win
                        n_lstm = n_not_overlap_wins - l_seq + 1
                        if n_lstm <= 0:
                            continue

                        cur_seq = np.zeros((n_lstm, l_seq, l_win, subject_data.shape[1]), dtype=np.float16)
                        for i in range(n_lstm):
                            for j in range(l_seq):
                                start = k + l_win * (j + i)
                                end = start + l_win
                                cur_seq[i, j] = subject_data[start:end]

                        lstm_sequences.append(cur_seq)

                    if lstm_sequences:
                        subject_sequences.append(np.concatenate(lstm_sequences, axis=0, dtype=np.float16))
            elif mode == 'test':
                data_all = np.concatenate(list(data[mode].values()), axis=0)
                n_sample = len(data_all)
                lstm_sequences = []

                for k in range(l_win):
                    n_not_overlap_wins = (n_sample - k) // l_win
                    n_lstm = n_not_overlap_wins - l_seq + 1
                    if n_lstm <= 0:
                        continue

                    cur_seq = np.zeros((n_lstm, l_seq, l_win, data_all.shape[1]), dtype=np.float16)
                    for i in range(n_lstm):
                        for j in range(l_seq):
                            start = k + l_win * (j + i)
                            end = start + l_win
                            cur_seq[i, j] = data_all[start:end]

                    lstm_sequences.append(cur_seq)

                if lstm_sequences:
                    subject_sequences.append(np.concatenate(lstm_sequences, axis=0, dtype=np.float16))

            if subject_sequences:
                lstm_sequences_dict[mode] = np.concatenate(subject_sequences, axis=0, dtype=np.float16)
            else:
                lstm_sequences_dict[mode] = np.zeros((1, l_seq, l_win, list(data[mode].values())[0].shape[1]), dtype=np.float16)

        self.idx_train, self.idx_val, n_train, n_val = self.separate_train_and_val_set(lstm_sequences_dict['training'].shape[0])

        self.train_set_lstm = {'data': lstm_sequences_dict['training'][self.idx_train]}
        self.val_set_lstm = {'data': lstm_sequences_dict['training'][self.idx_val]}
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
    
    def get_embeddings_datasets(self):
        return self.train_set_lstm['data'], self.val_set_lstm['data']
    
    def get_embeddings_dataloaders(self, batch_size):
        train_data = torch.from_numpy(self.train_set_lstm['data']).float()
        val_data = torch.from_numpy(self.val_set_lstm['data']).float()

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader