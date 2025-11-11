import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split

class VAEDataset(Dataset):
    def __init__(self, data, l_win):
        self.data = data
        self.l_win = l_win
        
        self.indices = []
        for sid, subject_data in self.data.items():
            n_sample = len(subject_data)
            n_vae = n_sample - l_win + 1
            for i in range(n_vae):
                self.indices.append((sid, i))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sid, i = self.indices[idx]
        sample = self.data[sid][i:i+self.l_win]
        return torch.from_numpy(sample).float()

class LSTMDataset(Dataset):
    def __init__(self, data, l_win, l_seq):
        self.data = data
        self.l_win = l_win
        self.l_seq = l_seq
        
        self.indices = []
        for sid, subject_data in self.data.items():
            n_sample = len(subject_data)
            for k in range(l_win):
                n_not_overlap_wins = (n_sample - k) // l_win
                n_lstm = n_not_overlap_wins - l_seq + 1
                for i in range(n_lstm):
                    self.indices.append((sid, i, k))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        sid, i, k = self.indices[idx]
        sample = np.zeros((self.l_seq, self.l_win, self.data[sid].shape[1]), dtype=np.float32)
        for j in range(self.l_seq):
            start = k + self.l_win * (j + i)
            end = start + self.l_win
            sample[j] = self.data[sid][start:end]
        return torch.from_numpy(sample).float()
    
class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.load_dataset()
        self.create_vae_sets()
        self.create_lstm_sets()
        
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
        self.data = {
            'training': {
                sid: group.to_numpy()
                for sid, group in train_df_normalized.groupby(train_subjects)
            },
            'test': {
                'test_sid': test_df_normalized.to_numpy() # Test set은 subject_id를 구분하지 않고 모두 합쳐서 사용
            }
        }

    def create_vae_sets(self):
        dataset = VAEDataset(self.data['training'], self.config['l_win'])
        n_total = len(dataset)
        n_val = int(n_total * 0.1)
        n_train = n_total - n_val
        self.train_set_vae, self.val_set_vae = random_split(dataset, [n_train, n_val])
        self.test_set_vae = VAEDataset(self.data['test'], self.config['l_win'])
    
    def create_lstm_sets(self):
        dataset = LSTMDataset(self.data['training'], self.config['l_win'], self.config['l_seq'])
        n_total = len(dataset)
        n_val = int(n_total * 0.1)
        n_train = n_total - n_val
        self.train_set_lstm, self.val_set_lstm = random_split(dataset, [n_train, n_val])
        self.test_set_lstm = LSTMDataset(self.data['test'], self.config['l_win'], self.config['l_seq'])
    
    def get_vae_dataloaders(self, test=False):
        train_loader = DataLoader(self.train_set_vae, batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(self.val_set_vae, batch_size=self.config['batch_size'], shuffle=False)
        
        if test:
            test_loader = DataLoader(self.test_set_vae, batch_size=self.config['batch_size'], shuffle=False)
            return train_loader, val_loader, test_loader
        
        return train_loader, val_loader
    
    def get_lstm_dataloaders(self, test=False):
        train_loader = DataLoader(self.train_set_lstm, batch_size=self.config['batch_size_lstm'], shuffle=True)
        val_loader = DataLoader(self.val_set_lstm, batch_size=self.config['batch_size_lstm'], shuffle=False)
        
        if test:
            test_loader = DataLoader(self.test_set_lstm, batch_size=self.config['batch_size_lstm'], shuffle=False)
            return train_loader, val_loader, test_loader
        
        return train_loader, val_loader