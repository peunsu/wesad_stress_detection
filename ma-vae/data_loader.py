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
        n_train = int(np.floor((n_win * 0.8)))
        n_val = n_win - n_train
        idx_train = random.sample(range(n_win), n_train)
        idx_val = list(set(idx_train) ^ set(range(n_win)))
        return idx_train, idx_val, n_train, n_val

    # 데이터 셋 정규화 후, 학습에 사용할 dataset 생성
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
    
    def _create_vae_sets(self, data):
        window_size = self.config['window_size']
        window_shift = self.config['window_shift']
        
        rolling_windows_dict = {}
        for mode in ['training', 'test']:
            subject_windows = []
            for sid, subject_data in data[mode].items():
                n_sample = subject_data.shape[0]
                n_vae = int((n_sample - window_size) / window_shift + 1)
                if n_vae <= 0:
                    continue
                windows = np.zeros((n_vae, window_size, subject_data.shape[1]))
                for window in range(n_vae):
                    for feature in range(subject_data.shape[1]):
                        windows[window, :, feature] = subject_data[window * window_shift:window_size + window * window_shift, feature]
                subject_windows.append(windows)

            if subject_windows:
                rolling_windows_dict[mode] = np.concatenate(subject_windows, axis=0)
            else:
                rolling_windows_dict[mode] = np.zeros((1, window_size, list(data[mode].values())[0].shape[1]))

        self.idx_train, self.idx_val, n_train, n_val = self.separate_train_and_val_set(rolling_windows_dict['training'].shape[0])

        self.train_set_vae = {'data': rolling_windows_dict['training'][self.idx_train]}
        self.val_set_vae = {'data': rolling_windows_dict['training'][self.idx_val]}
        self.test_set_vae = {'data': rolling_windows_dict['test']}

    def get_vae_datasets(self):
        return self.train_set_vae['data'], self.val_set_vae['data']

    def get_vae_dataloaders(self):
        batch_size = self.config['batch_size']
        
        train_data = torch.from_numpy(self.train_set_vae['data']).float()
        val_data = torch.from_numpy(self.val_set_vae['data']).float()

        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader
