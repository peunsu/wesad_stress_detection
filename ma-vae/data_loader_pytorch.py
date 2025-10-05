import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.load_dataset()

    # 데이터 셋 정규화 후, 학습에 사용할 dataset 생성
    def load_dataset(self):
        data_dir = Path('../data')
        train_df = pd.read_csv(data_dir / 'train.csv')
        val_df = pd.read_csv(data_dir / 'val.csv')
        test_df = pd.read_csv(data_dir / 'test.csv')
        
        # subject_id, label 분리
        train_subjects = train_df.pop('subject_id')
        val_subjects = val_df.pop('subject_id')
        test_subjects = test_df.pop('subject_id')
        train_df = train_df.drop(columns=['label'])
        val_df = val_df.drop(columns=['label'])
        test_df = test_df.drop(columns=['label'])
        
        # 평균, 표준편차 계산하여 정규화
        train_m = train_df.mean()
        train_std = train_df.std()
        train_df_normalized = (train_df - train_m) / train_std
        val_df_normalized = (val_df - train_m) / train_std
        test_df_normalized = (test_df - train_m) / train_std
        
        # subject_id별로 그룹화해서 딕셔너리 형태로 저장
        data = {
            'training': {
                sid: group.to_numpy()
                for sid, group in train_df_normalized.groupby(train_subjects)
            },
            'validation': {
                sid: group.to_numpy()
                for sid, group in val_df_normalized.groupby(val_subjects)
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
        for mode in ['training', 'validation', 'test']:
            subject_windows = []
            for sid, subject_data in data[mode].items():
                n_sample = len(subject_data)
                n_vae = n_sample - self.config['l_win'] + 1
                if n_vae <= 0:
                    continue
                windows = np.zeros((n_vae, self.config['l_win'], subject_data.shape[1]))
                for i in range(n_vae):
                    windows[i] = subject_data[i:i + self.config['l_win']]
                subject_windows.append(windows)

            if subject_windows:
                rolling_windows_dict[mode] = np.concatenate(subject_windows, axis=0)
            else:
                rolling_windows_dict[mode] = np.zeros((1, self.config['l_win'], list(data[mode].values())[0].shape[1]))

        self.train_set_vae = {'data': rolling_windows_dict['training']}
        self.val_set_vae = {'data': rolling_windows_dict['validation']}
        self.test_set_vae = {'data': rolling_windows_dict['test']}

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
