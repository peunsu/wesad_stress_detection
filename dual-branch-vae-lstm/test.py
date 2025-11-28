import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pyplot import plot, ion, show, savefig, cla, figure
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import VAE_LSTM_MA, VAE_LSTM_Linear, VAE_LSTM_Local, MA_VAE
from data_loader import DataGenerator
from utils import process_config, create_dirs
from train import load_latest_vae_checkpoint, set_random_seeds

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # keep GPU ordering consistent
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load VAE model
config = process_config('config.json')
# create the experiments dirs
create_dirs([config['result_dir'], config['checkpoint_dir']])
# create your data generator
data = DataGenerator(config)

seed = config.get('random_seed', 42)
set_random_seeds(seed)

# create a VAE model and load the latest checkpoint
if config['exp_name'] == 'vae-lstm-dual-ma':
    model = VAE_LSTM_MA(config, beta=1e-8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.95))
elif config['exp_name'] == 'vae-lstm-dual-linear':
    model = VAE_LSTM_Linear(config, beta=1e-8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.95))
elif config['exp_name'] == 'vae-lstm-local-only':
    model = VAE_LSTM_Local(config, beta=1e-8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0004, betas=(0.9, 0.95))
elif config['exp_name'] == 'ma-vae':
    model = MA_VAE(config, beta=1e-8).to(device)
    optimizer = optim.Adam(model.parameters(), amsgrad=True)
else:
    raise ValueError(f"Unknown experiment name: {config['exp_name']}")

if not load_latest_vae_checkpoint(model, optimizer, device, config['checkpoint_dir']):
    raise RuntimeError("No checkpoint found to load.")
model.eval()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(data_loader):
    batch_recons_error = []
    
    for batch_data in tqdm(data_loader):
        if isinstance(batch_data, (list, tuple)):
            batch_data = batch_data[0]
        batch_data = batch_data.to(device)
        with torch.no_grad():
            reconstruction = model(batch_data)
            Xhat = reconstruction[0]
                        
            batch_data = batch_data.view(Xhat.size(0), -1, Xhat.size(2), Xhat.size(3))
            batch_data = batch_data[:, 1:, :, :]  # Align batch_data with the number of windows
            
            recon_loss = torch.sum((batch_data - Xhat).pow(2), dim=[1, 2, 3])
            
            batch_recons_error.append(recon_loss.cpu().numpy())
    
    batch_recons_error = np.concatenate(batch_recons_error, axis=0)
    
    return batch_recons_error

def evaluate_model_ma_vae(data_loader):
    batch_recons_error = []
    
    for batch_data in tqdm(data_loader):
        if isinstance(batch_data, (list, tuple)):
            batch_data = batch_data[0]
        batch_data = batch_data.to(device)
        with torch.no_grad():
            reconstruction = model(batch_data)
            Xhat = reconstruction[2]
            
            recon_loss = torch.sum((batch_data - Xhat).pow(2), dim=(1,2))            
            batch_recons_error.append(recon_loss.cpu().numpy())
    
    batch_recons_error = np.concatenate(batch_recons_error, axis=0)
    
    return batch_recons_error

def return_anomaly_idx_by_threshold(test_anomaly_metric, threshold):
    idx_error = np.flatnonzero(test_anomaly_metric > threshold)
    if len(idx_error.shape) == 0:
        idx_error = np.expand_dims(idx_error, 0)
    return idx_error

def augment_detected_idx(idx_detected_anomaly, anomaly_index):
    idx_detected_anomaly_set = set(idx_detected_anomaly)
    idx_detected_anomaly_extended = set(idx_detected_anomaly)
    
    # anomaly_index는 이상 구간(리스트의 리스트)이라고 가정
    for anomaly_win in anomaly_index:
        # 현재 이상 구간(anomaly_win)이 기존 탐지에 포함되었는지 확인
        # set intersection이 빠름
        if idx_detected_anomaly_set.intersection(anomaly_win):
            # 포함되었다면 해당 구간 전체를 확장된 set에 추가
            idx_detected_anomaly_extended.update(anomaly_win)

    # numpy로 변환 후 정렬
    return np.sort(np.array(list(idx_detected_anomaly_extended), dtype=int))

def count_TP_FP_FN(idx_detected_anomaly, anomaly_index, test_labels):  
    detected_labels = test_labels[idx_detected_anomaly]
    n_TP = np.sum(detected_labels == 1)
    n_FP = np.sum(detected_labels == 0)
    n_TN = np.sum(test_labels == 0) - n_FP
    n_FN = np.sum(test_labels == 1) - n_TP
    
    return n_TP, n_TN, n_FP, n_FN, 

def compute_precision_and_recall(idx_detected_anomaly, anomaly_index, test_labels):
    n_TP, n_TN, n_FP, n_FN = count_TP_FP_FN(idx_detected_anomaly, anomaly_index, test_labels)
    
    precision = n_TP / (n_TP + n_FP) if (n_TP + n_FP) > 0 else 1.0
    recall = n_TP / (n_TP + n_FN) if (n_TP + n_FN) > 0 else 0.0
    accuracy = (n_TP + n_TN) / (n_TP + n_FP + n_FN + n_TN) if (n_TP + n_FP + n_FN + n_TN) > 0 else 0.0
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, accuracy, F1, n_TP, n_TN, n_FP, n_FN

def main():
    print(f'The model has {count_parameters(model):,} trainable parameters')

    if config['load_dir'] == 'default' or config['load_dir'] == 'experiments':
        data_dir = Path('../data')
        fold_id = config.get('fold_id', 0)
        test_df = pd.read_csv(data_dir / f'test_fold{fold_id}.csv')
    else:
        raise ValueError("Invalid load_dir in config.json")

    test_subjects = test_df.pop('subject_id')
    test_labels = test_df.pop('label')

    train_loader, val_loader, test_loader = data.get_dataloaders(test=True)

    if config['exp_name'] == 'ma-vae':
        train_score = evaluate_model_ma_vae(train_loader)
    else:
        train_score = evaluate_model(train_loader)
        
    if config['exp_name'] == 'ma-vae':
        val_score = evaluate_model_ma_vae(val_loader)
    else:
        val_score = evaluate_model(val_loader)
        
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    if config['exp_name'] == 'ma-vae':
        test_score = evaluate_model_ma_vae(test_loader)
    else:
        test_score = evaluate_model(test_loader)

    end_event.record()

    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Test Set Inference Time: {elapsed_time * 1e-3} seconds")

    idx_anomaly_test = test_labels[test_labels == 1].index.to_numpy()
    n_test = len(data.test_set)
    test_labels_vae = np.zeros(n_test)
    anomaly_index_vae = []
    for i in tqdm(range(len(idx_anomaly_test))):
        idx_start = (idx_anomaly_test[i] - config['window_size']) // config['window_shift']
        idx_end = idx_anomaly_test[i] // config['window_shift'] + 1
        
        if idx_start < 0:
            idx_start = 0
        
        if idx_end > n_test:
            idx_end = n_test
        
        anomaly_index_vae.append(np.arange(idx_start,idx_end))
        test_labels_vae[idx_start:idx_end] = 1

    percentile_list = np.arange(0, 100.1, 1)
    threshold_list = np.percentile(test_score, percentile_list)

    n_threshold = len(threshold_list)

    precision = np.zeros(n_threshold)
    recall = np.zeros(n_threshold)
    accuracy = np.zeros(n_threshold)
    F1 = np.zeros(n_threshold)

    precision_aug = np.zeros(n_threshold)
    recall_aug = np.zeros(n_threshold)
    accuracy_aug = np.zeros(n_threshold)
    F1_aug = np.zeros(n_threshold)

    for i, threshold in tqdm(enumerate(threshold_list)):
        #print(threshold_list[i])
        idx_detection_vae = return_anomaly_idx_by_threshold(test_score, threshold)
        precision[i], recall[i], accuracy[i], F1[i], _, _, _, _ = compute_precision_and_recall(idx_detection_vae, 
                                                                            anomaly_index_vae, 
                                                                            test_labels_vae)
        # augment the detection using the ground truth labels
        # a method to discount the factor one anomaly appears in multiple consecutive windows
        # introduced in "Unsupervised anomaly detection via variational auto-encoder for seasonal kpis in web applications"
        idx_detection_vae_augmented = augment_detected_idx(idx_detection_vae, anomaly_index_vae)
        precision_aug[i], recall_aug[i], accuracy_aug[i], F1_aug[i], _, _, _, _ = compute_precision_and_recall(idx_detection_vae_augmented, 
                                                                                        anomaly_index_vae, 
                                                                                        test_labels_vae)
        
    # Now select a threshold
    threshold = threshold_list[np.squeeze(np.argmax(F1_aug))]

    print("Threshold is {}".format(threshold))
    idx_detection = return_anomaly_idx_by_threshold(test_score, threshold)
    idx_detection_augmented = augment_detected_idx(idx_detection, anomaly_index_vae)
    precision, recall, accuracy, F1, n_TP, n_TN, n_FP, n_FN = compute_precision_and_recall(idx_detection_augmented, 
                                                                        anomaly_index_vae, 
                                                                        test_labels_vae)
    print("\nPR evaluation using augmented detection:")
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Accuracy: {}".format(accuracy))
    print("F1: {}".format(F1))
    print("TP: {}".format(n_TP))
    print("TN: {}".format(n_TN))
    print("FP: {}".format(n_FP))
    print("FN: {}".format(n_FN))
    
if __name__ == "__main__":
    main()