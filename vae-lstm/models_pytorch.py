import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def depth_to_space_2d(x, block_size):
    """Replicates TensorFlow's depth_to_space for NCHW tensors."""
    b, c, h, w = x.shape
    new_c = c // (block_size ** 2)
    x = x.view(b, new_c, block_size, block_size, h, w)
    # Rearrange to (b, new_c, h * block_size, w * block_size)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, new_c, h * block_size, w * block_size)


class VAEmodel(nn.Module):
    # feature을 따로 처리하지 않고 하나로 합쳐서 input vector로 처리
    def __init__(self, config):
        super(VAEmodel, self).__init__()
        self.config = config
        self.input_dims = config['l_win'] * config['n_channel'] # input data= window length * channel(feature) 수
        self.code_size = config['code_size'] # latent variable 차원
        self.num_hidden_units = config['num_hidden_units'] 
        self.l_win = config['l_win'] # window length
        self.n_channel = config['n_channel'] # input data channel 수
        self.sigma2_offset = config['sigma2_offset'] # variance offset 설정 => 수치적 안정성
        self.train_sigma = (config['TRAIN_sigma'] == 1) # sigma 파라미터 학습 여부

        self._build_encoder()
        self._build_decoder()

        if self.train_sigma:
            self.sigma = nn.Parameter(torch.tensor(config['sigma'], dtype=torch.float32))
        else:
            self.register_buffer('sigma', torch.tensor(config['sigma'], dtype=torch.float32))

    # 4개의 Conv2d layer
    # channel 수는 늘어나고, feature map의 크기는 감소 + 시계열 특성에 맞게 kernel size 조절(직사각형)
    # 시간축에서 여러 시점을 한 번에 보면서 feature 간 상관관계를 효과적으로 추출할
    def _build_encoder(self):
        if self.l_win == 24:
            self._encoder_symmetric_pad = True
            self.enc_conv1 = nn.Conv2d(self.n_channel, self.num_hidden_units // 16,
                                       kernel_size=(3, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
            self.enc_conv2 = nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 8,
                                       kernel_size=(3, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
            self.enc_conv3 = nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 4,
                                       kernel_size=(3, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
            self.enc_conv4 = nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units,
                                       kernel_size=(4, 1),
                                       stride=(1, 1),
                                       padding=0)
        elif self.l_win == 48:
            self._encoder_symmetric_pad = False
            self.enc_conv1 = nn.Conv2d(self.n_channel, self.num_hidden_units // 16,
                                       kernel_size=(3, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
            self.enc_conv2 = nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 8,
                                       kernel_size=(3, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
            self.enc_conv3 = nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 4,
                                       kernel_size=(3, 1),
                                       stride=(2, 1),
                                       padding=(1, 0))
            self.enc_conv4 = nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units,
                                       kernel_size=(6, 1),
                                       stride=(1, 1),
                                       padding=0)
        elif self.l_win == 144:
            self._encoder_symmetric_pad = False
            self.enc_conv1 = nn.Conv2d(self.n_channel, self.num_hidden_units // 16,
                                       kernel_size=(3, 1),
                                       stride=(4, 1),
                                       padding=(1, 0))
            self.enc_conv2 = nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 8,
                                       kernel_size=(3, 1),
                                       stride=(4, 1),
                                       padding=(1, 0))
            self.enc_conv3 = nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 4,
                                       kernel_size=(3, 1),
                                       stride=(3, 1),
                                       padding=(1, 0))
            self.enc_conv4 = nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units,
                                       kernel_size=(3, 1),
                                       stride=(1, 1),
                                       padding=0)
        else:
            raise ValueError(f"Unsupported window length: {self.l_win}")

        self.enc_fc = nn.Linear(self.num_hidden_units, self.code_size * 4)
        self.enc_fc_mean = nn.Linear(self.code_size * 4, self.code_size) # mean vector
        self.enc_fc_std = nn.Linear(self.code_size * 4, self.code_size) # std vector

    def _build_decoder(self):
        self.dec_fc = nn.Linear(self.code_size, self.num_hidden_units)

        if self.l_win == 24:
            self.dec_conv1 = nn.Conv2d(self.num_hidden_units, self.num_hidden_units,
                                       kernel_size=(1, 1), padding=0)
            self.dec_conv2 = nn.Conv2d(self.num_hidden_units // 4, self.num_hidden_units // 4,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_conv3 = nn.Conv2d(self.num_hidden_units // 8, self.num_hidden_units // 8,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_conv4 = nn.Conv2d(self.num_hidden_units // 16, self.num_hidden_units // 16,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_out = nn.Conv2d(16, self.n_channel,
                                      kernel_size=(9, 1), padding=(4, 0))
        elif self.l_win == 48:
            self.dec_conv1 = nn.Conv2d(self.num_hidden_units, 256 * 3,
                                       kernel_size=(1, 1), padding=0)
            self.dec_conv2 = nn.Conv2d(256, 256,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_conv3 = nn.Conv2d(128, 128,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_conv4 = nn.Conv2d(32, 32,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_out = nn.Conv2d(16, self.n_channel,
                                      kernel_size=(5, 1), padding=(2, 0))
        elif self.l_win == 144:
            self.dec_conv1 = nn.Conv2d(self.num_hidden_units, 32 * 27,
                                       kernel_size=(1, 1), padding=0)
            self.dec_conv2 = nn.Conv2d(32 * 9, 32 * 9,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_conv3 = nn.Conv2d(32 * 3, 32 * 3,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_conv4 = nn.Conv2d(24, 24,
                                       kernel_size=(3, 1), padding=(1, 0))
            self.dec_out = nn.Conv2d(6, self.n_channel,
                                      kernel_size=(9, 1), padding=(4, 0))
        else:
            raise ValueError(f"Unsupported window length: {self.l_win}")

    def encode(self, x):        
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = x.view(-1, self.l_win, self.n_channel) # (B, l_win, n_channel)
        x = x.permute(0, 2, 1)  # (B, n_channel, l_win)

        x = x.unsqueeze(-1)  # (B, n_channel, l_win, 1)

        if getattr(self, '_encoder_symmetric_pad', False):
            x = F.pad(x, (0, 0, 4, 4), mode='reflect')

        x = F.leaky_relu(self.enc_conv1(x), negative_slope=0.2) # 0보다 작을 때 기울기 => hyperparameter
        x = F.leaky_relu(self.enc_conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.enc_conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.enc_conv4(x), negative_slope=0.2)

        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.enc_fc(x), negative_slope=0.2)

        mean = self.enc_fc_mean(x)
        std = F.relu(self.enc_fc_std(x)) + 1e-2

        return mean, std

    def reparameterize(self, mean, std):
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        x = F.leaky_relu(self.dec_fc(z), negative_slope=0.2)
        x = x.view(-1, self.num_hidden_units, 1, 1)

        if self.l_win == 24:
            x = F.leaky_relu(self.dec_conv1(x), negative_slope=0.2)
            b = x.size(0)
            x = x.view(b, self.num_hidden_units // 4, 4, 1)

            x = F.leaky_relu(self.dec_conv2(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, self.num_hidden_units // 8, 8, 1)

            x = F.leaky_relu(self.dec_conv3(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, self.num_hidden_units // 16, 16, 1)

            x = F.leaky_relu(self.dec_conv4(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 16, self.l_win, 1)

            x = self.dec_out(x)
        elif self.l_win == 48:
            x = F.leaky_relu(self.dec_conv1(x), negative_slope=0.2)
            b = x.size(0)
            x = x.view(b, 256, 3, 1)

            x = F.leaky_relu(self.dec_conv2(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 128, 6, 1)

            x = F.leaky_relu(self.dec_conv3(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 32, 24, 1)

            x = F.leaky_relu(self.dec_conv4(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 16, self.l_win, 1)

            x = self.dec_out(x)
        elif self.l_win == 144:
            x = F.leaky_relu(self.dec_conv1(x), negative_slope=0.2)
            b = x.size(0)
            x = x.view(b, 32 * 9, 3, 1)

            x = F.leaky_relu(self.dec_conv2(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 3)
            x = x.view(b, 32 * 3, 9, 1)

            x = F.leaky_relu(self.dec_conv3(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 24, 36, 1)

            x = F.leaky_relu(self.dec_conv4(x), negative_slope=0.2)
            x = depth_to_space_2d(x, 2)
            x = x.view(b, 6, self.l_win, 1)

            x = self.dec_out(x)
        else:
            raise ValueError(f"Unsupported window length: {self.l_win}")

        # Convert back to (batch, l_win, n_channel)
        x = x.squeeze(-1)  # remove width dim
        x = x.permute(0, 2, 1)  # (B, l_win, n_channel))
        return x

    def forward(self, x):
        mean, std = self.encode(x)
        z = self.reparameterize(mean, std)
        recon = self.decode(z)
        return recon, mean, std

    # 학습 중 VAE의 분산값에 안정성과 유연성을 추가
    def get_sigma2(self):
        sigma2 = torch.square(self.sigma)
        if self.train_sigma:
            sigma2 = sigma2 + self.sigma2_offset
        return sigma2


class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.config = config
        self.code_size = config['code_size']
        self.num_hidden_units_lstm = config['num_hidden_units_lstm']

        # lstm layer 3개
        self.lstm1 = nn.LSTM(self.code_size, self.num_hidden_units_lstm, batch_first=True)
        self.lstm2 = nn.LSTM(self.num_hidden_units_lstm, self.num_hidden_units_lstm, batch_first=True)
        self.lstm3 = nn.LSTM(self.num_hidden_units_lstm, self.code_size, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        return x


class LSTMTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate_lstm'])
        self.criterion = nn.MSELoss()

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        if val_loader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x)
                predictions.append(output.cpu().numpy())
        return np.concatenate(predictions, axis=0)
