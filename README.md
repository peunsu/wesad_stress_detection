## WESAD Stress Detection
### Dual-Branch VAE-LSTM for Anomaly Detection
- Create and activate virtual environment, then install dependencies. You can skip this step if you are running in Google Colab or other environments where packages are pre-installed.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

- Configure model and training parameters in `/dual-branch-vae-lstm/config.json`

```json
{
    "exp_name": "vae-lstm-dual-ma", // Options: "vae-lstm-dual-ma", "vae-lstm-dual-linear", "vae-lstm-local-only", "ma-vae".
    "load_dir": "default",
    "window_size": 288, // Window length. Options: 576, 288, 144, 72.
    "small_window_size": 48, // Segment length for each window. Options: 48, 24.
    "window_shift": 1, // Window shift.
    "features": 67, // Number of input features.
    "hidden_dim": 512, // Hidden dimension for Encoder/Decoder.
    "hidden_dim_lstm": 64, // Hidden dimension for LSTM layers.
    "latent_dim": 12, // Latent dimension.
    "batch_size": 512, // Batch size.
    "epochs": 100, // Number of training epochs.
    "patience": 10, // Early stopping patience.
    "grace_period": 25, // KL annealing grace period.
    "annealing_epochs": 25, // KL annealing epochs.
    "seed": 42, // Random seed.
    "TRAIN_VAE": 1 // Set to 1 to train the model, 0 to skip training.
}
```

- Train model

```bash
$ cd dual-branch-vae-lstm
$ python train.py --config config.json
```

- Anomaly Detection: Open `anomaly-detection.ipynb` and run all code blocks

### References
- [WESAD Dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html): Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection," ICMI 2018, Boulder, USA, 2018.
- [VAE-LSTM for Anomaly Detection](https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection): S. Lin, R. Clark, R. Birke, S. Schönborn, N. Trigoni and S. Roberts, "Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 4322-4326, doi: 10.1109/ICASSP40776.2020.9053558.
- [MA-VAE](https://github.com/lcs-crr/MA-VAE): Correia, Lucas & Goos, Jan-Christoph & Klein, Philipp & Bäck, Thomas & Kononova, Anna, "MA-VAE: Multi-head Attention-based Variational Autoencoder Approach for Anomaly Detection in Multivariate Time-series Applied to Automotive Endurance Powertrain Testing," doi: 10.48550/arXiv.2309.02253.