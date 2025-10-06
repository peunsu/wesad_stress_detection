## WESAD Stress Detection
### VAE-LSTM for Anomaly Detection
- Train model

```bash
$ cd vae-lstm
$ python train_pytorch.py --config config.json
```

- Anomaly Detection: Open `anomaly-detection.ipynb` and run all code blocks

### MA-VAE for Anomaly Detection
- Train model

```bash
$ cd ma-vae
$ python train.py --config config.json
```

- Anomaly Detection: Open `anomaly-detection.ipynb` and run all code blocks

### References
- [WESAD Dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html): Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven, "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection," ICMI 2018, Boulder, USA, 2018.
- [VAE-LSTM for Anomaly Detection](https://github.com/lin-shuyu/VAE-LSTM-for-anomaly-detection): S. Lin, R. Clark, R. Birke, S. Schönborn, N. Trigoni and S. Roberts, "Anomaly Detection for Time Series Using VAE-LSTM Hybrid Model," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 4322-4326, doi: 10.1109/ICASSP40776.2020.9053558.
- [MA-VAE](https://github.com/lcs-crr/MA-VAE): Correia, Lucas & Goos, Jan-Christoph & Klein, Philipp & Bäck, Thomas & Kononova, Anna, "MA-VAE: Multi-head Attention-based Variational Autoencoder Approach for Anomaly Detection in Multivariate Time-series Applied to Automotive Endurance Powertrain Testing," doi: 10.48550/arXiv.2309.02253.