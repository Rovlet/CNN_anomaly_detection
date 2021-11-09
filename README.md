# CNN anomaly detection


<!-- ABOUT THE PROJECT -->
## About The Project

The goal of this project is to detect anomalies from log data using CNN (Convolutional neural network)

The app will be deployed based on the following approaches:
* [Intrusion Detection Using Convolutional Neural Networks for Representation Learning](https://www.researchgate.net/publication/320687642_Intrusion_Detection_Using_Convolutional_Neural_Networks_for_Representation_Learning)
* [An Encoding Technique for CNN-based Network Anomaly Detection](https://ieeexplore.ieee.org/document/8622568)
* [Log Anomaly Detection](https://github.com/WraySmith/log-anomaly)

Datasets:
* [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
* TBC


<!-- GETTING STARTED -->
## Getting Started



### Prerequisites

At this point, the project files should work on Windows and Linux. 

### How to use this project?

1. Clone the repo
   ```
   git clone https://github.com/Rovlet//CNN_anomaly_detection.git
   ```
2. Create virtualenv/venv and install requirements
    ```
   virtualenv cnn_anomaly_detection_env
   source cnn_anomaly_detection_env/bin/activate
   pip install -r requirements.txt
    ```
3. Download labeled data and copy it to /data directory. Link:
   [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
4. Check your settings in settings.py file. If you want to display corr matrices and save encoded pictures, change these values:
   ```
    DISPLAY_CORR_MATRICES = False
    SAVE_ENCODED_PICTURES = False
   ```
5. Run preprocessing
   ```
    python preprocessing.py
   ```
###TBC

<!-- ROADMAP -->
## Roadmap

- [x] Add Preprocessing
- [x] Add Encoding
- [ ] Add CNN Model
- [ ] Add methods to analyze results
- [ ] Add new datasets
