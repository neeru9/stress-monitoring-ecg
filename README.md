# Stress Detection using ECG Signals

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Pre-Processing](#data-pre-processing)
- [Results on Testing Set](#results-on-testing-set)
- [Experimental Results](#experimental-results)
- [Installation](#installation)
- [Usage](#usage)
- [References](#references)

## Introduction
This project aims to develop a deep learning model capable of predicting the emotional state (stress/no stress) of an individual based on their ElectroCardiogram (ECG) signals. The model has been trained on the PhysioNet Driver’s Stress Detection dataset.

## Dataset
The dataset, obtained from the PhysioNet repository [[1]](#1), contains ECG data collected from drivers under stress conditions. The data is sampled at a frequency of 700Hz and includes recordings from multiple subjects. The dataset was challenging to process due to errors and limited data, but optimization techniques were applied to ensure model robustness.

## Data Pre-Processing
The raw ECG data was preprocessed using:
- KNN Imputation: To handle missing values
- Standard Scaling: For normalization
- Power Transformation: To stabilize variance and make data more Gaussian-like

Features extracted include heart rate variability (HRV), mean heart rate, and other temporal features. Data augmentation was performed to enhance model generalization.

## Model and Training
The ensemble model combines predictions from XGBoost, LSTM, and a Transformer-based neural network. The architecture of each component model is as follows:
- XGBoost: Used for feature importance analysis and as a robust baseline
- LSTM: Captures temporal dependencies in ECG signals
- Transformer: Learns long-range dependencies with self-attention
## Ensemble Method
The ensemble model aggregates predictions from XGBoost, LSTM, and Transformer models. Weighted averaging is applied to combine the predictions, giving more importance to XGBoost due to its stable performance in stress detection. The weights were determined through cross-validation experiments to maximize accuracy and F1 score.

The final prediction equation is given by:

\\[
y_{ensemble} = w_1 \, y_{xgboost} + w_2 \, y_{lstm} + w_3 \, y_{transformer}
\\]

Where \\(w_1 > w_2\\) and \\(w_1 > w_3\\), indicating XGBoost's higher influence.

The final prediction is a weighted ensemble, giving more importance to XGBoost due to its stability.

### Model Performance Metrics
The performance metrics (MSE, RMSE) of the individual models and the weighted ensemble on both training and testing sets are as follows:

| Model            | R²    | MSE (Training) | RME (Testing) |
|-----------------|-------|----------------|---------------|
| XGBoost          | 0.792 | 0.2084         | 0.4566        |
| LSTM             | 0.802 | 0.1982         | 0.4471        |
| Transformer      | 0.769 | 0.3211         | 0.4808        |
| Weighted Ensemble| 0.812 | 0.1808         | 0.4253        |

## Experimental Results
Below are the results from specific test samples, showing the predicted stress levels based on physiological signals:

| EMG_mean | HANDGSR_mean | HR_mean | RESP_mean | Predicted Stress Level |
|---------|--------------|--------|----------|-------------------------|
| 0.1     | 0.2          | 6      | 2        | 1.16 (Relaxed)           |
| 0.4     | 0.35         | 15     | 5.5      | 3.11 (Medium Stress)     |
| 0.9     | 0.7          | 30     | 11       | 5.12 (High Stress)       |

## References
<a id="1">[1]</a> PhysioNet Driver’s Stress Detection Dataset. Available at: <a href="https://physionet.org/content/drivedb/1.0.0/">https://physionet.org/content/drivedb/1.0.0/ /br
[2] A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. Ch. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals," *Circulation*, vol. 101, no. 23, pp. e215–e220, Jun. 2000, doi: [10.1161/01.CIR.101.23.e215](https://doi.org/10.1161/01.CIR.101.23.e215). 
</a>

## Installation
```bash
# Clone the repository
git clone https://github.com/username/stress-detection.git
cd stress-detection

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
