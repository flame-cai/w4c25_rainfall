# Weather4Cast 2025 — Precipitation Nowcasting  

This repository contains the implementation of our submission for the **NeurIPS 2025 Weather4Cast Challenge**, focusing on **short-term precipitation nowcasting** using SEVIRI satellite data.  
Our pipeline leverages **ConvGRU** and **ConvLSTM** architectures to model spatiotemporal dependencies in brightness temperature imagery for future rainfall prediction.

---

## Setup Instructions  

### 1. Create a Python Environment  

It is recommended to use **Python 3.10+** and create a dedicated environment:

```bash
# Create a new virtual environment
python -m venv w4c_env

# Activate the environment
# On Windows:
w4c_env\Scripts\activate
# On macOS/Linux:
source w4c_env/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

## Training

You can retrain or finetune the models as follows:
### ConvGRU
```
python models/ConvGRU/training_convgru.py # Trains for hour 1
python models/ConvGRU/training_convgru_future.py # Trains for the next 3 hours
```
### ConvLSTM
```
python models/ConvLSTM/training_convlstm.py # Trains for hour 1
python models/ConvLSTM/training_convlstm_future.py # Trains for the next 3 hours
```

## Evaluation

Evaluate trained models on test data using:
```
python evaluation/evaluate_gru.py
python evaluation/evaluate_lstm.py
```

### Metrics computed include:
- RMSE (Root Mean Squared Error)
- SSIM (Structural Similarity Index)

## Submission Generation

To generate predictions formatted for the official leaderboard:
```
python submission_scripts/submission_event_prediction.py
python submission_scripts/submission_cumulative_rainfall.py
```
