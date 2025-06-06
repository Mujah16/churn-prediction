# Telecom Customer Churn Prediction (PyTorch)

This project uses PyTorch to predict customer churn in the telecom industry based on service usage and account information. It includes support for static and sequential model training (LSTM).

## 📁 Project Structure

```
churn-prediction/
├── data/
│   └── customer_sequences.csv
├── models/
│   └── lstm_churn_model.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── utils/
│   └── data_loader.py
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
Place your dataset file under the `data/` directory and name it `customer_sequences.csv`.

### 3. Train the Model
```bash
python scripts/train.py
```

### 4. Evaluate the Model
```bash
python scripts/evaluate.py
```

### 5. Make Predictions
```bash
python scripts/predict.py
```

## 📌 Notes
- Adjust input dimensions in `train.py` and `predict.py` based on your feature set.
- You can modify the LSTM architecture in `models/lstm_churn_model.py`.

## 🔒 License
This project is for educational and internal research use.
