# 📉 Customer Churn Prediction using LSTM

This project predicts customer churn using an LSTM neural network built with PyTorch. It includes:

- ✅ Data preprocessing (encoding, scaling)
- 🏋️‍♀️ Training and Optuna tuning
- 📊 Evaluation with ROC and confusion matrix
- 📦 Experiment tracking via MLflow
- 🧠 Inference on new samples
- 🚀 Azure ML deployment (optional)

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction

# Create Conda environment
conda env create -f python_env.yaml
conda activate churn-prediction

# (Optional) Install Azure ML dependencies
pip install azure-ai-ml azure-identity

🏋️‍♂️ Training

# Method 1: Using MLflow entry point
mlflow run . -P n_trials=10 --env-manager=local

# Method 2: Direct Python script
python scripts/train.py --n_trials 10

📈 Evaluation
python scripts/evaluate.py

🔍 Inference
python scripts/predict.py

# Azure ML Deployment
python deploy_to_azure.py

🧱 Project Structure

churn-prediction/
├── artifacts/                    # Exported model artifacts
├── data/                         # Input CSV data
├── mlruns/                       # MLflow runs
├── models/                       # Saved model definitions
├── scripts/                      # Core Python scripts
│   ├── train.py                  # Training with Optuna & MLflow
│   ├── evaluate.py               # Evaluation of registered model
│   └── predict.py                # Inference on single input
├── deploy_to_azure.py           # Script to deploy to Azure ML
├── utils/                        # Optional utility code
├── sample_input.json            # Example input for inference
├── confusion_matrix.png         # Logged confusion matrix image
├── roc_curve.png                # Logged ROC curve image
├── conda.yaml                   # Conda environment definition
├── requirements.txt             # Python dependencies
├── python_env.yaml              # MLflow-compatible Conda env
├── MLproject                    # MLflow project file
├── README.md                    # You're reading it!

⚙️ MLflow Usage

# Launch MLflow UI
mlflow ui

# Serve model locally
mlflow models serve -m "models:/ChurnLSTM/1" --no-conda

# Register manually (if needed)
mlflow models register -m runs:/<RUN_ID>/model -n ChurnLSTM
