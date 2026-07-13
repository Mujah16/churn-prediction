# Customer Churn Prediction

PyTorch LSTM workflow for customer churn prediction with MLflow tracking, reproducible preprocessing, local inference, evaluation, and optional Azure ML deployment.

## What Is Enterprise-Ready Here

- Fixed feature contract in `utils/preprocessing.py` to prevent target leakage.
- Train/test split occurs before fitting scalers and encoders.
- Inference loads the fitted preprocessor instead of refitting on a single request.
- Azure deployment configuration is supplied through CLI arguments or environment variables.
- CI runs lint and tests on every push and pull request.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

Install the platform-specific PyTorch build first. The command above uses CPU wheels, which is the right default for local checks and CI.

For Azure ML deployment support:

```bash
python -m pip install -r requirements-azure.txt
```

## Training

```bash
python -m scripts.train --n_trials 10
```

Or run through MLflow:

```bash
mlflow run . -P n_trials=10 --env-manager=local
```

Training writes the fitted preprocessing artifact to `artifacts/preprocessor.joblib` and registers the PyTorch model as `ChurnLSTM` by default.
The default MLflow backend is `sqlite:///mlflow.db` because recent MLflow versions discourage filesystem tracking stores.

## Evaluation

```bash
python -m scripts.evaluate \
  --model-uri models:/ChurnLSTM/1 \
  --preprocessor-path artifacts/preprocessor.joblib
```

## Inference

```bash
python -m scripts.predict \
  --input-path sample_input.json \
  --model-uri models:/ChurnLSTM/1 \
  --preprocessor-path artifacts/preprocessor.joblib
```

The input JSON can be a single object or a list of objects using the feature columns defined in `utils/preprocessing.py`.

## Azure ML Deployment

```bash
export AZURE_SUBSCRIPTION_ID="<subscription-id>"
export AZURE_RESOURCE_GROUP="<resource-group>"
export AZURE_ML_WORKSPACE="<workspace-name>"

python deploy_to_azure.py \
  --model-name ChurnLSTM \
  --model-version 1
```

## Quality Gates

```bash
ruff check .
pytest
```

## Project Structure

```text
churn-prediction/
├── .github/workflows/ci.yml
├── data/
├── models/
├── scripts/
├── tests/
├── utils/
├── deploy_to_azure.py
├── MLproject
├── pyproject.toml
├── requirements.txt
├── requirements-azure.txt
└── requirements-dev.txt
```
