from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_churn_model import LSTMChurnModel
from utils.preprocessing import (
    build_preprocessor,
    fit_transform_features,
    load_churn_dataframe,
    split_features_target,
    target_to_tensor,
    to_sequence_tensor,
    transform_features,
)


@dataclass
class TrainingData:
    x_train: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    input_dim: int
    preprocessor: object


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_training_data(data_path: str, test_size: float, seed: int) -> TrainingData:
    df = load_churn_dataframe(data_path)
    features, target = split_features_target(df)
    if target is None:
        raise ValueError("Training requires a target column.")

    x_train_raw, x_test_raw, y_train_raw, y_test_raw = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=seed,
    )

    preprocessor = build_preprocessor()
    x_train = fit_transform_features(preprocessor, x_train_raw)
    x_test = transform_features(preprocessor, x_test_raw)

    return TrainingData(
        x_train=to_sequence_tensor(x_train),
        x_test=to_sequence_tensor(x_test),
        y_train=target_to_tensor(y_train_raw),
        y_test=target_to_tensor(y_test_raw),
        input_dim=x_train.shape[1],
        preprocessor=preprocessor,
    )


def get_dataloader(x_tensor: torch.Tensor, y_tensor: torch.Tensor, batch_size: int) -> DataLoader:
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
) -> None:
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")


def evaluate(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int,
    device: torch.device,
    artifact_dir: Path,
) -> tuple[float, float, float]:
    model.eval()
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    probabilities: list[float] = []
    targets: list[float] = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = torch.sigmoid(model(batch_x))
            probabilities.extend(outputs.view(-1).cpu().numpy().tolist())
            targets.extend(batch_y.view(-1).cpu().numpy().tolist())

    predictions = (np.asarray(probabilities) > 0.5).astype(int)
    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    roc_auc = roc_auc_score(targets, probabilities)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = artifact_dir / "confusion_matrix.png"
    roc_curve_path = artifact_dir / "roc_curve.png"

    cm = confusion_matrix(targets, predictions)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path)
    plt.close()
    mlflow.log_artifact(str(confusion_matrix_path))

    fpr, tpr, _ = roc_curve(targets, probabilities)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(roc_curve_path)
    plt.close()
    mlflow.log_artifact(str(roc_curve_path))

    return accuracy, f1, roc_auc


def positive_class_weight(y_train: torch.Tensor) -> torch.Tensor:
    positives = torch.count_nonzero(y_train == 1).item()
    negatives = torch.count_nonzero(y_train == 0).item()
    if positives == 0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def objective(
    trial: optuna.Trial,
    data: TrainingData,
    device: torch.device,
    artifact_dir: Path,
) -> float:
    hidden_dim = trial.suggest_int("hidden_dim", 64, 192)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    epochs = trial.suggest_int("epochs", 10, 20)

    train_loader = get_dataloader(data.x_train, data.y_train, batch_size)
    model = LSTMChurnModel(
        input_dim=data.input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight(data.y_train).to(device))

    with mlflow.start_run(nested=True):
        mlflow.log_params(
            {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            },
        )

        train_model(model, train_loader, criterion, optimizer, epochs, device)
        accuracy, f1, roc_auc = evaluate(
            model,
            data.x_test,
            data.y_test,
            batch_size,
            device,
            artifact_dir,
        )

        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc})
        print(f"Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    trial.set_user_attr("trained_model", model.to("cpu"))
    return f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register the churn LSTM model.")
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--data-path", default="data/customer_sequences.csv")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="ChurnPrediction_v1")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--registered-model-name", default="ChurnLSTM")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(args.artifact_dir)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    data = prepare_training_data(args.data_path, args.test_size, args.seed)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, data, device, artifact_dir),
        n_trials=args.n_trials,
    )

    print("Best trial:", study.best_trial.params)
    best_model = study.best_trial.user_attrs["trained_model"]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = artifact_dir / "preprocessor.joblib"
    joblib.dump(data.preprocessor, preprocessor_path)

    with mlflow.start_run(run_name=f"best_trial_{study.best_trial.number}"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_param("input_dim", data.input_dim)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing")
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=args.registered_model_name or None,
            serialization_format=mlflow.pytorch.SERIALIZATION_FORMAT_PICKLE,
        )


if __name__ == "__main__":
    main()
