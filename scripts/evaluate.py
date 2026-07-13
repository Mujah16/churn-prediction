from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from utils.preprocessing import (
    load_churn_dataframe,
    split_features_target,
    target_to_tensor,
    to_sequence_tensor,
    transform_features,
)


def load_evaluation_tensors(
    data_path: str,
    preprocessor_path: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    df = load_churn_dataframe(data_path)
    features, target = split_features_target(df)
    if target is None:
        raise ValueError("Evaluation requires a target column.")

    preprocessor = joblib.load(preprocessor_path)
    x_matrix = transform_features(preprocessor, features)
    return to_sequence_tensor(x_matrix), target_to_tensor(target)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    artifact_dir: Path,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    targets: list[float] = []
    probabilities: list[float] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            probability = torch.sigmoid(logits).view(-1)
            prediction = (probability > threshold).int()

            probabilities.extend(probability.tolist())
            predictions.extend(prediction.tolist())
            targets.extend(y_batch.view(-1).tolist())

    print("\nClassification Report:")
    print(classification_report(targets, predictions, zero_division=0))

    cm = confusion_matrix(targets, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "f1_score": f1_score(targets, predictions),
        "roc_auc": roc_auc_score(targets, probabilities),
    }
    print(f"\nF1 Score: {metrics['f1_score']:.4f}")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path = artifact_dir / "confusion_matrix.png"
    roc_curve_path = artifact_dir / "roc_curve.png"

    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path)
    plt.close()
    mlflow.log_artifact(str(confusion_matrix_path))

    fpr, tpr, _ = roc_curve(targets, probabilities)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(roc_curve_path)
    plt.close()
    mlflow.log_artifact(str(roc_curve_path))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained churn LSTM model.")
    parser.add_argument("--data-path", default="data/customer_sequences.csv")
    parser.add_argument("--model-uri", default="models:/ChurnLSTM/1")
    parser.add_argument("--preprocessor-path", default="artifacts/preprocessor.joblib")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", default="ChurnPrediction_v1")
    parser.add_argument("--artifact-dir", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print(f"Loading model from {args.model_uri}...")
    model = mlflow.pytorch.load_model(args.model_uri)

    print("Loading evaluation data...")
    x_tensor, y_tensor = load_evaluation_tensors(args.data_path, args.preprocessor_path)
    test_loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=args.batch_size)

    with mlflow.start_run(run_name="evaluate_model"):
        metrics = evaluate_model(model, test_loader, Path(args.artifact_dir), args.threshold)
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
