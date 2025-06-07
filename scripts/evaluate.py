import torch
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders.hashing import HashingEncoder
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def preprocess_data(path):
    df = pd.read_csv(path)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    X_num = num_pipeline.fit_transform(X[num_cols])

    cat_encoder = HashingEncoder(n_components=16)
    X_cat = cat_encoder.fit_transform(X[cat_cols])

    X_processed = np.hstack([X_num, X_cat.values])
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    return X_tensor, y_tensor


def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    predictions, targets, probs = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            logits = model(x_batch)
            prob = torch.sigmoid(logits).view(-1)
            pred = (prob > threshold).int()

            probs.extend(prob.tolist())
            predictions.extend(pred.tolist())
            targets.extend(y_batch.view(-1).tolist())

    print("\n📊 Classification Report:")
    print(classification_report(targets, predictions))

    print("\n🧮 Confusion Matrix:")
    cm = confusion_matrix(targets, predictions)
    print(cm)

    print(f"\n✅ F1 Score: {f1_score(targets, predictions):.4f}")
    print(f"🎯 ROC AUC Score: {roc_auc_score(targets, probs):.4f}")

    # Log confusion matrix
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Log ROC curve
    fpr, tpr, _ = roc_curve(targets, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(targets, probs):.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")


if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ChurnPrediction_v1")

    print("🔄 Loading latest model from MLflow registry...")
    model = mlflow.pytorch.load_model("models:/ChurnLSTM/1")

    print("📦 Loading test data...")
    X_tensor, y_tensor = preprocess_data("data/customer_sequences.csv")
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64)

    with mlflow.start_run(run_name="evaluate_model"):
        evaluate_model(model, test_loader)
