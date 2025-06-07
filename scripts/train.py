import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

import mlflow
import mlflow.pytorch
import optuna
from models.lstm_churn_model import LSTMChurnModel
from category_encoders.hashing import HashingEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"]

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    X_num = num_pipeline.fit_transform(X[numeric])

    cat_encoder = HashingEncoder(n_components=16)
    X_cat = cat_encoder.fit_transform(X[categorical])

    X_processed = np.hstack([X_num, X_cat.values])
    input_dim = X_processed.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    return (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor), input_dim, X_train, X_test, y_test


def get_dataloaders(X_train, y_train, batch_size):
    dataset = TensorDataset(X_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")


def evaluate(model, X_test, y_test, batch_size):
    model.eval()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    preds, targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).view(-1, 1)
            outputs = torch.sigmoid(model(batch_x))
            preds.extend(outputs.view(-1).cpu().numpy())
            targets.extend(batch_y.view(-1).cpu().numpy())

    y_pred = (np.array(preds) > 0.5).astype(int)
    acc = accuracy_score(targets, y_pred)
    f1 = f1_score(targets, y_pred)
    auc_score = roc_auc_score(targets, preds)

    # Confusion Matrix
    cm = confusion_matrix(targets, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(targets, preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

    return acc, f1, auc_score


def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 64, 192)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    epochs = trial.suggest_int("epochs", 10, 20)

    (X_train, X_test, y_train, y_test), input_dim, X_train_raw, _, _ = load_data("data/customer_sequences.csv")
    train_loader = get_dataloaders(X_train, y_train, batch_size)

    model = LSTMChurnModel(input_dim=input_dim,
                           hidden_dim=hidden_dim,
                           num_layers=num_layers,
                           dropout=dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train.numpy()), y=y_train.numpy())
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs
        })

        train_model(model, train_loader, criterion, optimizer, epochs)
        acc, f1, auc_score = evaluate(model, X_test, y_test, batch_size)

        mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "roc_auc": auc_score})
        print(f"✅ Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, ROC AUC: {auc_score:.4f}")

    trial.set_user_attr("trained_model", model)
    trial.set_user_attr("input_dim", input_dim)
    trial.set_user_attr("input_example", X_train_raw[0:1].astype(np.float32))

    return f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=10)
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")

    experiment_name = "ChurnPrediction_v1"
    client = mlflow.tracking.MlflowClient()

    if not client.get_experiment_by_name(experiment_name):
        client.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("✅ Best trial:", study.best_trial.params)

    best_model = study.best_trial.user_attrs["trained_model"]
    input_example = study.best_trial.user_attrs["input_example"]

    with mlflow.start_run(run_name=f"best_trial_{study.best_trial.number}"):
        mlflow.log_params(study.best_trial.params)
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="ChurnLSTM",
            input_example=np.expand_dims(input_example, axis=1)
        )


if __name__ == "__main__":
    main()
