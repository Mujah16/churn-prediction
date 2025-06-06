import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

from models.lstm_churn_model import LSTMChurnModel

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Clean data: remove leading/trailing whitespace and drop missing/invalid rows
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    required_columns = [
        "Tenure Months", "Phone Service", "Multiple Lines", "Internet Service",
        "Online Security", "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
        "Payment Method", "Monthly Charges", "Total Charges", "Churn Value"
    ]
    df = df.dropna(subset=required_columns)
    df = df[~df[required_columns].isin(["", " "]).any(axis=1)]

    features = [
        "Tenure Months", "Phone Service", "Multiple Lines", "Internet Service",
        "Online Security", "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
        "Payment Method", "Monthly Charges", "Total Charges"
    ]
    target = "Churn Value"

    numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges"]
    categorical_features = list(set(features) - set(numeric_features))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X = df[features]
    y = df[target]
    X_processed = pipeline.fit_transform(X)

    # Save pipeline for inference
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/preprocessor.joblib")

    X_tensor = torch.tensor(X_processed, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    return train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42), X_tensor.shape[-1]

def train(model, loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/20 - Loss: {running_loss/len(loader):.4f}")

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        predictions, targets = [], []
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            predictions.extend(outputs.round().squeeze().tolist())
            targets.extend(batch_y.squeeze().tolist())
    accuracy = np.mean(np.array(predictions) == np.array(targets))
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")

def save_model(model, path="models/lstm_churn_model.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def main():
    (X_train, X_test, y_train, y_test), input_dim = load_data("data/customer_sequences.csv")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    model = LSTMChurnModel(input_dim=input_dim, hidden_dim=128)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n🚀 Training started...")
    train(model, train_loader, criterion, optimizer)

    print("\n🔍 Evaluating...")
    evaluate(model, test_loader)
    save_model(model)

if __name__ == "__main__":
    main()
