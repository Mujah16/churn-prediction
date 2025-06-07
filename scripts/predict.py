import torch
import mlflow.pytorch
import pandas as pd
from category_encoders.hashing import HashingEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def preprocess_input(df):
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    num_pipeline = StandardScaler().fit(df[num_cols])
    cat_encoder = HashingEncoder(n_components=16).fit(df[cat_cols])

    X_num = num_pipeline.transform(df[num_cols])
    X_cat = cat_encoder.transform(df[cat_cols])
    return np.hstack([X_num, X_cat.values]), cat_encoder, num_pipeline


def load_model():
    return mlflow.pytorch.load_model("models:/ChurnLSTM/1") # or use "runs:/<RUN_ID>/model"


def predict_single(example_df):
    model = load_model()
    X_proc, _, _ = preprocess_input(example_df)
    X_tensor = torch.tensor(X_proc, dtype=torch.float32).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    return preds.numpy(), probs.numpy()


if __name__ == "__main__":
    sample = pd.read_csv("data/customer_sequences.csv").drop(columns=["Churn Value"]).iloc[[0]]
    pred, prob = predict_single(sample)
    print(f"Prediction: {pred[0][0]}, Probability: {prob[0][0]:.4f}")
