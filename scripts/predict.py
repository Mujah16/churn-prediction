import sys, os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import argparse
import joblib
from models.lstm_churn_model import LSTMChurnModel


def predict_single(input_dict, model_path, preprocessor_path, input_dim, hidden_dim):
    # Load model
    model = LSTMChurnModel(input_dim=input_dim, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load preprocessor
    preprocessor = joblib.load(preprocessor_path)

    # Prepare input
    df = pd.DataFrame([input_dict])
    X_processed = preprocessor.transform(df)

    # Convert to tensor
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    if X_tensor.ndim == 2:
        X_tensor = X_tensor.unsqueeze(1)  # shape: [1, 1, input_dim]
    elif X_tensor.ndim == 1:
        X_tensor = X_tensor.unsqueeze(0).unsqueeze(1)

    with torch.no_grad():
        output = model(X_tensor)
        prediction = output.item()
        return {
            "churn_probability": round(prediction, 4),
            "will_churn": prediction > 0.5
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON input file")
    parser.add_argument("--model", required=True, help="Path to saved model (.pth)")
    parser.add_argument("--preprocessor", default="models/preprocessor.joblib", help="Path to saved preprocessor")
    parser.add_argument("--input_dim", type=int, required=True, help="Input dimension used in model")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for LSTM")

    args = parser.parse_args()

    with open(args.input, "r") as f:
        input_dict = json.load(f)

    result = predict_single(input_dict, args.model, args.preprocessor, args.input_dim, args.hidden_dim)
    print(json.dumps(result, indent=2))
