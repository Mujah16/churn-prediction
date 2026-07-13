from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import pandas as pd
import torch

from utils.preprocessing import (
    split_features_target,
    strip_string_values,
    to_sequence_tensor,
    transform_features,
)


def load_input_frame(input_path: str) -> pd.DataFrame:
    with Path(input_path).open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be an object or a list of objects.")

    return strip_string_values(pd.DataFrame(payload))


def predict_frame(
    example_df: pd.DataFrame,
    *,
    model_uri: str,
    preprocessor_path: str,
    threshold: float,
) -> list[dict[str, float | int]]:
    preprocessor = joblib.load(preprocessor_path)
    features, _ = split_features_target(example_df, require_target=False)
    x_matrix = transform_features(preprocessor, features)
    x_tensor = to_sequence_tensor(x_matrix)

    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_tensor)).view(-1)
        predictions = (probabilities > threshold).int()

    return [
        {"prediction": int(prediction.item()), "probability": float(probability.item())}
        for prediction, probability in zip(predictions, probabilities, strict=True)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn inference for JSON input.")
    parser.add_argument("--input-path", default="sample_input.json")
    parser.add_argument("--model-uri", default="models:/ChurnLSTM/1")
    parser.add_argument("--preprocessor-path", default="artifacts/preprocessor.joblib")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    example_df = load_input_frame(args.input_path)
    predictions = predict_frame(
        example_df,
        model_uri=args.model_uri,
        preprocessor_path=args.preprocessor_path,
        threshold=args.threshold,
    )
    print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
