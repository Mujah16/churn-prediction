import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset

def load_sequence_data(data):
    # Accept either file path or DataFrame
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Input must be a filepath or pandas DataFrame.")

    features = [
        "Tenure Months", "Phone Service", "Multiple Lines", "Internet Service",
        "Online Security", "Online Backup", "Device Protection", "Tech Support",
        "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
        "Payment Method", "Monthly Charges", "Total Charges"
    ]
    target = "Churn Value" if "Churn Value" in df.columns else None

    numeric_features = ["Tenure Months", "Monthly Charges", "Total Charges"]
    categorical_features = list(set(features) - set(numeric_features))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X = df[features]
    y = df[target].values if target else None

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
    X_processed = pipeline.fit_transform(X)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1) if y is not None else None

    return TensorDataset(X_tensor, y_tensor) if y is not None else TensorDataset(X_tensor, torch.zeros(X_tensor.size(0))), X_tensor.shape[-1]
