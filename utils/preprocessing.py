from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn Value"

FEATURE_COLUMNS = [
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]

NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_FEATURES = [column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES]
LEAKAGE_COLUMNS = ["Churn Label", "Churn Score", "Churn Reason"]


def load_churn_dataframe(path: str | Path) -> pd.DataFrame:
    return strip_string_values(pd.read_csv(path))


def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(
        lambda column: column.map(
            lambda value: value.strip() if isinstance(value, str) else value,
        ),
    )


def split_features_target(
    df: pd.DataFrame,
    *,
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    missing_features = sorted(set(FEATURE_COLUMNS) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")

    if require_target and TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    features = df.loc[:, FEATURE_COLUMNS].copy()
    features.loc[:, NUMERIC_FEATURES] = features.loc[:, NUMERIC_FEATURES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    target = df[TARGET_COLUMN].astype(np.float32) if TARGET_COLUMN in df.columns else None
    return features, target


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ],
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _one_hot_encoder()),
        ],
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def fit_transform_features(preprocessor: ColumnTransformer, features: pd.DataFrame) -> np.ndarray:
    return _as_float32_array(preprocessor.fit_transform(features))


def transform_features(preprocessor: ColumnTransformer, features: pd.DataFrame) -> np.ndarray:
    return _as_float32_array(preprocessor.transform(features))


def to_sequence_tensor(features: np.ndarray) -> torch.Tensor:
    return torch.tensor(features, dtype=torch.float32).unsqueeze(1)


def target_to_tensor(target: pd.Series | Iterable[float]) -> torch.Tensor:
    return torch.tensor(np.asarray(target, dtype=np.float32), dtype=torch.float32)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _as_float32_array(values: object) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)
