from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    fit_transform_features,
    split_features_target,
    strip_string_values,
    target_to_tensor,
    to_sequence_tensor,
)


def load_sequence_data(data: str | pd.DataFrame) -> tuple[TensorDataset, int]:
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a filepath or pandas DataFrame.")

    df = strip_string_values(df)
    features, target = split_features_target(df, require_target=TARGET_COLUMN in df.columns)
    preprocessor = build_preprocessor()
    x_matrix = fit_transform_features(preprocessor, features)
    x_tensor = to_sequence_tensor(x_matrix)
    y_tensor = target_to_tensor(target) if target is not None else torch.zeros(x_tensor.size(0))

    return TensorDataset(x_tensor, y_tensor), x_tensor.shape[-1]
