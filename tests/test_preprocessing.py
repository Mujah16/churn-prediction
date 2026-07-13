import numpy as np
import pandas as pd

from utils.preprocessing import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    build_preprocessor,
    fit_transform_features,
    split_features_target,
    transform_features,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Tenure Months": 1,
                "Phone Service": "Yes",
                "Multiple Lines": "No",
                "Internet Service": "DSL",
                "Online Security": "Yes",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": "Month-to-month",
                "Paperless Billing": "Yes",
                "Payment Method": "Mailed check",
                "Monthly Charges": 53.85,
                "Total Charges": "108.15",
                "Churn Value": 1,
                "Churn Label": "Yes",
                "Churn Score": 86,
                "Churn Reason": "Competitor made better offer",
            },
            {
                "Tenure Months": 24,
                "Phone Service": "Yes",
                "Multiple Lines": "Yes",
                "Internet Service": "Fiber optic",
                "Online Security": "No",
                "Online Backup": "Yes",
                "Device Protection": "Yes",
                "Tech Support": "Yes",
                "Streaming TV": "Yes",
                "Streaming Movies": "Yes",
                "Contract": "Two year",
                "Paperless Billing": "No",
                "Payment Method": "Credit card",
                "Monthly Charges": 89.10,
                "Total Charges": "2138.40",
                "Churn Value": 0,
                "Churn Label": "No",
                "Churn Score": 12,
                "Churn Reason": np.nan,
            },
        ],
    )


def test_split_features_target_uses_explicit_non_leaking_contract() -> None:
    features, target = split_features_target(_sample_frame())

    assert list(features.columns) == FEATURE_COLUMNS
    assert not set(LEAKAGE_COLUMNS).intersection(features.columns)
    assert target is not None
    assert target.tolist() == [1.0, 0.0]


def test_preprocessor_fits_train_schema_and_transforms_inference_rows() -> None:
    features, _ = split_features_target(_sample_frame())
    preprocessor = build_preprocessor()

    train_matrix = fit_transform_features(preprocessor, features)
    inference_matrix = transform_features(preprocessor, features.iloc[[0]])

    assert train_matrix.dtype == np.float32
    assert inference_matrix.dtype == np.float32
    assert inference_matrix.shape[0] == 1
    assert inference_matrix.shape[1] == train_matrix.shape[1]
