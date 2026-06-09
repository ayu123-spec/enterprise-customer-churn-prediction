"""Unit tests for the churn pipeline transforms."""
import pandas as pd

from src.preprocessing.data_preprocessing import clean
from src.features.feature_engineering import encode, engineer


def _raw() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "gender": ["Female", "Male", "Male"],
        "tenure": [1, 0, 24],
        "Contract": ["Month-to-month", "Two year", "One year"],
        "TotalCharges": ["29.85", " ", "1840.75"],   # note the blank (tenure-0 case)
        "MonthlyCharges": [29.85, 20.0, 76.0],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "No", "Yes"],
        "TechSupport": ["No", "Yes", "No"],
        "StreamingTV": ["No", "No", "Yes"],
        "StreamingMovies": ["No", "Yes", "Yes"],
        "Churn": ["No", "Yes", "No"],
    })


def test_clean_drops_id_and_encodes_target():
    out = clean(_raw())
    assert "customerID" not in out.columns
    assert set(out["Churn"].unique()).issubset({0, 1})
    assert out["TotalCharges"].isnull().sum() == 0          # blank got filled
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])


def test_feature_engineering_adds_columns():
    feat = engineer(clean(_raw()))
    assert {"avg_monthly_spend", "is_month_to_month", "num_addon_services"} <= set(feat.columns)
    assert feat.loc[0, "is_month_to_month"] == 1            # row 0 is month-to-month


def test_encoded_data_is_all_numeric():
    encoded = encode(engineer(clean(_raw())))
    assert "Churn" in encoded.columns
    non_numeric = [c for c in encoded.columns if not pd.api.types.is_numeric_dtype(encoded[c])]
    assert non_numeric == []