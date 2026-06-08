"""Feature engineering for the Telco customer churn project.

Takes the cleaned dataset and turns it into a fully numeric, model-ready table:
  - engineers new features informed by the EDA
  - one-hot encodes the remaining categorical columns
Output: data/processed/telco_churn_features.csv

Run from the project root (with venv active):
    python src/features/feature_engineering.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_features.csv"

TARGET = "Churn"
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create new predictive features based on EDA findings."""
    df = df.copy()

    # 1. Average lifetime monthly spend (tenure 0 -> divide by 1 to avoid /0).
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # 2. High-risk contract flag: month-to-month churns ~43%.
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    # 3. How many add-on services the customer subscribes to.
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)

    # 4. Tenure band: churn drops sharply after the first year.
    df["tenure_band"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "12-24m", "24-48m", "48-72m"], include_lowest=True,
    )

    logger.info("Engineered 4 new features.")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; keep the target as a 0/1 column."""
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    X[TARGET] = y.values
    logger.info("Encoded categoricals -> %d total columns (incl. target).", X.shape[1])
    return X


def run() -> pd.DataFrame:
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(
            f"Clean data not found at {CLEAN_PATH}. "
            f"Run: python src/preprocessing/data_preprocessing.py first."
        )
    df = pd.read_csv(CLEAN_PATH)
    logger.info("Loaded clean data: %d rows, %d columns.", df.shape[0], df.shape[1])

    featured = encode(engineer(df))

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(FEATURES_PATH, index=False)
    logger.info("Saved model-ready data to %s", FEATURES_PATH)
    return featured


if __name__ == "__main__":
    df = run()
    print("Final shape:", df.shape)
    print("\nNew engineered columns sample:")
    print(df[["avg_monthly_spend", "is_month_to_month", "num_addon_services"]].head())
    print("\nAll columns:")
    print(list(df.columns))