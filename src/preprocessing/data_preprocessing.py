"""Preprocessing for the Telco customer churn project.

Turns the raw data into a tidy, model-agnostic dataset:
  - fixes TotalCharges (text -> numeric, blanks -> 0)
  - drops the customerID identifier
  - encodes the target Churn (Yes/No -> 1/0)
Encoding of categorical FEATURES and scaling are NOT done here -- that is
the feature-engineering stage's job.

Run from the project root (with venv active):
    python src/preprocessing/data_preprocessing.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_clean.csv"

TARGET = "Churn"
ID_COLUMN = "customerID"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the unambiguous cleaning steps every model needs."""
    df = df.copy()

    # 1. TotalCharges is stored as text and is blank for tenure-0 customers.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    logger.info("TotalCharges: converted to numeric, filled %d blanks with 0.", n_missing)

    # 2. customerID is a unique identifier, not a predictive feature.
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
        logger.info("Dropped identifier column '%s'.", ID_COLUMN)

    # 3. Encode the target as 1 = churned, 0 = stayed.
    df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    logger.info("Encoded target '%s' as 1=churn, 0=stay.", TARGET)

    # 4. Trim stray whitespace in any remaining text columns (defensive).
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    logger.info("Clean dataset: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def run() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_PATH}. "
            f"Run: python src/ingestion/data_ingestion.py first."
        )
    df = pd.read_csv(RAW_PATH)
    logger.info("Loaded raw data: %d rows, %d columns.", df.shape[0], df.shape[1])

    cleaned = clean(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    logger.info("Saved cleaned data to %s", PROCESSED_PATH)
    return cleaned


if __name__ == "__main__":
    df = run()
    print(df.head())
    print("\nTarget distribution (0 = stayed, 1 = churned):")
    print(df["Churn"].value_counts())