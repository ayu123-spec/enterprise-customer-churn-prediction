"""Data ingestion for the Telco customer churn project.

Downloads the raw dataset (once), loads it, and validates its schema.
Cleaning/encoding is NOT done here -- that is the preprocessing stage's job.

Run from the project root (with venv active):
    python src/ingestion/data_ingestion.py
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- config (later we can move these into configs/config.yaml) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
TARGET = "Churn"
ID_COLUMN = "customerID"
EXPECTED_ROW_COUNT = 7043
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def download() -> None:
    """Download the raw CSV into data/raw/ once. Skips if already present."""
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists():
        logger.info("Raw data already present at %s, skipping download.", RAW_PATH)
        return
    logger.info("Downloading dataset to %s ...", RAW_PATH)
    urllib.request.urlretrieve(SOURCE_URL, RAW_PATH)
    logger.info("Downloaded %d bytes.", RAW_PATH.stat().st_size)


def load_raw() -> pd.DataFrame:
    """Read the raw CSV into a DataFrame."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_PATH}. Run download() first.")
    df = pd.read_csv(RAW_PATH)
    logger.info("Loaded raw data: %d rows, %d columns", df.shape[0], df.shape[1])
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, target, row count; warn on known anomalies."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    if df.shape[0] != EXPECTED_ROW_COUNT:
        logger.warning("Row count %d differs from expected %d", df.shape[0], EXPECTED_ROW_COUNT)

    blanks = (df["TotalCharges"].astype(str).str.strip() == "").sum()
    if blanks:
        logger.warning("TotalCharges has %d blank values (tenure=0) to handle in preprocessing.", blanks)

    dupes = df[ID_COLUMN].duplicated().sum()
    if dupes:
        logger.warning("Found %d duplicate customer IDs.", dupes)

    churn_rate = (df[TARGET] == "Yes").mean()
    logger.info("Validation passed. Churn rate: %.1f%%", churn_rate * 100)
    return df


def run() -> pd.DataFrame:
    """Full ingestion: download -> load -> validate."""
    download()
    return validate(load_raw())


if __name__ == "__main__":
    df = run()
    print(df.head())
    