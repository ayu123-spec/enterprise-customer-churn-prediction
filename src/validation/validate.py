"""Validation gates for the Telco customer churn project.

Two checks that would gate a deployment in a real MLOps setup:
  1. Data quality  - the model-ready table is complete, numeric, and sane.
  2. Model quality - every trained model clears a minimum ROC-AUC bar.

Exits with code 1 if any check fails, so it can be used in CI.

Run from the project root (with venv active):
    python src/validation/validate.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_features.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

TARGET = "Churn"
MIN_ROWS = 1000
MIN_AUC = 0.80


def validate_data() -> list[str]:
    issues: list[str] = []
    if not FEATURES_PATH.exists():
        return [f"features file missing: {FEATURES_PATH}"]

    df = pd.read_csv(FEATURES_PATH)
    if df.shape[0] < MIN_ROWS:
        issues.append(f"too few rows: {df.shape[0]} < {MIN_ROWS}")
    if df.isnull().any().any():
        issues.append("null values present in features")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        issues.append("target is not strictly 0/1")
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"non-numeric columns present: {non_numeric}")
    return issues


def validate_model() -> list[str]:
    if not METRICS_PATH.exists():
        return [f"metrics file missing: {METRICS_PATH}"]
    metrics = json.loads(METRICS_PATH.read_text())
    issues = [f"{name} ROC-AUC {m['roc_auc']} below bar {MIN_AUC}"
              for name, m in metrics.items() if m["roc_auc"] < MIN_AUC]
    return issues


def run() -> None:
    issues = validate_data() + validate_model()
    if issues:
        logger.error("VALIDATION FAILED:")
        for i in issues:
            logger.error("  - %s", i)
        sys.exit(1)
    logger.info("All validation checks passed.")


if __name__ == "__main__":
    run()