"""End-to-end pipeline for the Enterprise Customer Churn Prediction project.

Runs every stage in order with one command:
    ingestion -> preprocessing -> features -> training -> evaluation -> validation

Run from the project root (with venv active):
    python main.py
"""
from __future__ import annotations

import logging

from src.ingestion.data_ingestion import run as run_ingestion
from src.preprocessing.data_preprocessing import run as run_preprocessing
from src.features.feature_engineering import run as run_features
from src.training.train import run as run_training
from src.evaluation.evaluation import run as run_evaluation
from src.validation.validate import run as run_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")

STAGES = [
    ("Ingestion", run_ingestion),
    ("Preprocessing", run_preprocessing),
    ("Feature engineering", run_features),
    ("Training", run_training),
    ("Evaluation", run_evaluation),
    ("Validation", run_validation),
]


def main() -> None:
    total = len(STAGES)
    for i, (name, stage) in enumerate(STAGES, start=1):
        logger.info("=" * 60)
        logger.info("Stage %d/%d: %s", i, total, name)
        logger.info("=" * 60)
        stage()
    logger.info("=" * 60)
    logger.info("Pipeline complete. Model and reports are ready.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()