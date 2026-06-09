"""Training for the Telco customer churn project.

Loads the model-ready features, splits into train/test (stratified),
trains a baseline Logistic Regression and an XGBoost model -- both with
class-imbalance handling -- reports validation metrics, and saves the
trained models to models/.

Run from the project root (with venv active):
    python src/training/train.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_xy():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features not found at {FEATURES_PATH}. "
            f"Run: python src/features/feature_engineering.py first."
        )
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    logger.info("Loaded features: %d rows, %d feature columns.", X.shape[0], X.shape[1])
    return X, y


def train_logreg(X_train, y_train) -> Pipeline:
    """Logistic Regression needs scaling, so wrap it in a pipeline."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train, y_train) -> XGBClassifier:
    """XGBoost: weight the positive (churn) class to fight imbalance."""
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / pos, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(name, model, X_test, y_test) -> float:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, proba)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, preds, target_names=["stay", "churn"]))
    return auc


def run():
    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    logreg = train_logreg(X_train, y_train)
    results["logistic_regression"] = (logreg, evaluate("Logistic Regression", logreg, X_test, y_test))

    xgb = train_xgb(X_train, y_train)
    results["xgboost"] = (xgb, evaluate("XGBoost", xgb, X_test, y_test))

    for name, (model, auc) in results.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
        logger.info("Saved %s (ROC-AUC %.3f).", name, auc)

    # Save the feature column order so the prediction API can rebuild inputs.
    joblib.dump(list(X.columns), MODELS_DIR / "feature_columns.joblib")

    best = max(results, key=lambda k: results[k][1])
    logger.info("Best model: %s (ROC-AUC %.3f).", best, results[best][1])


if __name__ == "__main__":
    run()