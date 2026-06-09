"""Evaluation for the Telco customer churn project.

Loads the saved models, evaluates them on the held-out test set, and writes:
  - metrics (ROC-AUC, precision, recall, F1) to reports/metrics.json
  - ROC curves, confusion matrix, feature importance, and a SHAP summary
    (explainability) to reports/figures/

Run from the project root (with venv active):
    python src/evaluation/evaluate.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    classification_report, roc_auc_score,
)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_test_set():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    # Same split as training, so the test set matches.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return X_test, y_test


def metrics_for(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "churn_precision": round(report["1"]["precision"], 4),
        "churn_recall": round(report["1"]["recall"], 4),
        "churn_f1": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
    }


def plot_roc(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC curves")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=120)
    plt.close(fig)


def plot_confusion(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["stay", "churn"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (XGBoost)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_importance(model, X_test):
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top = importances.sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    top.plot.barh(ax=ax)
    ax.set_title("Top 15 features (XGBoost importance)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def plot_shap(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("SHAP: what drives churn predictions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()


def run():
    if not (MODELS_DIR / "xgboost.joblib").exists():
        raise FileNotFoundError("Models not found. Run: python src/training/train.py first.")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_test, y_test = load_test_set()

    models = {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.joblib"),
        "xgboost": joblib.load(MODELS_DIR / "xgboost.joblib"),
    }

    all_metrics = {name: metrics_for(m, X_test, y_test) for name, m in models.items()}
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))
    logger.info("Metrics:\n%s", json.dumps(all_metrics, indent=2))

    plot_roc(models, X_test, y_test)
    xgb = models["xgboost"]
    plot_confusion(xgb, X_test, y_test)
    plot_importance(xgb, X_test)
    plot_shap(xgb, X_test)
    logger.info("Saved 4 figures to %s and metrics to %s", FIGURES_DIR, METRICS_PATH)


if __name__ == "__main__":
    run()