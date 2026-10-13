"""
EV Battery Mission Control — Model Evaluation Module
=====================================================
Calculates comprehensive safety-critical classification metrics,
ROC-AUC, Precision-Recall curves, confusion matrices, and risk thresholds.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import joblib


def evaluate_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate trained classifier on held-out test telemetry.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= decision_threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, probs))
    pr_auc = float(average_precision_score(y_test, probs))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "decision_threshold": decision_threshold,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "critical_safety_recall": round(float(recall), 4),
        "false_alarm_rate": round(float(fp / max(1, fp + tn)), 4),
    }


def find_optimal_safety_threshold(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float = 0.95,
) -> float:
    """
    Find threshold guaranteeing high recall (>= min_recall) to prevent
    unflagged battery thermal runaway incidents.
    """
    probs = model.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = 0.0

    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        preds = (probs >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="binary", zero_division=0
        )
        if recall >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return round(best_threshold, 3)
