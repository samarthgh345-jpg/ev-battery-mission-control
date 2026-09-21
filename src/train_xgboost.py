"""
XGBoost Temperature Model Training
====================================
Train, evaluate, and save the XGBoost regression model.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import joblib

from src.preprocessing import (
    load_dataset,
    get_features_and_target,
    split_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """Train an XGBoost classifier."""
    # Calculate scale_pos_weight
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Compute evaluation metrics on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred)), 4),
        "Recall": round(float(recall_score(y_test, y_pred)), 4),
        "F1_Score": round(float(f1_score(y_test, y_pred)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
        "PR_AUC": round(float(average_precision_score(y_test, y_prob)), 4),
    }


def save_model(model: XGBClassifier, model_dir: Path) -> Path:
    """Save trained model to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_failure_model.joblib"
    joblib.dump(model, model_path)
    return model_path


def save_metadata(
    metrics: dict,
    n_train: int,
    n_test: int,
    feature_names: list,
    model_dir: Path,
    y_test: pd.Series = None,
    y_pred: np.ndarray = None,
) -> Path:
    """Save model metadata and metrics."""
    metadata = {
        "model_type": "XGBoost Classifier (Baseline)",
        "dataset": "200K EV Battery Failure Dataset",
        "disclaimer": "Prototype model trained on synthetic data. Not experimentally validated.",
        "target": TARGET_COLUMN,
        "features": feature_names,
        "n_features": len(feature_names),
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "n_total_samples": n_train + n_test,
        "test_size": 0.20,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(),
    }

    meta_path = model_dir / "model_metadata_xgb.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    return meta_path


def run_training_pipeline(data_path: str, model_dir: str) -> dict:
    """Full training pipeline: load → split → train → evaluate → save."""
    data_path = Path(data_path)
    model_dir = Path(model_dir)

    print("\n[INFO] Loading dataset...")
    df = load_dataset(str(data_path))
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[INFO] Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train)

    print("\n[INFO] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    y_pred = model.predict(X_test)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n[INFO] Saving model and metadata...")
    model_path = save_model(model, model_dir)
    meta_path = save_metadata(
        metrics, len(X_train), len(X_test), FEATURE_COLUMNS, model_dir, y_test, y_pred
    )
    print(f"  Model: {model_path}")
    print(f"  Metadata: {meta_path}")

    return {"model": model, "metrics": metrics, "X_test": X_test, "y_test": y_test}
