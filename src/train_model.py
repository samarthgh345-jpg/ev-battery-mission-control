"""
EV Battery Mission Control — Model Training Pipeline
=====================================================
Trains baseline and XGBoost classification models for battery failure prediction.
Saves model artifacts, scalers, and metadata to models/ directory.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

from src.data_preprocessing import (
    check_dataset_availability,
    load_raw_dataset,
    handle_missing_values,
    investigate_target_leakage,
    split_features_and_target,
    create_stratified_splits,
    DEFAULT_RAW_DATA_PATH,
)

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")


def train_models(
    data_path: Optional[Path] = None,
    output_dir: Path = MODELS_DIR,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train baseline and XGBoost failure prediction models.

    Raises:
        FileNotFoundError: If the dataset is not yet present.
    """
    path = Path(data_path) if data_path else DEFAULT_RAW_DATA_PATH

    if not check_dataset_availability(path):
        raise FileNotFoundError(
            f"Cannot train models: Raw dataset not found at '{path}'.\n"
            f"The application is running in DEMO MODE.\n"
            f"Add the dataset to '{path}' and rerun this script."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading telemetry dataset from {path}...")
    df = load_raw_dataset(path, nrows=sample_size)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # Clean missing values
    df_clean = handle_missing_values(df)

    # Check for target leakage
    leakage_info = investigate_target_leakage(df_clean)
    cols_to_drop = [col for col, _ in leakage_info.get("high_correlation_leakage", [])]
    if cols_to_drop:
        print(f"Excluding suspect leakage columns: {cols_to_drop}")

    # Feature target separation
    X, y = split_features_and_target(df_clean, drop_leakage_cols=cols_to_drop)

    # Convert non-numeric if any
    X_numeric = pd.get_dummies(X, drop_first=True)
    feature_names = list(X_numeric.columns)

    # Stratified splits
    splits = create_stratified_splits(X_numeric, y, train_size=0.70, val_size=0.15, test_size=0.15)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    # 1. Baseline Model
    print("Training Baseline Dummy Classifier...")
    baseline = DummyClassifier(strategy="stratified", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_val_preds = baseline.predict_proba(X_val)[:, 1]
    baseline_auc = roc_auc_score(y_val, baseline_val_preds)

    # 2. XGBoost Model
    print("Training XGBoost Classifier...")
    pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    xgb_val_probs = xgb.predict_proba(X_val)[:, 1]
    xgb_val_preds = xgb.predict(X_val)
    xgb_auc = roc_auc_score(y_val, xgb_val_probs)
    xgb_f1 = f1_score(y_val, xgb_val_preds)

    print(f"Validation ROC-AUC: Baseline={baseline_auc:.4f} | XGBoost={xgb_auc:.4f}")
    print(f"Validation F1-Score: XGBoost={xgb_f1:.4f}")

    # Save artifacts
    model_path = output_dir / "xgboost_failure_model.joblib"
    joblib.dump(xgb, model_path)
    print(f"Saved XGBoost model to {model_path}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "XGBClassifier",
        "num_features": len(feature_names),
        "features": feature_names,
        "validation_metrics": {
            "baseline_roc_auc": round(float(baseline_auc), 4),
            "xgboost_roc_auc": round(float(xgb_auc), 4),
            "xgboost_f1": round(float(xgb_f1), 4),
        },
        "hyperparameters": xgb.get_params(),
    }

    meta_path = output_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {meta_path}")

    return {
        "model": xgb,
        "metadata": metadata,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    try:
        train_models()
    except FileNotFoundError as e:
        print(f"\n[DEMO MODE INFO]\n{e}")
