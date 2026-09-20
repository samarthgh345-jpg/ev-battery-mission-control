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
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.preprocessing import (
    load_dataset,
    get_features_and_target,
    split_data,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def train_xgboost_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


def evaluate_model(
    model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """Compute evaluation metrics on the test set."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def save_model(model: XGBRegressor, model_dir: Path) -> Path:
    """Save trained model to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_temperature_model.joblib"
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
        "model_type": "XGBoost Regressor",
        "dataset": "Synthetic / Simulation-Inspired BTMS Dataset",
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
    if y_test is not None and y_pred is not None:
        metadata["test_actual"] = [round(float(v), 4) for v in y_test.values.tolist()]
        metadata["test_predicted"] = [round(float(v), 4) for v in y_pred.tolist()]

    meta_path = model_dir / "model_metadata.json"
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
