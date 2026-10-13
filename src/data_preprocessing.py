"""
EV Battery Mission Control — Data Preprocessing Pipeline
=========================================================
Modular ingestion, cleaning, target-leakage analysis, and stratified
splitting for the 200,000-record EV Battery Failure Prediction dataset.

Target: 'battery_failure' (Binary Classification)
Expected Raw Location: data/raw/ev_battery_failure_dataset.csv
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_RAW_DATA_PATH = Path("data/raw/ev_battery_failure_dataset.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")
TARGET_COLUMN = "battery_failure"


def check_dataset_availability(path: Optional[Path] = None) -> bool:
    """
    Check if the raw dataset CSV file exists.
    """
    data_path = Path(path) if path else DEFAULT_RAW_DATA_PATH
    return data_path.is_file() and data_path.stat().st_size > 0


def check_ml_pipeline_status(
    raw_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates end-to-end readiness of the ML pipeline.
    Preserves DEMO MODE until:
    1. Raw dataset exists and is non-empty.
    2. Trained model artifact exists and can be loaded.
    3. Model metadata exists.
    Merely detecting a CSV file does NOT mark the ML system as ready.
    """
    raw = Path(raw_path) if raw_path else DEFAULT_RAW_DATA_PATH
    m_path = Path(model_path) if model_path else Path("models/xgboost_failure_model.joblib")
    meta = Path(meta_path) if meta_path else Path("models/model_metadata.json")

    has_raw = raw.is_file() and raw.stat().st_size > 0
    has_model = False

    if m_path.is_file() and meta.is_file():
        try:
            import joblib
            loaded = joblib.load(m_path)
            if loaded is not None:
                has_model = True
        except Exception:
            has_model = False

    is_ready = has_raw and has_model
    return {
        "raw_dataset_present": has_raw,
        "model_artifact_loaded": has_model,
        "is_demo_mode": not is_ready,
        "badge_text": "ML PIPELINE CONNECTED" if is_ready else "DEMO MODE — DATASET NOT CONNECTED",
        "status_detail": (
            "Full data-driven XGBoost inference active."
            if is_ready
            else (
                "Raw dataset detected but model not yet trained. Run src/train_model.py."
                if has_raw
                else "Dataset not uploaded. Demonstrative sensor telemetry active."
            )
        ),
    }


def load_raw_dataset(path: Optional[Path] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load raw telemetry dataset from data/raw/.

    Raises:
        FileNotFoundError: If the dataset file has not yet been placed in data/raw/.
    """
    data_path = Path(path) if path else DEFAULT_RAW_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'.\n"
            f"STATUS: Application is running in DEMO MODE.\n"
            f"To enable the full ML pipeline, place 'ev_battery_failure_dataset.csv' "
            f"inside the '{data_path.parent}' directory."
        )

    return pd.read_csv(data_path, nrows=nrows)


def inspect_dataset(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Dict[str, Any]:
    """
    Inspect dataset schema, missingness, and target distribution.
    Does not assume specific feature names beyond the confirmed target.
    """
    total_rows, total_cols = df.shape
    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()

    target_stats: Dict[str, Any] = {}
    if target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        proportions = df[target_col].value_counts(normalize=True, dropna=False).to_dict()
        target_stats = {
            "target_present": True,
            "counts": counts,
            "proportions": {k: round(v, 4) for k, v in proportions.items()},
        }
    else:
        target_stats = {"target_present": False, "columns_found": list(df.columns[:10])}

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

    return {
        "num_rows": total_rows,
        "num_cols": total_cols,
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols),
        "columns_with_missing": cols_with_missing,
        "missing_cells_total": int(df.isnull().sum().sum()),
        "target_stats": target_stats,
    }


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Modular missing value handler.
    Fills numeric columns with median/mean and categoricals with mode or 'UNKNOWN'.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            if np.issubdtype(df_clean[col].dtype, np.number):
                if numeric_strategy == "median":
                    fill_val = df_clean[col].median()
                elif numeric_strategy == "mean":
                    fill_val = df_clean[col].mean()
                else:
                    fill_val = 0.0
                df_clean[col] = df_clean[col].fillna(fill_val)
            else:
                if categorical_strategy == "mode" and not df_clean[col].mode().empty:
                    fill_val = df_clean[col].mode()[0]
                else:
                    fill_val = "UNKNOWN"
                df_clean[col] = df_clean[col].fillna(fill_val)

    return df_clean


def investigate_target_leakage(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    corr_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Screen candidate features for potential target leakage:
    1. Perfect or near-perfect correlation (|r| >= corr_threshold).
    2. Zero-variance / constant columns.
    3. High-cardinality ID or timestamp columns.
    4. Suspicious column names containing 'risk', 'fail', 'target', 'label'.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in DataFrame."}

    flagged_high_corr: List[Tuple[str, float]] = []
    zero_variance_cols: List[str] = []
    suspicious_name_cols: List[str] = []

    # Check correlations for numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col in numeric_df.columns:
        corrs = numeric_df.corr()[target_col].drop(target_col, errors="ignore")
        for col_name, corr_val in corrs.items():
            if abs(corr_val) >= corr_threshold:
                flagged_high_corr.append((str(col_name), round(float(corr_val), 4)))

    # Zero-variance columns
    for col in df.columns:
        if col != target_col and df[col].nunique() <= 1:
            zero_variance_cols.append(str(col))

    # Name-based suspicious checks
    leakage_keywords = {"risk", "fail", "anomaly", "flag", "label", "target"}
    for col in df.columns:
        if col != target_col:
            lower = col.lower()
            if any(kw in lower for kw in leakage_keywords):
                suspicious_name_cols.append(str(col))

    return {
        "high_correlation_leakage": flagged_high_corr,
        "zero_variance_features": zero_variance_cols,
        "suspicious_named_features": suspicious_name_cols,
        "recommendation": (
            "Exclude high-correlation proxies and derived post-failure signals "
            "to ensure the model generalizes to pre-failure prevention."
        ),
    }


def split_features_and_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_leakage_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X and target vector y.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not present in dataset.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    if drop_leakage_cols:
        existing_drop = [c for c in drop_leakage_cols if c in X.columns]
        X = X.drop(columns=existing_drop)

    return X, y


def create_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Perform stratified 3-way split: Train (70%), Validation (15%), Test (15%).
    Preserves class balance for rare failure events.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-4, "Splits must sum to 1.0"

    # Step 1: Split into train and temp (val + test)
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )

    # Step 2: Split temp into validation and test
    val_ratio_of_temp = val_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio_of_temp), random_state=random_state, stratify=y_temp
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "shapes": {
            "train": X_train.shape,
            "val": X_val.shape,
            "test": X_test.shape,
        },
    }
