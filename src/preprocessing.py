"""
Data Preprocessing
==================
Feature/target split, train/test split, and feature scaling utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Input features for the EV Battery Failure model
FEATURE_COLUMNS = [
    "cycle_count",
    "state_of_charge",
    "depth_of_discharge",
    "cell_voltage_avg",
    "cell_voltage_std",
    "cell_temperature_avg",
    "cell_temperature_max",
    "internal_resistance",
    "charge_efficiency",
    "fast_charge_ratio",
    "average_charge_power_kw",
    "charging_interruptions",
    "cooling_system_health",
    "average_ambient_temperature",
]

TARGET_COLUMN = "battery_failure"

# Features used for anomaly detection
ANOMALY_FEATURES = FEATURE_COLUMNS.copy()


def load_dataset(path: str) -> pd.DataFrame:
    """Load the BTMS dataset from CSV."""
    return pd.read_csv(path)


def get_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split into features (X) and target (y). No target-derived columns in X."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """80/20 train/test split with stratification by binary target."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_feature_display_names() -> dict:
    """Human-readable names for features."""
    return {
        "cycle_count": "Cycle Count",
        "state_of_charge": "State of Charge (%)",
        "depth_of_discharge": "Depth of Discharge (%)",
        "cell_voltage_avg": "Avg Cell Voltage (V)",
        "cell_voltage_std": "Cell Voltage Imbalance (Std)",
        "cell_temperature_avg": "Avg Cell Temp (°C)",
        "cell_temperature_max": "Max Cell Temp (°C)",
        "internal_resistance": "Internal Resistance (mOhm)",
        "charge_efficiency": "Charge Efficiency",
        "fast_charge_ratio": "Fast Charge Ratio",
        "average_charge_power_kw": "Avg Charge Power (kW)",
        "charging_interruptions": "Charging Interruptions",
        "cooling_system_health": "Cooling System Health (%)",
        "average_ambient_temperature": "Ambient Temp (°C)",
    }
