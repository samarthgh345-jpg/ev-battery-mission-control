"""
Data Preprocessing
==================
Feature/target split, train/test split, and feature scaling utilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Input features for the XGBoost temperature model
FEATURE_COLUMNS = [
    "battery_current_A",
    "battery_voltage_V",
    "state_of_charge_percent",
    "ambient_temperature_C",
    "battery_temperature_C",
    "coolant_inlet_temperature_C",
    "coolant_flow_rate_kg_s",
    "coolant_pressure_Pa",
    "nanoparticle_concentration_percent",
    "reynolds_number",
    "heat_transfer_coefficient_W_m2K",
    "microchannel_width_mm",
    "microchannel_height_mm",
    "discharge_rate_C",
]

TARGET_COLUMN = "max_battery_temperature_C"

# Features used for anomaly detection
ANOMALY_FEATURES = [
    "battery_current_A",
    "battery_voltage_V",
    "ambient_temperature_C",
    "battery_temperature_C",
    "coolant_flow_rate_kg_s",
    "coolant_inlet_temperature_C",
    "max_battery_temperature_C",
]


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
    """80/20 train/test split with stratification by risk bracket."""
    # Create risk brackets for stratified split
    bins = [0, 35, 42, 50, 100]
    labels_strat = pd.cut(y, bins=bins, labels=["low", "med", "high", "vhigh"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=labels_strat
    )
    return X_train, X_test, y_train, y_test


def get_feature_display_names() -> dict:
    """Human-readable names for features."""
    return {
        "battery_current_A": "Battery Current (A)",
        "battery_voltage_V": "Battery Voltage (V)",
        "state_of_charge_percent": "State of Charge (%)",
        "ambient_temperature_C": "Ambient Temperature (°C)",
        "battery_temperature_C": "Battery Temperature (°C)",
        "coolant_inlet_temperature_C": "Coolant Inlet Temp (°C)",
        "coolant_flow_rate_kg_s": "Coolant Flow Rate (kg/s)",
        "coolant_pressure_Pa": "Coolant Pressure (Pa)",
        "nanoparticle_concentration_percent": "Nanoparticle Conc. (%)",
        "reynolds_number": "Reynolds Number",
        "heat_transfer_coefficient_W_m2K": "Heat Transfer Coeff (W/m²K)",
        "microchannel_width_mm": "Microchannel Width (mm)",
        "microchannel_height_mm": "Microchannel Height (mm)",
        "discharge_rate_C": "Discharge Rate (C)",
    }
