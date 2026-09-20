"""
Simulation Engine
==================
Generates simulated sensor streams for live monitoring mode.

Modes:
  - NORMAL: Stable operating conditions
  - HIGH_LOAD: Increasing current and discharge rate
  - THERMAL_STRESS: Increasing current, increasing ambient temp, decreasing coolant flow

DISCLAIMER: This is a simulated sensor stream for demonstration.
Not based on actual battery hardware.
"""

import numpy as np
from src.prediction_engine import get_default_features


def get_simulation_state(mode: str, step: int, seed: int = 42) -> dict:
    """
    Generate a simulated sensor state for a given mode and time step.

    Args:
        mode: "NORMAL", "HIGH_LOAD", or "THERMAL_STRESS"
        step: Current simulation step (0, 1, 2, ...)
        seed: Random seed for reproducibility

    Returns:
        Feature dictionary representing current sensor readings.
    """
    rng = np.random.default_rng(seed + step)
    features = get_default_features()
    noise = lambda scale=0.3: rng.normal(0, scale)
    
    # Pseudo-random realistic drift using multiple frequencies
    drift1 = np.sin(step * 0.1)
    drift2 = np.sin(step * 0.035) * 1.5
    drift3 = np.cos(step * 0.01) * 2.0
    combined_drift = (drift1 + drift2 + drift3) / 3.0

    if mode == "NORMAL":
        features["battery_current_A"] = 4.0 + combined_drift * 1.5 + noise(0.15)
        features["ambient_temperature_C"] = 27.0 + combined_drift * 0.8 + noise(0.2)
        features["battery_temperature_C"] = 33.0 + combined_drift * 1.2 + noise(0.25)
        features["coolant_flow_rate_kg_s"] = 0.025 + combined_drift * 0.002 + noise(0.0005)
        features["coolant_inlet_temperature_C"] = 22.0 + combined_drift * 0.5 + noise(0.2)
        features["discharge_rate_C"] = 1.0 + combined_drift * 0.2 + noise(0.05)
        features["battery_voltage_V"] = 3.8 - combined_drift * 0.05 + noise(0.02)

    elif mode == "HIGH_LOAD":
        ramp = min(step * 0.3, 8.0)
        features["battery_current_A"] = 5.0 + ramp + combined_drift * 1.0 + noise(0.2)
        features["ambient_temperature_C"] = 28.0 + combined_drift * 0.5 + noise(0.2)
        features["battery_temperature_C"] = 34.0 + ramp * 0.6 + combined_drift * 1.0 + noise(0.3)
        features["coolant_flow_rate_kg_s"] = 0.022 + noise(0.001)
        features["coolant_inlet_temperature_C"] = 23.0 + combined_drift * 0.5 + noise(0.3)
        features["discharge_rate_C"] = 1.5 + min(step * 0.1, 2.5) + noise(0.1)
        features["battery_voltage_V"] = 3.7 - min(step * 0.01, 0.3) + noise(0.03)

    elif mode == "THERMAL_STRESS":
        ramp = min(step * 0.4, 10.0)
        features["battery_current_A"] = 6.0 + ramp + combined_drift * 1.5 + noise(0.25)
        features["ambient_temperature_C"] = 30.0 + min(step * 0.3, 12.0) + combined_drift * 0.5 + noise(0.2)
        features["battery_temperature_C"] = 36.0 + ramp * 0.9 + combined_drift * 1.0 + noise(0.3)
        features["coolant_flow_rate_kg_s"] = max(0.005, 0.025 - step * 0.0008) + noise(0.0005)
        features["coolant_inlet_temperature_C"] = 25.0 + min(step * 0.2, 8.0) + noise(0.3)
        features["discharge_rate_C"] = 2.0 + min(step * 0.15, 2.5) + noise(0.1)
        features["battery_voltage_V"] = 3.6 - min(step * 0.015, 0.3) + noise(0.03)

    # Clamp to valid ranges
    features["battery_current_A"] = max(0.5, min(15.0, features["battery_current_A"]))
    features["battery_voltage_V"] = max(3.0, min(4.3, features["battery_voltage_V"]))
    features["ambient_temperature_C"] = max(15.0, min(50.0, features["ambient_temperature_C"]))
    features["battery_temperature_C"] = max(20.0, min(60.0, features["battery_temperature_C"]))
    features["coolant_flow_rate_kg_s"] = max(0.003, min(0.04, features["coolant_flow_rate_kg_s"]))
    features["coolant_inlet_temperature_C"] = max(15.0, min(42.0, features["coolant_inlet_temperature_C"]))
    features["discharge_rate_C"] = max(0.5, min(5.0, features["discharge_rate_C"]))
    features["state_of_charge_percent"] = max(10, 80 - step * 0.5)

    return features


def get_thermal_stress_trajectory(n_steps: int = 15, seed: int = 42) -> list:
    """
    Generate a pre-computed thermal stress trajectory for the demo.

    Returns list of feature dicts showing progressive thermal stress.
    """
    trajectory = []
    for step in range(n_steps):
        state = get_simulation_state("THERMAL_STRESS", step, seed)
        trajectory.append(state)
    return trajectory
