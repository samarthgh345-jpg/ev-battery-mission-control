"""
Simulation Engine
==================
Generates simulated sensor streams for live monitoring mode based on the new 14 features.
"""

import numpy as np
from src.prediction_engine import get_default_features


def get_simulation_state(mode: str, step: int, seed: int = 42) -> dict:
    """
    Generate a simulated sensor state for a given mode and time step.
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
        features["cell_temperature_avg"] = 30.0 + combined_drift * 1.2 + noise(0.2)
        features["cell_temperature_max"] = features["cell_temperature_avg"] + 1.5 + noise(0.1)
        features["internal_resistance"] = 1.5 + combined_drift * 0.05 + noise(0.01)
        features["cooling_system_health"] = 98.0 + noise(1.0)
        features["average_charge_power_kw"] = 45.0 + combined_drift * 5 + noise(2.0)

    elif mode == "HIGH_LOAD":
        ramp = min(step * 0.5, 20.0)
        features["average_charge_power_kw"] = 60.0 + ramp + combined_drift * 5 + noise(3.0)
        features["cell_temperature_avg"] = 32.0 + ramp * 0.5 + combined_drift * 1.5 + noise(0.5)
        features["cell_temperature_max"] = features["cell_temperature_avg"] + 2.5 + ramp * 0.1 + noise(0.2)
        features["internal_resistance"] = 1.6 + ramp * 0.02 + noise(0.05)
        features["cooling_system_health"] = 95.0 - ramp * 0.2 + noise(1.0)

    elif mode == "THERMAL_STRESS":
        ramp = min(step * 1.0, 40.0)
        features["average_charge_power_kw"] = 70.0 + ramp * 0.5 + combined_drift * 5 + noise(3.0)
        features["cell_temperature_avg"] = 35.0 + ramp * 0.8 + combined_drift * 2.0 + noise(0.8)
        features["cell_temperature_max"] = features["cell_temperature_avg"] + 5.0 + ramp * 0.3 + noise(0.5)
        features["internal_resistance"] = 1.8 + ramp * 0.05 + noise(0.1)
        features["cooling_system_health"] = max(10.0, 90.0 - ramp * 1.5) + noise(2.0)
        features["fast_charge_ratio"] = 0.8 + noise(0.05)

    # Clamp to valid ranges
    features["cell_temperature_avg"] = max(15.0, min(80.0, features["cell_temperature_avg"]))
    features["cell_temperature_max"] = max(features["cell_temperature_avg"], min(85.0, features["cell_temperature_max"]))
    features["internal_resistance"] = max(0.5, min(10.0, features["internal_resistance"]))
    features["cooling_system_health"] = max(0.0, min(100.0, features["cooling_system_health"]))
    features["average_charge_power_kw"] = max(0.0, min(200.0, features["average_charge_power_kw"]))
    features["state_of_charge"] = max(0.0, min(100.0, 80.0 - step * 0.2))
    features["fast_charge_ratio"] = max(0.0, min(1.0, features["fast_charge_ratio"]))

    return features

def get_thermal_stress_trajectory(n_steps: int = 15, seed: int = 42) -> list:
    """Generate a pre-computed thermal stress trajectory for the demo."""
    trajectory = []
    for step in range(n_steps):
        state = get_simulation_state("THERMAL_STRESS", step, seed)
        trajectory.append(state)
    return trajectory
