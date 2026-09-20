"""
Synthetic / Simulation-Inspired BTMS Dataset Generator
======================================================
Generates a synthetic dataset for EV battery thermal management research.

DISCLAIMER: This is synthetic/simulation-inspired data for demonstration purposes.
It is NOT experimentally validated battery sensor data.

The dataset models physically sensible (but simplified) relationships between
battery operating conditions, cooling system parameters, and thermal outcomes.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_btms_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic BTMS dataset with physically sensible relationships."""
    rng = np.random.default_rng(seed)

    # ── Operating condition scenarios ──────────────────────────────
    # Create a mix of normal, caution, high-risk, and critical scenarios
    scenario_probs = [0.50, 0.25, 0.15, 0.10]  # normal, caution, high, critical
    scenarios = rng.choice(4, size=n_samples, p=scenario_probs)

    # ── Base feature generation per scenario ──────────────────────
    battery_current = np.zeros(n_samples)
    ambient_temperature = np.zeros(n_samples)
    discharge_rate = np.zeros(n_samples)
    coolant_flow_rate = np.zeros(n_samples)
    coolant_inlet_temperature = np.zeros(n_samples)

    for i in range(n_samples):
        s = scenarios[i]
        if s == 0:  # Normal
            battery_current[i] = rng.uniform(1.0, 5.0)
            ambient_temperature[i] = rng.uniform(20.0, 30.0)
            discharge_rate[i] = rng.uniform(0.5, 1.5)
            coolant_flow_rate[i] = rng.uniform(0.015, 0.040)
            coolant_inlet_temperature[i] = rng.uniform(18.0, 25.0)
        elif s == 1:  # Caution
            battery_current[i] = rng.uniform(4.0, 9.0)
            ambient_temperature[i] = rng.uniform(25.0, 35.0)
            discharge_rate[i] = rng.uniform(1.0, 2.5)
            coolant_flow_rate[i] = rng.uniform(0.010, 0.030)
            coolant_inlet_temperature[i] = rng.uniform(22.0, 30.0)
        elif s == 2:  # High risk
            battery_current[i] = rng.uniform(7.0, 13.0)
            ambient_temperature[i] = rng.uniform(30.0, 40.0)
            discharge_rate[i] = rng.uniform(2.0, 4.0)
            coolant_flow_rate[i] = rng.uniform(0.005, 0.020)
            coolant_inlet_temperature[i] = rng.uniform(28.0, 35.0)
        else:  # Critical
            battery_current[i] = rng.uniform(10.0, 15.0)
            ambient_temperature[i] = rng.uniform(35.0, 45.0)
            discharge_rate[i] = rng.uniform(3.0, 5.0)
            coolant_flow_rate[i] = rng.uniform(0.005, 0.015)
            coolant_inlet_temperature[i] = rng.uniform(30.0, 40.0)

    # ── Other features ────────────────────────────────────────────
    battery_voltage = rng.uniform(3.2, 4.2, n_samples)
    state_of_charge = rng.uniform(10.0, 100.0, n_samples)
    coolant_pressure = rng.uniform(80000, 200000, n_samples)
    nanoparticle_concentration = rng.uniform(0.0, 5.0, n_samples)
    microchannel_width = rng.uniform(0.3, 1.0, n_samples)
    microchannel_height = rng.uniform(0.2, 0.8, n_samples)

    # Reynolds number (correlated with flow rate and channel dimensions)
    hydraulic_diameter = (2 * microchannel_width * microchannel_height) / (
        microchannel_width + microchannel_height
    )
    reynolds_number = (
        coolant_flow_rate * hydraulic_diameter * 1000 / (0.001 * microchannel_width * microchannel_height)
    )
    reynolds_number = np.clip(reynolds_number + rng.normal(0, 50, n_samples), 200, 8000)

    # Heat transfer coefficient (correlated with Re, nanoparticles, flow)
    base_htc = 500 + 2.5 * reynolds_number + 100 * nanoparticle_concentration
    heat_transfer_coefficient = base_htc + rng.normal(0, 80, n_samples)
    heat_transfer_coefficient = np.clip(heat_transfer_coefficient, 300, 25000)

    # ── Battery temperature (correlated with operating conditions) ─
    # Starts from ambient, modified by current and discharge
    battery_temperature = (
        ambient_temperature
        + 0.8 * battery_current
        + 0.5 * discharge_rate
        + rng.normal(0, 1.0, n_samples)
    )
    battery_temperature = np.clip(battery_temperature, 20.0, 55.0)

    # ── Target: max_battery_temperature_C ─────────────────────────
    # Nonlinear heat generation model
    # Q_gen ∝ I² * R_internal (Joule heating, quadratic in current)
    internal_resistance = 0.05 + 0.02 * (1 - state_of_charge / 100)  # higher R at low SoC
    heat_generation = battery_current**2 * internal_resistance

    # Additional heat from discharge rate (exponential effect at high C-rates)
    discharge_heat = 0.3 * discharge_rate**1.5

    # Cooling effectiveness
    cooling_effectiveness = (
        0.001 * heat_transfer_coefficient
        * coolant_flow_rate
        * (1 + 0.1 * nanoparticle_concentration)
        / (0.5 + 0.001 * coolant_inlet_temperature)
    )

    # Temperature rise model (nonlinear)
    temperature_rise = (
        (heat_generation + discharge_heat)
        / (0.5 + cooling_effectiveness)
        + 0.3 * (ambient_temperature - 25.0)
        + rng.normal(0, 0.8, n_samples)
    )
    temperature_rise = np.clip(temperature_rise, 0.0, 30.0)

    max_battery_temperature = ambient_temperature + temperature_rise
    max_battery_temperature = np.clip(max_battery_temperature, 22.0, 65.0)

    # ── Derived risk metrics ──────────────────────────────────────
    # Thermal risk score (0–100)
    thermal_risk_score = np.clip(
        (max_battery_temperature - 35) / (55 - 35) * 100
        + 10 * (battery_current / 15)
        + rng.normal(0, 3, n_samples),
        0, 100
    )

    # Thermal risk label
    thermal_risk_label = np.where(
        thermal_risk_score < 25, "NORMAL",
        np.where(
            thermal_risk_score < 50, "CAUTION",
            np.where(thermal_risk_score < 75, "HIGH", "CRITICAL")
        )
    )

    # Hotspot risk score (0–100)
    hotspot_risk_score = np.clip(
        0.6 * thermal_risk_score
        + 15 * (battery_current / 15)
        + 10 * (discharge_rate / 5)
        - 20 * (coolant_flow_rate / 0.04)
        + rng.normal(0, 5, n_samples),
        0, 100
    )

    # ── Assemble DataFrame ────────────────────────────────────────
    df = pd.DataFrame({
        "battery_current_A": np.round(battery_current, 3),
        "battery_voltage_V": np.round(battery_voltage, 3),
        "state_of_charge_percent": np.round(state_of_charge, 2),
        "ambient_temperature_C": np.round(ambient_temperature, 2),
        "battery_temperature_C": np.round(battery_temperature, 2),
        "coolant_inlet_temperature_C": np.round(coolant_inlet_temperature, 2),
        "coolant_flow_rate_kg_s": np.round(coolant_flow_rate, 5),
        "coolant_pressure_Pa": np.round(coolant_pressure, 1),
        "nanoparticle_concentration_percent": np.round(nanoparticle_concentration, 3),
        "reynolds_number": np.round(reynolds_number, 1),
        "heat_transfer_coefficient_W_m2K": np.round(heat_transfer_coefficient, 2),
        "microchannel_width_mm": np.round(microchannel_width, 3),
        "microchannel_height_mm": np.round(microchannel_height, 3),
        "discharge_rate_C": np.round(discharge_rate, 3),
        "max_battery_temperature_C": np.round(max_battery_temperature, 2),
        "temperature_rise_C": np.round(temperature_rise, 2),
        "thermal_risk_score": np.round(thermal_risk_score, 2),
        "thermal_risk_label": thermal_risk_label,
        "hotspot_risk_score": np.round(hotspot_risk_score, 2),
    })

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Print validation summary of the generated dataset."""
    print("=" * 60)
    print("  SYNTHETIC BTMS DATASET — VALIDATION REPORT")
    print("=" * 60)
    print(f"  Rows            : {len(df):,}")
    print(f"  Features        : {df.shape[1]}")
    print(f"  Missing values  : {df.isnull().sum().sum()}")
    print("-" * 60)
    print(f"  Min temperature : {df['max_battery_temperature_C'].min():.2f} °C")
    print(f"  Max temperature : {df['max_battery_temperature_C'].max():.2f} °C")
    print(f"  Mean temperature: {df['max_battery_temperature_C'].mean():.2f} °C")
    print(f"  Std temperature : {df['max_battery_temperature_C'].std():.2f} °C")
    print("-" * 60)
    print("  Risk distribution:")
    risk_counts = df["thermal_risk_label"].value_counts()
    for label in ["NORMAL", "CAUTION", "HIGH", "CRITICAL"]:
        count = risk_counts.get(label, 0)
        pct = count / len(df) * 100
        print(f"    {label:<10}: {count:>5} ({pct:5.1f}%)")
    print("=" * 60)

    # Sanity checks
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values!"
    assert (df["max_battery_temperature_C"] >= 20).all(), "Impossible low temperature!"
    assert (df["max_battery_temperature_C"] <= 70).all(), "Impossible high temperature!"
    assert (df["battery_current_A"] >= 0).all(), "Negative current!"
    assert (df["coolant_flow_rate_kg_s"] > 0).all(), "Non-positive coolant flow!"
    print("\n  ✓ All validation checks passed.\n")
