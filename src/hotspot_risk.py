"""
Hotspot Risk Estimation
========================
Estimates hotspot risk score and generates a simulated cell temperature grid
for the digital twin visualization.

DISCLAIMER: This is a simulated/estimated hotspot risk based on operating
conditions. It is NOT based on actual cell-level thermal measurements.
"""

import numpy as np


def estimate_hotspot_risk(
    current_temp: float,
    battery_current: float,
    discharge_rate: float,
    coolant_flow: float,
    ambient_temp: float,
) -> dict:
    """
    Estimate hotspot risk score from operating conditions.

    Returns:
        {
            "hotspot_risk_percent": float (0-100),
            "status": "LOW" | "MODERATE" | "ELEVATED" | "HIGH" | "CRITICAL",
            "color": str
        }
    """
    # Normalized contributions
    temp_factor = max(0, (current_temp - 35) / 25) * 35
    current_factor = (battery_current / 15) * 20
    discharge_factor = (discharge_rate / 5) * 15
    flow_penalty = max(0, (0.025 - coolant_flow) / 0.025) * 15
    ambient_factor = max(0, (ambient_temp - 25) / 20) * 15

    risk = temp_factor + current_factor + discharge_factor + flow_penalty + ambient_factor
    risk = max(0, min(100, risk))

    if risk < 20:
        status, color = "LOW", "#00E676"
    elif risk < 40:
        status, color = "MODERATE", "#FFAB00"
    elif risk < 60:
        status, color = "ELEVATED", "#FF9100"
    elif risk < 80:
        status, color = "HIGH", "#FF6D00"
    else:
        status, color = "CRITICAL", "#FF1744"

    return {
        "hotspot_risk_percent": round(risk, 1),
        "status": status,
        "color": color,
    }


def generate_cell_temperature_grid(
    base_temp: float,
    hotspot_risk_percent: float,
    rows: int = 3,
    cols: int = 4,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a simulated cell temperature grid for digital twin visualization.

    This is a SIMULATED cell thermal map — not based on actual cell-level
    thermal sensor data.

    The grid creates a realistic-looking temperature distribution where
    cells near the center tend to be warmer, with one "hotspot" cell
    whose temperature is influenced by the hotspot risk score.
    """
    rng = np.random.default_rng(seed)

    grid = np.full((rows, cols), base_temp, dtype=float)

    # Add spatial variation (center cells warmer)
    for r in range(rows):
        for c in range(cols):
            dist_from_center = abs(r - rows / 2 + 0.5) + abs(c - cols / 2 + 0.5)
            max_dist = rows / 2 + cols / 2
            spatial_factor = 1 - (dist_from_center / max_dist) * 0.5
            grid[r, c] += rng.normal(0, 1.0) + spatial_factor * 3

    # Add a "hotspot" cell proportional to risk
    if hotspot_risk_percent > 30:
        # Hotspot tends to be near center
        hot_r, hot_c = rows // 2, cols // 2
        hotspot_boost = (hotspot_risk_percent / 100) * 8
        grid[hot_r, hot_c] += hotspot_boost

    # Add minor noise
    grid += rng.normal(0, 0.5, (rows, cols))

    # Shift grid so its max exactly equals base_temp
    current_max = grid.max()
    grid = grid - (current_max - base_temp)

    return np.round(grid, 1)
