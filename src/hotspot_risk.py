"""
Hotspot Risk Estimation
========================
Estimates hotspot risk score and generates a simulated cell temperature grid
for the digital twin visualization based on avg and max cell temperatures.
"""

import numpy as np

def estimate_hotspot_risk(
    cell_temp_avg: float,
    cell_temp_max: float
) -> dict:
    """
    Estimate hotspot risk from temperature variance.
    """
    variance = cell_temp_max - cell_temp_avg
    
    # Normalized risk based on variance (0-5C is normal, >10C is critical)
    risk = (variance / 15.0) * 100
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
    cell_temp_avg: float,
    cell_temp_max: float,
    rows: int = 3,
    cols: int = 4,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a simulated cell temperature grid for digital twin visualization.
    """
    rng = np.random.default_rng(seed)

    # Base grid at average temperature
    grid = np.full((rows, cols), cell_temp_avg, dtype=float)

    # Add minor noise around average
    grid += rng.normal(0, 0.5, (rows, cols))

    # Add a "hotspot" cell at exactly the max temperature
    hot_r, hot_c = rows // 2, cols // 2
    grid[hot_r, hot_c] = cell_temp_max
    
    # Ensure average doesn't drift too far, though in a 12 cell pack,
    # if one cell is max, others must be slightly lower than average to maintain true average.
    # We ignore strict arithmetic average for the sake of the visualization feeling intuitive.

    return np.round(grid, 1)
