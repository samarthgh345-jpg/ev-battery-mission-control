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
    Generate an illustrative cell temperature grid for scenario visualization.
    Places the hotspot peak at the center cell and balances background cells so that
    the grid's arithmetic mean strictly equals cell_temp_avg and maximum equals cell_temp_max.
    """
    rng = np.random.default_rng(seed)
    n = rows * cols
    effective_max = max(cell_temp_avg, cell_temp_max)

    if abs(effective_max - cell_temp_avg) < 1e-4:
        noise = rng.normal(0, 0.2, (rows, cols))
        noise -= noise.mean()
        grid = cell_temp_avg + noise
        return np.round(grid, 1)

    hot_r, hot_c = rows // 2, cols // 2
    rem_n = n - 1
    rem_mean = (n * cell_temp_avg - effective_max) / rem_n

    # Generate small zero-mean noise for remaining cells
    noise = rng.normal(0, 0.25, rem_n)
    noise -= noise.mean()
    rem_vals = rem_mean + noise
    rem_vals = np.clip(rem_vals, 5.0, effective_max)
    # Ensure remaining cells sum exactly to (n * avg - max)
    rem_vals = rem_vals + (rem_mean - rem_vals.mean())

    grid = np.zeros((rows, cols), dtype=float)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if r == hot_r and c == hot_c:
                grid[r, c] = effective_max
            else:
                grid[r, c] = rem_vals[idx]
                idx += 1

    return np.round(grid, 1)
