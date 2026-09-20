# Synthetic / Simulation-Inspired BTMS Dataset

> **⚠️ DISCLAIMER:** This is a synthetic/simulation-inspired dataset created for demonstration
> and educational purposes. It is **NOT** experimentally validated battery sensor data.

## Overview

This dataset models a Battery Thermal Management System (BTMS) for electric vehicles.
It contains 10,000 synthetic data points representing various battery operating conditions
and their associated thermal outcomes.

## Features

| Feature | Unit | Description |
|---|---|---|
| `battery_current_A` | A | Battery discharge/charge current |
| `battery_voltage_V` | V | Battery terminal voltage |
| `state_of_charge_percent` | % | State of charge (0–100) |
| `ambient_temperature_C` | °C | Ambient/environmental temperature |
| `battery_temperature_C` | °C | Current battery pack temperature |
| `coolant_inlet_temperature_C` | °C | Coolant inlet temperature |
| `coolant_flow_rate_kg_s` | kg/s | Coolant mass flow rate |
| `coolant_pressure_Pa` | Pa | Coolant system pressure |
| `nanoparticle_concentration_percent` | % | Nanofluid additive concentration |
| `reynolds_number` | — | Reynolds number in microchannel |
| `heat_transfer_coefficient_W_m2K` | W/(m²·K) | Effective heat transfer coefficient |
| `microchannel_width_mm` | mm | Microchannel width |
| `microchannel_height_mm` | mm | Microchannel height |
| `discharge_rate_C` | C | C-rate of discharge |

## Target

| Target | Unit | Description |
|---|---|---|
| `max_battery_temperature_C` | °C | Maximum battery temperature (prediction target) |

## Derived Columns

| Column | Description |
|---|---|
| `temperature_rise_C` | Temperature rise above ambient |
| `thermal_risk_score` | Risk score (0–100) |
| `thermal_risk_label` | Risk category: NORMAL / CAUTION / HIGH / CRITICAL |
| `hotspot_risk_score` | Hotspot risk score (0–100) |

## Generation Methodology

1. **Scenario mixing**: Data is generated as a mix of four operating scenarios
   (Normal 50%, Caution 25%, High-Risk 15%, Critical 10%)
2. **Nonlinear heat generation**: Uses I²R Joule heating model with SoC-dependent internal resistance
3. **Cooling model**: Cooling effectiveness depends on heat transfer coefficient, flow rate,
   nanoparticle concentration, and coolant inlet temperature
4. **Temperature rise**: Computed as heat generation divided by cooling effectiveness,
   with ambient temperature offset
5. **Noise**: Gaussian noise added to simulate sensor variability
6. **Validation**: All values are clipped to physically plausible ranges

## Assumptions

- Internal resistance follows a simplified linear model w.r.t. SoC
- Heat generation is dominated by Joule heating (I²R)
- Cooling effectiveness is a simplified lumped-parameter model
- Feature correlations approximate real-world trends but are not calibrated to any specific battery chemistry

## Limitations

- **Not experimentally validated** — relationships are physically inspired but simplified
- No temporal dynamics (each row is independent)
- No degradation / aging effects
- Simplified cooling model (no CFD)
- Single battery chemistry assumed
