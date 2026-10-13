# EV Battery Mission Control — Manual Run & Demonstration Guide

This guide provides step-by-step instructions for running, testing, and demonstrating the **EV Battery Mission Control** application. It documents the three active production pages and reflects the current project architecture, controls, and deterministic decision logic.

---

## 1. PROJECT STARTUP & ENVIRONMENT SETUP

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Virtual environment (`venv`)

### Step-by-Step Launch Instructions

1. **Clone or Navigate to the Project Root**:
   ```bash
   cd ev-battery-mission-control
   ```

2. **Create and Activate Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     *(If script execution is blocked on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
   - **macOS / Linux**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Automated Test Suite**:
   ```bash
   python -m pytest
   ```
   *Expected result: 67 passing tests, confirming ODE simulation stability, decision engine rules, and model inference pipelines.*

5. **Start the Application**:
   - **Option A (System Initialization Launcher)**:
     ```bash
     python run.py
     ```
     *Validates environment prerequisites and dataset presence before launching Streamlit.*
   - **Option B (Direct Streamlit Launch)**:
     ```bash
     streamlit run app/main.py
     ```

6. **Access the Dashboard**:
   Open your browser to:
   ```text
   http://localhost:8501
   ```

---

## 2. NAVIGATION & SYSTEM STATUS

Use the **Navigation** selector in the left sidebar to navigate between the three active pages:

| Page | Icon | Purpose |
| :--- | :---: | :--- |
| **Mission Control** | 🎛️ | Primary 24-cell thermal pack console, scenario switching, coolant flow controls, and deterministic AI decision support. |
| **Thermal Analysis** | 🌡️ | Continuous lumped-capacitance ODE simulation streaming, forward stepping, live time-series plots, and multi-horizon failure forecasting. |
| **What-If Simulator** | 🧪 | Interactive 6-parameter perturbation sandbox with real-time physics explanations, delta calculations, and side-by-side thermal grids. |

### System Status Badges (Sidebar)
The sidebar displays real-time health indicators for core ML components:
- **XGBoost**: `● ONLINE` (Hold-out test accuracy: 91.68%, ROC-AUC: 0.9789)
- **PyTorch MLP**: `● ONLINE` (Deep neural network failure classifier)
- **Autoencoder**: `● ONLINE` (Unsupervised multivariate anomaly detection, threshold: 0.1929)
- **Dataset**: `ONLINE (200,000 ROWS)` (Telemetry connectivity status)

---

## 3. PAGE 1: MISSION CONTROL

### Overview
Mission Control is the primary operator console. It features a real-time 24-cell (4×6 planar module) thermal grid with safety bands, localized hotspot tracking (Cell C7, index 6), physical cooling controls, and deterministic rule-based decision support.

### Key Visual & Telemetry Elements
- **24-Cell Thermal Grid (4×6 Planar Module)**:
  - Cell IDs 1 through 24 with individual temperatures.
  - Cell 7 (C7) is the designated localized hotspot cell with elevated internal resistance (0.038 Ω vs. 0.015 Ω nominal).
- **Safety Color Bands**:
  - `Normal`: $< 36.0^\circ\text{C}$ (Green `#16866A`)
  - `Moderate`: $36.0 - 42.0^\circ\text{C}$ (Blue `#326CE5`)
  - `Caution`: $42.0 - 48.0^\circ\text{C}$ (Orange `#C7771B`)
  - `Critical`: $\ge 48.0^\circ\text{C}$ (Red `#D94841` — exothermic self-heating onset)
- **Top KPI Cards**:
  - `Pack Average Temp`: Mean temperature across all 24 cells.
  - `Hotspot Temp`: Maximum cell temperature (Cell C7).
  - `Pack Load Current`: Operating current in Amperes.
  - `Failure Probability`: PyTorch MLP and XGBoost failure risk percentage.
  - `Thermal Risk Score`: Multi-factor composite index (0–100 scale).
  - `Cooling System Health`: Effective BTMS convective health percentage.

### Interactive Controls to Demonstrate

1. **Operating Scenario Switching**:
   - **Where**: "Scenario Mode" selector in the controls panel.
   - **Options**: `NORMAL` (10A current), `HIGH_LOAD` (25A current), `THERMAL_STRESS` (45A current with degraded cooling).
   - **Action**: Select `THERMAL_STRESS`.
   - **Observe**: Hotspot temperature rises toward the caution/critical envelope, failure probability increases, and risk level updates immediately.

2. **Coolant Flow Rate Slider (0.010 – 0.050 kg/s)**:
   - **Where**: "Coolant Flow Rate (kg/s)" slider.
   - **Action**: Move slider from `0.020 kg/s` to `0.040 kg/s`.
   - **Observe**: Immediate quasi-steady convective recalculation reduces hotspot temperature and pack average without advancing the simulation clock.

3. **Apply Suggested Cooling Button**:
   - **Where**: Primary button next to the coolant slider.
   - **Action**: Click **Apply Suggested Cooling**.
   - **Observe**:
     - System executes the deterministic recommendation (+10%, +18%, or +35% flow boost).
     - Slider updates to the newly calculated flow rate.
     - Hotspot temperature and pack average immediately decrease.
     - A confirmation toast displays: `Cooling flow updated to X.XXX kg/s (Hotspot: -X.X°C)`.

4. **Deterministic Thermal AI Decision Support Panel**:
   - Inspect the structured rule-engine output:
     - **Observation**: Clear physical summary of current pack state.
     - **Active Triggers**: List of explicit safety boundaries crossed.
     - **Predicted Outcome**: Physical trajectory if unaddressed.
     - **Recommended Action**: Prescribed cooling intervention.
     - **Physical Engineering Rationale**: Thermodynamic basis for the action.

5. **Autoencoder Anomaly Detection**:
   - Displays reconstruction error (MSE) against the calibrated threshold (0.1929).
   - Status badge shows `NORMAL` or `ANOMALY`.

6. **Reset to Baseline**:
   - **Action**: Click **Reset to Baseline**.
   - **Observe**: Restores calibrated nominal conditions (30°C ambient, 22°C coolant, 10A current, 0.020 kg/s flow rate).

---

## 4. PAGE 2: THERMAL ANALYSIS

### Overview
Thermal Analysis provides dynamic time-series simulation monitoring, ODE forward integration, and multi-horizon failure forecasts (+1m, +3m, +5m).

### Controls & Navigation
- **Telemetry Simulation Mode**: Select `NORMAL`, `HIGH_LOAD`, or `THERMAL_STRESS`.
- **Stream Controls**:
  - `▶ Start Stream`: Launches dynamic 50-step simulation stream.
  - `⏹ Pause Stream`: Halts stream at current step.
  - `⏭ Single Step`: Manually advances simulation by one time-step (+2s).
  - `🔄 Reset Monitor`: Restores monitor to step 0 and clears history.

### Live Telemetry & Visualizations to Inspect

1. **Synchronized Time-Series Charts**:
   - **Temperature Trend Plot**: Plots Maximum Cell Temp (red curve) and Average Cell Temp (blue curve) over time with horizontal dashed safety thresholds:
     - Caution Boundary: $42.0^\circ\text{C}$ (Orange)
     - Critical Boundary: $48.0^\circ\text{C}$ (Red)
   - **Failure Risk Horizon Plot**: Plots PyTorch MLP failure probability over time with shaded risk fill and threshold lines (Caution: 15%, High: 40%).

2. **Multi-Horizon Failure Forecast Card**:
   - Displays real-time model projections across future time horizons:
     - Current State failure probability
     - `+1 min Horizon`
     - `+3 min Horizon`
     - `+5 min Horizon`

3. **Autoencoder & Risk Engine Summary**:
   - Live MSE reconstruction error vs. threshold.
   - Composite Thermal Risk level (NORMAL, CAUTION, HIGH, CRITICAL) with 0–100 numerical score.

### Demonstration Procedure
1. Select **THERMAL_STRESS** from the dropdown.
2. Click **Start Stream**.
3. Watch the progress bar advance toward 50 steps.
4. Observe the temperature curves and failure probability climb.
5. Watch the toast notification appear upon completion: `✅ Live simulation reached 50 steps.`

---

## 5. PAGE 3: WHAT-IF SIMULATOR

### Overview
The What-If Simulator provides an interactive parameter perturbation sandbox. Operators can adjust six battery operational variables, receive real-time physical explanations of each adjustment, and compare baseline vs. hypothetical outcomes side-by-side.

### Operating Controls (6 Sliders)
1. **Average Cell Temperature**: Range 15.0 – 60.0°C (Baseline: 21.0°C, Step: 0.5°C)
2. **Maximum Hotspot Temperature**: Range 15.0 – 85.0°C (Baseline: 28.5°C, Step: 0.5°C)
3. **Average Charge Power**: Range 0.0 – 200.0 kW (Baseline: 40.0 kW, Step: 5.0 kW)
4. **Internal Resistance**: Range 0.2 – 6.0 mΩ (Baseline: 0.53 mΩ, Step: 0.05 mΩ)
5. **Cooling System Health**: Range 0.0 – 100.0% (Baseline: 74.0%, Step: 1.0%)
6. **State of Charge**: Range 5.0 – 100.0% (Baseline: 55.0%, Step: 1.0%)

### Real-Time Live Parameter Explanation Panel
- Adjusting any slider immediately updates this panel on the fly.
- For each changed variable, the system explains:
  - **What this means**: Physical definition of the change.
  - **Why it matters**: Thermodynamic risk mechanism.
  - **Expected effect**: Anticipated impact on temperature and failure likelihood.
  - **Suggested adjustment**: Operational mitigation advice.

### Demonstration Scenarios

#### Scenario A: Thermal Stress Case
1. Adjust **Maximum Hotspot Temperature** to `52.0°C`.
2. Adjust **Average Charge Power** to `120.0 kW`.
3. Adjust **Cooling System Health** down to `35%`.
4. Observe the **Live Parameter Explanation** card displaying 3 changed parameters with color-coded risk tags.
5. Click **Run Simulation & Compare**:
   - **Baseline Failure Prob (MLP)**: Displays calibrated baseline (~0.26%).
   - **Scenario Failure Prob (MLP)**: Displays elevated scenario risk.
   - **Difference (Delta)**: Displays percentage-point increase in red.
   - **Calculated outcome**: Detailed model explanation of the primary drivers.
   - **Recommended next adjustment**: Concrete cooling or power-shed action.
6. Inspect the **Side-by-Side Thermal Grids**:
   - Baseline Nominal Heatmap (cool, uniform green/blue).
   - What-If Hypothetical Heatmap (concentrated red/orange hotspot).

#### Scenario B: Reset Action
1. Click **Reset to Baseline**.
2. All 6 sliders instantly return to calibrated default values.
3. The Live Explanation panel returns to `NOMINAL BASELINE`.

---

## 6. DETERMINISTIC REPEATABILITY VERIFICATION

The safety decision pipeline is purely deterministic (rule-based evaluation and calibrated model inferences), ensuring reproducible behavior across runs.

### Repeatability Test Procedure
1. Navigate to **Mission Control**.
2. Select **HIGH_LOAD** mode.
3. Record the following values:
   - Recommended Action text
   - Recommended Boost Percentage
   - Hotspot Temperature
4. Click **Reset to Baseline**, then select **HIGH_LOAD** again.
5. Confirm that the exact same observation, triggers, action, and boost are produced.

---

## 7. RECOMMENDED DEMONSTRATION FLOW (10-MINUTE WALKTHROUGH)

For presentations, academic reviews, or live evaluations, follow this structured flow:

```text
1. Mission Control Overview (3 min)
   ├── Show 24-cell grid, safety bands, and Cell C7 hotspot
   ├── Switch Scenario: NORMAL → THERMAL_STRESS
   ├── Adjust Coolant Flow slider (0.010 → 0.040 kg/s)
   ├── Click "Apply Suggested Cooling" (demonstrate automatic flow boost & hotspot relief)
   └── Review Deterministic Decision Support card (Observation, Triggers, Rationale)

2. Thermal Analysis Simulation (3 min)
   ├── Select THERMAL_STRESS profile
   ├── Click "Start Stream" (watch 50-step ODE progression)
   ├── Inspect dual Plotly charts (Temperature trend & Failure risk curve)
   └── Inspect Multi-Horizon Forecast (+1m, +3m, +5m) and Autoencoder MSE

3. What-If Parameter Sandbox (3 min)
   ├── Perturb sliders (Hotspot Temp ↑, Charge Power ↑, Cooling Health ↓)
   ├── Highlight Live Parameter Explanation panel updating instantly
   ├── Click "Run Simulation & Compare" (inspect delta metrics and outcome text)
   ├── Compare Side-by-Side Thermal Grids (Baseline vs Hypothetical)
   └── Click "Reset to Baseline"

4. Architecture & Test Validation (1 min)
   ├── Mention 67 passing automated tests (physics invariants, decision rules, ML)
   └── Point out System Status badges in the sidebar
```

---

## 8. VERIFICATION CHECKLIST

Use this checklist during manual runs:

- [ ] Streamlit starts with 0 errors (`streamlit run app/main.py`)
- [ ] Sidebar displays all 4 system status badges as `ONLINE`
- [ ] Mission Control: 24-cell grid renders with Cell 7 as hotspot
- [ ] Mission Control: Coolant slider smoothly adjusts from 0.010 to 0.050 kg/s
- [ ] Mission Control: "Apply Suggested Cooling" boosts flow and lowers hotspot
- [ ] Mission Control: "Reset to Baseline" restores 0.020 kg/s and nominal state
- [ ] Thermal Analysis: "Start Stream" advances 50 steps with live chart updates
- [ ] Thermal Analysis: Multi-horizon forecast displays +1m, +3m, +5m predictions
- [ ] What-If Simulator: Sliders update Live Parameter Explanations in real time
- [ ] What-If Simulator: "Run Simulation & Compare" computes delta percentage points
- [ ] What-If Simulator: Side-by-side heatmaps render Baseline vs. Hypothetical
- [ ] What-If Simulator: "Reset to Baseline" resets all 6 sliders cleanly
