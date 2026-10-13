# ⚡ EV Battery Mission Control

> **AI-Powered Thermal Safety and Battery Failure Prediction System**

A modern industrial mission control dashboard for Electric Vehicle Battery Thermal Management Systems (BTMS). It combines physics-based multi-cell simulation, explainable AI (SHAP), trained machine learning predictors (PyTorch MLP, XGBoost, Autoencoder), and deterministic rule-based decision support to monitor battery health and mitigate thermal runaway risk.

![Dashboard Preview](screenshots/dashboard_preview.png)

---

## 🚀 Key Features

- **Mission Control Console**: Real-time 24-cell (4×6 planar module) thermal grid with color-coded safety bands (Normal <36°C, Moderate 36–42°C, Caution 42–48°C, Critical ≥48°C) and localized hotspot tracking (Cell C7).
- **Cooling Controls & Quasi-Steady Recalculation**: Adjust liquid coolant mass flow rate (0.010–0.050 kg/s) with immediate quasi-steady convective thermal recalculation, or 1-click **Apply Suggested Cooling**.
- **Operating Scenario Switching**: Toggle operating profiles (`NORMAL`, `HIGH_LOAD`, `THERMAL_STRESS`) or restore calibrated nominal parameters via **Reset to Baseline**.
- **Deterministic Thermal AI Decision Support**: Transparent rule-based decision engine (Observation, Reasoning, Prediction, Action) evaluating cell gradients, failure probabilities, and anomaly scores.
- **Thermal Analysis & Dynamic Simulation**: Continuous lumped-capacitance ODE integration with forward stepping (+5s step, time integration), live temperature trends, and multi-horizon failure forecasts (+1m, +3m, +5m).
- **Autoencoder Anomaly Detection**: Unsupervised PyTorch Autoencoder tracking reconstruction error (MSE) against a calibrated threshold to detect multivariate operational drift.
- **What-If Simulator with Live Explainability**: Real-time parameter perturbation sandbox (Hotspot temperature, load current, cooling health, resistance, SoC) that explains each specific physical change immediately in clear language.
- **Side-by-Side Thermal Grids**: Visual comparison between Nominal Baseline and Hypothetical What-If thermal heatmaps.
- **Explainable AI (SHAP)**: Feature attributions explaining the physical drivers of elevated failure probability (temperature, internal resistance, charge power, cooling health).
- **Engineering Prototype Notice**: Clear disclaimers distinguishing deterministic physics ODEs and calibrated machine learning models from physical vehicle hardware actuators.

---

## 🛠️ Project Architecture

```text
ev-battery-mission-control/
│
├── app/
│   ├── main.py                     # Unified mission control dashboard & navigation router
│   ├── mission_control.py          # Primary battery console, 24-cell grid, cooling & scenario controls
│   ├── prediction.py               # Thermal Analysis page: forward ODE stepping & multi-horizon forecast
│   ├── what_if.py                  # What-If Simulator: live parameter explainability & side-by-side grids
│   ├── ui_components.py            # Reusable KPI metric cards, headers, status badges
│   └── styles.py                   # Centralized industrial design system & clean CSS
│
├── data/
│   ├── ev_battery_failure_dataset.csv  # 200,000-row telemetry dataset (77.2 MB)
│   ├── btms_dataset.csv                # BTMS thermal telemetry dataset
│   ├── data_dictionary.csv             # Feature definitions and units
│   └── feature_descriptions.md         # Domain documentation for BTMS features
│
├── models/
│   ├── failure_predictor/          # Trained PyTorch MLP classifier, scaler, imputer
│   ├── autoencoder/                # Trained PyTorch Autoencoder anomaly detector & threshold
│   ├── xgboost_failure_model.joblib # Serialized XGBoost failure prediction model
│   ├── model_metadata.json         # MLP model schema and performance metrics
│   └── model_metadata_xgb.json     # XGBoost hyperparameters and training metrics
│
├── src/
│   ├── __init__.py                 # Python package initializer
│   ├── anomaly_detection.py        # PyTorch Autoencoder reconstruction & thresholding
│   ├── decision_engine.py          # Deterministic safety rules & cooling recommendations
│   ├── forecasting.py              # Multi-horizon (+1m, +3m, +5m) forward feature projection
│   ├── prediction_engine.py        # Model loading & inference abstraction (MLP / XGBoost)
│   ├── risk_engine.py              # Multi-factor thermal risk assessment & boundary rules
│   ├── shap_explainer.py           # Model explainability & feature attributions
│   ├── simulation_engine.py        # Scenario state generator (NORMAL, HIGH_LOAD, THERMAL_STRESS)
│   ├── thermal_simulator.py        # 24-cell lumped-capacitance ODE heat conduction physics engine
│   └── what_if_explainer.py        # Parameter perturbation mechanics & explainability engine
│
├── tests/
│   ├── test_audit_functional_correctness.py  # End-to-end model response & cooling tests
│   ├── test_decision_engine.py               # Deterministic rule engine validation
│   ├── test_mission_control_system.py        # System integration & mode anomaly tests
│   ├── test_risk_threshold_boundaries.py     # Continuous risk boundary threshold tests
│   ├── test_thermal_simulator.py             # 24-cell ODE dynamics & runaway unit tests
│   └── test_what_if_explainer.py             # Live parameter explainability & schema tests
│
├── .gitignore                      # Git rules for venv, cache, datasets, and models
├── requirements.txt                # Python package dependencies
├── README.md                       # Project documentation & setup instructions
└── run.py                          # Launcher script with environment & dataset verification
```

---

## 💻 Quick Start

### 1. Clone & Navigate to Project Directory
```bash
git clone <repository-url>
cd ev-battery-mission-control
```

### 2. Create and Activate Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell script execution is restricted, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```bash
python -m pytest
```

### 5. Launch the Dashboard
Using the system initialization launcher:
```bash
python run.py
```
Or directly via Streamlit:
```bash
streamlit run app/main.py
```

The Mission Control dashboard will open in your browser at `http://localhost:8501`.

---

## 📂 Dataset Connectivity & Demonstration Modes

The launcher script `run.py` and application automatically verify dataset presence:
- **Connected Mode:** When `data/ev_battery_failure_dataset.csv` is present (77.2 MB, 200,000 records), the dashboard initializes with `Dataset Status: CONNECTED` and utilizes full background distributions for SHAP explainability.
- **Calibrated Demo Mode:** If the dataset is absent, the application gracefully operates using pre-calibrated baseline telemetry vectors, trained serialized model artifacts, and the physical ODE thermal simulation.

---

## 🔬 Thermal Physics Modeling

The 24-cell pack dynamics are modeled using forward sub-stepped Euler integration of lumped-capacitance heat transfer:

$$\frac{dT_i}{dt} = \frac{I^2 R_i + Q_{\text{exo}, i} + \sum_{j \in \mathcal{N}_i} k_{\text{cond}} (T_j - T_i) - h_{\text{cool}} (T_i - T_{\text{cool}}) - h_{\text{amb}} A (T_i - T_{\text{amb}})}{C_{\text{thermal}}}$$

- **$I^2 R$ Joule Heating**: Quadratic heat generation proportional to instantaneous load current.
- **Inter-cell Conduction**: 4-connected grid conduction modeling thermal migration to adjacent planar cells.
- **Coolant Convection**: Active liquid heat extraction modulated by coolant mass flow rate $\dot{m}$.
- **Exothermic Runaway Trigger**: Activated when cell temperatures surpass $48^\circ\text{C}$, accelerating cell self-heating exponentially.
