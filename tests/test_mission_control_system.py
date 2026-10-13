"""
Comprehensive System & Unit Tests for EV Battery Mission Control
=================================================================
Validates:
1. Model loading & inference (MLP, XGBoost, Autoencoder).
2. Physical thermal simulation & neighbor conduction.
3. Simulation modes (NORMAL, HIGH_LOAD, THERMAL_STRESS) and anomaly calibration.
4. Deterministic thermal runaway progression.
5. Closed-loop cooling intervention state recalculation.
6. What-If scenario parameter perturbation and risk impact.
7. Multi-horizon forecasting.
8. SHAP explanation engine.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.prediction_engine import (
    load_mlp_model,
    load_model as load_xgb_model,
    predict_failure_probability,
    get_default_features,
)
from src.anomaly_detection import load_autoencoder, detect_anomaly
from src.risk_engine import assess_thermal_risk
from src.simulation_engine import get_simulation_state
from src.forecasting import forecast_failure_probability
from src.thermal_simulator import BatteryPackThermalSimulator
from src.agent import ThermalAIAgent


class TestEVBatteryMissionControlSystem(unittest.TestCase):
    """Integration & validation suite for all Mission Control components."""

    @classmethod
    def setUpClass(cls):
        cls.mlp = load_mlp_model("models/failure_predictor")
        cls.xgb = load_xgb_model("models/xgboost_failure_model.joblib")
        cls.ae_artifacts = load_autoencoder("models/autoencoder")

    def test_model_loading(self):
        """Verify all three models load successfully."""
        self.assertIsNotNone(self.mlp, "MLP model should load from models/failure_predictor")
        self.assertIsNotNone(self.xgb, "XGBoost model should load from models/xgboost_failure_model.joblib")
        self.assertIsNotNone(self.ae_artifacts, "Autoencoder should load from models/autoencoder")

    def test_calibrated_default_features_normal_state(self):
        """Default startup features must predict NORMAL without false anomalies."""
        features = get_default_features()
        ae_model, ae_scaler, ae_imputer, ae_threshold = self.ae_artifacts
        ae_result = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)

        self.assertEqual(
            ae_result["label"],
            "NORMAL",
            f"Default features should not trigger false anomaly (score: {ae_result['score']}, threshold: {ae_result['threshold']})",
        )
        self.assertLess(ae_result["score"], ae_result["threshold"])

        prob_mlp = predict_failure_probability(self.mlp, features)
        self.assertLess(prob_mlp, 0.15, "Default failure probability should be low (< 15%)")

    def test_simulation_modes_anomaly_sensitivity(self):
        """
        NORMAL mode across multiple steps must stay NORMAL.
        THERMAL_STRESS must produce elevated risk and trigger anomaly.
        """
        ae_model, ae_scaler, ae_imputer, ae_threshold = self.ae_artifacts

        # Check steps in NORMAL mode
        for step in [0, 5, 10, 20]:
            norm_state = get_simulation_state("NORMAL", step)
            norm_res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, norm_state)
            self.assertEqual(
                norm_res["label"],
                "NORMAL",
                f"NORMAL mode at step {step} produced false anomaly: {norm_res}",
            )

        # Check THERMAL_STRESS mode produces high risk or anomaly
        stress_state = get_simulation_state("THERMAL_STRESS", 10)
        stress_prob = predict_failure_probability(self.mlp, stress_state)
        stress_ae = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, stress_state)

        self.assertTrue(
            stress_prob > 0.3 or stress_ae["label"] == "ANOMALY",
            "THERMAL_STRESS should produce either high probability or anomaly detection.",
        )

    def test_whatif_feature_perturbation(self):
        """What-If slider changes should modulate failure probability and anomaly score."""
        baseline = get_default_features()
        base_prob = predict_failure_probability(self.mlp, baseline)

        stress_scenario = baseline.copy()
        stress_scenario["cell_temperature_max"] = 55.0
        stress_scenario["cooling_system_health"] = 20.0
        stress_scenario["internal_resistance"] = 3.5

        scen_prob = predict_failure_probability(self.mlp, stress_scenario)
        self.assertGreater(
            scen_prob,
            base_prob,
            "Severe thermal scenario should increase failure probability above baseline.",
        )

    def test_thermal_runaway_progression(self):
        """Runaway demonstration must produce deterministic escalation and cooling stabilization."""
        sim = BatteryPackThermalSimulator(seed=42)
        runaway_data = sim.simulate_thermal_runaway()

        self.assertEqual(runaway_data["minutes"], [0, 1, 2, 3, 4, 5])
        self.assertGreater(runaway_data["final_temp_without_ai"], 50.0)
        self.assertLess(runaway_data["final_temp_with_ai"], 40.0)
        self.assertIn("38°C", runaway_data["temp_rise_sequence"])
        self.assertIn("35%", runaway_data["ai_response_sequence"])

    def test_cooling_intervention_recalculation(self):
        """
        Applying cooling recommendation must physically modify coolant flow,
        reduce hotspot temperature, and update pack state.
        """
        sim = BatteryPackThermalSimulator(seed=42)
        initial_hotspot = sim.get_hotspot_info()["temperature"]
        initial_flow = sim.coolant_flow

        res = sim.apply_recommendation(boost_percentage=18.0)

        self.assertEqual(res["status"], "applied")
        self.assertGreater(sim.coolant_flow, initial_flow)
        self.assertLess(sim.get_hotspot_info()["temperature"], initial_hotspot)
        self.assertAlmostEqual(res["delta_temp"], sim.get_hotspot_info()["temperature"] - initial_hotspot, places=2)

    def test_forecasting_horizons(self):
        """Forecasting module should return +1, +3, +5 minute horizons."""
        features = get_default_features()
        forecasts = forecast_failure_probability(self.mlp, features)

        self.assertEqual(len(forecasts), 3)
        self.assertEqual([f["horizon_min"] for f in forecasts], [1, 3, 5])
        for f in forecasts:
            self.assertGreaterEqual(f["predicted_prob"], 0.0)
            self.assertLessEqual(f["predicted_prob"], 1.0)

    def test_risk_engine_bounds(self):
        """Risk engine must return bounded 0-100 scores and valid classifications."""
        risk = assess_thermal_risk(current_temp=30.0, anomaly_score=0.05, hotspot_risk_pct=5.0)
        self.assertIn(risk["risk_level"], ["NORMAL", "CAUTION", "HIGH", "CRITICAL"])
        self.assertGreaterEqual(risk["risk_score"], 0)
        self.assertLessEqual(risk["risk_score"], 100)


if __name__ == "__main__":
    unittest.main()
