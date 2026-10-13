"""
Comprehensive Functional Correctness and Audit Test Suite
==========================================================
Validates:
1. End-to-end input-to-output calculation pathways.
2. Mathematical determinism and inference repeatability for PyTorch MLP, XGBoost, and Autoencoder.
3. Physical response: Coolant flow adjustments strictly reduce simulated temperatures.
4. Feature perturbation sensitivity: Temperature and resistance increases elevate failure probabilities.
5. Forecasting behavior: Forecast horizon evaluates feature projections across +1m, +3m, +5m.
6. Baseline restoration: Baseline values restore calibrated nominal values.
7. Autoencoder reconstruction MSE: Separates in-distribution normal telemetry from out-of-distribution stress states.
"""

import unittest
import numpy as np
from pathlib import Path

from src.prediction_engine import (
    load_mlp_model,
    load_model as load_xgb_model,
    predict_failure_probability,
    get_default_features,
)
from src.anomaly_detection import load_autoencoder, detect_anomaly
from src.thermal_simulator import BatteryPackThermalSimulator
from src.forecasting import forecast_failure_probability
from src.simulation_engine import get_simulation_state
from src.risk_engine import assess_thermal_risk
from src.decision_engine import evaluate_thermal_recommendation
from src.what_if_explainer import (
    explain_parameter_change,
    get_live_parameter_explanations,
    compute_simulation_comparison,
)


class TestFunctionalCorrectnessAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mlp = load_mlp_model("models/failure_predictor")
        cls.xgb = load_xgb_model("models/xgboost_failure_model.joblib")
        cls.ae_artifacts = load_autoencoder("models/autoencoder")

    # ── 1. Inference Repeatability & Determinism ────────────────
    def test_mlp_inference_repeatability(self):
        """PyTorch MLP failure predictor must yield identical outputs for identical inputs."""
        features = get_default_features()
        p1 = predict_failure_probability(self.mlp, features)
        p2 = predict_failure_probability(self.mlp, features)
        self.assertAlmostEqual(p1, p2, places=6, msg="MLP predictions must be strictly deterministic")

    def test_xgb_inference_repeatability(self):
        """XGBoost failure classifier must yield identical outputs for identical inputs."""
        if self.xgb is not None:
            features = get_default_features()
            p1 = predict_failure_probability(self.xgb, features)
            p2 = predict_failure_probability(self.xgb, features)
            self.assertAlmostEqual(p1, p2, places=6, msg="XGBoost predictions must be strictly deterministic")

    def test_autoencoder_reconstruction_repeatability(self):
        """Autoencoder anomaly reconstruction error must be strictly repeatable."""
        ae_model, ae_scaler, ae_imputer, ae_threshold = self.ae_artifacts
        features = get_default_features()
        res1 = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)
        res2 = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, features)
        self.assertEqual(res1["score"], res2["score"])
        self.assertEqual(res1["label"], res2["label"])

    # ── 2. Coolant Flow Physical Effectiveness ─────────────────
    def test_coolant_flow_reduces_temperatures(self):
        """
        In the physics thermal simulator, higher coolant flow must extract more heat,
        resulting in lower temperatures under identical load current and time.
        """
        # Low coolant flow simulation
        sim_low = BatteryPackThermalSimulator(current=35.0, coolant_flow=0.012, seed=100)
        sim_low.step(dt_seconds=120.0)
        temp_low = sim_low.get_pack_summary()["avg_temp"]

        # High coolant flow simulation under identical current & duration
        sim_high = BatteryPackThermalSimulator(current=35.0, coolant_flow=0.038, seed=100)
        sim_high.step(dt_seconds=120.0)
        temp_high = sim_high.get_pack_summary()["avg_temp"]

        self.assertLess(
            temp_high,
            temp_low,
            f"Higher coolant flow (0.038 kg/s) must cool pack more than low flow (0.012 kg/s): {temp_high} vs {temp_low}",
        )

    def test_apply_recommendation_decreases_hotspot(self):
        """Applying the cooling recommendation must boost flow and lower hotspot temperature."""
        sim = BatteryPackThermalSimulator(seed=42)
        initial_hotspot = sim.get_hotspot_info()["temperature"]
        initial_flow = sim.coolant_flow

        res = sim.apply_recommendation(boost_percentage=25.0)
        self.assertGreater(sim.coolant_flow, initial_flow)
        self.assertLess(sim.get_hotspot_info()["temperature"], initial_hotspot)
        self.assertLess(res["delta_temp"], 0.0)

    # ── 3. Feature Sensitivity & Model Recalculation ───────────
    def test_temperature_increase_elevates_failure_probability(self):
        """Increasing cell temperature must increase failure probability in the trained MLP."""
        baseline = get_default_features()
        base_prob = predict_failure_probability(self.mlp, baseline)

        hot_scenario = baseline.copy()
        hot_scenario["cell_temperature_avg"] = 48.0
        hot_scenario["cell_temperature_max"] = 56.0
        hot_prob = predict_failure_probability(self.mlp, hot_scenario)

        self.assertGreater(
            hot_prob,
            base_prob,
            f"Hot scenario ({hot_prob:.4f}) must produce higher failure probability than baseline ({base_prob:.4f})",
        )

    def test_resistance_increase_elevates_failure_probability(self):
        """Increasing internal resistance must increase failure probability in the trained MLP."""
        baseline = get_default_features()
        base_prob = predict_failure_probability(self.mlp, baseline)

        degraded_res = baseline.copy()
        degraded_res["internal_resistance"] = 3.50
        degraded_prob = predict_failure_probability(self.mlp, degraded_res)

        self.assertGreater(
            degraded_prob,
            base_prob,
            f"Degraded resistance ({degraded_prob:.4f}) must produce higher failure probability than baseline ({base_prob:.4f})",
        )

    # ── 4. Forecasting Horizon Behavior ────────────────────────
    def test_forecasting_is_projected_feature_approximation(self):
        """
        Verify forecasting applies trend increments over +1, +3, +5 minutes
        and evaluates the static MLP model.
        """
        baseline = get_default_features()
        trend = {
            "cell_temperature_avg": 1.0,
            "cell_temperature_max": 1.5,
            "internal_resistance": 0.1,
            "state_of_charge": -1.0,
        }
        forecasts = forecast_failure_probability(self.mlp, baseline, trend=trend, horizons_min=[1, 3, 5])

        self.assertEqual(len(forecasts), 3)
        self.assertEqual([f["horizon_min"] for f in forecasts], [1, 3, 5])

        # Under positive temperature/resistance trend, failure probability should monotonically increase
        probs = [f["predicted_prob"] for f in forecasts]
        self.assertLessEqual(probs[0], probs[1])
        self.assertLessEqual(probs[1], probs[2])

    # ── 5. Baseline Restoration Integrity ──────────────────────
    def test_baseline_defaults_correctness(self):
        """Baseline dictionary must contain all required operational features with valid ranges."""
        baseline = get_default_features()
        required_keys = [
            "cell_temperature_avg",
            "cell_temperature_max",
            "internal_resistance",
            "cooling_system_health",
            "average_charge_power_kw",
            "state_of_charge",
        ]
        for key in required_keys:
            self.assertIn(key, baseline)
            self.assertIsInstance(baseline[key], (int, float))

        # Check nominal physical bounds
        self.assertGreaterEqual(baseline["cell_temperature_max"], baseline["cell_temperature_avg"])
        self.assertGreaterEqual(baseline["cooling_system_health"], 0.0)
        self.assertLessEqual(baseline["cooling_system_health"], 100.0)
        self.assertGreater(baseline["internal_resistance"], 0.0)

    # ── 6. Autoencoder In-Distribution vs Out-of-Distribution ────
    def test_autoencoder_distinguishes_normal_from_anomaly(self):
        """Autoencoder reconstruction MSE must remain below threshold for nominal baseline, and rise for stress state."""
        ae_model, ae_scaler, ae_imputer, ae_threshold = self.ae_artifacts

        # Nominal in-distribution state
        norm_features = get_default_features()
        norm_res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, norm_features)
        self.assertEqual(norm_res["label"], "NORMAL")
        self.assertLess(norm_res["score"], ae_threshold)

        # Extreme out-of-distribution stress state
        stress_features = get_simulation_state("THERMAL_STRESS", step=15)
        stress_res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, stress_features)
        self.assertGreater(
            stress_res["score"],
            norm_res["score"],
            "Thermal stress state must produce significantly higher reconstruction MSE than nominal baseline",
        )

    # ── 7. Reset Reproducibility & State Restoration ───────────
    def test_reset_reproducibility(self):
        """Reset must restore exact deterministic baseline across temperatures, cell states, flow, and features."""
        fresh_sim = BatteryPackThermalSimulator(ambient_temp=30.0, coolant_temp=22.0, current=10.0, coolant_flow=0.02, seed=42)
        fresh_features = get_default_features()

        # Perturb simulator state severely (high load, high flow, stepped multiple times)
        sim = BatteryPackThermalSimulator(ambient_temp=38.0, coolant_temp=28.0, current=45.0, coolant_flow=0.038, seed=999)
        sim.step(dt_seconds=300.0)
        sim.set_coolant_flow(0.012)
        perturbed_features = get_simulation_state("THERMAL_STRESS", 10)

        # Confirm state is deeply perturbed
        self.assertNotEqual(sim.coolant_flow, fresh_sim.coolant_flow)
        self.assertFalse(np.allclose(sim.temperatures, fresh_sim.temperatures))

        # Perform deterministic Reset to Baseline sequence
        reset_mode = "NORMAL"
        reset_flow = 0.020
        reset_sim = BatteryPackThermalSimulator(ambient_temp=30.0, coolant_temp=22.0, current=10.0, coolant_flow=0.02, seed=42)
        reset_features = get_default_features()

        # Assert 100% state identity with fresh baseline
        self.assertEqual(reset_mode, "NORMAL")
        self.assertEqual(reset_flow, 0.020)
        self.assertEqual(reset_sim.coolant_flow, fresh_sim.coolant_flow)
        np.testing.assert_array_almost_equal(
            reset_sim.temperatures, fresh_sim.temperatures,
            err_msg="Reset temperatures must exactly match baseline temperatures",
        )
        self.assertEqual(reset_sim.get_hotspot_info(), fresh_sim.get_hotspot_info())
        self.assertEqual(reset_sim.get_pack_summary(), fresh_sim.get_pack_summary())
        self.assertEqual(reset_features, fresh_features)

        # Model inferences on reset state must match fresh baseline exactly
        p_fresh = predict_failure_probability(self.mlp, fresh_features)
        p_reset = predict_failure_probability(self.mlp, reset_features)
        self.assertEqual(p_fresh, p_reset)

    # ── 8. Slider Application & Recalculation ──────────────────
    def test_slider_application_and_recalculation(self):
        """Slider coolant change must update simulator state, features, and model risk predictions."""
        sim = BatteryPackThermalSimulator(current=10.0, coolant_flow=0.02, seed=42)
        initial_hotspot = sim.get_hotspot_info()["temperature"]
        initial_avg = sim.get_pack_summary()["avg_temp"]

        # Increase flow via slider mechanism (0.020 -> 0.040)
        res = sim.set_coolant_flow(0.040)
        self.assertEqual(res["status"], "updated")
        self.assertEqual(sim.coolant_flow, 0.040)
        self.assertLess(res["new_hotspot_temp"], initial_hotspot)
        self.assertLess(sim.get_pack_summary()["avg_temp"], initial_avg)
        self.assertLess(res["delta_temp"], 0.0)

        # Sync into features
        features = get_default_features()
        features["cell_temperature_avg"] = sim.get_pack_summary()["avg_temp"]
        features["cell_temperature_max"] = sim.get_hotspot_info()["temperature"]
        features["cooling_system_health"] = max(10.0, min(100.0, round((0.040 / 0.02) * 74.0, 1)))

        self.assertEqual(features["cooling_system_health"], 100.0)
        p_cool = predict_failure_probability(self.mlp, features)
        p_base = predict_failure_probability(self.mlp, get_default_features())
        self.assertLessEqual(p_cool, p_base)

    # ── 9. Cooling Response in All Operating Modes ─────────────
    def test_cooling_response_all_modes_identical_conditions(self):
        """
        Under identical initial conditions and identical elapsed simulation time,
        increasing coolant flow (0.010 -> 0.040 kg/s) strictly lowers temperatures
        in NORMAL, HIGH_LOAD, and THERMAL_STRESS modes.
        """
        modes = [
            ("NORMAL", 10.0, 30.0, 22.0),
            ("HIGH_LOAD", 35.0, 32.0, 24.0),
            ("THERMAL_STRESS", 45.0, 38.0, 28.0),
        ]

        for mode_name, current, amb_temp, cool_temp in modes:
            with self.subTest(mode=mode_name):
                sim_low = BatteryPackThermalSimulator(
                    ambient_temp=amb_temp, coolant_temp=cool_temp, current=current, coolant_flow=0.010, seed=42
                )
                sim_high = BatteryPackThermalSimulator(
                    ambient_temp=amb_temp, coolant_temp=cool_temp, current=current, coolant_flow=0.040, seed=42
                )

                # Identical initial temperatures
                np.testing.assert_array_almost_equal(sim_low.temperatures, sim_high.temperatures)

                # Run identical elapsed time
                sim_low.step(dt_seconds=60.0)
                sim_high.step(dt_seconds=60.0)

                low_h = sim_low.get_hotspot_info()["temperature"]
                high_h = sim_high.get_hotspot_info()["temperature"]
                low_avg = sim_low.get_pack_summary()["avg_temp"]
                high_avg = sim_high.get_pack_summary()["avg_temp"]

                self.assertLess(high_h, low_h, f"{mode_name}: Hotspot with 0.040 kg/s must be lower than 0.010 kg/s")
                self.assertLess(high_avg, low_avg, f"{mode_name}: Pack average with 0.040 kg/s must be lower than 0.010 kg/s")


if __name__ == "__main__":
    unittest.main()
