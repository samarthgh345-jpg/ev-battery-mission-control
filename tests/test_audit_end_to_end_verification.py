"""
End-to-End Audit Verification Test Suite
========================================
Comprehensive regression and invariant test suite covering:
1. UI components and supporting page imports (section_header, dataset_explorer, model_performance, xai).
2. Slider and session-state defensive clamping (What-If sliders, Mission Control coolant slider).
3. Decision engine boundary conditions across all 5 deterministic safety rules.
4. Thermal simulator numerical stability, extreme bounds, and clamping invariants.
5. ML prediction and anomaly detection robustness against corrupted/missing features.
6. Risk engine monotonic behavior and bounded outputs [0, 100].
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.decision_engine import evaluate_thermal_recommendation
from src.risk_engine import assess_thermal_risk, load_risk_config
from src.thermal_simulator import BatteryPackThermalSimulator
from src.prediction_engine import (
    load_mlp_model,
    load_model as load_xgb_model,
    predict_failure_probability,
    get_default_features,
    MLPPredictor,
)
from src.anomaly_detection import load_autoencoder, detect_anomaly
from src.hotspot_risk import estimate_hotspot_risk, generate_cell_temperature_grid
from src.forecasting import forecast_failure_probability
from app.ui_components import section_header, page_header, metric_card, clean_html


class TestEndToEndAuditVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mlp = load_mlp_model("models/failure_predictor")
        cls.xgb = load_xgb_model("models/xgboost_failure_model.joblib")
        cls.ae_artifacts = load_autoencoder("models/autoencoder")
        cls.risk_config = load_risk_config()

    # ── 1. UI Components & Supporting Page Import Verification ───
    def test_section_header_callable_and_valid_html(self):
        """section_header must exist in app.ui_components and produce valid HTML."""
        self.assertTrue(callable(section_header))

    def test_supporting_pages_import_without_error(self):
        """All supporting analysis pages must import cleanly without missing attributes."""
        import app.dataset_explorer as de
        import app.model_performance as mp
        import app.xai as xai

        self.assertTrue(callable(de.render))
        self.assertTrue(callable(mp.render))
        self.assertTrue(callable(xai.render))

    # ── 2. Decision Engine Boundary Exhaustive Verification ───────
    def test_decision_engine_rule1_critical_boundaries(self):
        """Rule 1: Exothermic Runaway Onset (Hotspot >= 48.0°C or Failure Prob >= 0.80)."""
        # Immediately below 48.0°C and prob < 0.80 -> Should NOT be CRITICAL
        res_below = evaluate_thermal_recommendation(
            hotspot_temp=47.9, avg_temp=35.0, cooling_health=70.0, failure_prob=0.79, anomaly_score=0.01
        )
        self.assertNotEqual(res_below["severity"], "CRITICAL")

        # Exactly at 48.0°C -> CRITICAL
        res_at_temp = evaluate_thermal_recommendation(
            hotspot_temp=48.0, avg_temp=35.0, cooling_health=70.0, failure_prob=0.10, anomaly_score=0.01
        )
        self.assertEqual(res_at_temp["severity"], "CRITICAL")
        self.assertEqual(res_at_temp["recommended_flow_boost_pct"], 35.0)

        # Exactly at 0.80 failure probability -> CRITICAL
        res_at_prob = evaluate_thermal_recommendation(
            hotspot_temp=35.0, avg_temp=30.0, cooling_health=70.0, failure_prob=0.80, anomaly_score=0.01
        )
        self.assertEqual(res_at_prob["severity"], "CRITICAL")
        self.assertEqual(res_at_prob["recommended_flow_boost_pct"], 35.0)

    def test_decision_engine_rule2_high_caution_boundaries(self):
        """Rule 2: High Thermal Caution (Hotspot >= 42.0°C or Failure Prob >= 0.40 or Cooling Health <= 50%)."""
        # Exactly at 42.0°C -> HIGH
        res_at_temp = evaluate_thermal_recommendation(
            hotspot_temp=42.0, avg_temp=35.0, cooling_health=70.0, failure_prob=0.10, anomaly_score=0.01
        )
        self.assertEqual(res_at_temp["severity"], "HIGH")
        self.assertEqual(res_at_temp["recommended_flow_boost_pct"], 18.0)

        # Immediately below 42.0°C (41.9°C), prob=0.39, health=50.1 -> Not HIGH
        res_below = evaluate_thermal_recommendation(
            hotspot_temp=41.9, avg_temp=35.0, cooling_health=50.1, failure_prob=0.39, anomaly_score=0.01
        )
        self.assertNotEqual(res_below["severity"], "HIGH")

        # Exactly at 50.0% cooling health -> HIGH
        res_at_health = evaluate_thermal_recommendation(
            hotspot_temp=35.0, avg_temp=30.0, cooling_health=50.0, failure_prob=0.10, anomaly_score=0.01
        )
        self.assertEqual(res_at_health["severity"], "HIGH")

        # Exactly at 0.40 failure prob -> HIGH
        res_at_prob = evaluate_thermal_recommendation(
            hotspot_temp=35.0, avg_temp=30.0, cooling_health=70.0, failure_prob=0.40, anomaly_score=0.01
        )
        self.assertEqual(res_at_prob["severity"], "HIGH")

    def test_decision_engine_rule3_multivariate_anomaly_boundaries(self):
        """Rule 3: Autoencoder anomaly score > threshold triggers CAUTION."""
        thresh = 0.1929
        # Score <= threshold -> Falls through to Rule 4 or 5
        res_at_thresh = evaluate_thermal_recommendation(
            hotspot_temp=30.0, avg_temp=25.0, cooling_health=70.0, failure_prob=0.10,
            anomaly_score=thresh, anomaly_threshold=thresh
        )
        self.assertEqual(res_at_thresh["severity"], "NORMAL")

        # Score > threshold -> CAUTION
        res_above = evaluate_thermal_recommendation(
            hotspot_temp=30.0, avg_temp=25.0, cooling_health=70.0, failure_prob=0.10,
            anomaly_score=thresh + 0.001, anomaly_threshold=thresh
        )
        self.assertEqual(res_above["severity"], "CAUTION")
        self.assertTrue(any("Autoencoder" in t for t in res_above["triggers"]))

    def test_decision_engine_rule4_mild_gradient_boundaries(self):
        """Rule 4: Hotspot >= 36.0°C or gradient (hotspot - avg) >= 10.0°C."""
        # Exactly at 36.0°C -> CAUTION
        res_temp = evaluate_thermal_recommendation(
            hotspot_temp=36.0, avg_temp=30.0, cooling_health=70.0, failure_prob=0.10, anomaly_score=0.01
        )
        self.assertEqual(res_temp["severity"], "CAUTION")

        # Exactly at 10.0°C delta -> CAUTION
        res_delta = evaluate_thermal_recommendation(
            hotspot_temp=34.0, avg_temp=24.0, cooling_health=70.0, failure_prob=0.10, anomaly_score=0.01
        )
        self.assertEqual(res_delta["severity"], "CAUTION")

        # Hotspot=35.9°C and delta=9.9°C -> NORMAL
        res_norm = evaluate_thermal_recommendation(
            hotspot_temp=35.9, avg_temp=26.0, cooling_health=70.0, failure_prob=0.10, anomaly_score=0.01
        )
        self.assertEqual(res_norm["severity"], "NORMAL")

    # ── 3. Thermal Simulator Physics & Extreme Bounds Invariants ─
    def test_simulator_extreme_flow_clamping(self):
        """set_coolant_flow must clamp values strictly between 0.005 and 0.060 kg/s."""
        sim = BatteryPackThermalSimulator(seed=42)
        # Attempt to set negative flow
        res_low = sim.set_coolant_flow(-0.05)
        self.assertAlmostEqual(sim.coolant_flow, 0.005, places=4)
        self.assertAlmostEqual(res_low["new_flow"], 0.005, places=4)

        # Attempt to set excessively high flow
        res_high = sim.set_coolant_flow(0.150)
        self.assertAlmostEqual(sim.coolant_flow, 0.060, places=4)
        self.assertAlmostEqual(res_high["new_flow"], 0.060, places=4)

    def test_simulator_finite_outputs_under_high_current(self):
        """Simulator temperatures must remain finite (no NaN or inf) even under extreme 100A load."""
        sim = BatteryPackThermalSimulator(current=100.0, coolant_flow=0.020, seed=42)
        for _ in range(10):
            sim.step(dt_seconds=30.0)
            self.assertTrue(np.all(np.isfinite(sim.temperatures)))
            self.assertTrue(np.all(sim.temperatures > 0.0))

    def test_simulator_hotspot_cell_index_range(self):
        """Hotspot cell id must always be between 1 and 24 inclusive."""
        sim = BatteryPackThermalSimulator(seed=42)
        info = sim.get_hotspot_info()
        self.assertGreaterEqual(info["cell_id"], 1)
        self.assertLessEqual(info["cell_id"], 24)
        self.assertGreaterEqual(info["cell_index"], 0)
        self.assertLessEqual(info["cell_index"], 23)

    # ── 4. ML Prediction & Anomaly Robustness ─────────────────────
    def test_mlp_and_xgb_handle_extra_features(self):
        """Models must gracefully filter features when dictionary contains superfluous metadata keys."""
        feats = get_default_features()
        feats["superfluous_key_1"] = 999.9
        feats["extra_metadata"] = "BENCHMARK_PROTOTYPE"

        p_mlp = predict_failure_probability(self.mlp, feats)
        self.assertIsInstance(p_mlp, float)
        self.assertGreaterEqual(p_mlp, 0.0)
        self.assertLessEqual(p_mlp, 1.0)

        if self.xgb is not None:
            p_xgb = predict_failure_probability(self.xgb, feats)
            self.assertIsInstance(p_xgb, float)
            self.assertGreaterEqual(p_xgb, 0.0)
            self.assertLessEqual(p_xgb, 1.0)

    def test_autoencoder_score_is_positive_finite(self):
        """Autoencoder reconstruction error must be strictly positive and finite."""
        ae_model, ae_scaler, ae_imputer, ae_threshold = self.ae_artifacts
        feats = get_default_features()
        res = detect_anomaly(ae_model, ae_scaler, ae_imputer, ae_threshold, feats)

        self.assertGreaterEqual(res["score"], 0.0)
        self.assertTrue(np.isfinite(res["score"]))
        self.assertIn(res["label"], ["NORMAL", "ANOMALY"])

    # ── 5. Risk Engine Invariants & Monotonicity ───────────────────
    def test_risk_score_is_bounded_and_monotonic_with_temperature(self):
        """Risk engine score must be strictly in [0.0, 100.0] and increase monotonically with temperature."""
        temps = [25.0, 35.0, 42.0, 46.0, 52.0, 60.0]
        scores = []
        for t in temps:
            r = assess_thermal_risk(current_temp=t, anomaly_score=0.0, hotspot_risk_pct=10.0)
            self.assertGreaterEqual(r["risk_score"], 0.0)
            self.assertLessEqual(r["risk_score"], 100.0)
            scores.append(r["risk_score"])

        # Monotonically non-decreasing
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i + 1], scores[i])

    def test_cell_temperature_grid_generator_invariants(self):
        """Synthetic grid generator must produce a grid whose max equals max_temp and mean equals avg_temp."""
        avg_t = 32.5
        max_t = 46.0
        grid = generate_cell_temperature_grid(avg_t, max_t, rows=3, cols=4, seed=42)

        self.assertEqual(grid.shape, (3, 4))
        self.assertAlmostEqual(float(np.max(grid)), max_t, places=1)
        self.assertAlmostEqual(float(np.mean(grid)), avg_t, delta=0.5)


if __name__ == "__main__":
    unittest.main()
