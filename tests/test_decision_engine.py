"""
Unit tests for Deterministic Thermal Decision Engine
=====================================================
Validates all rule conditions, threshold boundaries, and severity escalations.
"""

import unittest
from src.decision_engine import evaluate_thermal_recommendation


class TestThermalDecisionEngine(unittest.TestCase):
    """Test suite for deterministic thermal decision rules."""

    def test_nominal_safe_state(self):
        """Nominal inputs should yield NORMAL severity with no alerts."""
        res = evaluate_thermal_recommendation(
            hotspot_temp=28.5,
            avg_temp=21.0,
            cooling_health=74.0,
            failure_prob=0.01,
            anomaly_score=0.003,
            anomaly_threshold=0.1929,
        )
        self.assertEqual(res["severity"], "NORMAL")
        self.assertEqual(res["suggested_action"], "MAINTAIN NOMINAL OPERATION")
        self.assertEqual(res["recommended_flow_boost_pct"], 0.0)

    def test_caution_on_mild_temperature_elevation(self):
        """Hotspot temp >= 36°C should trigger CAUTION with preemptive cooling."""
        res = evaluate_thermal_recommendation(
            hotspot_temp=38.0,
            avg_temp=25.0,
            cooling_health=70.0,
            failure_prob=0.05,
            anomaly_score=0.05,
        )
        self.assertEqual(res["severity"], "CAUTION")
        self.assertIn("PREEMPTIVE", res["suggested_action"])
        self.assertEqual(res["recommended_flow_boost_pct"], 10.0)

    def test_high_caution_on_hotspot_boundary(self):
        """Hotspot temp >= 42°C should trigger HIGH severity with 18% flow boost."""
        res = evaluate_thermal_recommendation(
            hotspot_temp=43.5,
            avg_temp=35.0,
            cooling_health=65.0,
            failure_prob=0.15,
            anomaly_score=0.10,
        )
        self.assertEqual(res["severity"], "HIGH")
        self.assertIn("INCREASE COOLANT FLOW", res["suggested_action"])
        self.assertEqual(res["recommended_flow_boost_pct"], 18.0)
        self.assertTrue(any("42.0°C" in t for t in res["triggers"]))

    def test_high_caution_on_degraded_cooling_health(self):
        """Cooling health <= 50% should trigger HIGH severity even if temp is moderate."""
        res = evaluate_thermal_recommendation(
            hotspot_temp=32.0,
            avg_temp=25.0,
            cooling_health=40.0,
            failure_prob=0.05,
            anomaly_score=0.02,
        )
        self.assertEqual(res["severity"], "HIGH")
        self.assertTrue(any("Cooling system health" in t for t in res["triggers"]))

    def test_critical_runaway_boundary(self):
        """Hotspot temp >= 48°C or failure prob >= 0.80 should trigger CRITICAL severity."""
        res_temp = evaluate_thermal_recommendation(
            hotspot_temp=49.0,
            avg_temp=40.0,
            cooling_health=60.0,
            failure_prob=0.20,
            anomaly_score=0.15,
        )
        self.assertEqual(res_temp["severity"], "CRITICAL")
        self.assertIn("EMERGENCY MAXIMUM COOLING", res_temp["suggested_action"])
        self.assertEqual(res_temp["recommended_flow_boost_pct"], 35.0)

        res_prob = evaluate_thermal_recommendation(
            hotspot_temp=35.0,
            avg_temp=30.0,
            cooling_health=70.0,
            failure_prob=0.85,
            anomaly_score=0.10,
        )
        self.assertEqual(res_prob["severity"], "CRITICAL")

    def test_multivariate_anomaly_trigger(self):
        """Anomaly score > threshold with normal temp should trigger CAUTION."""
        res = evaluate_thermal_recommendation(
            hotspot_temp=30.0,
            avg_temp=22.0,
            cooling_health=75.0,
            failure_prob=0.02,
            anomaly_score=0.35,
            anomaly_threshold=0.1929,
        )
        self.assertEqual(res["severity"], "CAUTION")
        self.assertTrue(any("Autoencoder reconstruction error" in t for t in res["triggers"]))


if __name__ == "__main__":
    unittest.main()
