"""
Unit Tests for What-If Explainer, Decision Engine Status, and UI Helpers
========================================================================
Validates:
1. Decision engine schema compliance across all rules (presence of status, severity, actions).
2. What-If dynamic explanation generation (baseline, thermal stress, cooling improvements).
3. Safe division in percentage delta calculations (zero denominators).
4. One-Variable-at-a-Time (OVAT) sensitivity rankings and physical mechanisms.
5. Model comparison consensus and disagreement detection.
6. UI component HTML cleaning.
"""

import unittest
from typing import Dict, Any

from src.decision_engine import evaluate_thermal_recommendation
from src.what_if_explainer import (
    generate_what_if_explanation,
    explain_parameter_change,
    get_live_parameter_explanations,
    compute_simulation_comparison,
)
from src.prediction_engine import load_mlp_model, load_model as load_xgb_model, get_default_features
from src.anomaly_detection import load_autoencoder
from app.styles import clean_html


class TestWhatIfExplainerAndDecisionSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mlp = load_mlp_model("models/failure_predictor")
        cls.xgb = load_xgb_model("models/xgboost_failure_model.joblib")
        cls.ae_artifacts = load_autoencoder("models/autoencoder")

    def test_decision_engine_schema_all_rules(self):
        """Verify all decision engine rules return required keys with valid strings."""
        test_cases = [
            # (hotspot, avg, cool, prob, anomaly, current, flow)
            (52.0, 42.0, 30.0, 0.85, 0.05, 40.0, 0.012),  # Rule 1: Critical runaway
            (44.0, 38.0, 40.0, 0.45, 0.05, 30.0, 0.020),  # Rule 2: High caution
            (32.0, 28.0, 70.0, 0.10, 0.25, 10.0, 0.020),  # Rule 3: Multivariate anomaly
            (38.0, 26.0, 70.0, 0.10, 0.05, 15.0, 0.020),  # Rule 4: Moderate elevation
            (25.0, 22.0, 74.0, 0.05, 0.01, 10.0, 0.020),  # Rule 5: Nominal
        ]

        required_keys = [
            "status",
            "severity",
            "suggested_action",
            "recommended_action",
            "reason",
            "observation",
            "triggers",
            "predicted_outcome",
            "recommended_flow_boost_pct",
        ]

        for case in test_cases:
            res = evaluate_thermal_recommendation(
                hotspot_temp=case[0],
                avg_temp=case[1],
                cooling_health=case[2],
                failure_prob=case[3],
                anomaly_score=case[4],
                coolant_flow=case[6],
                current=case[5],
            )
            for k in required_keys:
                self.assertIn(k, res, f"Decision engine output missing key '{k}' for case {case}")
            self.assertEqual(res["status"], res["severity"])
            self.assertEqual(res["suggested_action"], res["recommended_action"])
            self.assertIn(res["status"], ["NORMAL", "CAUTION", "HIGH", "CRITICAL"])
            self.assertIsInstance(res["recommended_flow_boost_pct"], (int, float))

    def test_what_if_baseline_scenario_identical(self):
        """When scenario equals baseline, all deltas must be zero and risk direction nominal."""
        baseline = get_default_features()
        scenario = baseline.copy()

        exp = generate_what_if_explanation(
            baseline=baseline,
            scenario=scenario,
            mlp_model=self.mlp,
            xgb_model=self.xgb,
            ae_artifacts=self.ae_artifacts,
        )

        self.assertEqual(exp["summary"]["direction"], "STABLE / NOMINAL")
        self.assertAlmostEqual(exp["summary"]["mlp_delta_pp"], 0.0, places=3)
        self.assertAlmostEqual(exp["summary"]["risk_score_delta"], 0.0, places=1)

        # Check all table rows show UNCHANGED
        for row in exp["parameter_table"]:
            self.assertEqual(row["direction"], "UNCHANGED")
            self.assertEqual(row["indicator"], "━")
            self.assertAlmostEqual(row["abs_delta_val"], 0.0, places=3)

    def test_what_if_severe_thermal_stress_scenario(self):
        """Severe thermal stress scenario must increase risk, rank hotspot/resistance, and prescribe intervention."""
        baseline = get_default_features()
        scenario = baseline.copy()
        scenario["cell_temperature_max"] = 56.0
        scenario["internal_resistance"] = 3.5
        scenario["cooling_system_health"] = 20.0
        scenario["average_charge_power_kw"] = 120.0

        exp = generate_what_if_explanation(
            baseline=baseline,
            scenario=scenario,
            mlp_model=self.mlp,
            xgb_model=self.xgb,
            ae_artifacts=self.ae_artifacts,
        )

        self.assertEqual(exp["summary"]["direction"], "INCREASED RISK")
        self.assertGreater(exp["summary"]["mlp_delta_pp"], 10.0)
        self.assertGreater(exp["summary"]["scen_mlp_pct"], exp["summary"]["base_mlp_pct"])

        # Check sensitivity ranking
        exps = exp["physics_explanations"]
        self.assertTrue(len(exps) > 0)
        # Verify drivers are ranked by absolute marginal impact
        impacts = [abs(e["marginal_impact_pp"]) for e in exps]
        self.assertEqual(impacts, sorted(impacts, reverse=True))

        # Check recommendation
        rec = exp["recommendation"]
        self.assertIn("Immediate", rec["title"])
        self.assertIn("Simulated recommendation only", rec["disclaimer"])

    def test_what_if_cooling_improvement_scenario(self):
        """Enhanced cooling and moderate power scenario must reduce or maintain low risk."""
        baseline = get_default_features()
        scenario = baseline.copy()
        scenario["cooling_system_health"] = 95.0
        scenario["average_charge_power_kw"] = 20.0

        exp = generate_what_if_explanation(
            baseline=baseline,
            scenario=scenario,
            mlp_model=self.mlp,
            xgb_model=self.xgb,
            ae_artifacts=self.ae_artifacts,
        )

        self.assertIn(exp["summary"]["direction"], ["DECREASED RISK", "STABLE / NOMINAL"])
        self.assertLessEqual(exp["summary"]["mlp_delta_pp"], 0.5)

    def test_what_if_safe_division_zero_denominator(self):
        """Check that zero baseline values handle percentage delta division safely."""
        baseline = get_default_features()
        baseline["average_charge_power_kw"] = 0.0  # Zero denominator
        scenario = baseline.copy()
        scenario["average_charge_power_kw"] = 50.0

        exp = generate_what_if_explanation(
            baseline=baseline,
            scenario=scenario,
            mlp_model=self.mlp,
            xgb_model=self.xgb,
            ae_artifacts=self.ae_artifacts,
        )

        power_row = next(r for r in exp["parameter_table"] if r["key"] == "average_charge_power_kw")
        self.assertEqual(power_row["direction"], "INCREASE")
        self.assertEqual(power_row["indicator"], "▲")
        self.assertEqual(power_row["pct_delta"], "N/A")  # Handled safely

    def test_clean_html_strips_indented_lines(self):
        """clean_html must eliminate leading whitespace on every line to prevent code block interpretation."""
        indented_html = """
        <div class="metric-tile">
            <span>Label</span>
        </div>
        """
        cleaned = clean_html(indented_html)
        self.assertNotIn("    ", cleaned)
        self.assertTrue(cleaned.startswith('<div class="metric-tile">'))
        self.assertTrue(cleaned.endswith("</div>"))

    def test_explain_parameter_change_all_sliders(self):
        """Verify individual parameter explanations for all 6 sliders."""
        sliders = [
            ("cell_temperature_avg", 21.0, 32.0, "INCREASE"),
            ("cell_temperature_avg", 21.0, 18.0, "DECREASE"),
            ("cell_temperature_max", 28.5, 45.0, "INCREASE"),
            ("cell_temperature_max", 28.5, 24.0, "DECREASE"),
            ("internal_resistance", 0.53, 1.80, "INCREASE"),
            ("internal_resistance", 0.53, 0.35, "DECREASE"),
            ("cooling_system_health", 74.0, 40.0, "DECREASE"),
            ("cooling_system_health", 74.0, 90.0, "INCREASE"),
            ("average_charge_power_kw", 40.0, 95.0, "INCREASE"),
            ("average_charge_power_kw", 40.0, 20.0, "DECREASE"),
            ("state_of_charge", 55.0, 85.0, "INCREASE"),
            ("state_of_charge", 55.0, 25.0, "DECREASE"),
        ]

        required_fields = [
            "param_key",
            "param_name",
            "baseline_val",
            "current_val",
            "delta_val",
            "change_summary",
            "direction",
            "what_this_means",
            "why_it_matters",
            "expected_effect",
            "suggested_adjustment",
        ]

        for key, base, curr, expected_dir in sliders:
            res = explain_parameter_change(key, base, curr)
            self.assertIsNotNone(res, f"Expected explanation for {key} changed from {base} to {curr}")
            for field in required_fields:
                self.assertIn(field, res, f"Missing field {field} in explanation for {key}")
            self.assertEqual(res["direction"], expected_dir)
            self.assertTrue(len(res["what_this_means"]) > 10)
            self.assertTrue(len(res["why_it_matters"]) > 10)
            self.assertTrue(len(res["expected_effect"]) > 10)
            self.assertTrue(len(res["suggested_adjustment"]) > 10)

    def test_explain_parameter_change_threshold_transitions(self):
        """Verify hotspot and cooling thresholds trigger the expected caution/critical text."""
        # Hotspot critical (>= 48.0°C)
        crit_res = explain_parameter_change("cell_temperature_max", 28.5, 52.0)
        self.assertIn("48.0°C", crit_res["what_this_means"])
        self.assertIn("critical", crit_res["expected_effect"].lower())

        # Hotspot caution (>= 42.0°C and < 48.0°C)
        caut_res = explain_parameter_change("cell_temperature_max", 28.5, 43.5)
        self.assertIn("42.0°C", caut_res["what_this_means"])
        self.assertIn("caution", caut_res["expected_effect"].lower())

        # Cooling health degraded (<= 50.0%)
        cool_deg = explain_parameter_change("cooling_system_health", 74.0, 40.0)
        self.assertIn("50%", cool_deg["what_this_means"])

    def test_explain_parameter_change_unchanged_returns_none(self):
        """When parameter value equals baseline within tolerance, returns None."""
        self.assertIsNone(explain_parameter_change("cell_temperature_avg", 21.0, 21.0))
        self.assertIsNone(explain_parameter_change("internal_resistance", 0.53, 0.53001))

    def test_get_live_parameter_explanations_empty_when_all_baseline(self):
        """When all sliders match baseline, live explanations list must be empty."""
        baseline = get_default_features()
        current = baseline.copy()
        exps = get_live_parameter_explanations(baseline, current)
        self.assertEqual(len(exps), 0)

    def test_get_live_parameter_explanations_multiple_changes(self):
        """When multiple sliders are changed, all changed parameters must be returned."""
        baseline = get_default_features()
        current = baseline.copy()
        current["cell_temperature_avg"] = 30.0
        current["cell_temperature_max"] = 44.0
        current["cooling_system_health"] = 45.0

        exps = get_live_parameter_explanations(baseline, current)
        self.assertEqual(len(exps), 3)
        keys = [e["param_key"] for e in exps]
        self.assertEqual(keys, ["cell_temperature_avg", "cell_temperature_max", "cooling_system_health"])

    def test_compute_simulation_comparison(self):
        """Verify before/after simulation comparison outputs probabilities and compact explanation."""
        baseline = get_default_features()
        scenario = baseline.copy()
        scenario["cell_temperature_max"] = 46.0
        scenario["average_charge_power_kw"] = 90.0

        comp = compute_simulation_comparison(
            baseline=baseline,
            scenario=scenario,
            mlp_model=self.mlp,
            xgb_model=self.xgb,
            ae_artifacts=self.ae_artifacts,
        )

        self.assertIn("base_mlp_pct", comp)
        self.assertIn("scen_mlp_pct", comp)
        self.assertIn("mlp_delta_pp", comp)
        self.assertIn("base_xgb_pct", comp)
        self.assertIn("scen_xgb_pct", comp)
        self.assertIn("result_explanation", comp)
        self.assertIn("recommended_action", comp)
        self.assertIn("disclaimer", comp)

        self.assertGreater(comp["mlp_delta_pp"], 0.0)
        self.assertIn("percentage points", comp["result_explanation"])
        self.assertIn("statistical risk estimates", comp["disclaimer"])


if __name__ == "__main__":
    unittest.main()
