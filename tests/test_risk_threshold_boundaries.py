import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.risk_engine import assess_thermal_risk, load_risk_config


class TestRiskThresholdBoundaries(unittest.TestCase):
    """
    Focused verification tests for boundary thresholds in risk_engine:
    Testing immediately below, exactly at, and immediately above every configured threshold.
    """

    def setUp(self):
        self.config = load_risk_config()

    def test_current_temperature_boundaries(self):
        # Config: normal_max=40.0, caution_max=45.0, critical_above=50.0
        # Immediately below normal_max: 39.9°C -> NORMAL (0)
        r_below_norm = assess_thermal_risk(current_temp=39.9, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_below_norm["risk_level"], "NORMAL")

        # Exactly at normal_max: 40.0°C -> NORMAL (0)
        r_at_norm = assess_thermal_risk(current_temp=40.0, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_at_norm["risk_level"], "NORMAL")

        # Immediately above normal_max: 40.1°C -> CAUTION (1)
        r_above_norm = assess_thermal_risk(current_temp=40.1, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_above_norm["risk_level"], "CAUTION")

        # Immediately below caution_max: 44.9°C -> CAUTION (1)
        r_below_caut = assess_thermal_risk(current_temp=44.9, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_below_caut["risk_level"], "CAUTION")

        # Exactly at caution_max: 45.0°C -> CAUTION (1)
        r_at_caut = assess_thermal_risk(current_temp=45.0, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_at_caut["risk_level"], "CAUTION")

        # Immediately above caution_max: 45.1°C -> HIGH (2)
        r_above_caut = assess_thermal_risk(current_temp=45.1, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_above_caut["risk_level"], "HIGH")

        # Immediately below critical_above: 49.9°C -> HIGH (2)
        r_below_crit = assess_thermal_risk(current_temp=49.9, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_below_crit["risk_level"], "HIGH")

        # Exactly at critical_above: 50.0°C -> HIGH (2)
        r_at_crit = assess_thermal_risk(current_temp=50.0, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_at_crit["risk_level"], "HIGH")

        # Immediately above critical_above: 50.1°C -> CRITICAL (3)
        r_above_crit = assess_thermal_risk(current_temp=50.1, anomaly_score=0.0, hotspot_risk_pct=0.0)
        self.assertEqual(r_above_crit["risk_level"], "CRITICAL")

    def test_hotspot_risk_percent_boundaries(self):
        # Config: normal_max=30, caution_max=55, critical_above=75
        # Set current_temp to 30.0 so temp_risk is 0
        # Immediately below normal_max: 29.9% -> NORMAL
        r_below_norm = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=29.9)
        self.assertEqual(r_below_norm["risk_level"], "NORMAL")

        # Exactly at normal_max: 30.0% -> NORMAL
        r_at_norm = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=30.0)
        self.assertEqual(r_at_norm["risk_level"], "NORMAL")

        # Immediately above normal_max: 30.1% -> CAUTION
        r_above_norm = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=30.1)
        self.assertEqual(r_above_norm["risk_level"], "CAUTION")

        # Immediately below caution_max: 54.9% -> CAUTION
        r_below_caut = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=54.9)
        self.assertEqual(r_below_caut["risk_level"], "CAUTION")

        # Exactly at caution_max: 55.0% -> CAUTION
        r_at_caut = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=55.0)
        self.assertEqual(r_at_caut["risk_level"], "CAUTION")

        # Immediately above caution_max: 55.1% -> HIGH
        r_above_caut = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=55.1)
        self.assertEqual(r_above_caut["risk_level"], "HIGH")

        # Immediately below critical_above: 74.9% -> HIGH
        r_below_crit = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=74.9)
        self.assertEqual(r_below_crit["risk_level"], "HIGH")

        # Exactly at critical_above: 75.0% -> HIGH
        r_at_crit = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=75.0)
        self.assertEqual(r_at_crit["risk_level"], "HIGH")

        # Immediately above critical_above: 75.1% -> CRITICAL
        r_above_crit = assess_thermal_risk(current_temp=30.0, anomaly_score=0.0, hotspot_risk_pct=75.1)
        self.assertEqual(r_above_crit["risk_level"], "CRITICAL")

    def test_predicted_temperature_boundaries(self):
        # Config: normal_max=40.0, caution_max=45.0, critical_above=50.0
        r_at_norm = assess_thermal_risk(current_temp=30.0, predicted_temp_5min=40.0)
        self.assertEqual(r_at_norm["risk_level"], "NORMAL")

        r_above_norm = assess_thermal_risk(current_temp=30.0, predicted_temp_5min=40.1)
        self.assertEqual(r_above_norm["risk_level"], "CAUTION")

        r_at_caut = assess_thermal_risk(current_temp=30.0, predicted_temp_5min=45.0)
        self.assertEqual(r_at_caut["risk_level"], "CAUTION")

        r_above_caut = assess_thermal_risk(current_temp=30.0, predicted_temp_5min=45.1)
        self.assertEqual(r_above_caut["risk_level"], "HIGH")

        r_at_crit = assess_thermal_risk(current_temp=30.0, predicted_temp_5min=50.0)
        self.assertEqual(r_at_crit["risk_level"], "HIGH")

        r_above_crit = assess_thermal_risk(current_temp=30.0, predicted_temp_5min=50.1)
        self.assertEqual(r_above_crit["risk_level"], "CRITICAL")

    def test_temperature_rate_boundaries(self):
        # Config: normal_max=0.5, caution_max=1.5, critical_above=3.0
        r_at_norm = assess_thermal_risk(current_temp=30.0, temp_rate_per_min=0.5)
        self.assertEqual(r_at_norm["risk_level"], "NORMAL")

        r_above_norm = assess_thermal_risk(current_temp=30.0, temp_rate_per_min=0.51)
        self.assertEqual(r_above_norm["risk_level"], "CAUTION")

        r_at_caut = assess_thermal_risk(current_temp=30.0, temp_rate_per_min=1.5)
        self.assertEqual(r_at_caut["risk_level"], "CAUTION")

        r_above_caut = assess_thermal_risk(current_temp=30.0, temp_rate_per_min=1.51)
        self.assertEqual(r_above_caut["risk_level"], "HIGH")

        r_at_crit = assess_thermal_risk(current_temp=30.0, temp_rate_per_min=3.0)
        self.assertEqual(r_at_crit["risk_level"], "HIGH")

        r_above_crit = assess_thermal_risk(current_temp=30.0, temp_rate_per_min=3.01)
        self.assertEqual(r_above_crit["risk_level"], "CRITICAL")


if __name__ == "__main__":
    unittest.main()
