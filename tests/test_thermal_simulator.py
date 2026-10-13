"""
Unit tests for BatteryPackThermalSimulator
=========================================
Tests the physical modeling, neighbor conduction, hotspot detection,
coolant flow sensitivity, and runaway dynamics.
"""

import unittest
import numpy as np
from src.thermal_simulator import BatteryPackThermalSimulator


class TestBatteryPackThermalSimulator(unittest.TestCase):
    """Test suite for the 24-cell thermal simulator."""

    def setUp(self):
        self.sim = BatteryPackThermalSimulator(
            ambient_temp=30.0,
            coolant_temp=22.0,
            current=10.0,
            coolant_flow=0.02,
            seed=42,
        )

    def test_initialization(self):
        """Verify initial pack geometry and temperature vector."""
        self.assertEqual(len(self.sim.temperatures), 24)
        self.assertEqual(self.sim.NUM_CELLS, 24)
        self.assertEqual(self.sim.ROWS, 4)
        self.assertEqual(self.sim.COLS, 6)
        self.assertEqual(len(self.sim.resistances), 24)

    def test_neighbor_connectivity(self):
        """Verify grid neighbor graph connections for corners, edges, and center cells."""
        # Top-left corner (index 0) has 2 neighbors: right (1) and down (6)
        corner_neighbors = self.sim._get_neighbors(0)
        self.assertCountEqual(corner_neighbors, [1, 6])

        # Bottom-right corner (index 23) has 2 neighbors: left (22) and up (17)
        br_neighbors = self.sim._get_neighbors(23)
        self.assertCountEqual(br_neighbors, [22, 17])

        # Center cell (e.g., cell index 7: row 1, col 1) has 4 neighbors
        center_neighbors = self.sim._get_neighbors(7)
        self.assertCountEqual(center_neighbors, [1, 13, 6, 8])

    def test_hotspot_identification(self):
        """Cell 7 (index 6) should be identified as initial hotspot."""
        info = self.sim.get_hotspot_info()
        self.assertEqual(info["cell_id"], 7)
        self.assertAlmostEqual(info["temperature"], 45.3, places=1)
        self.assertIn("Middle", info["location"])

    def test_pack_summary_metrics(self):
        """Check calculation of avg, max, min, risk levels."""
        summary = self.sim.get_pack_summary()
        self.assertAlmostEqual(summary["max_temp"], 45.3, places=1)
        self.assertGreater(summary["avg_temp"], 30.0)
        self.assertLess(summary["avg_temp"], 46.0)
        self.assertEqual(summary["total_cells"], 24)
        self.assertEqual(summary["active_cells"], 24)
        self.assertEqual(summary["faults"], 0)
        self.assertEqual(summary["risk_level"], "MODERATE")

    def test_step_simulation(self):
        """Verify stepping forward in time records history and updates temps."""
        initial_history_len = len(self.sim.history)
        new_temps = self.sim.step(dt_seconds=60.0)
        self.assertEqual(len(new_temps), 24)
        self.assertEqual(len(self.sim.history), initial_history_len + 1)
        self.assertEqual(self.sim.time_elapsed_seconds, 60.0)

    def test_cooling_flow_effect(self):
        """Higher coolant flow should yield lower temperatures under the same load."""
        sim_low_flow = BatteryPackThermalSimulator(current=12.0, coolant_flow=0.01, seed=42)
        sim_high_flow = BatteryPackThermalSimulator(current=12.0, coolant_flow=0.03, seed=42)

        for _ in range(5):
            sim_low_flow.step(dt_seconds=60.0)
            sim_high_flow.step(dt_seconds=60.0)

        self.assertGreater(
            np.mean(sim_low_flow.temperatures),
            np.mean(sim_high_flow.temperatures),
            "Low coolant flow should lead to higher pack temperatures than high coolant flow.",
        )

    def test_trajectory_forecasting(self):
        """Simulate forward trajectory for 5 minutes."""
        traj = self.sim.simulate_trajectory(duration_minutes=5, dt_minutes=1.0)
        self.assertEqual(len(traj), 6)  # 0, 1, 2, 3, 4, 5 min
        self.assertEqual(traj[0]["minute"], 0)
        self.assertEqual(traj[-1]["minute"], 5)
        # Hotspot temperature should be positive and bounded
        for entry in traj:
            self.assertGreater(entry["hotspot_temp"], 30.0)
            self.assertLess(entry["hotspot_temp"], 100.0)

    def test_apply_recommendation(self):
        """Applying AI recommendation should increase coolant flow and reduce hotspot temp."""
        old_hotspot_temp = self.sim.get_hotspot_info()["temperature"]
        old_flow = self.sim.coolant_flow

        res = self.sim.apply_recommendation(boost_percentage=18.0)
        self.assertEqual(res["status"], "applied")
        self.assertGreater(self.sim.coolant_flow, old_flow)
        self.assertLess(self.sim.get_hotspot_info()["temperature"], old_hotspot_temp)

    def test_thermal_runaway_simulation(self):
        """Verify runaway simulation comparison structure."""
        runaway_data = self.sim.simulate_thermal_runaway()
        self.assertIn("minutes", runaway_data)
        self.assertIn("without_ai", runaway_data)
        self.assertIn("with_ai", runaway_data)
        self.assertIn("temp_rise_sequence", runaway_data)
        self.assertIn("ai_response_sequence", runaway_data)
        # Without AI should end higher than With AI
        self.assertGreater(
            runaway_data["without_ai"][-1],
            runaway_data["with_ai"][-1],
        )

    def test_set_coolant_flow_adjusts_temperatures(self):
        """Slider flow adjustment should dynamically update coolant flow and cell temperatures."""
        initial_hotspot = self.sim.get_hotspot_info()["temperature"]
        initial_avg = self.sim.get_pack_summary()["avg_temp"]

        # Increasing flow from 0.02 to 0.04 kg/s should reduce hotspot and pack avg
        res_up = self.sim.set_coolant_flow(0.04)
        self.assertEqual(res_up["status"], "updated")
        self.assertEqual(self.sim.coolant_flow, 0.04)
        self.assertLess(res_up["new_hotspot_temp"], initial_hotspot)
        self.assertLess(res_up["delta_temp"], 0.0)
        self.assertLess(self.sim.get_pack_summary()["avg_temp"], initial_avg)

        # Decreasing flow back to 0.01 kg/s should increase temperatures
        cur_hotspot = self.sim.get_hotspot_info()["temperature"]
        res_down = self.sim.set_coolant_flow(0.01)
        self.assertEqual(res_down["status"], "updated")
        self.assertEqual(self.sim.coolant_flow, 0.01)
        self.assertGreater(res_down["new_hotspot_temp"], cur_hotspot)
        self.assertGreater(res_down["delta_temp"], 0.0)

    def test_cooling_response_all_modes(self):
        """
        Verify that higher coolant flow strictly lowers hotspot and average temperatures
        across NORMAL, HIGH_LOAD, and THERMAL_STRESS modes under identical initial conditions
        and elapsed simulation time.
        """
        scenarios = [
            ("NORMAL", 10.0, 30.0, 22.0),
            ("HIGH_LOAD", 35.0, 32.0, 24.0),
            ("THERMAL_STRESS", 45.0, 38.0, 28.0),
        ]

        for mode, current, amb_temp, cool_temp in scenarios:
            with self.subTest(mode=mode):
                # Low coolant flow simulator
                sim_low = BatteryPackThermalSimulator(
                    ambient_temp=amb_temp,
                    coolant_temp=cool_temp,
                    current=current,
                    coolant_flow=0.010,
                    seed=42,
                )
                # High coolant flow simulator
                sim_high = BatteryPackThermalSimulator(
                    ambient_temp=amb_temp,
                    coolant_temp=cool_temp,
                    current=current,
                    coolant_flow=0.040,
                    seed=42,
                )

                # Verify identical initial conditions
                np.testing.assert_array_almost_equal(
                    sim_low.temperatures, sim_high.temperatures,
                    err_msg=f"{mode}: Initial temperatures must be identical",
                )

                # Run both for identical elapsed simulation time
                sim_low.step(dt_seconds=60.0)
                sim_high.step(dt_seconds=60.0)

                low_hotspot = sim_low.get_hotspot_info()["temperature"]
                high_hotspot = sim_high.get_hotspot_info()["temperature"]
                low_avg = sim_low.get_pack_summary()["avg_temp"]
                high_avg = sim_high.get_pack_summary()["avg_temp"]

                self.assertLess(
                    high_hotspot,
                    low_hotspot,
                    f"{mode}: High flow hotspot ({high_hotspot}°C) must be lower than low flow hotspot ({low_hotspot}°C)",
                )
                self.assertLess(
                    high_avg,
                    low_avg,
                    f"{mode}: High flow avg ({high_avg}°C) must be lower than low flow avg ({low_avg}°C)",
                )


if __name__ == "__main__":
    unittest.main()
