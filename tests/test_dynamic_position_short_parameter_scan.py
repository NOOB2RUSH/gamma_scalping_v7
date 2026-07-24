import unittest

import pandas as pd

from scripts.research.dynamic_position.dynamic_position_short_parameter_scan import (
    parameter_grid,
    window_statistics,
)


class DynamicPositionShortParameterScanTest(unittest.TestCase):
    def test_parameter_grid_builds_cartesian_product(self):
        grid = parameter_grid(
            pmin_values=[8, 10],
            pmax_values=[14, 16],
            iv_max_values=[0.35, 0.40],
            iv_spike_values=[0.03],
            short_steps_values=[5, 10],
        )
        self.assertEqual(len(grid), 16)
        self.assertIn(
            {
                "min_qty": 10,
                "max_qty": 16,
                "iv_max": 0.40,
                "iv_spike": 0.03,
                "short_steps": 10,
            },
            grid,
        )

    def test_window_statistics_uses_only_requested_interval(self):
        daily = pd.DataFrame(
            {"nav": [100.0, 110.0, 99.0, 105.0, 90.0]},
            index=pd.date_range("2024-09-20", periods=5, freq="B"),
        )
        stats = window_statistics(daily, "2024-09-23", "2024-09-25")
        self.assertAlmostEqual(stats["shock_return"], 105.0 / 110.0 - 1.0)
        self.assertAlmostEqual(stats["shock_max_drawdown"], 99.0 / 110.0 - 1.0)


if __name__ == "__main__":
    unittest.main()
