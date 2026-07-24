import unittest

import pandas as pd

from scripts.research.dynamic_position.iv_daily_absolute_change_distribution import (
    build_daily_absolute_changes,
    build_fixed_threshold_summary,
    build_quantile_summary,
)


class IvDailyAbsoluteChangeDistributionTest(unittest.TestCase):
    def setUp(self):
        self.atm_iv = pd.Series(
            [0.10, 0.12, 0.11, None, 0.15, 0.20],
            index=pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-12",
                ]
            ),
        )
        self.samples = build_daily_absolute_changes(self.atm_iv)

    def test_uses_only_adjacent_trading_day_rows(self):
        valid = self.samples.dropna(subset=["atm_iv_absolute_change"])
        self.assertEqual(len(valid), 3)
        self.assertEqual(valid["date"].tolist(), ["2026-01-06", "2026-01-07", "2026-01-12"])
        self.assertAlmostEqual(valid.iloc[0]["atm_iv_absolute_change"], 0.02)
        self.assertAlmostEqual(valid.iloc[1]["atm_iv_absolute_change_signed"], -0.01)
        self.assertAlmostEqual(valid.iloc[2]["atm_iv_absolute_change"], 0.05)

    def test_fixed_threshold_summary_counts_tail_direction(self):
        summary = build_fixed_threshold_summary(self.samples, thresholds=(0.015,))
        row = summary.iloc[0]
        self.assertEqual(row["event_days"], 2)
        self.assertEqual(row["upward_event_days"], 2)
        self.assertEqual(row["downward_event_days"], 0)
        self.assertAlmostEqual(row["event_share"], 2 / 3)

    def test_quantile_summary_returns_iv_units_and_percentage_points(self):
        summary = build_quantile_summary(self.samples, quantiles=(0.50,))
        row = summary.iloc[0]
        self.assertAlmostEqual(row["ivspike_threshold"], 0.02)
        self.assertAlmostEqual(row["ivspike_threshold_percentage_points"], 2.0)


if __name__ == "__main__":
    unittest.main()
