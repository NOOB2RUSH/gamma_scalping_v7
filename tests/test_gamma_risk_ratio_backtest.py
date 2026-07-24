import unittest

import pandas as pd

from scripts.research.gamma_risk_ratio_backtest import (
    fill_inactive_account_days,
    performance_stats,
    summarize_account,
)


class GammaRiskRatioBacktestTest(unittest.TestCase):
    def test_fill_inactive_days_carries_nav_without_cost_or_margin(self):
        all_dates = pd.date_range("2026-01-05", periods=3, freq="B")
        sparse = pd.DataFrame(
            {
                "nav": [1_000_010.0, 1_000_030.0],
                "cash": [900_000.0, 920_000.0],
                "option_margin": [100.0, 130.0],
                "hedge_margin": [50.0, 70.0],
                "option_fee": [10.0, 3.0],
                "etf_fee": [2.0, 1.0],
            },
            index=all_dates[[0, 2]],
        )

        filled = fill_inactive_account_days(sparse, all_dates, 1_000_000.0)

        self.assertEqual(filled["nav"].tolist(), [
            1_000_010.0,
            1_000_010.0,
            1_000_030.0,
        ])
        self.assertEqual(filled["cash"].tolist(), [
            900_000.0,
            900_000.0,
            920_000.0,
        ])
        self.assertEqual(filled.loc[all_dates[1], "option_margin"], 0.0)
        self.assertEqual(filled.loc[all_dates[1], "hedge_margin"], 0.0)
        self.assertEqual(filled.loc[all_dates[1], "option_fee"], 0.0)
        self.assertEqual(filled.loc[all_dates[1], "etf_fee"], 0.0)

    def test_summary_describes_one_independent_account(self):
        index = pd.date_range("2026-01-05", periods=3, freq="B")
        daily = pd.DataFrame(
            {
                "nav": [999_980.0, 1_000_100.0, 1_000_250.0],
                "cash": [900_000.0, 910_000.0, 920_000.0],
                "option_margin": [100.0, 120.0, 130.0],
                "hedge_margin": [50.0, 60.0, 70.0],
                "option_fee": [10.0, 0.0, 0.0],
                "etf_fee": [10.0, 0.0, 0.0],
            },
            index=index,
        )
        trades = pd.DataFrame(
            {
                "type": [
                    "open_straddle",
                    "close_straddle",
                    "open_short_straddle",
                ]
            }
        )

        summary = summarize_account(
            "500etf",
            10,
            daily,
            trades,
            1_000_000.0,
        )

        self.assertEqual(summary["initial_cash"], 1_000_000.0)
        self.assertEqual(summary["final_nav"], 1_000_250.0)
        self.assertEqual(summary["total_pnl"], 250.0)
        self.assertEqual(summary["total_fee"], 20.0)
        self.assertEqual(summary["pnl_before_fee"], 270.0)
        self.assertEqual(summary["long_entries"], 1)
        self.assertEqual(summary["short_entries"], 1)
        self.assertEqual(summary["max_total_margin"], 200.0)
        self.assertAlmostEqual(summary["max_margin_to_nav"], 200 / 1_000_250)

    def test_performance_stats_uses_only_the_supplied_account_nav(self):
        nav = pd.Series(
            [1_000_000.0, 900_000.0, 950_000.0],
            index=pd.date_range("2026-01-05", periods=3, freq="B"),
        )

        stats = performance_stats(nav, 1_000_000.0)

        self.assertEqual(stats["total_pnl"], -50_000.0)
        self.assertAlmostEqual(stats["total_return"], -0.05)
        self.assertAlmostEqual(stats["max_drawdown"], -0.10)


if __name__ == "__main__":
    unittest.main()
