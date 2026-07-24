import math
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.research.gamma_exposure_report import (
    approximate_gamma_pnl,
    build_daily_gamma_exposure,
    build_scenario_table,
    calculate_cash_gamma,
    summarize_product,
)


class GammaExposureReportTest(unittest.TestCase):
    def test_cash_gamma_and_gamma_pnl_follow_requested_formula(self):
        exposure = calculate_cash_gamma(
            call_gamma=0.20,
            put_gamma=0.30,
            call_multiplier=10_000,
            put_multiplier=10_000,
            spot=4.0,
        )

        self.assertAlmostEqual(exposure["call_cash_gamma"], 32_000.0)
        self.assertAlmostEqual(exposure["put_cash_gamma"], 48_000.0)
        self.assertAlmostEqual(exposure["pair_cash_gamma"], 80_000.0)
        self.assertAlmostEqual(
            approximate_gamma_pnl(exposure["pair_cash_gamma"], 0.02),
            16.0,
        )

    def test_daily_exposure_maps_close_gamma_to_next_trading_day_return(self):
        dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
        ohlc = pd.DataFrame(
            {
                "open": [4.0, 4.1, 4.0],
                "high": [4.1, 4.2, 4.1],
                "low": [3.9, 4.0, 3.9],
                "close": [4.0, 4.1, 4.0],
                "volume": [1_000, 1_000, 1_000],
            },
            index=dates,
        )
        chain = pd.DataFrame({"placeholder": [1]})
        enriched = {date: chain for date in dates}

        def atm_for_day(_chain, spot):
            call = pd.Series(
                {
                    "order_book_id": f"C{spot}",
                    "gamma": 0.20,
                    "contract_multiplier": 10_000,
                }
            )
            put = pd.Series(
                {
                    "order_book_id": f"P{spot}",
                    "gamma": 0.30,
                    "contract_multiplier": 10_000,
                }
            )
            return {
                "call": call,
                "put": put,
                "strike": spot,
                "expiry": pd.Timestamp("2026-02-25"),
                "dte": 20,
                "call_iv": 0.20,
                "put_iv": 0.21,
            }

        with patch(
            "scripts.research.gamma_exposure_report."
            "core.vol_engine.select_atm_from_chain",
            side_effect=atm_for_day,
        ):
            daily = build_daily_gamma_exposure("50etf", ohlc, enriched)

        expected_return = math.log(4.1 / 4.0)
        expected_cash_gamma = (0.20 + 0.30) * 10_000 * 4.0**2
        self.assertAlmostEqual(daily.loc[0, "next_day_log_return"], expected_return)
        self.assertAlmostEqual(daily.loc[0, "pair_cash_gamma"], expected_cash_gamma)
        self.assertAlmostEqual(
            daily.loc[0, "long_pair_gamma_pnl_next_day"],
            0.5 * expected_cash_gamma * expected_return**2,
        )
        self.assertAlmostEqual(
            daily.loc[0, "short_pair_gamma_pnl_next_day"],
            -daily.loc[0, "long_pair_gamma_pnl_next_day"],
        )
        self.assertTrue(math.isnan(daily.loc[2, "next_day_log_return"]))
        self.assertTrue(
            math.isnan(daily.loc[2, "long_pair_gamma_pnl_next_day"])
        )

    def test_summary_and_scenarios_include_rms_and_tail_moves(self):
        dates = pd.date_range("2026-01-05", periods=5, freq="B")
        daily = pd.DataFrame(
            {
                "date": dates,
                "product": ["50etf"] * 5,
                "spot": [4.0] * 5,
                "log_return": [math.nan, 0.01, -0.02, 0.03, -0.04],
                "atm_available": [True] * 5,
                "dte": [20] * 5,
                "call_gamma": [0.2] * 5,
                "put_gamma": [0.3] * 5,
                "pair_gamma": [0.5] * 5,
                "pair_cash_gamma": [80_000.0] * 5,
                "next_day_log_return": [0.01, -0.02, 0.03, -0.04, math.nan],
                "long_pair_gamma_pnl_next_day": [4.0, 16.0, 36.0, 64.0, math.nan],
            }
        )

        summary = pd.DataFrame([summarize_product(daily)])
        scenarios = build_scenario_table(summary)

        self.assertAlmostEqual(
            summary.loc[0, "log_return_rms"],
            math.sqrt((0.01**2 + 0.02**2 + 0.03**2 + 0.04**2) / 4),
        )
        self.assertEqual(set(scenarios["scenario"]), {
            "median_abs",
            "p75_abs",
            "p90_abs",
            "p95_abs",
            "p99_abs",
            "rms",
        })
        rms = scenarios.loc[scenarios["scenario"].eq("rms")].iloc[0]
        self.assertAlmostEqual(
            rms["long_pair_gamma_pnl"],
            0.5 * 80_000.0 * summary.loc[0, "log_return_rms"] ** 2,
        )
        self.assertAlmostEqual(
            rms["short_pair_gamma_pnl"],
            -rms["long_pair_gamma_pnl"],
        )


if __name__ == "__main__":
    unittest.main()
