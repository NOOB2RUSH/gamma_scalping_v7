import unittest

import pandas as pd

from core.live import account_report


class AccountReportGreeksPnlTest(unittest.TestCase):
    def test_live_report_uses_endpoint_average_for_all_greeks(self):
        history = pd.DataFrame(
            [
                {
                    "日期": "2026-06-09",
                    "账户ID": "default",
                    "标的价格": 100.0,
                    "期权总盈亏": 0.0,
                    "对冲总盈亏": 0.0,
                    "对冲最新价": 100.0,
                    "对冲持仓": 0.0,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                    "Call Delta": 1.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 2.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 3.0,
                    "Put Vega": 0.0,
                    "Call Theta": 4.0,
                    "Put Theta": 0.0,
                },
                {
                    "日期": "2026-06-10",
                    "账户ID": "default",
                    "标的价格": 110.0,
                    "期权总盈亏": 0.0,
                    "对冲总盈亏": 0.0,
                    "对冲最新价": 110.0,
                    "对冲持仓": 0.0,
                    "Call IV": 0.30,
                    "Put IV": 0.20,
                    "Call Delta": 3.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 4.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 5.0,
                    "Put Vega": 0.0,
                    "Call Theta": 6.0,
                    "Put Theta": 0.0,
                },
            ]
        )

        result = account_report._add_summary_greeks_pnl_for_account(history)
        row = result.iloc[-1]

        self.assertEqual(row["期权单日DeltaPnL"], 20.0)
        self.assertEqual(row["期权单日GammaPnL"], 150.0)
        self.assertAlmostEqual(row["期权单日VegaPnL"], 40.0)
        self.assertEqual(row["期权单日ThetaPnL"], 5.0)
        self.assertEqual(row["GreeksPnL口径"], "endpoint_average")

    def test_live_report_does_not_apply_intraday_override(self):
        history = pd.DataFrame(
            [
                {"日期": "2026-06-09", "账户ID": "default"},
                {"日期": "2026-06-10", "账户ID": "default"},
            ]
        )

        result = account_report._add_summary_greeks_pnl(history)

        self.assertNotIn("5min", str(result.iloc[-1]["GreeksPnL口径"]))


if __name__ == "__main__":
    unittest.main()
