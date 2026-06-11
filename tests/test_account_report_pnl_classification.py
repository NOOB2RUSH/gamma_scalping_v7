import unittest

import pandas as pd

from core.live import account_report


class AccountReportPnlClassificationTest(unittest.TestCase):
    def test_new_intraday_position_pnl_is_left_unexplained(self):
        result = account_report._daily_position_pnl_breakdown(
            current_qty=80,
            current_side="short",
            current_price=0.0556,
            previous_qty=0,
            previous_side=None,
            previous_price=None,
            previous_cost=None,
            trade_rows=[
                {
                    "买卖": "卖",
                    "成交数量": 80,
                    "成交价格": 0.0517,
                    "成交时间(日)": "20260609 13:21:54",
                }
            ],
            multiplier=10000,
        )

        self.assertEqual(result["holding_pnl"], 0.0)
        self.assertEqual(result["mark_to_market_trade_pnl"], 0.0)
        self.assertEqual(result["daily_pnl_decomposition"], 0.0)

    def test_previous_close_position_pnl_remains_holding_pnl(self):
        result = account_report._daily_position_pnl_breakdown(
            current_qty=80,
            current_side="short",
            current_price=0.04825,
            previous_qty=80,
            previous_side="short",
            previous_price=0.0556,
            previous_cost=0.0517,
            trade_rows=[],
            multiplier=10000,
        )

        self.assertAlmostEqual(result["holding_pnl"], 5880.0)
        self.assertEqual(result["mark_to_market_trade_pnl"], 0.0)
        self.assertAlmostEqual(result["daily_pnl_decomposition"], 5880.0)

    def test_first_history_day_uses_cumulative_pnl_since_reset(self):
        history = pd.DataFrame(
            [
                {
                    "日期": "2026-06-09",
                    "账户ID": "default",
                    "期权总盈亏": 1560.0,
                    "对冲总盈亏": -53.1,
                    "期权单日盈亏": 0.0,
                    "对冲单日盈亏": 0.0,
                },
                {
                    "日期": "2026-06-10",
                    "账户ID": "default",
                    "期权总盈亏": -2560.0,
                    "对冲总盈亏": -637.2,
                },
            ]
        )

        result = account_report._add_summary_greeks_pnl_for_account(history)

        self.assertEqual(result.iloc[0]["期权单日盈亏"], 1560.0)
        self.assertEqual(result.iloc[0]["对冲单日盈亏"], -53.1)
        self.assertEqual(result.iloc[0]["总单日盈亏"], 1506.9)


if __name__ == "__main__":
    unittest.main()
