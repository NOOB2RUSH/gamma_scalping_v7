import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import account, account_report


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

    def test_closed_previous_position_uses_fill_for_holding_pnl(self):
        result = account_report._daily_position_pnl_breakdown(
            current_qty=0,
            current_side="short",
            current_price=0.1354,
            previous_qty=10,
            previous_side="short",
            previous_price=0.1170,
            previous_cost=0.1189,
            trade_rows=[
                {"买卖": "买", "成交数量": 10, "成交价格": 0.1349}
            ],
            multiplier=10000,
        )

        self.assertAlmostEqual(result["holding_pnl"], -1790.0)
        self.assertAlmostEqual(result["realized_cost_pnl"], -1600.0)
        self.assertEqual(result["mark_to_market_trade_pnl"], 0.0)
        self.assertAlmostEqual(result["daily_pnl_decomposition"], -1790.0)

    def test_partially_closed_previous_position_uses_fill_and_latest_price(self):
        result = account_report._daily_position_pnl_breakdown(
            current_qty=6,
            current_side="short",
            current_price=0.1300,
            previous_qty=10,
            previous_side="short",
            previous_price=0.1170,
            previous_cost=0.1189,
            trade_rows=[
                {"买卖": "买", "成交数量": 4, "成交价格": 0.1250}
            ],
            multiplier=10000,
        )

        expected = (0.1170 - 0.1250) * 4 * 10000
        expected += (0.1170 - 0.1300) * 6 * 10000
        self.assertAlmostEqual(result["holding_pnl"], expected)
        self.assertEqual(result["mark_to_market_trade_pnl"], 0.0)
        self.assertAlmostEqual(result["daily_pnl_decomposition"], expected)

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

    def test_option_realized_pnl_is_replayed_when_close_fill_has_no_realized_pnl(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = Path(temp_dir) / "account.sqlite"
            with mock.patch.object(account.storage, "account_db_path", return_value=db_path):
                account.record_fill(
                    "kc50etf",
                    {
                        "action": "open_short_straddle",
                        "side": "short",
                        "date": "2026-06-09",
                        "call_code": "10010393",
                        "put_code": "10010394",
                        "strike": 1.75,
                        "expiry": "2026-06-24",
                        "call_qty": 80,
                        "put_qty": 80,
                        "entry_call_price": 0.0517,
                        "entry_put_price": 0.0717,
                        "contract_multiplier": 10000,
                        "entry_option_value": 98720.0,
                        "option_margin": 376608.0,
                    },
                )
                account.record_fill(
                    "kc50etf",
                    {
                        "action": "close_short_straddle",
                        "side": "short",
                        "date": "2026-06-17",
                        "call_code": "10010393",
                        "put_code": "10010394",
                        "call_qty": 80,
                        "put_qty": 80,
                        "call_price": 0.1853,
                        "put_price": 0.0045,
                        "contract_multiplier": 10000,
                    },
                )

                realized = account_report._cumulative_option_realized_pnl_for_report(
                    "kc50etf",
                    "default",
                    "2026-06-17",
                )

        self.assertAlmostEqual(realized, -53120.0)


if __name__ == "__main__":
    unittest.main()
