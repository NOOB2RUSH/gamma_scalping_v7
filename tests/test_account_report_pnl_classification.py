import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import account, account_report


class AccountReportPnlClassificationTest(unittest.TestCase):
    def test_reopened_etf_does_not_inherit_stale_pre_flat_position(self):
        payload = {
            "product": "500etf",
            "account_id": "default",
            "date": "2026-07-21",
            "spot": 7.792,
            "current_chain_metadata": {},
            "position_history": pd.DataFrame(
                [
                    {
                        "日期": "2026-07-14",
                        "账户ID": "default",
                        "方向": "hedge",
                        "合约代码": "510500",
                        "合约名称": "510500.XSHG",
                        "总持仓": 142_400,
                        "最新价": 8.090,
                        "持仓均价": 8.100,
                    },
                    {
                        "日期": "2026-07-21",
                        "账户ID": "default",
                        "方向": "hedge",
                        "合约代码": "510500",
                        "合约名称": "510500.XSHG",
                        "总持仓": 8_200,
                        "最新价": 7.804,
                        "持仓均价": 7.804,
                    },
                ]
            ),
            "summary_history": pd.DataFrame(
                [
                    {
                        "日期": "2026-07-15",
                        "账户ID": "default",
                        "对冲持仓": 0,
                    },
                    {
                        "日期": "2026-07-20",
                        "账户ID": "default",
                        "对冲持仓": 0,
                    },
                ]
            ),
            "trade_rows": [
                {
                    "日期": "2026-07-21",
                    "合约代码": "510500",
                    "合约名称": "中证500ETF南方",
                    "买卖": "买",
                    "成交价格": 7.804,
                    "成交数量": 8_200,
                    "成交时间": "14:55:55",
                    "类型": "ETF对冲",
                }
            ],
        }

        with mock.patch.object(
            account_report,
            "_option_contract_adjustments_by_code",
            return_value={},
        ):
            result = account_report._position_report_frame(payload)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["今日变化"], 8_200)
        self.assertEqual(row["持仓盈亏"], 0.0)
        self.assertAlmostEqual(row["交易盈亏"], -98.4)
        self.assertAlmostEqual(row["当日盈亏分解合计"], -98.4)

    def test_new_intraday_position_pnl_is_transaction_to_close(self):
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
        self.assertAlmostEqual(result["realized_cost_pnl"], -3120.0)
        self.assertAlmostEqual(result["mark_to_market_trade_pnl"], -3120.0)
        self.assertAlmostEqual(result["daily_pnl_decomposition"], -3120.0)

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

        self.assertAlmostEqual(result["holding_pnl"], -1840.0)
        self.assertAlmostEqual(result["realized_cost_pnl"], 50.0)
        self.assertAlmostEqual(result["mark_to_market_trade_pnl"], 50.0)
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

        self.assertAlmostEqual(result["holding_pnl"], -1300.0)
        self.assertAlmostEqual(result["realized_cost_pnl"], 200.0)
        self.assertAlmostEqual(result["mark_to_market_trade_pnl"], 200.0)
        self.assertAlmostEqual(result["daily_pnl_decomposition"], -1100.0)

    def test_close_snapshot_option_daily_pnl_uses_closes_and_carry_positions(self):
        code_col = "\u5408\u7ea6\u4ee3\u7801"
        qty_col = "\u603b\u6301\u4ed3"
        side_col = "\u65b9\u5411"
        last_col = "\u6700\u65b0\u4ef7"
        name_col = "\u5408\u7ea6\u540d\u79f0"
        previous = pd.DataFrame(
            [
                {
                    code_col: "100001",
                    name_col: "50ETF\u8d2d7\u67082800",
                    side_col: "short",
                    qty_col: 4,
                    last_col: 0.22925,
                },
                {
                    code_col: "100002",
                    name_col: "50ETF\u6cbd7\u67083100",
                    side_col: "short",
                    qty_col: 10,
                    last_col: 0.11295,
                },
            ]
        )
        current = pd.DataFrame(
            [
                {
                    code_col: "100001",
                    name_col: "50ETF\u8d2d7\u67082800",
                    side_col: "short",
                    qty_col: 2,
                    last_col: 0.25875,
                },
                {
                    code_col: "100002",
                    name_col: "50ETF\u6cbd7\u67083100",
                    side_col: "short",
                    qty_col: 10,
                    last_col: 0.09030,
                },
            ]
        )
        trades = [
            {
                code_col: "100001",
                "\u4e70\u5356": "\u4e70",
                "\u5f00\u5e73": "\u5e73\u4ed3",
                "\u6210\u4ea4\u4ef7\u683c": 0.25990,
                "\u6210\u4ea4\u6570\u91cf": 4,
                "\u5e73\u4ed3\u76c8\u4e8f": -1180.0,
                "\u6210\u4ea4\u65f6\u95f4": "14:51:55",
                "\u7c7b\u578b": "\u671f\u6743",
            },
            {
                code_col: "100001",
                "\u4e70\u5356": "\u5356",
                "\u5f00\u5e73": "\u5f00\u4ed3",
                "\u6210\u4ea4\u4ef7\u683c": 0.25680,
                "\u6210\u4ea4\u6570\u91cf": 2,
                "\u6210\u4ea4\u65f6\u95f4": "14:52:28",
                "\u7c7b\u578b": "\u671f\u6743",
            },
        ]

        result = account_report._close_snapshot_option_daily_pnl(
            "50etf",
            "2026-06-25",
            previous,
            current,
            trades,
        )

        expected = -1180.0
        expected += -10 * (0.09030 - 0.11295) * 10000
        self.assertAlmostEqual(result, expected)

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

    def test_option_realized_pnl_is_replayed_for_rebalance_leg_close(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            db_path = Path(temp_dir) / "account.sqlite"
            with mock.patch.object(account.storage, "account_db_path", return_value=db_path):
                account.record_fill(
                    "300etf",
                    {
                        "action": "open_short_straddle",
                        "side": "short",
                        "date": "2026-06-22",
                        "call_code": "10011704",
                        "put_code": "10011713",
                        "strike": 5.0,
                        "expiry": "2026-07-22",
                        "call_qty": 10,
                        "put_qty": 10,
                        "entry_call_price": 0.1625,
                        "entry_put_price": 0.0839,
                        "contract_multiplier": 10000,
                    },
                )
                account.record_fill(
                    "300etf",
                    {
                        "action": "rebalance_straddle_legs",
                        "side": "short",
                        "date": "2026-06-23",
                        "call_code": "10011704",
                        "put_code": "10011713",
                        "strike": 5.0,
                        "expiry": "2026-07-22",
                        "call_qty": 6,
                        "put_qty": 10,
                        "entry_call_price": 0.1622,
                        "entry_put_price": 0.0836,
                        "contract_multiplier": 10000,
                        "leg_adjustments": [
                            {
                                "leg": "call",
                                "qty_change": -4,
                                "price": 0.0843,
                                "qty": 4,
                            }
                        ],
                    },
                )

                realized = account_report._cumulative_option_realized_pnl_for_report(
                    "300etf",
                    "default",
                    "2026-06-23",
                )

        self.assertAlmostEqual(realized, 3128.0)


if __name__ == "__main__":
    unittest.main()
