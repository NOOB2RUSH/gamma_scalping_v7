import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import account_report


class AccountReportGreeksPnlTest(unittest.TestCase):
    def test_unapproved_dividend_fields_are_not_report_columns(self):
        forbidden = {"应收股利", "资金口径净单日盈亏"}

        self.assertTrue(forbidden.isdisjoint(account_report.SUMMARY_COLUMNS))
        self.assertTrue(
            forbidden.isdisjoint(account_report.DEFAULT_SUMMARY_REPORT_COLUMNS)
        )
        self.assertTrue(
            forbidden.isdisjoint(account_report.DIAGNOSE_SUMMARY_REPORT_COLUMNS)
        )

    def test_corporate_action_full_revaluation_matches_option_price_change(self):
        config = account_report.load_product_config("500etf")
        previous_spot = 8.413
        current_spot = 8.151
        distribution = 0.149
        new_strike = 8.104
        old_multiplier = 10000
        new_multiplier = 10180
        start_iv = 0.25
        end_iv = 0.30
        start_dte = 6.0
        end_dte = 5.0
        adjusted_start_price = account_report._black_scholes_price(
            "c",
            previous_spot - distribution,
            new_strike,
            start_dte / config.vol.annual_days,
            config.vol.risk_free_rate,
            start_iv,
        )
        current_price = account_report._black_scholes_price(
            "c",
            current_spot,
            new_strike,
            end_dte / config.vol.annual_days,
            config.vol.risk_free_rate,
            end_iv,
        )
        previous_positions = pd.DataFrame(
            [
                {
                    "日期": "2026-07-14",
                    "方向": "short",
                    "合约代码": "10011720",
                    "合约名称": "500ETF购7月8250",
                    "总持仓": 10,
                    "最新价": adjusted_start_price * new_multiplier / old_multiplier,
                    "行权价": 8.25,
                    "到期日": "2026-07-22",
                    "合约乘数": old_multiplier,
                    "IV": start_iv,
                }
            ]
        )
        adjustments = {
            "10011720": {
                "new_strike": new_strike,
                "new_contract_multiplier": new_multiplier,
            }
        }

        with (
            mock.patch.object(
                account_report,
                "_option_close_price_from_quote_snapshot",
                return_value=current_price,
            ),
            mock.patch.object(
                account_report,
                "_option_close_dte",
                side_effect=[start_dte, end_dte],
            ),
        ):
            parts = account_report._corporate_action_option_revaluation_parts(
                "500etf",
                "2026-07-15",
                previous_spot,
                current_spot,
                distribution,
                previous_positions,
                adjustments,
            )

        expected = -10 * new_multiplier * (
            current_price - adjusted_start_price
        )
        self.assertAlmostEqual(sum(parts.values()), expected, places=6)

    def test_ex_dividend_close_day_uses_total_return_and_previous_greeks(self):
        history = pd.DataFrame(
            [
                {
                    "\u65e5\u671f": "2026-07-14",
                    "\u8d26\u6237ID": "default",
                    "\u6807\u7684\u4ef7\u683c": 8.413,
                    "\u5bf9\u51b2\u6700\u65b0\u4ef7": 8.413,
                    "\u5bf9\u51b2\u6301\u4ed3": 34000,
                    "Call IV": 0.25,
                    "Put IV": 0.30,
                    "Call Delta": 1.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 2.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 0.0,
                    "Put Vega": 0.0,
                    "Call Theta": 4.0,
                    "Put Theta": 0.0,
                },
                {
                    "\u65e5\u671f": "2026-07-15",
                    "\u8d26\u6237ID": "default",
                    "\u6807\u7684\u4ef7\u683c": 8.151,
                    "\u5bf9\u51b2\u6700\u65b0\u4ef7": 8.151,
                    "\u5bf9\u51b2\u6301\u4ed3": 0,
                },
            ]
        )

        result = account_report._add_summary_greeks_pnl_for_account(
            history,
            product="500etf",
        )
        row = result.iloc[-1]
        total_return_change = 8.151 + 0.149 - 8.413

        self.assertAlmostEqual(row["ETF\u5355\u65e5\u76c8\u4e8f"], 34000 * total_return_change)
        self.assertAlmostEqual(row["\u671f\u6743\u5355\u65e5DeltaPnL"], total_return_change)
        self.assertAlmostEqual(
            row["\u671f\u6743\u5355\u65e5GammaPnL"],
            0.5 * 2.0 * total_return_change**2,
        )
        self.assertEqual(row["\u671f\u6743\u5355\u65e5ThetaPnL"], 4.0)

    def test_transaction_greeks_columns_are_internal_only(self):
        internal_columns = {
            "\u4ea4\u6613DeltaPnL",
            "\u4ea4\u6613GammaPnL",
            "\u4ea4\u6613VegaPnL",
            "\u4ea4\u6613ThetaPnL",
            "\u4ea4\u6613GreeksPnL",
            "\u6628\u4ed3GreeksPnL",
        }

        self.assertTrue(internal_columns.issubset(account_report.SUMMARY_COLUMNS))
        self.assertTrue(
            internal_columns.isdisjoint(account_report.DEFAULT_SUMMARY_REPORT_COLUMNS)
        )
        self.assertTrue(
            internal_columns.isdisjoint(account_report.DIAGNOSE_SUMMARY_REPORT_COLUMNS)
        )

    def test_live_report_uses_previous_close_for_all_greeks(self):
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

        self.assertEqual(row["期权单日DeltaPnL"], 10.0)
        self.assertEqual(row["期权单日GammaPnL"], 100.0)
        self.assertAlmostEqual(row["期权单日VegaPnL"], 30.0)
        self.assertEqual(row["期权单日ThetaPnL"], 4.0)
        self.assertEqual(row["GreeksPnL口径"], "previous_close")

    def test_theta_pnl_uses_actual_exchange_trading_days(self):
        history = pd.DataFrame(
            [
                {
                    "日期": "2026-06-18",
                    "账户ID": "default",
                    "标的价格": 100.0,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 0.0,
                    "Put Vega": 0.0,
                    "Call Theta": 4.0,
                    "Put Theta": 1.0,
                },
                {
                    "日期": "2026-06-22",
                    "账户ID": "default",
                    "标的价格": 100.0,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                },
            ]
        )

        calendar = pd.DatetimeIndex(["2026-06-18", "2026-06-22"])
        with mock.patch.object(
            account_report.market_data,
            "load_live_trading_calendar",
            return_value=calendar,
        ):
            result = account_report._add_summary_greeks_pnl_for_account(
                history,
                product="500etf",
            )

        self.assertEqual(result.iloc[-1]["期权单日ThetaPnL"], 5.0)

    def test_stale_position_greeks_are_revalued_when_calendar_changes_dte(self):
        positions = pd.DataFrame(
            [
                {
                    "日期": "2026-06-18",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "CALL",
                    "合约名称": "Call",
                    "总持仓": 10,
                    "到期日": "2026-06-23",
                    "剩余天数": 3,
                    "IV": 0.1,
                }
            ]
        )
        calendar = pd.DatetimeIndex(
            ["2026-06-18", "2026-06-22", "2026-06-23"]
        )
        etf = pd.DataFrame([{"close": 5.0}])
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "contract_multiplier": 10_000,
                }
            ]
        )
        enriched = chain.assign(
            dte=2,
            iv=0.25,
            delta=0.6,
            gamma=0.1,
            vega=0.2,
            theta=-0.01,
        )

        with (
            mock.patch.object(
                account_report.market_data,
                "load_live_trading_calendar",
                return_value=calendar,
            ),
            mock.patch.object(
                account_report.market_data,
                "load_latest_quote_snapshot",
                return_value={
                    "etf_snapshot": "etf.parquet",
                    "option_snapshot": "option.parquet",
                },
            ),
            mock.patch.object(
                account_report.pd,
                "read_parquet",
                side_effect=[etf, chain],
            ),
            mock.patch.object(
                account_report.core.vol_engine,
                "add_iv_for_day",
                return_value=enriched,
            ),
            mock.patch.object(
                account_report.core.vol_engine,
                "add_greeks_for_day",
                return_value=enriched,
            ),
        ):
            result = account_report._revalue_stale_position_greeks(
                positions,
                "300etf",
            )

        row = result.iloc[0]
        self.assertEqual(row["剩余天数"], 2)
        self.assertEqual(row["IV"], 0.25)
        self.assertEqual(row["单张Delta"], -0.6)
        self.assertEqual(row["Delta"], -60_000.0)
        self.assertEqual(row["Theta"], 1_000.0)

    def test_live_report_does_not_apply_intraday_override(self):
        history = pd.DataFrame(
            [
                {"日期": "2026-06-09", "账户ID": "default"},
                {"日期": "2026-06-10", "账户ID": "default"},
            ]
        )

        result = account_report._add_summary_greeks_pnl(history)

        self.assertNotIn("5min", str(result.iloc[-1]["GreeksPnL口径"]))

    def test_vega_pnl_uses_same_previous_contract_iv_when_position_rolls(self):
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
                    "Put IV": 0.30,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 3.0,
                    "Put Vega": 4.0,
                    "Call Theta": 0.0,
                    "Put Theta": 0.0,
                },
                {
                    "日期": "2026-06-10",
                    "账户ID": "default",
                    "标的价格": 100.0,
                    "期权总盈亏": 0.0,
                    "对冲总盈亏": 0.0,
                    "对冲最新价": 100.0,
                    "对冲持仓": 0.0,
                    "Call IV": 0.50,
                    "Put IV": 0.60,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 1.0,
                    "Put Vega": 1.0,
                    "Call Theta": 0.0,
                    "Put Theta": 0.0,
                },
            ]
        )
        position_history = pd.DataFrame(
            [
                {
                    "日期": "2026-06-09",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "OLD_CALL",
                    "合约名称": "旧购",
                    "总持仓": 10,
                    "IV": 0.20,
                    "Vega": 3.0,
                },
                {
                    "日期": "2026-06-09",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "OLD_PUT",
                    "合约名称": "旧沽",
                    "总持仓": 10,
                    "IV": 0.30,
                    "Vega": 4.0,
                },
            ]
        )
        current_position_report = pd.DataFrame(
            [
                {
                    "日期": "2026-06-10",
                    "合约代码": "OLD_CALL",
                    "合约名称": "旧购",
                    "到期日": "2026-06-24",
                    "IV": 0.25,
                },
                {
                    "日期": "2026-06-10",
                    "合约代码": "OLD_PUT",
                    "合约名称": "旧沽",
                    "到期日": "2026-06-24",
                    "IV": 0.21,
                },
            ]
        )

        result = account_report._add_summary_greeks_pnl(
            history,
            position_history=position_history,
            current_position_report=current_position_report,
        )
        row = result.iloc[-1]

        self.assertAlmostEqual(row["期权单日VegaPnL"], -21.0)
        self.assertAlmostEqual(row["单日VegaPnL"], -21.0)
        self.assertEqual(
            row["GreeksPnL说明"],
            "previous_close_same_contract_iv_for_vega",
        )

    def test_no_history_write_still_uses_existing_history_for_greeks(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.csv"
            pd.DataFrame(
                [
                    {
                        "日期": "2026-06-12",
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
                    }
                ],
                columns=account_report.SUMMARY_COLUMNS,
            ).to_csv(path, index=False, encoding="utf-8-sig")

            history = account_report._read_report_history_for_calculation(
                path,
                [
                    {
                        "日期": "2026-06-15",
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
                    }
                ],
                account_report.SUMMARY_COLUMNS,
                key_columns=["日期", "账户ID"],
            )

        result = account_report._add_summary_greeks_pnl(history)
        row = result.loc[result["日期"].eq("2026-06-15")].iloc[0]

        self.assertEqual(row["单日DeltaPnL"], 10.0)
        self.assertEqual(row["单日GammaPnL"], 100.0)
        self.assertAlmostEqual(row["单日VegaPnL"], 30.0)
        self.assertEqual(row["单日ThetaPnL"], 4.0)
        self.assertEqual(row["单日GreeksPnL"], 144.0)

    def test_hedge_daily_pnl_preserves_report_actual_pnl(self):
        history = pd.DataFrame(
            [
                {
                    "日期": "2026-06-25",
                    "账户ID": "default",
                    "标的价格": 10.0,
                    "对冲持仓": 100.0,
                    "对冲最新价": 10.0,
                    "ETF单日盈亏": 50.0,
                },
                {
                    "日期": "2026-06-26",
                    "账户ID": "default",
                    "标的价格": 11.0,
                    "对冲持仓": 180.0,
                    "对冲最新价": 11.0,
                    "ETF单日盈亏": 900.0,
                },
            ]
        )

        result = account_report._add_summary_greeks_pnl_for_account(history)
        row = result.iloc[-1]

        self.assertEqual(row["ETF单日盈亏"], 900.0)
        self.assertEqual(row["对冲单日GreeksPnL"], 100.0)
        self.assertEqual(row["总单日盈亏"], 900.0)

    def test_intraday_etf_trade_contributes_to_transaction_greeks_pnl(self):
        history = pd.DataFrame(
            [
                {
                    "日期": "2026-06-09",
                    "账户ID": "default",
                    "标的价格": 10.0,
                    "对冲最新价": 10.0,
                    "对冲持仓": 0.0,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 0.0,
                    "Put Vega": 0.0,
                    "Call Theta": 0.0,
                    "Put Theta": 0.0,
                },
                {
                    "日期": "2026-06-10",
                    "账户ID": "default",
                    "标的价格": 11.0,
                    "对冲最新价": 11.0,
                    "对冲持仓": 100.0,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 0.0,
                    "Put Vega": 0.0,
                    "Call Theta": 0.0,
                    "Put Theta": 0.0,
                },
            ]
        )
        trade_rows = [
            {
                "日期": "2026-06-10",
                "类型": "ETF对冲",
                "买卖": "买",
                "成交价格": 10.0,
                "成交数量": 100,
                "成交时间": "14:50:00",
            }
        ]

        result = account_report._add_summary_greeks_pnl_for_account(
            history,
            product="300etf",
            trade_rows=trade_rows,
        )
        row = result.iloc[-1]

        self.assertEqual(row["交易DeltaPnL"], 100.0)
        self.assertEqual(row["交易GreeksPnL"], 100.0)
        self.assertEqual(row["单日DeltaPnL"], 100.0)
        self.assertEqual(row["单日GreeksPnL"], 100.0)
        self.assertEqual(row["GreeksPnL口径"], "previous_close_plus_transaction_to_close")

    def test_intraday_option_trade_handles_current_position_series(self):
        date = "\u65e5\u671f"
        account = "\u8d26\u6237ID"
        spot = "\u6807\u7684\u4ef7\u683c"
        direction = "\u65b9\u5411"
        code = "\u5408\u7ea6\u4ee3\u7801"
        name = "\u5408\u7ea6\u540d\u79f0"
        qty = "\u603b\u6301\u4ed3"
        latest_price = "\u6700\u65b0\u4ef7"
        strike = "\u884c\u6743\u4ef7"
        expiry = "\u5230\u671f\u65e5"
        dte = "\u5269\u4f59\u5929\u6570"
        iv = "IV"

        history = pd.DataFrame(
            [
                {
                    date: "2026-06-09",
                    account: "default",
                    spot: 2.0,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 0.0,
                    "Put Vega": 0.0,
                    "Call Theta": 0.0,
                    "Put Theta": 0.0,
                },
                {
                    date: "2026-06-10",
                    account: "default",
                    spot: 2.1,
                    "Call IV": 0.20,
                    "Put IV": 0.20,
                    "Call Delta": 0.0,
                    "Put Delta": 0.0,
                    "Call Gamma": 0.0,
                    "Put Gamma": 0.0,
                    "Call Vega": 0.0,
                    "Put Vega": 0.0,
                    "Call Theta": 0.0,
                    "Put Theta": 0.0,
                },
            ]
        )
        position_history = pd.DataFrame(
            [
                {
                    date: "2026-06-10",
                    account: "default",
                    direction: "short",
                    code: "10000001",
                    name: "test call",
                    qty: 1,
                    latest_price: 0.12,
                    strike: 2.0,
                    expiry: "2026-07-22",
                    dte: 30,
                    iv: 0.20,
                }
            ]
        )
        trade_rows = [
            {
                date: "2026-06-10",
                code: "10000001",
                name: "test call",
                "\u4e70\u5356": "\u5356",
                "\u6210\u4ea4\u4ef7\u683c": 0.10,
                "\u6210\u4ea4\u6570\u91cf": 1,
                "\u6210\u4ea4\u65f6\u95f4": "14:50:00",
            }
        ]

        result = account_report._add_summary_greeks_pnl_for_account(
            history,
            product="kc50etf",
            position_history=position_history,
            trade_rows=trade_rows,
        )
        row = result.iloc[-1]

        self.assertTrue(pd.notna(row["交易GreeksPnL"]))
        self.assertEqual(row["GreeksPnL口径"], "previous_close_plus_transaction_to_close")

    def test_transaction_option_greeks_uses_two_endpoints_with_intraday_dte(self):
        code = "\u5408\u7ea6\u4ee3\u7801"
        name = "\u5408\u7ea6\u540d\u79f0"
        buy_sell = "\u4e70\u5356"
        trade_price = "\u6210\u4ea4\u4ef7\u683c"
        trade_qty = "\u6210\u4ea4\u6570\u91cf"
        trade_time = "\u6210\u4ea4\u65f6\u95f4"
        strike = "\u884c\u6743\u4ef7"
        dte = "\u5269\u4f59\u5929\u6570"

        rows_by_code = {
            "10000001": {
                "current": pd.Series(
                    {
                        code: "10000001",
                        name: "test call",
                        strike: 2.0,
                        dte: 30.0,
                        "IV": 0.20,
                    }
                )
            }
        }
        trade = {
            code: "10000001",
            name: "test call",
            buy_sell: "\u5356",
            trade_price: 0.10,
            trade_qty: 1,
            trade_time: "14:50:00",
        }
        calls = []

        def fake_greeks(product, row, price, spot, flag, signed_qty, node_dte, fallback_iv=None):
            calls.append(node_dte)
            iv = 0.20 if len(calls) == 1 else 0.21
            return {"iv": iv, "delta": 1.0, "gamma": 2.0, "vega": 3.0, "theta": 4.0}

        with (
            mock.patch.object(account_report, "_spot_from_intraday_minute", return_value=2.0),
            mock.patch.object(account_report, "_transaction_option_close_price", return_value=0.12),
            mock.patch.object(account_report, "_option_greeks_for_dte", side_effect=fake_greeks),
        ):
            parts = account_report._transaction_option_greeks_pnl(
                "kc50etf",
                "2026-06-10",
                trade,
                rows_by_code,
                2.1,
            )

        self.assertEqual(len(calls), 2)
        self.assertAlmostEqual(calls[0], 30.0 + 10.0 / 240.0)
        self.assertAlmostEqual(calls[1], 30.0)
        self.assertAlmostEqual(parts["vega_pnl"], 3.0)

    def test_remaining_trading_day_fraction_excludes_lunch_break(self):
        report_date = "2026-06-10"

        self.assertAlmostEqual(
            account_report._remaining_trading_day_fraction(
                "2026-06-10 10:30:00",
                report_date,
            ),
            0.75,
        )
        self.assertAlmostEqual(
            account_report._remaining_trading_day_fraction(
                "2026-06-10 12:00:00",
                report_date,
            ),
            0.50,
        )
        self.assertAlmostEqual(
            account_report._remaining_trading_day_fraction(
                "2026-06-10 14:50:00",
                report_date,
            ),
            10.0 / 240.0,
        )

    def test_option_close_dte_recomputes_from_expiry_for_previous_only_row(self):
        previous = pd.Series(
            {
                "日期": "2026-06-09",
                "到期日": "2026-06-17",
                "剩余天数": 6.0,
            }
        )

        with mock.patch.object(
            account_report.market_data,
            "load_live_trading_calendar",
            return_value=pd.DatetimeIndex(
                pd.to_datetime(
                    [
                        "2026-06-09",
                        "2026-06-10",
                        "2026-06-11",
                        "2026-06-12",
                        "2026-06-15",
                        "2026-06-16",
                        "2026-06-17",
                    ]
                )
            ),
        ):
            dte = account_report._option_close_dte(
                "500etf",
                "2026-06-10",
                {"previous": previous},
                previous,
            )

        self.assertEqual(dte, 5.0)

    def test_option_daily_pnl_preserves_report_actual_pnl(self):
        date = "\u65e5\u671f"
        account = "\u8d26\u6237ID"
        spot = "\u6807\u7684\u4ef7\u683c"
        direction = "\u65b9\u5411"
        code = "\u5408\u7ea6\u4ee3\u7801"
        name = "\u5408\u7ea6\u540d\u79f0"
        qty = "\u603b\u6301\u4ed3"
        latest_price = "\u6700\u65b0\u4ef7"
        option_daily = "\u671f\u6743\u5355\u65e5\u76c8\u4e8f"
        total_daily = "\u603b\u5355\u65e5\u76c8\u4e8f"

        history = pd.DataFrame(
            [
                {
                    date: "2026-06-25",
                    account: "default",
                    spot: 100.0,
                    option_daily: 0.0,
                },
                {
                    date: "2026-06-26",
                    account: "default",
                    spot: 101.0,
                    option_daily: 999.0,
                },
            ]
        )
        position_history = pd.DataFrame(
            [
                {
                    date: "2026-06-25",
                    account: "default",
                    direction: "short",
                    code: "10000001",
                    name: "old call",
                    qty: 10,
                    latest_price: 1.0,
                },
                {
                    date: "2026-06-26",
                    account: "default",
                    direction: "short",
                    code: "10000001",
                    name: "old call",
                    qty: 5,
                    latest_price: 1.2,
                },
                {
                    date: "2026-06-26",
                    account: "default",
                    direction: "short",
                    code: "10000002",
                    name: "new call",
                    qty: 4,
                    latest_price: 2.0,
                },
            ]
        )

        result = account_report._add_summary_greeks_pnl_for_account(
            history,
            product="300etf",
            position_history=position_history,
        )
        row = result.iloc[-1]

        self.assertAlmostEqual(row[option_daily], 999.0)
        self.assertAlmostEqual(row[total_daily], 999.0)

    def test_stale_intraday_minute_is_not_used_for_transaction_spot(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            minute_dir = root / "data" / "live" / "500etf" / "intraday" / "20260707"
            minute_dir.mkdir(parents=True)
            (minute_dir / "etf_510500_1m.csv").write_text(
                "\n".join(
                    [
                        "symbol,timestamp,close",
                        "510500,2026-07-07 10:43:00,8.758",
                    ]
                ),
                encoding="utf-8",
            )

            with mock.patch.object(account_report.storage, "PROJECT_ROOT", root):
                spot = account_report._spot_from_intraday_minute(
                    "500etf",
                    "2026-07-07",
                    pd.Timestamp("2026-07-07 14:51:00"),
                )

        self.assertIsNone(spot)


if __name__ == "__main__":
    unittest.main()
