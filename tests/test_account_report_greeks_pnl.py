import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import account_report


class AccountReportGreeksPnlTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
