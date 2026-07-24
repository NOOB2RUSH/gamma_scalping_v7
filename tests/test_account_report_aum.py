import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core.live import account_report


class AccountReportAumTest(unittest.TestCase):
    def test_closing_day_uses_opening_option_aum(self):
        summary = pd.DataFrame(
            [
                {"日期": "2026-07-14", "标的价格": 8.413},
                {"日期": "2026-07-15", "标的价格": 8.151},
            ]
        )
        positions = pd.DataFrame(
            [
                {
                    "日期": "2026-07-14",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "10011720",
                    "总持仓": 10,
                    "行权价": 8.25,
                    "到期日": "2026-07-22",
                    "合约乘数": 10000,
                }
            ]
        )
        payload = {
            "product": "500etf",
            "date": "2026-07-15",
            "spot": 8.151,
            "summary_history": summary,
            "position_history": positions,
            "trade_rows": [
                {"日期": "2026-07-15", "合约代码": "10011720"}
            ],
        }

        aum = account_report._summary_aum_by_date(payload)

        self.assertAlmostEqual(aum["2026-07-15"], 10 * 10000 * 8.413)

    def test_summary_reports_gross_daily_pnl_as_aum_ratio(self):
        summary = pd.DataFrame(
            [
                {
                    "日期": "2026-06-16",
                    "账户ID": "default",
                    "标的价格": 4.9,
                    "当日手续费": 100.0,
                    "总单日盈亏": 1_000.0,
                }
            ]
        )
        positions = pd.DataFrame(
            [
                {
                    "日期": "2026-06-16",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "call",
                    "总持仓": 7,
                    "行权价": 4.9,
                    "到期日": "2026-07-22",
                },
                {
                    "日期": "2026-06-16",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "put",
                    "总持仓": 10,
                    "行权价": 4.9,
                    "到期日": "2026-07-22",
                },
            ]
        )
        payload = {
            "product": "300etf",
            "date": "2026-06-16",
            "spot": 4.9,
            "summary_history": summary,
            "position_history": positions,
        }

        aum_by_date = account_report._summary_aum_by_date(payload)
        report = account_report._summary_report_frame(
            summary,
            aum_by_date=aum_by_date,
        )

        self.assertAlmostEqual(aum_by_date["2026-06-16"], 490_000.0)
        self.assertAlmostEqual(report.iloc[0]["单日盈亏/AUM"], 1_000.0 / 490_000.0)
        net_index = report.columns.get_loc("净单日盈亏")
        self.assertEqual(report.columns[net_index + 1], "单日盈亏/AUM")

    def test_option_pair_rows_share_aum_from_larger_leg_qty(self):
        position = {
            "call_code": "call",
            "put_code": "put",
            "call_qty": 7,
            "put_qty": 10,
            "entry_call_price": 0.10,
            "entry_put_price": 0.20,
            "entry_option_value": 27_000.0,
            "contract_multiplier": 10_000,
            "option_margin": 1_000.0,
        }
        call = pd.Series(
            {
                "order_book_id": "call",
                "contract_symbol": "call",
                "mid": 0.11,
                "strike_price": 4.9,
                "maturity_date": "2026-07-22",
                "dte": 30,
                "iv": 0.20,
                "delta": 0.5,
                "underlying_close": 4.9,
            }
        )
        put = pd.Series(
            {
                "order_book_id": "put",
                "contract_symbol": "put",
                "mid": 0.21,
                "strike_price": 4.9,
                "maturity_date": "2026-07-22",
                "dte": 30,
                "iv": 0.25,
                "delta": -0.5,
                "underlying_close": 4.9,
            }
        )
        greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "position_iv": 0.0,
            "call_iv": 0.20,
            "put_iv": 0.25,
            "call_delta": 0.0,
            "put_delta": 0.0,
            "call_gamma": 0.0,
            "put_gamma": 0.0,
            "call_vega": 0.0,
            "put_vega": 0.0,
            "call_theta": 0.0,
            "put_theta": 0.0,
        }
        account = SimpleNamespace(
            positions={"long": None, "short": position},
        )

        with (
            mock.patch.object(
                account_report.core.vol_engine,
                "resolve_position_pair",
                return_value=(call, put),
            ),
            mock.patch.object(
                account_report.core.strategy,
                "calc_position_greeks",
                return_value=greeks,
            ),
        ):
            rows, *_ = account_report._position_rows_from_account(
                account,
                pd.DataFrame(),
                "2026-06-16",
                "default",
            )

        for row in rows:
            self.assertAlmostEqual(row["AUM"], 490_000.0)
        self.assertEqual(
            account_report.DEFAULT_POSITION_REPORT_COLUMNS[5],
            "AUM",
        )

    def test_position_report_uses_payload_spot_when_chain_has_no_underlying_price(self):
        rows = pd.DataFrame(
            [
                {
                    "日期": "2026-06-16",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "call",
                    "合约名称": "call",
                    "总持仓": 7,
                    "最新价": 0.11,
                    "持仓均价": 0.10,
                    "行权价": 4.9,
                    "到期日": "2026-07-22",
                },
                {
                    "日期": "2026-06-16",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "put",
                    "合约名称": "put",
                    "总持仓": 10,
                    "最新价": 0.21,
                    "持仓均价": 0.20,
                    "行权价": 4.9,
                    "到期日": "2026-07-22",
                },
            ],
            columns=account_report.POSITION_COLUMNS,
        )
        payload = {
            "product": "300etf",
            "date": "2026-06-16",
            "spot": 4.9,
            "position_history": rows,
            "trade_rows": [],
            "current_chain_metadata": {},
        }

        report = account_report._position_report_frame(payload)

        for value in report["AUM"]:
            self.assertAlmostEqual(value, 490_000.0)


if __name__ == "__main__":
    unittest.main()
