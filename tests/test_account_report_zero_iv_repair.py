import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core.live import account_report


class AccountReportZeroIvRepairTest(unittest.TestCase):
    def test_akshare_minute_fallback_walks_back_from_close_until_positive_iv(self):
        option_minute = pd.DataFrame(
            {
                "date": ["2026-06-10", "2026-06-10", "2026-06-10"],
                "time": ["14:59:00", "15:00:00", "15:01:00"],
                "price": [0.02, 0.01, 0.03],
            }
        )
        etf_minute = pd.DataFrame(
            {
                "day": [
                    "2026-06-10 14:59:00",
                    "2026-06-10 15:00:00",
                    "2026-06-10 15:01:00",
                ],
                "close": [5.01, 5.00, 5.02],
            }
        )
        ak = SimpleNamespace(
            option_finance_minute_sina=lambda symbol: option_minute,
            option_sse_minute_sina=lambda symbol: option_minute,
            stock_zh_a_minute=lambda symbol, period, adjust: etf_minute,
            fund_etf_hist_min_em=lambda symbol, period, adjust: etf_minute,
        )

        def add_iv_for_day(chain, spot, trading_calendar=None):
            result = chain.copy()
            result["dte"] = 10
            result["iv"] = 0.0 if float(result.iloc[0]["bid"]) == 0.01 else 0.25
            return result

        def add_greeks_for_day(chain, spot):
            result = chain.copy()
            result["delta"] = 0.5
            result["gamma"] = 0.1
            result["vega"] = 0.2
            result["theta"] = -0.01
            return result

        with (
            TemporaryDirectory() as temp_dir,
            mock.patch.object(account_report.storage, "PROJECT_ROOT", Path(temp_dir)),
            mock.patch.dict("sys.modules", {"akshare": ak}),
            mock.patch.object(
                account_report.core.vol_engine,
                "add_iv_for_day",
                side_effect=add_iv_for_day,
            ),
            mock.patch.object(
                account_report.core.vol_engine,
                "add_greeks_for_day",
                side_effect=add_greeks_for_day,
            ),
        ):
            quote = account_report._latest_positive_intraday_option_quote(
                "500etf",
                "510500",
                "2026-06-10",
                "10000001",
                "call",
                5.0,
                "2026-06-24",
                pd.DatetimeIndex(["2026-06-10"]),
            )

        self.assertIsNotNone(quote)
        self.assertEqual(quote["iv"], 0.25)
        self.assertEqual(quote["intraday_timestamp"], pd.Timestamp("2026-06-10 14:59:00"))
        self.assertEqual(quote["intraday_option_price"], 0.02)
        self.assertEqual(quote["intraday_spot"], 5.01)

    def test_zero_iv_position_rows_are_revalued_before_greeks_pnl(self):
        position_rows = pd.DataFrame(
            [
                {
                    "日期": "2026-06-10",
                    "账户ID": "default",
                    "方向": "short",
                    "合约代码": "10000001",
                    "合约名称": "500ETF购6月5000",
                    "总持仓": 2,
                    "最新价": 0.01,
                    "行权价": 5.0,
                    "到期日": "2026-06-24",
                    "剩余天数": 10,
                    "IV": 0.0,
                    "单张Delta": 0.0,
                    "Delta": 0.0,
                    "Gamma": 0.0,
                    "Vega": 0.0,
                    "Theta": 0.0,
                }
            ],
            columns=account_report.POSITION_COLUMNS,
        )
        quote = {
            "dte": 9,
            "iv": 0.25,
            "delta": 0.5,
            "gamma": 0.1,
            "vega": 0.2,
            "theta": -0.01,
        }

        with (
            mock.patch.object(
                account_report.market_data,
                "load_live_trading_calendar",
                return_value=pd.DatetimeIndex(["2026-06-10"]),
            ),
            mock.patch.object(
                account_report,
                "_latest_positive_intraday_option_quote",
                return_value=quote,
            ),
        ):
            result = account_report._repair_zero_iv_position_rows_with_intraday_minutes(
                position_rows,
                "500etf",
            )

        row = result.iloc[0]
        self.assertEqual(row["IV"], 0.25)
        self.assertEqual(row["剩余天数"], 9)
        self.assertEqual(row["单张Delta"], -0.5)
        self.assertEqual(row["Delta"], -10000.0)
        self.assertEqual(row["Gamma"], -2000.0)
        self.assertEqual(row["Vega"], -4000.0)
        self.assertEqual(row["Theta"], 200.0)

    def test_segmented_node_uses_position_iv_when_implied_iv_is_zero(self):
        c_strike = "\u884c\u6743\u4ef7"
        c_dte = "\u5269\u4f59\u5929\u6570"
        row = pd.Series(
            {
                c_strike: 5.0,
                c_dte: 10,
                "IV": 0.25,
            }
        )
        funcs = {
            "implied_volatility": lambda **kwargs: pd.Series([0.0]),
            "delta": lambda **kwargs: pd.Series([0.5]),
            "gamma": lambda **kwargs: pd.Series([0.1]),
            "vega": lambda **kwargs: pd.Series([0.2]),
            "theta": lambda **kwargs: pd.Series([-0.01]),
        }

        with mock.patch.object(
            account_report.core.vol_engine,
            "_load_vollib_funcs",
            return_value=funcs,
        ):
            result = account_report._single_node_option_greeks(
                "500etf",
                {"previous": row},
                row,
                price=0.01,
                spot=5.1,
                flag="c",
                signed_qty=-2,
                node_index=0,
                node_count=2,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["iv"], 0.25)
        self.assertEqual(result["delta"], -10000.0)
        self.assertEqual(result["gamma"], -2000.0)
        self.assertEqual(result["vega"], -4000.0)
        self.assertAlmostEqual(result["theta"], 289.6825396825397)


if __name__ == "__main__":
    unittest.main()
