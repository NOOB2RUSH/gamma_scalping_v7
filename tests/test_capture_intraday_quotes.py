import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
import capture_intraday_quotes


class CaptureIntradayQuotesTest(unittest.TestCase):
    def test_capture_once_skips_option_greeks_snapshot_by_default(self):
        with TemporaryDirectory() as tmpdir:
            fake_ak = SimpleNamespace(
                option_sse_spot_price_sina=object(),
                option_sse_greeks_sina=object(),
            )
            args = SimpleNamespace(
                product="50etf",
                account_id="default",
                output_dir=tmpdir,
                option_code=[],
                no_account_positions=True,
                save_option_greeks_snapshot=False,
            )
            snapshot_sources = []

            def fake_snapshot(_func, _code, _captured_at, source):
                snapshot_sources.append(source)
                return pd.DataFrame({"symbol": ["10000001"]})

            with (
                mock.patch.dict(sys.modules, {"akshare": fake_ak}),
                mock.patch.object(
                    capture_intraday_quotes.market_data,
                    "fetch_quote_snapshot",
                    return_value={"ok": True},
                ),
                mock.patch.object(
                    capture_intraday_quotes.market_data,
                    "SSE_ETF_OPTION_SPECS",
                    {"50etf": SimpleNamespace(etf_symbol="510050")},
                ),
                mock.patch.object(
                    capture_intraday_quotes, "_option_codes", return_value=["10000001"]
                ),
                mock.patch.object(
                    capture_intraday_quotes,
                    "_fetch_etf_minute",
                    return_value=pd.DataFrame({"timestamp": [pd.Timestamp.now()]}),
                ),
                mock.patch.object(
                    capture_intraday_quotes,
                    "_fetch_option_minute",
                    return_value=pd.DataFrame({"timestamp": [pd.Timestamp.now()]}),
                ),
                mock.patch.object(
                    capture_intraday_quotes,
                    "_fetch_option_snapshot",
                    side_effect=fake_snapshot,
                ),
                mock.patch.object(
                    capture_intraday_quotes, "_append_dedup_csv", return_value=1
                ),
            ):
                result = capture_intraday_quotes.capture_once(args)

        self.assertEqual(snapshot_sources, ["option_sse_spot_price_sina"])
        self.assertEqual(result["option_greeks_rows"], {})

    def test_capture_once_saves_option_greeks_snapshot_when_enabled(self):
        with TemporaryDirectory() as tmpdir:
            fake_ak = SimpleNamespace(
                option_sse_spot_price_sina=object(),
                option_sse_greeks_sina=object(),
            )
            args = SimpleNamespace(
                product="50etf",
                account_id="default",
                output_dir=tmpdir,
                option_code=[],
                no_account_positions=True,
                save_option_greeks_snapshot=True,
            )
            snapshot_sources = []

            def fake_snapshot(_func, _code, _captured_at, source):
                snapshot_sources.append(source)
                return pd.DataFrame({"symbol": ["10000001"]})

            with (
                mock.patch.dict(sys.modules, {"akshare": fake_ak}),
                mock.patch.object(
                    capture_intraday_quotes.market_data,
                    "fetch_quote_snapshot",
                    return_value={"ok": True},
                ),
                mock.patch.object(
                    capture_intraday_quotes.market_data,
                    "SSE_ETF_OPTION_SPECS",
                    {"50etf": SimpleNamespace(etf_symbol="510050")},
                ),
                mock.patch.object(
                    capture_intraday_quotes, "_option_codes", return_value=["10000001"]
                ),
                mock.patch.object(
                    capture_intraday_quotes,
                    "_fetch_etf_minute",
                    return_value=pd.DataFrame({"timestamp": [pd.Timestamp.now()]}),
                ),
                mock.patch.object(
                    capture_intraday_quotes,
                    "_fetch_option_minute",
                    return_value=pd.DataFrame({"timestamp": [pd.Timestamp.now()]}),
                ),
                mock.patch.object(
                    capture_intraday_quotes,
                    "_fetch_option_snapshot",
                    side_effect=fake_snapshot,
                ),
                mock.patch.object(
                    capture_intraday_quotes, "_append_dedup_csv", return_value=1
                ),
            ):
                result = capture_intraday_quotes.capture_once(args)

        self.assertEqual(
            snapshot_sources,
            ["option_sse_spot_price_sina", "option_sse_greeks_sina"],
        )
        self.assertEqual(result["option_greeks_rows"], {"10000001": 1})

    def test_etf_minute_keeps_current_capture_date_only(self):
        raw = pd.DataFrame(
            {
                "day": [
                    "2026-07-06 14:59:00",
                    "2026-07-07 09:30:00",
                    "2026-07-07 09:31:00",
                ],
                "open": [3.0, 3.1, 3.2],
                "high": [3.1, 3.2, 3.3],
                "low": [2.9, 3.0, 3.1],
                "close": [3.0, 3.1, 3.2],
                "volume": [100, 200, 300],
                "amount": [300.0, 620.0, 960.0],
            }
        )
        ak = SimpleNamespace(
            stock_zh_a_minute=lambda symbol, period, adjust: raw,
        )

        result = capture_intraday_quotes._fetch_etf_minute(
            ak,
            "510050",
            pd.Timestamp("2026-07-07 10:00:00"),
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(
            set(pd.to_datetime(result["timestamp"]).dt.date),
            {pd.Timestamp("2026-07-07").date()},
        )

    def test_etf_minute_can_select_historical_target_date(self):
        raw = pd.DataFrame(
            {
                "day": [
                    "2026-07-06 14:59:00",
                    "2026-07-07 09:30:00",
                    "2026-07-07 09:31:00",
                ],
                "open": [3.0, 3.1, 3.2],
                "high": [3.1, 3.2, 3.3],
                "low": [2.9, 3.0, 3.1],
                "close": [3.0, 3.1, 3.2],
                "volume": [100, 200, 300],
                "amount": [300.0, 620.0, 960.0],
            }
        )
        ak = SimpleNamespace(
            stock_zh_a_minute=lambda symbol, period, adjust: raw,
            fund_etf_hist_min_em=lambda symbol, period, adjust: pd.DataFrame(),
        )

        result = capture_intraday_quotes._fetch_etf_minute(
            ak,
            "510050",
            pd.Timestamp("2026-07-10 10:00:00"),
            target_date=pd.Timestamp("2026-07-06").date(),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["timestamp"], pd.Timestamp("2026-07-06 14:59:00"))

    def test_append_dedup_csv_can_filter_existing_timestamp_date(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "etf_510050_1m.csv"
            existing = pd.DataFrame(
                {
                    "symbol": ["510050", "510050"],
                    "timestamp": ["2026-07-06 14:59:00", "2026-07-07 09:30:00"],
                    "close": [3.0, 3.1],
                }
            )
            existing.to_csv(path, index=False, encoding="utf-8-sig")
            incoming = pd.DataFrame(
                {
                    "symbol": ["510050"],
                    "timestamp": [pd.Timestamp("2026-07-07 09:31:00")],
                    "close": [3.2],
                }
            )

            rows = capture_intraday_quotes._append_dedup_csv(
                path,
                incoming,
                ["symbol", "timestamp"],
                timestamp_date=pd.Timestamp("2026-07-07").date(),
            )
            saved = pd.read_csv(path)

        self.assertEqual(rows, 2)
        self.assertEqual(
            set(pd.to_datetime(saved["timestamp"]).dt.date),
            {pd.Timestamp("2026-07-07").date()},
        )

    def test_stale_finance_data_falls_back_to_current_sse_data(self):
        finance = pd.DataFrame(
            {
                "date": ["2026-06-09"],
                "time": ["15:00:00"],
                "price": [0.05],
                "average_price": [0.05],
                "volume": [1],
            }
        )
        sse = pd.DataFrame(
            {
                "日期": ["2026-06-10"],
                "时间": ["13:30:00"],
                "价格": [0.06],
                "均价": [0.055],
                "成交": [2],
                "持仓": [100],
            }
        )
        ak = SimpleNamespace(
            option_finance_minute_sina=lambda symbol: finance,
            option_sse_minute_sina=lambda symbol: sse,
        )

        result = capture_intraday_quotes._fetch_option_minute(
            ak,
            "10010393",
            pd.Timestamp("2026-06-10 13:31:00"),
        )

        self.assertEqual(result.iloc[-1]["timestamp"], pd.Timestamp("2026-06-10 13:30:00"))
        self.assertEqual(result.iloc[-1]["source"], "option_sse_minute_sina")
        self.assertEqual(result.iloc[-1]["open_interest"], 100)

    def test_option_minute_can_select_historical_finance_date(self):
        finance = pd.DataFrame(
            {
                "date": ["2026-07-03", "2026-07-07"],
                "time": ["15:00:00", "09:30:00"],
                "price": [0.05, 0.06],
                "average_price": [0.05, 0.055],
                "volume": [1, 2],
            }
        )
        ak = SimpleNamespace(
            option_finance_minute_sina=lambda symbol: finance,
            option_sse_minute_sina=lambda symbol: pd.DataFrame(),
        )

        result = capture_intraday_quotes._fetch_option_minute(
            ak,
            "10010393",
            pd.Timestamp("2026-07-10 13:31:00"),
            target_date=pd.Timestamp("2026-07-03").date(),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["timestamp"], pd.Timestamp("2026-07-03 15:00:00"))
        self.assertEqual(result.iloc[0]["source"], "option_finance_minute_sina")

    def test_both_stale_sources_raise_error(self):
        stale = pd.DataFrame(
            {
                "date": ["2026-06-09"],
                "time": ["15:00:00"],
                "price": [0.05],
                "average_price": [0.05],
                "volume": [1],
            }
        )
        ak = SimpleNamespace(
            option_finance_minute_sina=lambda symbol: stale,
            option_sse_minute_sina=lambda symbol: stale,
        )

        with self.assertRaisesRegex(ValueError, "no current-date option minute data"):
            capture_intraday_quotes._fetch_option_minute(
                ak,
                "10010393",
                pd.Timestamp("2026-06-10 13:31:00"),
            )

    def test_refresh_intraday_status_marks_missing_option_code(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            day = root / "data" / "live" / "50etf" / "intraday" / "20260707"
            day.mkdir(parents=True)
            pd.DataFrame({"timestamp": ["2026-07-07 09:30:00"]}).to_csv(
                day / "etf_510050_1m.csv",
                index=False,
            )
            pd.DataFrame({"timestamp": ["2026-07-07 09:30:00"]}).to_csv(
                day / "option_10000001_1m.csv",
                index=False,
            )

            with mock.patch.object(capture_intraday_quotes.storage, "PROJECT_ROOT", root):
                status = capture_intraday_quotes.refresh_intraday_status(
                    "50etf",
                    option_codes=["10000001", "10000002"],
                    etf_symbol="510050",
                )

        row = status["dates"]["2026-07-07"]
        self.assertFalse(row["complete"])
        self.assertEqual(row["missing_option_codes"], ["10000002"])
        self.assertEqual(row["etf_rows"], 1)

    def test_discover_backfill_dates_reports_complete_and_missing_dates(self):
        fake_ak = SimpleNamespace(
            stock_zh_a_minute=lambda symbol, period, adjust: pd.DataFrame(
                {
                    "day": ["2026-07-06 09:30:00", "2026-07-07 09:30:00"],
                    "open": [1.0, 1.1],
                    "high": [1.0, 1.1],
                    "low": [1.0, 1.1],
                    "close": [1.0, 1.1],
                    "volume": [10, 11],
                    "amount": [10, 12],
                }
            ),
            fund_etf_hist_min_em=lambda symbol, period, adjust: pd.DataFrame(),
            option_finance_minute_sina=lambda symbol: pd.DataFrame(
                {
                    "date": ["2026-07-06"],
                    "time": ["09:30:00"],
                    "price": [0.05],
                    "average_price": [0.05],
                    "volume": [1],
                }
            ),
            option_sse_minute_sina=lambda symbol: pd.DataFrame(),
        )

        with (
            mock.patch.dict(sys.modules, {"akshare": fake_ak}),
            mock.patch.object(
                capture_intraday_quotes.market_data,
                "SSE_ETF_OPTION_SPECS",
                {"50etf": SimpleNamespace(etf_symbol="510050")},
            ),
            mock.patch.object(
                capture_intraday_quotes,
                "_account_option_codes",
                return_value=[],
            ),
        ):
            result = capture_intraday_quotes.discover_backfill_dates(
                "50etf",
                option_codes=["10000001"],
            )

        by_date = {row["date"]: row for row in result["dates"]}
        self.assertTrue(by_date["2026-07-06"]["complete"])
        self.assertFalse(by_date["2026-07-07"]["complete"])
        self.assertEqual(by_date["2026-07-07"]["missing_option_codes"], ["10000001"])


if __name__ == "__main__":
    unittest.main()
