import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import market_data


class LiveQuoteSnapshotCacheTest(unittest.TestCase):
    def test_load_latest_complete_akshare_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_day = root / "data" / "live" / "50etf" / "quotes" / "20260620"
            new_day = root / "data" / "live" / "50etf" / "quotes" / "20260622"
            old_day.mkdir(parents=True)
            new_day.mkdir(parents=True)
            self._write_snapshot(old_day, "150000", "2026-06-20")
            self._write_snapshot(new_day, "100000", "2026-06-22")
            self._write_metadata(new_day, "103000", "2026-06-22")

            with mock.patch.object(market_data.storage, "PROJECT_ROOT", root):
                snapshot = market_data.fetch_quote_snapshot(
                    "50etf",
                    source="snapshot",
                    date="latest",
                )

        self.assertEqual(snapshot["source"], "snapshot")
        self.assertEqual(snapshot["snapshot_source"], "akshare")
        self.assertEqual(snapshot["snapshot_stamp"], "20260622_100000")
        self.assertTrue(snapshot["etf_snapshot"].endswith("100000_etf.parquet"))

    def test_invalid_latest_snapshot_raises_instead_of_falling_back(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            old_day = root / "data" / "live" / "50etf" / "quotes" / "20260620"
            new_day = root / "data" / "live" / "50etf" / "quotes" / "20260622"
            old_day.mkdir(parents=True)
            new_day.mkdir(parents=True)
            self._write_snapshot(old_day, "150000", "2026-06-20")
            self._write_snapshot(new_day, "100000", "2026-06-22", valid=False)

            with mock.patch.object(market_data.storage, "PROJECT_ROOT", root):
                with self.assertRaisesRegex(ValueError, "invalid.*positive prices"):
                    market_data.fetch_quote_snapshot(
                        "50etf",
                        source="snapshot",
                        date="latest",
                    )

    def test_missing_snapshot_has_actionable_error(self):
        with TemporaryDirectory() as temp_dir:
            with mock.patch.object(
                market_data.storage,
                "PROJECT_ROOT",
                Path(temp_dir),
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Fetch and save an AKShare snapshot first",
                ):
                    market_data.fetch_quote_snapshot(
                        "300etf",
                        source="snapshot",
                        date="latest",
                    )

    def test_load_previous_snapshot_excludes_current_date(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quote_root = root / "data" / "live" / "50etf" / "quotes"
            previous_day = quote_root / "20260618"
            current_day = quote_root / "20260622"
            previous_day.mkdir(parents=True)
            current_day.mkdir(parents=True)
            self._write_snapshot(previous_day, "150000", "2026-06-18")
            self._write_snapshot(current_day, "100000", "2026-06-22")

            with mock.patch.object(market_data.storage, "PROJECT_ROOT", root):
                snapshot = market_data.load_previous_quote_snapshot(
                    "50etf",
                    "2026-06-22",
                )

        self.assertEqual(snapshot["quote_date"], "2026-06-18")

    @classmethod
    def _write_snapshot(cls, day_dir, time_text, quote_date, valid=True):
        cls._write_metadata(day_dir, time_text, quote_date)
        close = 3.0 if valid else 0.0
        option_price = 0.1 if valid else 0.0
        pd.DataFrame(
            {
                "open": [close],
                "high": [close],
                "low": [close],
                "close": [close],
                "volume": [1000.0 if valid else 0.0],
            }
        ).to_parquet(day_dir / f"{time_text}_etf.parquet")
        pd.DataFrame(
            {
                "order_book_id": ["10000001"],
                "close": [option_price],
                "bid": [option_price],
                "ask": [option_price],
            }
        ).to_parquet(day_dir / f"{time_text}_option_chain.parquet")

    @staticmethod
    def _write_metadata(day_dir, time_text, quote_date):
        payload = {
            "product": "50etf",
            "source": "akshare",
            "snapshot_stamp": f"{quote_date.replace('-', '')}_{time_text}",
            "quote_date": quote_date,
        }
        (day_dir / f"{time_text}_metadata.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
