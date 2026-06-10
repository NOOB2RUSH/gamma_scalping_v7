import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
import promote_quote_snapshots


class PromoteQuoteSnapshotsTest(unittest.TestCase):
    def test_latest_pair_ignores_snapshot_with_wrong_quote_date(self):
        with TemporaryDirectory() as temp_dir:
            day_dir = Path(temp_dir)
            self._write_pair(day_dir, "150000", "2026-06-09")
            self._write_pair(day_dir, "160000", "2026-06-05")

            etf, option = promote_quote_snapshots._latest_snapshot_pair(
                day_dir,
                "2026-06-09",
            )

        self.assertEqual(etf.name, "150000_etf.parquet")
        self.assertEqual(option.name, "150000_option_chain.parquet")

    @staticmethod
    def _write_pair(day_dir, stamp, quote_date):
        (day_dir / f"{stamp}_metadata.json").write_text(
            json.dumps({"quote_date": quote_date}),
            encoding="utf-8",
        )
        (day_dir / f"{stamp}_etf.parquet").touch()
        (day_dir / f"{stamp}_option_chain.parquet").touch()


if __name__ == "__main__":
    unittest.main()
