import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
import generate_signal


class GenerateSignalScriptTest(unittest.TestCase):
    def test_main_loads_snapshot_before_generating_signal(self):
        args = SimpleNamespace(
            product="50etf",
            account_id="default",
            source="snapshot",
            date="latest",
        )
        snapshot = {
            "quote_date": "2026-07-09",
            "snapshot_stamp": "20260709_132929",
        }
        payload = {"product": "50etf", "date": "2026-07-09"}
        report_path = Path("signal_report.txt")

        with (
            mock.patch.object(generate_signal, "parse_args", return_value=args),
            mock.patch.object(
                generate_signal.market_data,
                "fetch_quote_snapshot",
                return_value=snapshot,
            ) as fetch_snapshot,
            mock.patch.object(
                generate_signal.signal_engine,
                "generate_signal",
                return_value=payload,
            ) as build_signal,
            mock.patch.object(
                generate_signal.report,
                "write_signal_report",
                return_value=report_path,
            ),
            mock.patch.object(
                generate_signal.report,
                "format_signal_summary",
                return_value=[],
            ),
            mock.patch.object(generate_signal.storage, "write_json") as write_json,
        ):
            generate_signal.main()

        fetch_snapshot.assert_called_once_with(
            "50etf",
            source="snapshot",
            date="latest",
        )
        build_signal.assert_called_once_with(
            "50etf",
            "default",
            "2026-07-09",
            quote_snapshot=snapshot,
        )
        self.assertIs(payload["quote_snapshot"], snapshot)
        write_json.assert_called_once_with(report_path.with_suffix(".json"), payload)


if __name__ == "__main__":
    unittest.main()
