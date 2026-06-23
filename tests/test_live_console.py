import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path

from core.live import account


class LiveConsoleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts_dir = Path(__file__).resolve().parents[1] / "scripts" / "live"
        sys.path.insert(0, str(scripts_dir))
        cls.live_console = importlib.import_module("live_console")

    @classmethod
    def tearDownClass(cls):
        sys.path.pop(0)

    def test_print_account_state_accepts_legacy_position_without_expiry(self):
        state = account.AccountState(
            product="kc50etf",
            positions={
                "long": None,
                "short": {
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "call_qty": 10,
                    "put_qty": 10,
                    "strike_price": 1.75,
                },
            },
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.live_console._print_account_state(state)

        self.assertIn(
            "position.short=CALL/PUT qty=10/10 strike=1.75 expiry=None",
            output.getvalue(),
        )

    def test_snapshot_time_text_is_human_readable(self):
        self.assertEqual(
            self.live_console._snapshot_time_text(
                {"snapshot_stamp": "20260622_102807"}
            ),
            "2026-06-22 10:28:07",
        )
        self.assertEqual(self.live_console._snapshot_time_text(None), "不可用")

    def test_auto_import_summary_shows_option_and_etf_changes_only(self):
        results = [
            {
                "kind": "options",
                "result": {
                    "dry_run": True,
                    "applied": [],
                    "skipped": [],
                    "warnings": [{"reason": "ignored"}],
                },
                "error": None,
            },
            {
                "kind": "etf",
                "result": {
                    "dry_run": True,
                    "applied": [
                        {
                            "dry_run": True,
                            "fill": {
                                "action": "delta_hedge",
                                "target_hedge_qty": 14500,
                                "trade_etf_qty": 8300,
                                "entry_price": 4.9342,
                                "latest_price": 4.981,
                                "cash_delta": -41068.4,
                                "unrealized_pnl": 714.1,
                            },
                        }
                    ],
                    "skipped": [],
                    "warnings": [],
                },
                "error": None,
            },
        ]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.live_console._print_auto_import_results(results)

        text = output.getvalue()
        self.assertIn("Option: 无变化", text)
        self.assertIn("ETF: 有变化 (预览)", text)
        self.assertIn("qty=6200->14500", text)
        self.assertIn("trade=8300", text)
        self.assertNotIn("holding_rows", text)
        self.assertNotIn("WARNING", text)

    def test_mark_updates_are_not_reported_as_position_changes(self):
        results = [
            {
                "kind": "etf",
                "result": {
                    "applied": [
                        {
                            "dry_run": True,
                            "fill": {
                                "action": "hedge_mark_update",
                                "target_hedge_qty": 21800,
                                "trade_etf_qty": 0,
                                "latest_price": 5.091,
                            },
                        }
                    ]
                },
                "error": None,
            }
        ]

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.live_console._print_auto_import_results(results)

        self.assertEqual(output.getvalue().strip(), "ETF: 无变化")


if __name__ == "__main__":
    unittest.main()
