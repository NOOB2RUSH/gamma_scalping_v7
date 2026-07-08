import contextlib
import importlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

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

    def test_intraday_pnl_action_prints_read_only_summary(self):
        session = {"products": ("kc50etf",), "account_id": "default"}
        payload = {"summary": {"product": "kc50etf"}}

        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(self.live_console, "_prompt_optional", return_value=None),
            mock.patch.object(
                self.live_console.intraday_pnl,
                "calculate_intraday_pnl",
                return_value=payload,
            ) as calculate,
            mock.patch.object(
                self.live_console.intraday_pnl,
                "format_intraday_pnl",
                return_value=["intraday summary"],
            ),
        ):
            self.live_console._action_intraday_pnl(session)

        calculate.assert_called_once_with(
            "kc50etf",
            account_id="default",
            date=None,
        )
        self.assertIn("== kc50etf ==", output.getvalue())
        self.assertIn("intraday summary", output.getvalue())

    def test_check_positions_action_prints_position_check_summary(self):
        session = {"products": ("500etf",), "account_id": "default"}
        payload = {"product": "500etf", "ok": False, "rows": []}

        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(self.live_console, "_prompt_optional", return_value="2026-06-29"),
            mock.patch.object(
                self.live_console.position_checker,
                "check_account_positions",
                return_value=payload,
            ) as check,
            mock.patch.object(
                self.live_console.position_checker,
                "format_position_check",
                return_value=["position check summary"],
            ),
        ):
            self.live_console._action_check_positions(session)

        check.assert_called_once_with(
            "500etf",
            account_id="default",
            date="2026-06-29",
        )
        self.assertIn("== 500etf ==", output.getvalue())
        self.assertIn("position check summary", output.getvalue())

    def test_reconcile_action_prints_fund_reconciliation_once_before_products(self):
        session = {"products": ("300etf", "50etf"), "account_id": "default"}
        payloads = [
            {"product": "300etf", "account_id": "default", "ok": True, "metrics": {}},
            {"product": "50etf", "account_id": "default", "ok": True, "metrics": {}},
        ]

        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(self.live_console, "_prompt_optional", return_value=None),
            mock.patch.object(self.live_console, "_prompt_float", side_effect=[0.0, 0.0]),
            mock.patch.object(
                self.live_console.reconciler,
                "fund_reconciliation_rows",
                return_value=[{"日期": "2026-07-01"}],
            ) as fund_rows,
            mock.patch.object(
                self.live_console.reconciler,
                "format_fund_reconciliation_terminal",
                return_value=["资金对账 once"],
            ),
            mock.patch.object(
                self.live_console.reconciler,
                "reconcile",
                side_effect=payloads,
            ),
            mock.patch.object(
                self.live_console.reconciler,
                "write_reconcile_report",
                return_value=Path("report.md"),
            ),
            mock.patch.object(self.live_console.storage, "write_json"),
            mock.patch.object(
                self.live_console.reconciler,
                "format_terminal_summary",
                side_effect=[["product 300"], ["product 50"]],
            ) as terminal_summary,
        ):
            self.live_console._action_reconcile(session)

        text = output.getvalue()
        self.assertEqual(text.count("资金对账 once"), 1)
        self.assertLess(text.index("资金对账 once"), text.index("== 300etf =="))
        fund_rows.assert_called_once_with(
            account_id="default",
            products=("300etf", "50etf"),
            start_date=None,
            end_date=None,
        )
        self.assertEqual(terminal_summary.call_count, 2)
        for call in terminal_summary.call_args_list:
            self.assertIs(call.kwargs["include_fund"], False)


if __name__ == "__main__":
    unittest.main()
