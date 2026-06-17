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


if __name__ == "__main__":
    unittest.main()
