import unittest
import sys
from pathlib import Path

import pandas as pd

from core.live import account_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
from scripts.live import reconcile_intraday_greeks


class IntradayGreeksIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.path = pd.DataFrame(
            {
                "spot": [100.0, 110.0, 100.0],
                "call_iv": [0.20, 0.20, 0.20],
                "put_iv": [0.20, 0.20, 0.20],
                "call_delta": [1.0, 2.0, 3.0],
                "put_delta": [0.0, 0.0, 0.0],
                "call_gamma": [0.0, 0.0, 0.0],
                "put_gamma": [0.0, 0.0, 0.0],
                "call_vega": [0.0, 0.0, 0.0],
                "put_vega": [0.0, 0.0, 0.0],
                "call_theta": [0.0, 0.0, 0.0],
                "put_theta": [0.0, 0.0, 0.0],
            }
        )

    def test_account_report_integrates_delta_over_each_interval(self):
        parts = account_report._integrate_intraday_option_greeks(self.path)

        self.assertEqual(parts["delta_pnl"], -10.0)
        self.assertEqual(parts["option_greeks_pnl"], -10.0)

    def test_reconcile_script_uses_same_interval_delta_integration(self):
        parts = reconcile_intraday_greeks._integrate_option_greeks(self.path)

        self.assertEqual(parts["delta_pnl"], -10.0)
        self.assertEqual(parts["option_greeks_pnl"], -10.0)


if __name__ == "__main__":
    unittest.main()
