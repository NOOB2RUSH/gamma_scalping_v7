import unittest

from core.live import report


class LiveSignalReportTest(unittest.TestCase):
    def test_executable_advice_reason_is_shown_before_execution_table(self):
        payload = {
            "advice": [
                {
                    "action": "CLOSE_SHORT_STRADDLE",
                    "side": "short",
                    "reason": "short_stop_loss",
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "call_qty": 10,
                    "put_qty": 10,
                    "estimated_call_price": 0.2,
                    "estimated_put_price": 0.05,
                }
            ]
        }

        lines = report.format_signal_summary(payload)

        self.assertEqual(
            lines[0],
            "reason: CLOSE_SHORT_STRADDLE=short_stop_loss",
        )
        self.assertTrue(lines[1].startswith("执行顺序"))


if __name__ == "__main__":
    unittest.main()
