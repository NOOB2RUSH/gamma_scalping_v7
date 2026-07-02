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

    def test_etf_trades_are_netted_in_execution_rows(self):
        payload = {
            "advice": [
                {
                    "action": "DELTA_HEDGE",
                    "priority": "action",
                    "reason": "reduce ETF hedge",
                    "trade_etf_qty": -4000,
                    "target_hedge_qty": 0,
                    "estimated_price": 4.938,
                    "underlying_order_book_id": "510300.XSHG",
                },
                {
                    "action": "OPTION_DELTA_HEDGE_COMBINATION",
                    "priority": "action",
                    "side": "short",
                    "reason": "open option hedge with ETF correction",
                    "open_legs": [
                        {
                            "order_book_id": "10011702",
                            "qty": 5,
                            "estimated_price": 0.1671,
                        }
                    ],
                    "trade_etf_qty": 3100,
                    "target_hedge_qty": 3100,
                    "estimated_price": 4.938,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ]
        }

        rows, notices = report._execution_rows(payload)

        row_values = [list(row.values()) for row in rows]
        self.assertEqual(row_values[0][0], "10011702")
        self.assertEqual(row_values[0][2], 5)
        self.assertEqual(row_values[1][0], "510300")
        self.assertEqual(row_values[1][2], 900.0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(notices, [])


if __name__ == "__main__":
    unittest.main()
