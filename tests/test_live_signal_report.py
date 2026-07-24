import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import mock

import pandas as pd

from core.live import report


def _row_values(row):
    values = list(row.values())
    return values[0], values[1], values[2], values[3]


class LiveSignalReportTest(unittest.TestCase):
    def test_plan_status_is_visible_in_markdown_and_terminal_summary(self):
        payload = {
            "plan_status": "ACTIONABLE_WITH_RESIDUAL",
            "execution_allowed": True,
            "advice": [
                {
                    "action": "RESIDUAL_RISK",
                    "priority": "notice",
                    "code": "DELTA_ROUNDING_RESIDUAL",
                    "reason": "A small delta residual remains.",
                    "blocking": False,
                }
            ],
        }

        terminal_lines = report.format_signal_summary(payload)
        with TemporaryDirectory() as tmpdir, mock.patch.object(
            report.storage,
            "output_dir",
            return_value=Path(tmpdir),
        ):
            markdown = report.write_signal_report("50etf", payload).read_text(
                encoding="utf-8"
            )

        self.assertEqual(
            terminal_lines[0],
            "plan_status=ACTIONABLE_WITH_RESIDUAL | execution_allowed=true",
        )
        self.assertIn("## Plan Status", markdown)
        self.assertIn("status: ACTIONABLE_WITH_RESIDUAL", markdown)
        self.assertIn("execution_allowed: true", markdown)

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

        self.assertEqual(lines[0], "reason: CLOSE_SHORT_STRADDLE=short_stop_loss")
        self.assertIn("CALL", "\n".join(lines))
        self.assertIn("PUT", "\n".join(lines))

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
                    "action": "FINAL_DELTA_HEDGE",
                    "priority": "action",
                    "reason": "correct ETF hedge",
                    "trade_etf_qty": 3100,
                    "target_hedge_qty": 3100,
                    "estimated_price": 4.938,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ]
        }

        rows, notices = report._execution_rows(payload)

        self.assertEqual(len(rows), 1)
        code, _direction, qty, price = _row_values(rows[0])
        self.assertEqual(code, "510300")
        self.assertEqual(qty, 900.0)
        self.assertEqual(price, 4.938)
        self.assertEqual(notices, [])

    def test_legacy_pre_roll_hedge_close_is_moved_after_option_roll(self):
        payload = {
            "product": "300etf",
            "account": {
                "positions": {
                    "short": {
                        "call_code": "OLD_CALL",
                        "put_code": "OLD_PUT",
                        "call_qty": 10,
                        "put_qty": 10,
                    }
                },
                "hedge": {
                    "underlying_order_book_id": "510300.XSHG",
                    "qty": 12_600,
                },
            },
            "account_greeks": {
                "delta": -50_095.58289447949,
                "gamma": -213_824.0,
                "vega": -971.0,
                "theta": 369.0,
            },
            "planned_account_greeks": {
                "delta": 1_170.1910488991125,
                "gamma": -265_401.0,
                "vega": -1_226.0,
                "theta": 471.0,
            },
            "current_account_delta": -37_495.58289447949,
            "planned_account_delta": 1_170.1910488991125,
            "account_delta_after_hedge": 1_170.1910488991125,
            "advice": [
                {
                    "action": "CLOSE_HEDGE_BEFORE_ROLL",
                    "priority": "action",
                    "reason": "Close the ETF hedge before rolling.",
                    "current_hedge_qty": 12_600,
                    "target_hedge_qty": 0,
                    "trade_etf_qty": -12_600,
                    "estimated_price": 4.786,
                    "underlying_order_book_id": "510300.XSHG",
                },
                {
                    "action": "ROLL_SHORT_STRADDLE",
                    "priority": "action",
                    "side": "short",
                    "reason": "held_strike_differs_from_current_atm",
                    "current_call_code": "OLD_CALL",
                    "current_put_code": "OLD_PUT",
                    "current_call_qty": 10,
                    "current_put_qty": 10,
                    "estimated_current_call_price": 0.2191,
                    "estimated_current_put_price": 0.06155,
                    "target_call_code": "NEW_CALL",
                    "target_put_code": "NEW_PUT",
                    "target_call_qty": 10,
                    "target_put_qty": 10,
                    "estimated_target_call_price": 0.1004,
                    "estimated_target_put_price": 0.14345,
                },
            ],
        }

        rows, notices = report._execution_rows(payload)
        greek_rows = report._expected_greek_target_rows(payload)
        with TemporaryDirectory() as tmpdir, mock.patch.object(
            report.storage,
            "output_dir",
            return_value=Path(tmpdir),
        ):
            markdown = report.write_signal_report("300etf", payload).read_text(
                encoding="utf-8"
            )

        self.assertEqual(
            [row["合约代码"] for row in rows],
            ["OLD_CALL", "OLD_PUT", "NEW_CALL", "NEW_PUT", "510300"],
        )
        self.assertEqual(rows[-1]["数量"], 12_600.0)
        self.assertEqual(rows[-1]["执行后预计数量"], 0)
        self.assertEqual(notices, [])
        self.assertAlmostEqual(greek_rows[0]["调整前"], -37_495.58289447949)
        self.assertAlmostEqual(greek_rows[0]["调整目标"], 1_170.1910488991125)
        self.assertGreater(markdown.index("510300"), markdown.index("NEW_PUT"))

    def test_zero_quantity_delta_hedge_is_not_shown_as_notice(self):
        payload = {
            "advice": [
                {
                    "action": "DELTA_HEDGE",
                    "priority": "action",
                    "reason": "Account delta exceeds tolerance.",
                    "trade_etf_qty": 0,
                    "target_hedge_qty": 0,
                    "estimated_price": 8.1,
                    "underlying_order_book_id": "510500.XSHG",
                }
            ]
        }

        rows, notices = report._execution_rows(payload)
        lines = report.format_signal_summary(payload)

        self.assertEqual(rows, [])
        self.assertEqual(notices, [])
        self.assertNotIn("Account delta exceeds tolerance.", "\n".join(lines))

    def test_etf_delta_hedge_shows_expected_greek_target(self):
        payload = {
            "account": {
                "hedge": {"underlying_order_book_id": "510300.XSHG", "qty": 16300},
            },
            "account_greeks": {
                "delta": -32400.10191871413,
                "gamma": -306320.48563286604,
                "vega": -812.21880266753,
                "theta": 680.582262387672,
            },
            "planned_account_greeks": {
                "delta": -32400.10191871413,
                "gamma": -306320.48563286604,
                "vega": -812.21880266753,
                "theta": 680.582262387672,
            },
            "account_delta_after_hedge": -16100.101918714128,
            "advice": [
                {
                    "action": "DELTA_HEDGE",
                    "priority": "action",
                    "reason": "Account delta exceeds tolerance.",
                    "option_delta": -32400.10191871413,
                    "current_hedge_qty": 16300.0,
                    "account_delta": -16100.101918714128,
                    "target_hedge_qty": 32400.0,
                    "trade_etf_qty": 16100.0,
                    "projected_account_delta_after_hedge": -0.10191871412802767,
                    "estimated_price": 4.893,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ],
        }

        rows = report._expected_greek_target_rows(payload)
        lines = report.format_signal_summary(payload)

        self.assertEqual([row["Greek"] for row in rows], ["Delta", "Gamma", "Vega", "Theta"])
        self.assertAlmostEqual(rows[0]["调整前"], -16100.101918714128)
        self.assertAlmostEqual(rows[0]["信号影响"], 16100.0)
        self.assertAlmostEqual(rows[0]["调整目标"], -0.10191871412802767)
        self.assertAlmostEqual(rows[1]["调整目标"], -306320.48563286604)
        self.assertTrue(any(line.startswith("预期Greeks目标 | Delta") for line in lines))

    def test_atm_straddle_rebalance_execution_rows(self):
        payload = {
            "account": {
                "positions": {
                    "short": {
                        "call_code": "10011704",
                        "put_code": "10011713",
                        "call_qty": 10,
                        "put_qty": 10,
                    }
                },
                "hedge": {"underlying_order_book_id": "510300.XSHG", "qty": 0},
            },
            "advice": [
                {
                    "action": "ATM_STRADDLE_DELTA_REBALANCE",
                    "priority": "action",
                    "side": "short",
                    "close_put_code": "10011713",
                    "close_put_qty": 1,
                    "estimated_close_put_price": 0.0839,
                    "open_call_code": "10011704",
                    "open_call_qty": 1,
                    "estimated_open_call_price": 0.1625,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ]
        }

        rows, notices = report._execution_rows(payload)

        self.assertEqual(len(rows), 2)
        self.assertEqual(_row_values(rows[0]), ("10011713", list(rows[0].values())[1], 1, 0.0839))
        self.assertEqual(_row_values(rows[1]), ("10011704", list(rows[1].values())[1], 1, 0.1625))
        self.assertEqual(rows[0]["执行后预计数量"], 9)
        self.assertEqual(rows[1]["执行后预计数量"], 11)
        self.assertEqual(notices, [])

    def test_signal_summary_shows_contract_symbol_next_to_code(self):
        with TemporaryDirectory() as tmpdir:
            option_snapshot = Path(tmpdir) / "option.parquet"
            pd.DataFrame(
                [
                    {
                        "order_book_id": "10011704",
                        "contract_symbol": "300ETF购7月5000",
                    }
                ]
            ).to_parquet(option_snapshot)
            payload = {
                "quote_snapshot": {"option_snapshot": str(option_snapshot)},
                "advice": [
                    {
                        "action": "ATM_STRADDLE_DELTA_REBALANCE",
                        "priority": "action",
                        "side": "short",
                        "open_call_code": "10011704",
                        "open_call_qty": 1,
                        "estimated_open_call_price": 0.1625,
                    },
                ],
            }

            rows, notices = report._execution_rows(payload)
            lines = report.format_signal_summary(payload)

        self.assertEqual(rows[0]["合约代码"], "10011704")
        self.assertEqual(rows[0]["contract_symbol"], "300ETF购7月5000")
        self.assertIn("10011704 (300ETF购7月5000)", "\n".join(lines))
        self.assertEqual(notices, [])

    def test_execution_rows_include_projected_quantity_after_each_order(self):
        payload = {
            "account": {
                "positions": {
                    "short": {
                        "call_code": "10011704",
                        "put_code": "10011713",
                        "call_qty": 1,
                        "put_qty": 10,
                    }
                },
                "hedge": {"underlying_order_book_id": "510300.XSHG", "qty": 7100},
            },
            "advice": [
                {
                    "action": "ATM_STRADDLE_DELTA_REBALANCE",
                    "priority": "action",
                    "side": "short",
                    "close_put_code": "10011713",
                    "close_put_qty": 1,
                    "estimated_close_put_price": 0.16635,
                    "open_call_code": "10011704",
                    "open_call_qty": 15,
                    "estimated_open_call_price": 0.0556,
                    "underlying_order_book_id": "510300.XSHG",
                },
                {
                    "action": "DELTA_HEDGE",
                    "priority": "action",
                    "trade_etf_qty": -7100,
                    "target_hedge_qty": 0,
                    "estimated_price": 4.903,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ],
        }

        rows, notices = report._execution_rows(payload)

        self.assertEqual(
            [(row["合约代码"], row["数量"], row["执行后预计数量"]) for row in rows],
            [
                ("10011713", 1, 9),
                ("10011704", 15, 16),
                ("510300", 7100.0, 0),
            ],
        )
        self.assertEqual(notices, [])

    def test_rebalance_uses_planned_greeks_without_readding_action_effect(self):
        payload = {
            "account_greeks": {
                "delta": 6804.0,
                "gamma": -276615.0,
                "vega": -1112.0,
                "theta": 579.0,
            },
            "planned_account_greeks": {
                "delta": -5333.0,
                "gamma": -236045.0,
                "vega": -945.5,
                "theta": 492.0,
            },
            "advice": [
                {
                    "action": "ATM_STRADDLE_DELTA_REBALANCE",
                    "priority": "action",
                    "side": "short",
                    "reason": "rebalance ATM short straddle delta",
                    "residual_delta_before_option_rebalance": 6804.0,
                    "estimated_delta_effect": -12137.0,
                    "estimated_gamma_effect": 40570.0,
                    "estimated_vega_effect": 166.5,
                    "estimated_theta_effect": -87.0,
                    "projected_account_delta_after_combined_hedge": 0.0,
                    "close_put_code": "10011713",
                    "close_put_qty": 1,
                    "estimated_close_put_price": 0.0839,
                    "open_call_code": "10011704",
                    "open_call_qty": 1,
                    "estimated_open_call_price": 0.1625,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ],
        }

        rows = report._expected_greek_target_rows(payload)
        lines = report.format_signal_summary(payload)

        self.assertEqual([row["Greek"] for row in rows], ["Delta", "Gamma", "Vega", "Theta"])
        gamma_values = list(rows[1].values())
        self.assertAlmostEqual(gamma_values[1], -276615.0)
        self.assertAlmostEqual(gamma_values[2], 40570.0)
        self.assertAlmostEqual(gamma_values[3], 40570.0 / -276615.0)
        self.assertAlmostEqual(gamma_values[4], -236045.0)
        self.assertEqual(
            sum(line.startswith("option hedge Greeks") for line in lines),
            0,
        )
        self.assertEqual(
            sum("Gamma" in line and "40570" in line for line in lines),
            1,
        )


if __name__ == "__main__":
    unittest.main()
