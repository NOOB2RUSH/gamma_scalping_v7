import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import core.hedge
from core.live import account_report


class AccountReportHedgePnlTest(unittest.TestCase):
    def test_option_position_uses_current_chain_marks_not_stale_account_marks(self):
        position = {
            "call_code": "call",
            "put_code": "put",
            "call_qty": 1,
            "put_qty": 1,
            "entry_call_price": 0.04,
            "entry_put_price": 0.06,
            "entry_option_value": 1000.0,
            "contract_multiplier": 10_000,
            "option_margin": 100.0,
            "last_call_price": 0.01,
            "last_put_price": 0.01,
            "last_mark_date": "2026-06-09",
        }
        call = pd.Series(
            {
                "order_book_id": "call",
                "contract_symbol": "call",
                "mid": 0.05,
                "strike_price": 1.75,
                "maturity_date": "2026-06-24",
                "dte": 10,
                "iv": 0.36,
                "delta": 0.5,
            }
        )
        put = pd.Series(
            {
                "order_book_id": "put",
                "contract_symbol": "put",
                "mid": 0.08,
                "strike_price": 1.75,
                "maturity_date": "2026-06-24",
                "dte": 10,
                "iv": 0.54,
                "delta": -0.5,
            }
        )
        greeks = {
            "delta": 0.0,
            "gamma": -1.0,
            "vega": -2.0,
            "theta": 3.0,
            "position_iv": 0.45,
            "call_iv": 0.36,
            "put_iv": 0.54,
            "call_delta": -0.5,
            "put_delta": 0.5,
            "call_gamma": -0.5,
            "put_gamma": -0.5,
            "call_vega": -1.0,
            "put_vega": -1.0,
            "call_theta": 1.5,
            "put_theta": 1.5,
        }
        account = SimpleNamespace(
            positions={"long": None, "short": position},
        )
        with (
            mock.patch.object(
                account_report.core.vol_engine,
                "resolve_position_pair",
                return_value=(call, put),
            ),
            mock.patch.object(
                account_report.core.strategy,
                "calc_position_greeks",
                return_value=greeks,
            ),
        ):
            rows, _, _, _, pnl = account_report._position_rows_from_account(
                account,
                pd.DataFrame(),
                "2026-06-10",
                "default",
            )

        self.assertEqual([row["最新价"] for row in rows], [0.05, 0.08])
        self.assertAlmostEqual(pnl, -300.0)

    def test_hedge_unrealized_pnl_uses_system_cost_not_broker_export(self):
        pnl = core.hedge.calc_unrealized_pnl(
            160_000,
            1.8205,
            1.823,
        )

        self.assertAlmostEqual(pnl, 400.0)

    def test_summary_backfill_overwrites_broker_hedge_pnl_with_system_pnl(self):
        summary = pd.DataFrame(
            [
                {
                    "日期": "2026-06-03",
                    "账户ID": "default",
                    "初始资金": 10_000_000.0,
                    "对冲持仓": 160_000.0,
                    "对冲成本": 1.8214,
                    "对冲最新价": 1.823,
                    "对冲估值价": 1.823,
                    "对冲浮盈亏": 254.36,
                    "期权浮盈亏": -300.0,
                    "手续费": 14.564,
                }
            ]
        )

        with (
            mock.patch.object(
                account_report,
                "_hedge_open_cost_for_report",
                return_value=1.8205,
            ),
            mock.patch.object(
                account_report,
                "_cumulative_hedge_realized_pnl_for_report",
                return_value=0.0,
            ),
            mock.patch.object(
                account_report,
                "_cumulative_option_realized_pnl_for_report",
                return_value=0.0,
            ),
        ):
            result = account_report._fill_summary_hedge_marks_and_pnl(
                summary,
                position_history=None,
                product="kc50etf",
            )

        row = result.iloc[0]
        self.assertAlmostEqual(row["对冲成本"], 1.8205)
        self.assertAlmostEqual(row["对冲浮盈亏"], 400.0)
        self.assertAlmostEqual(row["对冲总盈亏"], 400.0)
        self.assertAlmostEqual(row["估算权益"], 10_000_085.436)

    def test_hedge_row_uses_system_spot_without_reading_broker_holding(self):
        hedge = account_report.account_store.HedgeState(
            qty=160_000,
            entry_price=1.8205,
            underlying_order_book_id="588000.XSHG",
            latest_price=1.823,
            last_market_value=291_680.0,
            last_unrealized_pnl=254.36,
        )
        with (
            mock.patch.object(
                account_report,
                "_hedge_open_cost_for_report",
                return_value=1.8205,
            ),
        ):
            rows = account_report._hedge_rows_from_account(
                "kc50etf",
                hedge,
                "default",
                "2026-06-03",
                spot=1.815,
                prefer_spot_mark=True,
            )

        self.assertEqual(rows[0]["最新价"], 1.815)
        self.assertAlmostEqual(rows[0]["浮动盈亏"], -880.0)


if __name__ == "__main__":
    unittest.main()
