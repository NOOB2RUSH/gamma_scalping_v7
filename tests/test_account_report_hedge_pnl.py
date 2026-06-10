import unittest
from unittest import mock

import pandas as pd

from core.live import account_report


class AccountReportHedgePnlTest(unittest.TestCase):
    def test_hedge_unrealized_pnl_uses_system_cost_not_broker_export(self):
        pnl = account_report._hedge_unrealized_pnl_for_report(
            "kc50etf",
            qty=160_000,
            entry_price=1.8205,
            spot=1.823,
            report_date="2026-06-03",
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
