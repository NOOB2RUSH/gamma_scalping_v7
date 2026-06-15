import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from core.live import account_report


class AccountReportHistoryRestoreTest(unittest.TestCase):
    def test_restore_history_from_total_report(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            total_path = root / "20260612_150000_report.xlsx"
            summary_path = root / "summary.csv"
            position_path = root / "positions.csv"
            with pd.ExcelWriter(total_path, engine="openpyxl") as writer:
                pd.DataFrame(
                    [
                        {
                            "日期": "2026-06-11",
                            "估算权益": 1_000_100.0,
                            "当日手续费": 0.0,
                            "期权单日盈亏": 0.0,
                            "ETF单日盈亏": 100.0,
                            "总单日盈亏(手续费前)": 100.0,
                            "净单日盈亏": 100.0,
                            "账户Delta": 100.0,
                            "账户Gamma": 0.0,
                            "账户Vega": 0.0,
                            "账户Theta": 0.0,
                        }
                    ]
                ).to_excel(writer, sheet_name="账户总体情况", index=False)
                pd.DataFrame(
                    [
                        {
                            "日期": "2026-06-11",
                            "合约代码": "588000",
                            "合约名称": "588000.XSHG",
                            "交易方向": "多",
                            "总持仓张数": 100,
                            "今日变化": 0,
                            "最新价": 2.0,
                            "持仓均价": 1.0,
                            "持仓盈亏": 100.0,
                            "交易盈亏": 0.0,
                            "到期日": None,
                            "IV": None,
                        }
                    ]
                ).to_excel(writer, sheet_name="持仓记录", index=False)

            with (
                patch.object(
                    account_report.storage,
                    "account_report_summary_history_path",
                    return_value=summary_path,
                ),
                patch.object(
                    account_report.storage,
                    "account_report_position_history_path",
                    return_value=position_path,
                ),
                patch.object(
                    account_report,
                    "_add_summary_greeks_pnl",
                    side_effect=lambda summary, position, product=None: summary,
                ),
            ):
                result = account_report.restore_account_report_history_from_total(
                    "kc50etf",
                    total_path=total_path,
                )

            self.assertEqual(result["source_total"], total_path)
            summary = pd.read_csv(summary_path, encoding="utf-8-sig")
            positions = pd.read_csv(position_path, encoding="utf-8-sig")
            self.assertEqual(summary.iloc[0]["标的价格"], 2.0)
            self.assertEqual(summary.iloc[0]["对冲总盈亏"], 100.0)
            self.assertEqual(positions.iloc[0]["方向"], "hedge")
            self.assertEqual(positions.iloc[0]["浮动盈亏"], 100.0)

    def test_incomplete_history_fails_loudly(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary_path = root / "summary.csv"
            position_path = root / "positions.csv"
            summary_path.touch()
            with (
                patch.object(
                    account_report.storage,
                    "account_report_summary_history_path",
                    return_value=summary_path,
                ),
                patch.object(
                    account_report.storage,
                    "account_report_position_history_path",
                    return_value=position_path,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    account_report._ensure_account_report_history(
                        "kc50etf",
                        "default",
                    )

    def test_restore_range_respects_account_reset_date(self):
        frame = pd.DataFrame(
            [
                {"日期": "2026-06-05", "值": 5},
                {"日期": "2026-06-09", "值": 9},
                {"日期": "2026-06-12", "值": 12},
            ]
        )

        result = account_report._history_rows_between_dates(
            frame,
            from_date=pd.Timestamp("2026-06-08", tz="UTC"),
            through_date="2026-06-11",
        )

        self.assertEqual(result["值"].tolist(), [9])


if __name__ == "__main__":
    unittest.main()
