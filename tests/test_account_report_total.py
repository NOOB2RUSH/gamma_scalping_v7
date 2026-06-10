import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.live import account_report


class AccountReportTotalTest(unittest.TestCase):
    def test_total_report_only_replaces_current_date(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "total.xlsx"
            initial = {
                "账户总体情况": pd.DataFrame(
                    [{"日期": "2026-06-09", "值": 9}]
                ),
                "持仓记录": pd.DataFrame(
                    [{"日期": "2026-06-09", "值": "old position"}]
                ),
                "当日交易记录": pd.DataFrame(
                    [{"日期": "2026-06-09", "值": "old trade"}]
                ),
            }
            account_report._append_daily_frames_to_total_report(
                path,
                initial,
                "2026-06-09",
            )
            today = {
                "账户总体情况": pd.DataFrame(
                    [{"日期": "2026-06-10", "值": 10}]
                ),
                "持仓记录": pd.DataFrame(
                    [{"日期": "2026-06-10", "值": "new position"}]
                ),
                "当日交易记录": pd.DataFrame(
                    [{"日期": "2026-06-10", "值": "new trade"}]
                ),
            }
            account_report._append_daily_frames_to_total_report(
                path,
                today,
                "2026-06-10",
            )
            replacement = {
                sheet: frame.assign(值=frame["值"].astype(str) + " replaced")
                for sheet, frame in today.items()
            }
            account_report._append_daily_frames_to_total_report(
                path,
                replacement,
                "2026-06-10",
            )

            result = account_report._read_report_workbook(path)

        for frame in result.values():
            self.assertEqual(len(frame), 2)
            self.assertEqual(
                pd.to_datetime(frame["日期"]).dt.strftime("%Y-%m-%d").tolist(),
                ["2026-06-09", "2026-06-10"],
            )
            self.assertTrue(str(frame.iloc[-1]["值"]).endswith("replaced"))


if __name__ == "__main__":
    unittest.main()
