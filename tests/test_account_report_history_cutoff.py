import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.live import account_report


class AccountReportHistoryCutoffTest(unittest.TestCase):
    def test_total_report_discards_rows_before_history_start(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "total.xlsx"
            account_report._append_daily_frames_to_total_report(
                path,
                {
                    "账户总体情况": pd.DataFrame(
                        [
                            {"日期": "2026-06-05", "值": 5},
                            {"日期": "2026-06-09", "值": 9},
                        ]
                    )
                },
                "2026-06-09",
            )
            account_report._append_daily_frames_to_total_report(
                path,
                {"账户总体情况": pd.DataFrame([{"日期": "2026-06-10", "值": 10}])},
                "2026-06-10",
                start_date="2026-06-09",
            )

            result = account_report._read_report_workbook(path)["账户总体情况"]

        self.assertEqual(
            pd.to_datetime(result["日期"]).dt.strftime("%Y-%m-%d").tolist(),
            ["2026-06-09", "2026-06-10"],
        )


if __name__ == "__main__":
    unittest.main()
