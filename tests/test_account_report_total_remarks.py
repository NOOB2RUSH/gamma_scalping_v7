import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.live import account_report


class AccountReportTotalRemarksTest(unittest.TestCase):
    def test_total_report_preserves_manual_summary_remark(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "total.xlsx"
            account_report._append_daily_frames_to_total_report(
                path,
                {"账户总体情况": pd.DataFrame([{"日期": "2026-06-09", "值": 9}])},
                "2026-06-09",
            )
            frames = account_report._read_report_workbook(path)
            frames["账户总体情况"]["备注"] = frames["账户总体情况"]["备注"].astype(
                object
            )
            frames["账户总体情况"].loc[0, "备注"] = "账户重置后首日"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for sheet_name, frame in frames.items():
                    frame.to_excel(writer, sheet_name=sheet_name, index=False)

            account_report._append_daily_frames_to_total_report(
                path,
                {"账户总体情况": pd.DataFrame([{"日期": "2026-06-09", "值": 10}])},
                "2026-06-09",
            )
            result = account_report._read_report_workbook(path)["账户总体情况"]

        self.assertEqual(result.iloc[0]["值"], 10)
        self.assertEqual(result.iloc[0]["备注"], "账户重置后首日")


if __name__ == "__main__":
    unittest.main()
