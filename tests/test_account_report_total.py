import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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

    def test_timestamped_reports_inherit_latest_total_report(self):
        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            legacy_path = out_dir / "kc50etf_account_report_total.xlsx"
            account_report._append_daily_frames_to_total_report(
                legacy_path,
                {
                    "账户总体情况": pd.DataFrame(
                        [{"日期": "2026-06-09", "值": 9}]
                    )
                },
                "2026-06-09",
            )
            payload = {
                "date": "2026-06-10",
                "summary_history": pd.DataFrame([{"日期": "2026-06-09"}]),
            }
            daily_frames = {
                "账户总体情况": pd.DataFrame(
                    [{"日期": "2026-06-10", "值": 10}]
                )
            }

            with (
                patch.object(
                    account_report.storage,
                    "local_now_stamp",
                    return_value="20260611_131816",
                ),
                patch.object(
                    account_report.storage,
                    "output_dir",
                    return_value=out_dir,
                ),
                patch.object(
                    account_report,
                    "_daily_report_frames",
                    return_value=daily_frames,
                ),
                patch.object(account_report, "_json_payload", return_value={}),
            ):
                paths = account_report.write_live_account_report(
                    "kc50etf",
                    payload,
                )

            self.assertEqual(paths["total_excel"].name, "20260611_131816_report.xlsx")
            self.assertEqual(paths["json"].name, "20260611_131816_daily.json")
            self.assertNotIn("excel", paths)
            self.assertNotIn("csv", paths)
            self.assertNotIn("diagnostics", paths)
            total = account_report._read_report_workbook(paths["total_excel"])

        self.assertEqual(
            pd.to_datetime(total["账户总体情况"]["日期"])
            .dt.strftime("%Y-%m-%d")
            .tolist(),
            ["2026-06-09", "2026-06-10"],
        )

    def test_latest_total_report_is_mode_specific(self):
        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            default_path = out_dir / "20260611_130000_report.xlsx"
            diagnose_path = out_dir / "20260611_130100_report_diagnose.xlsx"
            default_path.touch()
            diagnose_path.touch()

            self.assertEqual(
                account_report._latest_total_report_path(
                    out_dir,
                    "kc50etf",
                    mode="default",
                ),
                default_path,
            )
            self.assertEqual(
                account_report._latest_total_report_path(
                    out_dir,
                    "kc50etf",
                    mode="diagnose",
                ),
                diagnose_path,
            )

    def test_diagnose_write_only_outputs_cumulative_report_and_json(self):
        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            payload = {
                "date": "2026-06-10",
                "summary_history": pd.DataFrame([{"日期": "2026-06-10"}]),
            }
            daily_frames = {
                "账户总体情况": pd.DataFrame(
                    [{"日期": "2026-06-10", "值": 10}]
                )
            }

            with (
                patch.object(
                    account_report.storage,
                    "local_now_stamp",
                    return_value="20260611_131816",
                ),
                patch.object(
                    account_report.storage,
                    "output_dir",
                    return_value=out_dir,
                ),
                patch.object(
                    account_report,
                    "_daily_report_frames",
                    return_value=daily_frames,
                ),
                patch.object(account_report, "_json_payload", return_value={}),
            ):
                paths = account_report.write_live_account_report(
                    "kc50etf",
                    payload,
                    mode="diagnose",
                )

        self.assertEqual(
            set(paths),
            {"total_excel", "json"},
        )
        self.assertEqual(
            paths["total_excel"].name,
            "20260611_131816_report_diagnose.xlsx",
        )
        self.assertEqual(
            paths["json"].name,
            "20260611_131816_daily_diagnose.json",
        )


if __name__ == "__main__":
    unittest.main()
