import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from unittest import mock

from core.live import account_report, etf_importer


class LiveEtfImporterTest(unittest.TestCase):
    def test_zero_local_and_zero_snapshot_match(self):
        self.assertTrue(
            etf_importer._same_target(
                {"qty": 0.0, "underlying_order_book_id": None},
                {"qty": 0.0, "underlying_order_book_id": "510300.XSHG"},
            )
        )

    def test_only_standard_etf_filenames_are_accepted(self):
        with self.assertRaisesRegex(ValueError, "ETF import requires"):
            etf_importer._resolve_file(
                "live_hold/证券持仓.csv",
                etf_importer.HOLDING_PREFIX,
            )

    def test_older_snapshot_is_not_a_newer_mark(self):
        self.assertFalse(
            etf_importer._is_newer_mark(
                {"date": "2026-06-09", "source_timestamp": "2026-06-09T15:00:00"},
                {"last_mark_date": "2026-06-11", "last_mark_source_timestamp": None},
            )
        )

    def test_historical_standard_files_parse_kc50etf(self):
        holding = pd.DataFrame(
            [
                {
                    "投资者账号": "account",
                    "证券代码": "588000",
                    "证券名称": "科创50ETF",
                    "持有数量": 53100,
                    "成本价": 1.7569,
                    "最新价": 1.756,
                    "市值": 93243.6,
                }
            ]
        )
        trade = pd.DataFrame(
            [
                {
                    "证券代码": "588000",
                    "证券名称": "科创50ETF",
                    "成交编号": "trade-1",
                    "报单编号": "order-1",
                    "买卖": "买",
                    "成交价格": 1.756,
                    "成交数量": 53100,
                    "日期": 20260609,
                    "成交时间(日)": "20260609 14:54:47",
                }
            ]
        )

        target = etf_importer._target_from_holding("kc50etf", holding, Path("holding.csv"))
        trades = etf_importer._trade_rows_for_target("kc50etf", trade, "2026-06-09")

        self.assertEqual(target["qty"], 53100)
        self.assertEqual(target["underlying_order_book_id"], "588000.XSHG")
        self.assertEqual(trades[0]["signed_qty"], 53100)
        self.assertEqual(trades[0]["cash_delta"], -93243.6)

    def test_account_report_reads_only_standard_etf_trade_prefix(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            standard = root / "证券委托查询_实时成交(信息导出)_2026_06_09-14_54_53.csv"
            legacy = root / "证券委托查询_2026_06_09-14_54_53.csv"
            frame = pd.DataFrame(
                [
                    {
                        "证券代码": "588000",
                        "证券名称": "科创50ETF",
                        "成交编号": "trade-1",
                        "买卖": "买",
                        "成交价格": 1.756,
                        "成交数量": 53100,
                        "日期": 20260609,
                    }
                ]
            )
            frame.to_csv(standard, index=False, encoding="gb18030")
            frame.to_csv(legacy, index=False, encoding="gb18030")
            with mock.patch.object(account_report, "_live_hold_dir", return_value=root):
                rows = account_report._all_etf_trade_rows_from_exports("kc50etf")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["成交编号"], "trade-1")

    def test_account_report_filters_shared_option_trade_detail_by_product(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "成交明细(信息导出)_2026_06_15-15_00_00.csv"
            pd.DataFrame(
                [
                    {
                        "合约代码": "300-call",
                        "合约名称": "300ETF购7月4900",
                        "成交数量": 10,
                        "成交价格": 0.1,
                        "日期": 20260615,
                    },
                    {
                        "合约代码": "500-call",
                        "合约名称": "500ETF购7月8500",
                        "成交数量": 10,
                        "成交价格": 0.2,
                        "日期": 20260615,
                    },
                ]
            ).to_csv(path, index=False, encoding="gb18030")

            rows = account_report._trade_rows_from_file(path, "300etf")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["合约代码"], "300-call")


if __name__ == "__main__":
    unittest.main()
