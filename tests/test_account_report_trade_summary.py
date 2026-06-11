import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from core.live import account_report


class AccountReportTradeSummaryTest(unittest.TestCase):
    def test_summary_trade_uses_manually_supplied_execution_time(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "成交汇总(信息导出)_2026_06_09-13_21_54.csv"
            pd.DataFrame(
                [
                    {
                        "投资者账号": "option-account",
                        "合约代码": "10010393",
                        "合约名称": "call",
                        "交易所": "SSE",
                        "卖开": 80,
                        "卖开均价": 0.0517,
                        "成交时间": "13:21:54",
                        "成交时间(日)": "20260609 13:21:54",
                    }
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            rows = account_report._trade_rows_from_summary_file(path, "kc50etf")

        self.assertEqual(rows[0]["成交时间"], "13:21:54")
        self.assertEqual(rows[0]["成交时间(日)"], "20260609 13:21:54")


if __name__ == "__main__":
    unittest.main()
