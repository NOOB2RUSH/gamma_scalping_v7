import unittest

from core.live import account_report


class AccountReportTradeColumnsTest(unittest.TestCase):
    def test_trade_report_hides_redundant_and_internal_columns(self):
        self.assertIn("成交时间", account_report.TRADE_COLUMNS)
        self.assertNotIn("成交时间(日)", account_report.TRADE_COLUMNS)
        self.assertNotIn("策略名称", account_report.TRADE_COLUMNS)

    def test_full_trade_time_remains_available_for_internal_sorting(self):
        row = {
            "成交时间": "13:21:54",
            "成交时间(日)": "20260609 13:21:54",
        }

        self.assertEqual(account_report._trade_sort_key(row), "20260609 13:21:54")


if __name__ == "__main__":
    unittest.main()
