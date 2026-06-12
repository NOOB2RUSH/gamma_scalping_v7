import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
import capture_intraday_quotes


class CaptureIntradayQuotesTest(unittest.TestCase):
    def test_stale_finance_data_falls_back_to_current_sse_data(self):
        finance = pd.DataFrame(
            {
                "date": ["2026-06-09"],
                "time": ["15:00:00"],
                "price": [0.05],
                "average_price": [0.05],
                "volume": [1],
            }
        )
        sse = pd.DataFrame(
            {
                "日期": ["2026-06-10"],
                "时间": ["13:30:00"],
                "价格": [0.06],
                "均价": [0.055],
                "成交": [2],
                "持仓": [100],
            }
        )
        ak = SimpleNamespace(
            option_finance_minute_sina=lambda symbol: finance,
            option_sse_minute_sina=lambda symbol: sse,
        )

        result = capture_intraday_quotes._fetch_option_minute(
            ak,
            "10010393",
            pd.Timestamp("2026-06-10 13:31:00"),
        )

        self.assertEqual(result.iloc[-1]["timestamp"], pd.Timestamp("2026-06-10 13:30:00"))
        self.assertEqual(result.iloc[-1]["source"], "option_sse_minute_sina")
        self.assertEqual(result.iloc[-1]["open_interest"], 100)

    def test_both_stale_sources_raise_error(self):
        stale = pd.DataFrame(
            {
                "date": ["2026-06-09"],
                "time": ["15:00:00"],
                "price": [0.05],
                "average_price": [0.05],
                "volume": [1],
            }
        )
        ak = SimpleNamespace(
            option_finance_minute_sina=lambda symbol: stale,
            option_sse_minute_sina=lambda symbol: stale,
        )

        with self.assertRaisesRegex(ValueError, "no current-date option minute data"):
            capture_intraday_quotes._fetch_option_minute(
                ak,
                "10010393",
                pd.Timestamp("2026-06-10 13:31:00"),
            )


if __name__ == "__main__":
    unittest.main()
