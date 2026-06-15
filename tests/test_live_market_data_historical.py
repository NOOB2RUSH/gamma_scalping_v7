import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core.live import market_data


class HistoricalAtmMarketDataTest(unittest.TestCase):
    def test_fetch_historical_atm_uses_exact_date_and_caches_result(self):
        etf = pd.DataFrame(
            {
                "\u65e5\u671f": ["2026-06-09"],
                "\u6536\u76d8": [1.755],
            }
        )
        risk = pd.DataFrame(
            {
                "CONTRACT_ID": [
                    "588000C2606M01750",
                    "588000P2606M01750",
                    "588000C2606M01800",
                    "588000P2606M01800",
                ],
                "CONTRACT_SYMBOL": [
                    "\u79d1\u521b50\u8d2d6\u67081750",
                    "\u79d1\u521b50\u6cbd6\u67081750",
                    "\u79d1\u521b50\u8d2d6\u67081800",
                    "\u79d1\u521b50\u6cbd6\u67081800",
                ],
            }
        )
        ak = SimpleNamespace(
            fund_etf_hist_em=mock.Mock(return_value=etf),
            option_risk_indicator_sse=mock.Mock(return_value=risk),
        )
        vol = SimpleNamespace(
            atm_target_dte=20,
            atm_target_dte_min=7,
            atm_target_dte_max=30,
        )
        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "historical_atm_cache.csv"
            with (
                mock.patch.dict("sys.modules", {"akshare": ak}),
                mock.patch.object(
                    market_data.storage,
                    "historical_atm_cache_path",
                    return_value=cache_path,
                ),
                mock.patch.object(
                    market_data,
                    "load_product_config",
                    return_value=SimpleNamespace(vol=vol),
                ),
            ):
                first = market_data.fetch_historical_atm_strike(
                    "kc50etf", "2026-06-09"
                )
                second = market_data.fetch_historical_atm_strike(
                    "kc50etf", "2026-06-09"
                )

        self.assertEqual(first["strike"], 1.75)
        self.assertEqual(first["source"], "akshare_historical_market_data")
        self.assertEqual(second["source"], "akshare_historical_atm_cache")
        ak.fund_etf_hist_em.assert_called_once_with(
            symbol="588000",
            period="daily",
            start_date="20260609",
            end_date="20260609",
            adjust="",
        )
        ak.option_risk_indicator_sse.assert_called_once_with(date="20260609")


if __name__ == "__main__":
    unittest.main()
