import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core.live import signal_engine


class SignalEngineRollTest(unittest.TestCase):
    def test_atm_strike_falls_back_to_akshare_historical_market_data(self):
        with (
            mock.patch.object(
                signal_engine.core.data_loader,
                "load_etf_series",
                side_effect=ValueError("ETF data is empty"),
            ),
            mock.patch.object(
                signal_engine.market_data,
                "fetch_historical_atm_strike",
                return_value={
                    "strike": 1.75,
                    "source": "akshare_historical_market_data",
                },
            ) as fallback,
        ):
            strike, source = signal_engine._atm_strike_for_roll_check(
                "kc50etf",
                pd.DataFrame(),
                pd.Timestamp("2026-06-09"),
                pd.Timestamp("2026-06-10"),
            )

        self.assertEqual(strike, 1.75)
        self.assertEqual(source, "akshare_historical_market_data")
        fallback.assert_called_once_with("kc50etf", pd.Timestamp("2026-06-09"))

    def test_atm_strike_reports_local_and_akshare_failures(self):
        with (
            mock.patch.object(
                signal_engine.core.data_loader,
                "load_etf_series",
                side_effect=ValueError("ETF data is empty"),
            ),
            mock.patch.object(
                signal_engine.market_data,
                "fetch_historical_atm_strike",
                side_effect=ValueError("AKShare unavailable"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "AKShare historical market data"):
                signal_engine._atm_strike_for_roll_check(
                    "kc50etf",
                    pd.DataFrame(),
                    pd.Timestamp("2026-06-09"),
                    pd.Timestamp("2026-06-10"),
                )

    def test_roll_check_skips_history_when_current_strike_matches_atm(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
                roll_strike_mismatch_days=2,
            ),
            backtest=SimpleNamespace(short_qty=80, long_qty=10),
        )
        strategy_state = SimpleNamespace(roll_cooldown_left={"short": 0})
        position = {"strike": 1.75}
        feature_row = pd.Series({"atm_strike": 1.75})

        with mock.patch.object(
            signal_engine,
            "_historical_strike_mismatch",
            side_effect=AssertionError("history should not be read"),
        ):
            result = signal_engine._roll_payload(
                config,
                "kc50etf",
                "short",
                position,
                pd.DataFrame(),
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-06-10"),
                None,
                1.746,
                10,
                strategy_state,
            )

        self.assertIsNone(result)

    def test_merge_latest_features_persists_date_column(self):
        history = pd.DataFrame(
            {"atm_iv": [0.20]},
            index=pd.DatetimeIndex(["2026-06-09"]),
        )
        latest = pd.DataFrame(
            {"atm_iv": [0.21]},
            index=pd.DatetimeIndex(["2026-06-10"]),
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "feature_history.parquet"
            with (
                mock.patch.object(signal_engine.storage, "feature_history_path", return_value=path),
                mock.patch.object(signal_engine, "_refresh_signal_columns", side_effect=lambda x: x),
            ):
                signal_engine._merge_latest_features(
                    "kc50etf",
                    history,
                    latest,
                    pd.Timestamp("2026-06-10"),
                )

            persisted = pd.read_parquet(path)

        self.assertEqual(
            persisted["date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2026-06-09", "2026-06-10"],
        )


if __name__ == "__main__":
    unittest.main()
