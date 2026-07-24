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
            ),
            backtest=SimpleNamespace(short_qty=80, long_qty=10),
        )
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
            )

        self.assertIsNone(result)

    def test_roll_waits_while_atm_is_less_than_one_strike_step_away(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
            ),
            backtest=SimpleNamespace(short_qty=10, long_qty=10),
        )
        position = {"strike": 1.75}
        feature_row = pd.Series({"atm_strike": 1.79})
        chain_df = pd.DataFrame(
            [
                {"strike_price": 1.70},
                {"strike_price": 1.75},
                {"strike_price": 1.80},
                {"strike_price": 1.85},
            ]
        )

        with mock.patch.object(
            signal_engine.core.vol_engine,
            "select_atm_from_chain",
            side_effect=AssertionError("target atm should not be selected"),
        ):
            result = signal_engine._roll_payload(
                config,
                "kc50etf",
                "short",
                position,
                chain_df,
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-06-17"),
                None,
                1.80,
                10,
            )

        self.assertIsNone(result)

    def test_roll_triggers_when_atm_is_exactly_one_strike_step_away(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
            ),
            backtest=SimpleNamespace(short_qty=10, long_qty=10),
        )
        position = {
            "strike": 8.5,
            "expiry": "2026-07-22",
            "call_code": "OLD_CALL",
            "put_code": "OLD_PUT",
        }
        feature_row = pd.Series({"atm_strike": 8.75})
        chain_df = pd.DataFrame(
            [
                {"strike_price": 8.25},
                {"strike_price": 8.50},
                {"strike_price": 8.75},
                {"strike_price": 9.00},
            ]
        )
        target_atm = {
            "strike": 8.75,
            "expiry": pd.Timestamp("2026-07-22"),
            "call": {"order_book_id": "NEW_CALL", "mid": 0.2},
            "put": {"order_book_id": "NEW_PUT", "mid": 0.15},
        }

        with mock.patch.object(
            signal_engine.core.vol_engine,
            "select_atm_from_chain_for_expiry",
            return_value=target_atm,
        ):
            result = signal_engine._roll_payload(
                config,
                "500etf",
                "short",
                position,
                chain_df,
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-07-09"),
                target_atm,
                8.797,
                9,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "held_strike_differs_from_current_atm")
        self.assertEqual(result["target_strike"], 8.75)
        self.assertEqual(result["target_call_code"], "NEW_CALL")
        self.assertEqual(result["target_put_code"], "NEW_PUT")

    def test_roll_waits_when_uneven_grid_atm_is_less_than_one_step_away(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
            ),
            backtest=SimpleNamespace(short_qty=10, long_qty=10),
        )
        position = {"strike": 3.1}
        feature_row = pd.Series({"atm_strike": 3.05})
        chain_df = pd.DataFrame(
            [
                {"strike_price": 2.95},
                {"strike_price": 3.00},
                {"strike_price": 3.10},
            ]
        )

        with mock.patch.object(
            signal_engine.core.vol_engine,
            "select_atm_from_chain",
            side_effect=AssertionError("target atm should not be selected"),
        ):
            result = signal_engine._roll_payload(
                config,
                "50etf",
                "short",
                position,
                chain_df,
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-07-02"),
                None,
                3.025,
                14,
            )

        self.assertIsNone(result)

    def test_roll_uses_current_atm_when_feature_atm_strike_is_missing(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
            ),
            backtest=SimpleNamespace(short_qty=10, long_qty=10),
        )
        position = {"strike": 1.75}
        feature_row = pd.Series({"atm_strike": pd.NA})
        chain_df = pd.DataFrame(
            [
                {"strike_price": 1.70},
                {"strike_price": 1.75},
                {"strike_price": 1.80},
                {"strike_price": 1.85},
            ]
        )

        with mock.patch.object(
            signal_engine.core.vol_engine,
            "select_atm_from_chain",
            side_effect=AssertionError("target atm should not be selected"),
        ):
            result = signal_engine._roll_payload(
                config,
                "kc50etf",
                "short",
                position,
                chain_df,
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-06-17"),
                {"strike": 1.79},
                1.80,
                10,
            )

        self.assertIsNone(result)

    def test_roll_ignores_adjusted_contract_strikes_for_step_size(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
            ),
            backtest=SimpleNamespace(short_qty=10, long_qty=10),
        )
        position = {"strike": 4.9}
        feature_row = pd.Series({"atm_strike": 4.95})
        chain_df = pd.DataFrame(
            [
                {"strike_price": 4.873, "contract_symbol": "300ETF购6月4873A"},
                {"strike_price": 4.873, "contract_symbol": "300ETF沽6月4873A"},
                {"strike_price": 4.9, "contract_symbol": "300ETF购7月4900"},
                {"strike_price": 4.9, "contract_symbol": "300ETF沽7月4900"},
                {"strike_price": 5.0, "contract_symbol": "300ETF购7月5000"},
                {"strike_price": 5.0, "contract_symbol": "300ETF沽7月5000"},
            ]
        )

        with mock.patch.object(
            signal_engine.core.vol_engine,
            "select_atm_from_chain",
            side_effect=AssertionError("target atm should not be selected"),
        ):
            result = signal_engine._roll_payload(
                config,
                "300etf",
                "short",
                position,
                chain_df,
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-06-18"),
                None,
                4.974,
                24,
            )

        self.assertIsNone(result)

    def test_roll_does_not_require_fresh_short_open_signal(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                roll_dte_threshold=7,
            ),
            backtest=SimpleNamespace(short_qty=10, long_qty=10),
        )
        position = {
            "strike": 1.75,
            "expiry": "2026-06-24",
            "call_code": "10010393",
            "put_code": "10010394",
        }
        feature_row = pd.Series(
            {
                "atm_strike": 1.90,
                "short_open_signal": False,
                "short_open_regime": None,
            }
        )
        target_atm = {
            "strike": 1.90,
            "expiry": pd.Timestamp("2026-06-24"),
            "call": {"order_book_id": "10011739", "mid": 0.09},
            "put": {"order_book_id": "10011748", "mid": 0.08},
        }
        chain_df = pd.DataFrame(
            [
                {"strike_price": 1.75},
                {"strike_price": 1.80},
                {"strike_price": 1.85},
                {"strike_price": 1.90},
            ]
        )

        with (
            mock.patch.object(
                signal_engine,
                "_historical_strike_mismatch",
                side_effect=AssertionError("history should not be read"),
            ) as history,
            mock.patch.object(
                signal_engine.core.vol_engine,
                "select_atm_from_chain_for_expiry",
                return_value=target_atm,
            ),
        ):
            result = signal_engine._roll_payload(
                config,
                "kc50etf",
                "short",
                position,
                chain_df,
                feature_row,
                pd.DataFrame(),
                pd.Timestamp("2026-06-17"),
                None,
                1.922,
                10,
            )

        self.assertIsNotNone(result)
        self.assertEqual(
            result["reason"],
            "held_strike_differs_from_current_atm",
        )
        history.assert_not_called()
        self.assertEqual(result["strike_mismatch_days"], 1)
        self.assertEqual(result["strike_mismatch_days_source"], "current_signal_row")
        self.assertEqual(result["target_call_code"], "10011739")
        self.assertEqual(result["target_put_code"], "10011748")
        self.assertEqual(result["target_call_qty"], 10)
        self.assertEqual(result["target_put_qty"], 10)

    def test_roll_cash_projection_includes_target_expiry(self):
        config = SimpleNamespace(
            backtest=SimpleNamespace(option_fee_per_contract=2.0),
            vol=SimpleNamespace(contract_multiplier=10000),
        )
        live_account = SimpleNamespace(
            cash=1_000_000.0,
            positions={
                "short": {
                    "contract_multiplier": 10000,
                    "option_margin": 20_000.0,
                }
            },
        )
        chain_df = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL_NEW",
                    "strike_price": 1.90,
                    "maturity_date": pd.Timestamp("2026-07-22"),
                    "mid": 0.09,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT_NEW",
                    "strike_price": 1.90,
                    "maturity_date": pd.Timestamp("2026-07-22"),
                    "mid": 0.08,
                    "contract_multiplier": 10000,
                },
            ]
        )
        option_actions = [
            {
                "action": "ROLL_SHORT_STRADDLE",
                "side": "short",
                "current_call_qty": 80,
                "current_put_qty": 80,
                "estimated_current_call_price": 0.20,
                "estimated_current_put_price": 0.01,
                "target_call_code": "CALL_NEW",
                "target_put_code": "PUT_NEW",
                "target_call_qty": 10,
                "target_put_qty": 10,
                "target_expiry": "2026-07-22",
            }
        ]

        projected = signal_engine._projected_cash_after_option_actions(
            config,
            live_account,
            option_actions,
            chain_df,
            1.922,
        )

        self.assertIsNotNone(projected)

    def test_missing_roll_target_closes_position_instead_of_falling_back_to_delta_hedge(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(short_stop_loss_enabled=False),
        )
        position = {
            "call_code": "CALL",
            "put_code": "PUT",
            "call_qty": 10,
            "put_qty": 10,
            "contract_multiplier": 10_000,
            "option_margin": 20_000.0,
        }
        chain_df = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "mid": 0.10,
                    "dte": 7,
                    "delta": 0.5,
                    "gamma": 0.1,
                    "vega": 0.1,
                    "theta": -0.1,
                },
                {
                    "order_book_id": "PUT",
                    "mid": 0.10,
                    "dte": 7,
                    "delta": -0.5,
                    "gamma": 0.1,
                    "vega": 0.1,
                    "theta": -0.1,
                },
            ]
        )
        roll_failure = {
            "close_current_position": True,
            "reason": "Roll is required, but no eligible replacement ATM straddle is available. Close the current position; do not substitute a delta hedge for this roll.",
            "roll_failure_reason": "roll_condition_active_but_no_valid_target_atm",
        }

        with (
            mock.patch.object(
                signal_engine.core.strategy,
                "get_short_close_reason",
                return_value=None,
            ),
            mock.patch.object(
                signal_engine.core.position,
                "has_short_volume_spike",
                return_value=False,
            ),
            mock.patch.object(
                signal_engine.core.strategy,
                "calc_position_greeks",
                return_value={},
            ),
            mock.patch.object(signal_engine, "_roll_payload", return_value=roll_failure),
        ):
            advice, _, _ = signal_engine._advice_for_existing_position(
                "500etf",
                "short",
                position,
                chain_df,
                pd.Series(),
                pd.DataFrame(),
                pd.Timestamp("2026-07-13"),
                None,
                8.316,
                None,
                config,
            )

        self.assertEqual(advice[0]["action"], "CLOSE_SHORT_STRADDLE")
        self.assertEqual(advice[0]["priority"], "action")
        self.assertEqual(
            advice[0]["roll_failure_reason"],
            "roll_condition_active_but_no_valid_target_atm",
        )

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

    def test_incremental_etf_start_uses_hv_window_not_backtest_start(self):
        calendar = pd.date_range("2026-01-01", periods=100, freq="B")
        config = SimpleNamespace(vol=SimpleNamespace(hv_windows=(60,)))

        start = signal_engine._incremental_etf_start_date(
            calendar,
            calendar[-1],
            config,
        )

        self.assertEqual(start, calendar[-62])


if __name__ == "__main__":
    unittest.main()
