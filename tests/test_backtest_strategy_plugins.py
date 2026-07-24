import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

from core import backtester, config, strategy
from core.backtest_strategies import (
    available_strategy_ids,
    create_strategy,
    deprecated_strategy_ids,
    resolve_strategy_config,
)


class BacktestStrategyPluginTest(unittest.TestCase):
    def setUp(self):
        self.runtime_config = config.load_config("50etf")
        self.plugin = create_strategy("iv_straddle_v1", self.runtime_config)

    def test_registry_exposes_iv_straddle_v1(self):
        self.assertEqual(
            available_strategy_ids(),
            (
                "dynamic_atm_iv_straddle",
                "dynamic_position_straddle",
                "iv_straddle_v1",
                "live_straddle",
                "original_atm_iv_straddle",
            ),
        )
        self.assertEqual(self.plugin.strategy_id, "iv_straddle_v1")

    def test_registry_marks_only_the_old_dynamic_prototype_deprecated(self):
        self.assertEqual(
            deprecated_strategy_ids(),
            ("dynamic_position_straddle",),
        )

        with self.assertWarnsRegex(
            FutureWarning,
            "deprecated legacy prototype",
        ):
            legacy = create_strategy(
                "dynamic_position_straddle",
                config.load_config("300etf"),
            )

        metadata = legacy.metadata()
        self.assertTrue(metadata["deprecated"])
        self.assertEqual(metadata["strategy_status"], "deprecated")
        self.assertEqual(
            metadata["replacement_strategy_id"],
            "dynamic_atm_iv_straddle",
        )

    def test_signal_frame_matches_current_iv_straddle_rules(self):
        features = pd.DataFrame(
            {
                "atm_iv": [0.12, 0.18, 0.30],
                "atm_iv_percentile": [0.10, 0.50, 0.90],
                "signal_iv": [pd.NA, 0.18, 0.30],
                "signal_iv_percentile": [pd.NA, 0.50, 0.90],
                "yz_hv60": [0.10, 0.12, 0.20],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
        )

        with patch.object(strategy, "CONFIG", self.runtime_config):
            expected = strategy.build_signals(features)
        actual = self.plugin.build_signals(features)

        columns = [
            "signal_iv",
            "signal_iv_percentile",
            "prev_atm_iv",
            "prev_signal_iv",
            "long_open_signal",
            "short_open_regime",
            "short_open_signal",
        ]
        pd.testing.assert_frame_equal(actual[columns], expected[columns])

    def test_plugin_uses_its_own_config_for_entry_and_exit_rules(self):
        low_iv = pd.Series({"atm_iv": 0.12, "yz_hv60": 0.10})
        high_iv = pd.Series({"atm_iv": 0.30, "yz_hv60": 0.10})

        self.assertEqual(self.plugin.entry_target_qty(low_iv, 10, "long"), 10)
        self.assertEqual(self.plugin.entry_target_qty(high_iv, 10, "long"), 0)
        self.assertEqual(self.plugin.get_close_reason(high_iv, 20), "iv_high")
        self.assertEqual(self.plugin.default_short_entry_regime, "absolute")


class OriginalAtmIvStraddleStrategyTest(unittest.TestCase):
    def setUp(self):
        default_config = config.load_config("300etf")
        self.runtime_config = replace(
            default_config,
            strategy=replace(
                default_config.strategy,
                enable_long_straddle=True,
                enable_short_straddle=True,
            ),
        )
        self.plugin = create_strategy(
            "original_atm_iv_straddle",
            self.runtime_config,
        )

    def test_original_strategy_profiles_disable_long_straddle(self):
        for product in ("50etf", "300etf", "500etf", "kc50etf"):
            with self.subTest(product=product):
                common_config = config.load_config(product)
                strategy_config = resolve_strategy_config(
                    common_config,
                    "original_atm_iv_straddle",
                )
                self.assertTrue(common_config.strategy.enable_long_straddle)
                self.assertFalse(
                    strategy_config.strategy.enable_long_straddle
                )

    def test_uses_only_absolute_atm_iv_open_and_close_thresholds(self):
        cfg = self.runtime_config.strategy
        features = pd.DataFrame(
            {
                "atm_iv": [
                    cfg.long_open_iv_threshold,
                    (
                        cfg.long_open_iv_threshold
                        + cfg.short_open_iv_threshold
                    )
                    / 2.0,
                    cfg.short_open_iv_threshold,
                ],
                "signal_iv": [0.99, 0.99, 0.01],
                "atm_iv_percentile": [0.99, 0.01, 0.01],
                "yz_hv60": [0.40, 0.01, 0.40],
            },
            index=pd.date_range("2026-01-05", periods=3, freq="B"),
        )

        signals = self.plugin.build_signals(features)

        self.assertEqual(signals["long_open_signal"].tolist(), [True, False, False])
        self.assertEqual(signals["short_open_signal"].tolist(), [False, False, True])
        self.assertTrue(pd.isna(signals["short_open_regime"].iloc[0]))
        self.assertTrue(pd.isna(signals["short_open_regime"].iloc[1]))
        self.assertEqual(signals["short_open_regime"].iloc[2], "absolute")
        self.assertEqual(
            self.plugin.get_close_reason(
                pd.Series({"atm_iv": cfg.long_close_iv_threshold}),
                position_dte=0,
            ),
            "iv_high",
        )
        self.assertEqual(
            self.plugin.get_short_close_reason(
                pd.Series({"atm_iv": cfg.short_close_iv_threshold}),
                position_dte=0,
            ),
            "short_iv_low",
        )

    def test_disables_all_non_core_strategy_behaviors(self):
        self.assertTrue(self.plugin.enable_roll)
        self.assertTrue(self.plugin.enable_strike_roll)
        self.assertTrue(self.plugin.attempt_roll_without_current_entry_signal)
        self.assertTrue(self.plugin.evaluate_roll_entry_on_candidate)
        self.assertTrue(self.plugin.close_if_roll_candidate_unavailable)
        self.assertEqual(
            self.plugin.roll_dte_threshold,
            self.runtime_config.strategy.roll_dte_threshold,
        )
        self.assertFalse(self.plugin.is_short_daily_loss_stop(-1_000_000, 1))
        self.assertFalse(self.plugin.has_short_volume_spike({}, {}, {}))
        self.assertEqual(self.plugin.short_cooldown_after_long_iv_high_exit_days, 0)
        self.assertIsNone(
            self.plugin.existing_position_target_qty(
                pd.Series({"atm_iv": 0.30}),
                "short",
            )
        )

    def test_requests_required_daily_delta_hedge_routes(self):
        self.assertTrue(self.plugin.enable_delta_hedge)
        self.assertEqual(self.plugin.delta_hedge_tolerance_ratio, 0.0)
        self.assertFalse(self.plugin.allow_etf_short_hedge)
        self.assertTrue(self.plugin.enable_atm_straddle_rebalance)
        self.assertFalse(self.plugin.enable_atm_straddle_shape_rebalance)
        self.assertEqual(
            self.plugin.config.strategy.delta_residual_abs_tolerance,
            0.0,
        )
        self.assertFalse(self.plugin.config.strategy.short_stop_loss_enabled)
        self.assertFalse(
            self.plugin.config.strategy.short_volume_spike_exit_enabled
        )

    def test_entry_uses_fixed_configured_quantity(self):
        cfg = self.runtime_config.strategy
        self.assertEqual(
            self.plugin.entry_target_qty(
                pd.Series({"atm_iv": cfg.long_open_iv_threshold}),
                7,
                "long",
            ),
            7,
        )
        self.assertEqual(
            self.plugin.entry_target_qty(
                pd.Series({"atm_iv": cfg.short_open_iv_threshold}),
                9,
                "short",
            ),
            9,
        )

    def test_dte_roll_trigger_does_not_require_strike_roll_signal(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.strategy_plugin = self.plugin
        engine.config = {
            "long_qty": 10,
            "short_qty": 10,
            "proportional_position_sizing_enabled": False,
        }
        state = backtester.BacktestState(
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "strike": 4.0,
                    "call_qty": 10,
                    "put_qty": 10,
                    "strategy_pair_qty": 10,
                },
            },
        )
        day = {
            "spot": 4.0,
            "feature_row": pd.Series({"atm_strike": pd.NA, "atm_iv": 0.14}),
            "chain_df": pd.DataFrame(),
        }

        self.assertTrue(
            engine._should_roll_position(
                day,
                state,
                "short",
                self.plugin.roll_dte_threshold,
            )
        )

    def test_strike_tracking_triggers_at_exactly_one_step(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.strategy_plugin = self.plugin
        engine.config = {
            "long_qty": 10,
            "short_qty": 10,
            "proportional_position_sizing_enabled": False,
        }
        state = backtester.BacktestState(
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "strike": 4.0,
                    "call_qty": 10,
                    "put_qty": 10,
                    "strategy_pair_qty": 10,
                },
            },
        )
        day = {
            "spot": 4.1,
            "feature_row": pd.Series({"atm_strike": 4.1, "atm_iv": 0.14}),
            "chain_df": pd.DataFrame(
                {"strike_price": [3.9, 4.0, 4.1, 4.2]}
            ),
        }

        self.assertEqual(
            engine._roll_trigger(day, state, "short", position_dte=20),
            "strike",
        )

    def test_failed_candidate_entry_rule_closes_original_position(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.strategy_plugin = self.plugin
        engine.config = {
            "long_qty": 10,
            "short_qty": 10,
            "proportional_position_sizing_enabled": False,
        }
        position = {
            "strike": 4.0,
            "call_qty": 10,
            "put_qty": 10,
            "strategy_pair_qty": 10,
        }
        state = backtester.BacktestState(
            cash=10_000_000,
            positions={"long": None, "short": position},
        )
        day = {
            "date": pd.Timestamp("2026-01-20"),
            "spot": 4.0,
            "feature_row": pd.Series({"atm_iv": 0.20, "atm_strike": 4.0}),
            "chain_df": pd.DataFrame(),
        }
        candidate = {
            "atm_iv": self.runtime_config.strategy.short_open_iv_threshold - 0.001,
            "strike": 4.0,
            "dte": 20,
        }

        with (
            patch.object(
                backtester.vol_engine,
                "select_atm_from_chain",
                return_value=candidate,
            ),
            patch.object(engine, "_close_hedge_before_roll"),
            patch.object(engine, "_close_position_after_failed_roll") as close,
        ):
            engine._roll_position(
                day,
                state,
                "short",
                pd.Series({"dte": self.plugin.roll_dte_threshold}),
                pd.Series(dtype=object),
                backtester.empty_greeks(),
            )

        self.assertEqual(close.call_count, 1)
        self.assertEqual(
            close.call_args.args[-1],
            "roll_entry_threshold_not_met",
        )


class DynamicAtmIvStraddleStrategyTest(unittest.TestCase):
    def setUp(self):
        runtime_config = config.load_config("300etf")
        self.original = create_strategy(
            "original_atm_iv_straddle",
            runtime_config,
        )
        self.dynamic = create_strategy(
            "dynamic_atm_iv_straddle",
            runtime_config,
        )

    def test_keeps_original_signals_and_exit_rules(self):
        features = pd.DataFrame(
            {
                "atm_iv": [0.08, 0.12, 0.18, 0.30, pd.NA],
                "signal_iv": [0.99, 0.99, 0.99, 0.01, 0.50],
            },
            index=pd.date_range("2026-01-05", periods=5, freq="B"),
        )

        original_signals = self.original.build_signals(features)
        dynamic_signals = self.dynamic.build_signals(features)
        pd.testing.assert_frame_equal(dynamic_signals, original_signals)

        for iv in [0.05, 0.10, 0.15, 0.25, pd.NA]:
            row = pd.Series({"atm_iv": iv})
            self.assertEqual(
                self.dynamic.entry_target_qty(row, 13, "long"),
                self.original.entry_target_qty(row, 13, "long"),
            )
            self.assertEqual(
                self.dynamic.get_close_reason(row, 0),
                self.original.get_close_reason(row, 0),
            )
            self.assertEqual(
                self.dynamic.get_short_close_reason(row, 0),
                self.original.get_short_close_reason(row, 0),
            )

    def test_short_quantity_uses_five_step_absolute_iv_ladder(self):
        open_iv = self.dynamic.config.strategy.short_open_iv_threshold
        full_iv = self.dynamic.position_ladder.full_position_iv
        interval = (full_iv - open_iv) / 5

        self.assertEqual(
            [
                self.dynamic.entry_target_qty(
                    pd.Series({"atm_iv": open_iv + level * interval}),
                    20,
                    "short",
                )
                for level in range(6)
            ],
            [10, 12, 14, 16, 18, 20],
        )
        self.assertEqual(
            self.dynamic.entry_target_qty(
                pd.Series({"atm_iv": open_iv - 0.0001}),
                20,
                "short",
            ),
            0,
        )
        self.assertEqual(
            self.dynamic.entry_target_qty(
                pd.Series({"atm_iv": full_iv + 0.10}),
                13,
                "short",
            ),
            13,
        )

    def test_existing_short_reduces_to_minimum_and_missing_iv_holds(self):
        open_iv = self.dynamic.config.strategy.short_open_iv_threshold
        self.assertEqual(
            self.dynamic.existing_position_target_qty(
                pd.Series({"atm_iv": open_iv - 0.001}),
                "short",
            ),
            10,
        )
        self.assertIsNone(
            self.dynamic.existing_position_target_qty(
                pd.Series({"atm_iv": pd.NA}),
                "short",
            )
        )

    def test_roll_candidate_is_sized_from_actual_new_contract_iv(self):
        below_open_iv = self.dynamic.config.strategy.short_open_iv_threshold - 0.01
        self.assertEqual(
            self.dynamic.roll_candidate_target_qty(
                pd.Series({"iv": 0.23}),
                pd.Series({"iv": 0.25}),
                20,
                "short",
            ),
            20,
        )
        self.assertEqual(
            self.dynamic.roll_candidate_target_qty(
                pd.Series({"iv": below_open_iv}),
                pd.Series({"iv": below_open_iv}),
                20,
                "short",
            ),
            0,
        )
        self.assertEqual(
            self.dynamic.roll_candidate_target_qty(
                pd.Series({"iv": pd.NA}),
                pd.Series({"iv": 0.25}),
                20,
                "short",
            ),
            0,
        )

    def test_engine_opens_roll_directly_at_candidate_ladder_qty(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.strategy_plugin = self.dynamic
        engine.config = {
            "long_qty": 20,
            "short_qty": 20,
            "proportional_position_sizing_enabled": False,
        }
        old_position = {
            "strike": 4.0,
            "expiry": pd.Timestamp("2026-02-25"),
            "call_qty": 7,
            "put_qty": 7,
            "strategy_pair_qty": 7,
            "short_entry_regime": "absolute",
        }
        state = backtester.BacktestState(
            cash=1_000_000,
            positions={"long": None, "short": old_position},
        )
        call = pd.Series({"iv": 0.23, "dte": 20})
        put = pd.Series({"iv": 0.25, "dte": 20})
        candidate = {
            "atm_iv": 0.24,
            "strike": 4.1,
            "dte": 20,
            "call": call,
            "put": put,
        }
        new_position = {
            "call_code": "NEW_CALL",
            "put_code": "NEW_PUT",
            "call_qty": 7,
            "put_qty": 7,
        }
        day = {
            "date": pd.Timestamp("2026-01-20"),
            "spot": 4.1,
            "feature_row": pd.Series({"atm_iv": 0.01, "atm_strike": 4.1}),
            "chain_df": pd.DataFrame(),
        }

        with (
            patch.object(engine, "_roll_trigger", return_value="strike"),
            patch.object(
                backtester.vol_engine,
                "select_atm_from_chain_for_expiry",
                return_value=candidate,
            ),
            patch.object(engine, "_dynamic_target_qty", side_effect=lambda *a, **k: a[3]),
            patch.object(engine, "_atm_underlying_price", return_value=4.1),
            patch.object(engine, "_project_cash_after_hedge", return_value=1_000_000),
            patch.object(engine, "_project_cash_after_option_close", return_value=1_000_000),
            patch.object(engine, "_project_cash_after_option_open", return_value=1_000_000),
            patch.object(
                backtester.strategy,
                "calc_position_greeks",
                return_value=backtester.empty_greeks(),
            ),
            patch.object(engine, "_has_cash_reserve", return_value=True),
            patch.object(engine, "_close_hedge_before_roll"),
            patch.object(backtester.opt_position, "close_trade", return_value=(1_000_000, 0.0)),
            patch.object(
                backtester.opt_position,
                "open_trade",
                return_value=(1_000_000, new_position, 0.0),
            ) as open_trade,
            patch.object(engine, "_add_new_option_fees"),
            patch.object(engine, "_set_side_eod"),
            patch.object(engine, "_resize_existing_position") as resize,
        ):
            engine._roll_position(
                day,
                state,
                "short",
                pd.Series({"dte": 20}),
                pd.Series(dtype=object),
                backtester.empty_greeks(),
            )

        self.assertEqual(open_trade.call_args.args[3:5], (20, 20))
        resize.assert_not_called()

    def test_engine_closes_old_without_opening_candidate_below_open(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.strategy_plugin = self.dynamic
        engine.config = {
            "long_qty": 20,
            "short_qty": 20,
            "proportional_position_sizing_enabled": False,
        }
        old_position = {
            "strike": 4.0,
            "expiry": pd.Timestamp("2026-02-25"),
            "call_qty": 7,
            "put_qty": 7,
            "strategy_pair_qty": 7,
            "short_entry_regime": "absolute",
        }
        state = backtester.BacktestState(
            cash=1_000_000,
            positions={"long": None, "short": old_position},
        )
        below_open_iv = self.dynamic.config.strategy.short_open_iv_threshold - 0.01
        call = pd.Series({"iv": below_open_iv, "dte": 20})
        put = pd.Series({"iv": below_open_iv, "dte": 20})
        candidate = {
            "atm_iv": below_open_iv,
            "strike": 4.1,
            "dte": 20,
            "call": call,
            "put": put,
        }
        new_position = {
            "call_code": "NEW_CALL",
            "put_code": "NEW_PUT",
            "call_qty": 7,
            "put_qty": 7,
        }
        day = {
            "date": pd.Timestamp("2026-01-20"),
            "spot": 4.1,
            "feature_row": pd.Series({"atm_iv": 0.30, "atm_strike": 4.1}),
            "chain_df": pd.DataFrame(),
        }

        def verify_rejected_candidate_close(*args):
            self.assertIs(state.positions["short"], old_position)
            self.assertEqual(args[-1], "roll_entry_threshold_not_met")

        with (
            patch.object(engine, "_roll_trigger", return_value="strike"),
            patch.object(
                backtester.vol_engine,
                "select_atm_from_chain_for_expiry",
                return_value=candidate,
            ),
            patch.object(engine, "_dynamic_target_qty", side_effect=lambda *a, **k: a[3]),
            patch.object(engine, "_atm_underlying_price", return_value=4.1),
            patch.object(engine, "_project_cash_after_hedge", return_value=1_000_000),
            patch.object(engine, "_project_cash_after_option_close", return_value=1_000_000),
            patch.object(engine, "_project_cash_after_option_open", return_value=1_000_000),
            patch.object(
                backtester.strategy,
                "calc_position_greeks",
                return_value=backtester.empty_greeks(),
            ),
            patch.object(engine, "_has_cash_reserve", return_value=True),
            patch.object(engine, "_close_hedge_before_roll"),
            patch.object(
                backtester.opt_position,
                "close_trade",
                return_value=(1_000_000, 0.0),
            ),
            patch.object(
                backtester.opt_position,
                "open_trade",
                return_value=(1_000_000, new_position, 0.0),
            ) as open_trade,
            patch.object(engine, "_add_new_option_fees"),
            patch.object(
                engine,
                "_close_position_after_failed_roll",
                side_effect=verify_rejected_candidate_close,
            ) as close,
            patch.object(engine, "_set_side_eod") as set_eod,
            patch.object(engine, "_resize_existing_position") as resize,
        ):
            engine._roll_position(
                day,
                state,
                "short",
                pd.Series({"dte": 20}),
                pd.Series(dtype=object),
                backtester.empty_greeks(),
            )

        open_trade.assert_not_called()
        self.assertEqual(close.call_count, 1)
        set_eod.assert_not_called()
        resize.assert_not_called()

    def test_has_an_independent_strategy_identity(self):
        self.assertEqual(
            self.dynamic.strategy_id,
            "dynamic_atm_iv_straddle",
        )
        self.assertEqual(
            self.dynamic.strategy_name,
            "动态 ATM IV 跨式策略",
        )
        self.assertEqual(
            self.dynamic.metadata()["strategy_id"],
            "dynamic_atm_iv_straddle",
        )
        self.assertEqual(
            self.dynamic.metadata()["strategy_status"],
            "active_development",
        )
        self.assertEqual(
            self.dynamic.metadata()["baseline_strategy_id"],
            "original_atm_iv_straddle",
        )
        self.assertEqual(
            self.dynamic.config.strategy,
            self.original.config.strategy,
        )
        self.assertEqual(
            self.dynamic.metadata()["absolute_iv_position_ladder"],
            {
                "min_qty": 10,
                "max_qty": 20,
                "steps": 5,
                "full_position_iv": 0.24,
            },
        )


if __name__ == "__main__":
    unittest.main()
