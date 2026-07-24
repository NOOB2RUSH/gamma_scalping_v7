import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import run
from core import config
from core.backtest_strategies import (
    available_strategy_config_ids,
    create_strategy,
    resolve_strategy_config,
)


def _args(**updates):
    values = {
        "product": "300etf",
        "strategy": "original_atm_iv_straddle",
        "start": None,
        "end": None,
        "test_date": None,
        "initial_cash": None,
        "dynamic_position_control": None,
        "proportional_position_sizing": None,
        "max_margin_to_nav_ratio": None,
        "long_open_iv_threshold": None,
        "long_close_iv_threshold": None,
        "short_open_iv_threshold": None,
        "short_close_iv_threshold": None,
        "atm_target_dte": None,
        "roll_dte_threshold": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


class StrategyConfigOverridesTest(unittest.TestCase):
    def test_active_dynamic_strategy_has_a_dedicated_profile(self):
        self.assertEqual(
            available_strategy_config_ids(),
            ("dynamic_atm_iv_straddle", "original_atm_iv_straddle"),
        )

    def test_plugin_fields_override_common_and_omitted_fields_fall_back(self):
        common = config.load_config("300etf")
        resolved = resolve_strategy_config(common, "original_atm_iv_straddle")

        self.assertTrue(common.strategy.enable_long_straddle)
        self.assertFalse(resolved.strategy.enable_long_straddle)
        self.assertEqual(common.strategy.short_open_iv_threshold, 0.155)
        self.assertEqual(resolved.strategy.short_open_iv_threshold, 0.20)
        self.assertEqual(common.vol.atm_target_dte, 20)
        self.assertEqual(resolved.vol.atm_target_dte, 25)

        self.assertEqual(resolved.data, common.data)
        self.assertEqual(resolved.backtest, common.backtest)
        self.assertEqual(
            resolved.strategy.short_volume_spike_multiplier,
            common.strategy.short_volume_spike_multiplier,
        )
        self.assertEqual(resolved.vol.contract_multiplier, common.vol.contract_multiplier)

    def test_strategy_without_profile_uses_common_config(self):
        common = config.load_config("300etf")
        resolved = resolve_strategy_config(common, "iv_straddle_v1")

        self.assertIs(resolved, common)
        self.assertTrue(resolved.strategy.enable_long_straddle)
        self.assertEqual(resolved.strategy.short_open_iv_threshold, 0.155)
        self.assertEqual(resolved.vol.atm_target_dte, 20)

    def test_each_original_profile_keeps_its_tuned_defaults(self):
        expected = {
            "50etf": (0.20, 0.175, 15, 7),
            "300etf": (0.20, 0.16, 25, 5),
            "500etf": (0.24, 0.18, 10, 7),
            "kc50etf": (0.35, 0.31, 15, 7),
        }
        for product, values in expected.items():
            with self.subTest(product=product):
                resolved = resolve_strategy_config(
                    config.load_config(product),
                    "original_atm_iv_straddle",
                )
                self.assertEqual(
                    (
                        resolved.strategy.short_open_iv_threshold,
                        resolved.strategy.short_close_iv_threshold,
                        resolved.vol.atm_target_dte,
                        resolved.strategy.roll_dte_threshold,
                    ),
                    values,
                )
                self.assertFalse(resolved.strategy.enable_long_straddle)

    def test_dynamic_profile_only_changes_original_target_quantities(self):
        expected_quantities = {
            "50etf": 35,
            "300etf": 20,
            "500etf": 10,
            "kc50etf": 40,
        }
        for product, quantity in expected_quantities.items():
            with self.subTest(product=product):
                common = config.load_config(product)
                original = resolve_strategy_config(
                    common,
                    "original_atm_iv_straddle",
                )
                dynamic = resolve_strategy_config(
                    common,
                    "dynamic_atm_iv_straddle",
                )

                self.assertEqual(dynamic.data, original.data)
                self.assertEqual(dynamic.strategy, original.strategy)
                self.assertEqual(dynamic.vol, original.vol)
                self.assertEqual(dynamic.report, original.report)
                self.assertEqual(dynamic.backtest.long_qty, quantity)
                self.assertEqual(dynamic.backtest.short_qty, quantity)
                self.assertEqual(
                    replace(
                        dynamic.backtest,
                        long_qty=original.backtest.long_qty,
                        short_qty=original.backtest.short_qty,
                    ),
                    original.backtest,
                )

    def test_cli_overrides_have_priority_over_plugin_profile(self):
        resolved = run.select_runtime_config(
            _args(
                short_open_iv_threshold=0.0,
                short_close_iv_threshold=0.0,
                atm_target_dte=17,
            )
        )

        self.assertEqual(resolved.strategy.short_open_iv_threshold, 0.0)
        self.assertEqual(resolved.strategy.short_close_iv_threshold, 0.0)
        self.assertEqual(resolved.vol.atm_target_dte, 17)
        self.assertEqual(resolved.strategy.roll_dte_threshold, 5)
        self.assertFalse(resolved.strategy.enable_long_straddle)

    def test_runtime_config_persists_plugin_effective_values(self):
        resolved = run.select_runtime_config(_args())
        plugin = create_strategy("original_atm_iv_straddle", resolved)

        with tempfile.TemporaryDirectory() as tmpdir:
            run.save_runtime_config(Path(tmpdir), plugin.config)
            saved = json.loads(
                (Path(tmpdir) / "runtime_config.json").read_text(encoding="utf-8")
            )

        self.assertFalse(saved["strategy"]["enable_long_straddle"])
        self.assertFalse(saved["strategy"]["short_stop_loss_enabled"])
        self.assertFalse(saved["strategy"]["short_volume_spike_exit_enabled"])
        self.assertEqual(saved["strategy"]["short_open_iv_threshold"], 0.20)
        self.assertEqual(saved["vol"]["atm_target_dte"], 25)


if __name__ == "__main__":
    unittest.main()
