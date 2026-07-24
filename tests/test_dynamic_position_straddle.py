import unittest

import pandas as pd

from core import backtester, config, strategy
from core.backtest_strategies import create_strategy


class DynamicPositionStraddleStrategyTest(unittest.TestCase):
    def setUp(self):
        self.runtime_config = config.load_config("300etf")
        self.plugin = create_strategy(
            "dynamic_position_straddle",
            self.runtime_config,
        )

    @staticmethod
    def _feature(atm_iv):
        return pd.Series({"atm_iv": atm_iv, "yz_hv60": 0.10})

    def test_short_ladder_uses_ten_downward_rounded_steps(self):
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.155), "short"),
            8,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.17449), "short"),
            8,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.17450), "short"),
            9,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.350), "short"),
            24,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.500), "short"),
            24,
        )

    def test_long_ladder_is_mirrored_and_configured_independently(self):
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.080), "long"),
            8,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.07701), "long"),
            8,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.07700), "long"),
            9,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.050), "long"),
            24,
        )
        self.assertEqual(
            self.plugin.target_position_qty(self._feature(0.030), "long"),
            24,
        )

    def test_existing_position_returns_to_minimum_until_old_close_threshold(self):
        self.assertEqual(
            self.plugin.existing_position_target_qty(
                self._feature(0.140),
                "short",
            ),
            8,
        )
        self.assertEqual(
            self.plugin.get_short_close_reason(
                self._feature(0.130),
                20,
                {"short_entry_regime": "absolute"},
            ),
            "short_iv_low",
        )
        self.assertEqual(
            self.plugin.existing_position_target_qty(
                self._feature(0.090),
                "long",
            ),
            8,
        )
        self.assertEqual(
            self.plugin.get_close_reason(self._feature(0.240), 20),
            "iv_high",
        )

    def test_entry_and_metadata_use_dynamic_configuration(self):
        self.assertEqual(
            self.plugin.entry_target_qty(self._feature(0.1745), 999, "short"),
            9,
        )
        self.assertEqual(
            self.plugin.entry_target_qty(self._feature(0.077), 999, "long"),
            9,
        )
        self.assertEqual(
            self.plugin.entry_target_qty(self._feature(0.10), 999, "long"),
            0,
        )
        metadata = self.plugin.metadata()
        self.assertEqual(metadata["strategy_id"], "dynamic_position_straddle")
        self.assertEqual(metadata["dynamic_position_config"]["min_qty"], 8)
        self.assertEqual(metadata["dynamic_position_config"]["max_qty"], 24)
        self.assertEqual(metadata["dynamic_position_config"]["short_steps"], 10)
        self.assertEqual(metadata["dynamic_position_config"]["long_steps"], 10)
        self.assertEqual(metadata["dynamic_position_config"]["iv_spike"], 0.03)
        self.assertEqual(
            metadata["dynamic_position_config"]["underlying_log_return_spike"],
            0.03,
        )

    def test_short_waits_through_spike_until_first_down_day(self):
        features = pd.DataFrame(
            {
                "atm_iv": [0.16, 0.19, 0.20, 0.195, 0.23, 0.24, 0.22],
                "yz_hv60": [0.10] * 7,
            },
            index=pd.date_range("2026-01-05", periods=7, freq="B"),
        )

        signals = self.plugin.build_signals(features)

        self.assertEqual(
            signals[self.plugin.SPIKE_EVENT_COL].tolist(),
            [False, True, False, False, True, False, False],
        )
        self.assertEqual(
            signals[self.plugin.SPIKE_WAITING_COL].tolist(),
            [False, True, True, False, True, True, False],
        )
        self.assertEqual(
            signals[self.plugin.SPIKE_PULLBACK_COL].tolist(),
            [False, False, False, True, False, False, True],
        )
        self.assertEqual(
            signals["short_open_signal"].tolist(),
            [True, False, False, True, False, False, True],
        )

    def test_underlying_close_shock_uses_absolute_log_return(self):
        features = pd.DataFrame(
            {
                "close": [4.00, 4.13, 4.12, 3.99, 4.00],
                "atm_iv": [0.16, 0.17, 0.165, 0.17, 0.165],
                "yz_hv60": [0.10] * 5,
            },
            index=pd.date_range("2026-01-05", periods=5, freq="B"),
        )

        signals = self.plugin.build_signals(features)

        self.assertEqual(
            signals[self.plugin.UNDERLYING_SPIKE_EVENT_COL].tolist(),
            [False, True, False, True, False],
        )
        self.assertEqual(
            signals[self.plugin.RISK_SPIKE_EVENT_COL].tolist(),
            [False, True, False, True, False],
        )
        self.assertEqual(
            signals[self.plugin.SPIKE_WAITING_COL].tolist(),
            [False, True, False, True, False],
        )
        self.assertEqual(
            signals["short_open_signal"].tolist(),
            [True, False, True, False, True],
        )

    def test_waiting_blocks_resize_but_preserves_qty_during_roll(self):
        waiting = self._feature(0.25)
        waiting[self.plugin.SPIKE_WAITING_COL] = True

        self.assertIsNone(
            self.plugin.existing_position_target_qty(waiting, "short")
        )
        self.assertEqual(
            self.plugin.roll_target_qty(waiting, 999, "short", current_qty=13),
            13,
        )
        self.assertEqual(
            self.plugin.existing_position_target_qty(waiting, "long"),
            8,
        )


class DynamicPositionResizeExecutionTest(unittest.TestCase):
    def setUp(self):
        self.engine = object.__new__(backtester.BacktestEngine)
        self.engine.config = {"min_cash_reserve": 0.0}
        self.engine.hedge_by_date = None
        self.engine.strategy_plugin = create_strategy(
            "dynamic_position_straddle",
            config.load_config("300etf"),
        )
        self.date = pd.Timestamp("2026-07-01")
        self.expiry = pd.Timestamp("2026-07-22")
        self.call_row = pd.Series(
            {
                "order_book_id": "CALL",
                "strike_price": 4.0,
                "mid": 0.10,
                "delta": 0.52,
                "gamma": 0.08,
                "vega": 0.02,
                "theta": -0.01,
                "iv": 0.20,
                "dte": 15,
                "volume": 10_000,
                "contract_multiplier": 10_000,
            }
        )
        self.put_row = pd.Series(
            {
                "order_book_id": "PUT",
                "strike_price": 4.0,
                "mid": 0.09,
                "delta": -0.48,
                "gamma": 0.07,
                "vega": 0.018,
                "theta": -0.009,
                "iv": 0.21,
                "dte": 15,
                "volume": 10_000,
                "contract_multiplier": 10_000,
            }
        )
        position = {
            "call_code": "CALL",
            "put_code": "PUT",
            "call_qty": 11,
            "put_qty": 9,
            "strategy_pair_qty": 10,
            "strike": 4.0,
            "expiry": self.expiry,
            "contract_multiplier": 10_000,
            "underlying_order_book_id": None,
            "side": "short",
            "short_entry_regime": "absolute",
            "option_margin": backtester.opt_position.calc_short_margin(
                self.call_row,
                self.put_row,
                11,
                9,
                4.0,
            ),
        }
        self.state = backtester.BacktestState(
            cash=10_000_000,
            positions={"long": None, "short": position},
        )
        self.day = self.engine._new_day(
            self.date,
            4.0,
            pd.Series({"atm_iv": 0.194}),
            pd.DataFrame([self.call_row, self.put_row]),
        )

    def test_resize_preserves_atm_rebalance_leg_imbalance(self):
        resized = self.engine._resize_existing_position(
            self.day,
            self.state,
            "short",
            self.call_row,
            self.put_row,
            12,
            15,
        )

        self.assertTrue(resized)
        position = self.state.positions["short"]
        self.assertEqual(position["call_qty"], 13)
        self.assertEqual(position["put_qty"], 11)
        self.assertEqual(position["strategy_pair_qty"], 12)
        self.assertEqual(self.state.trades[-1]["trade_call_qty"], 2)
        self.assertEqual(self.state.trades[-1]["trade_put_qty"], 2)
        self.assertEqual(
            self.state.trades[-1]["type"],
            "dynamic_position_increase_straddle",
        )

        resized = self.engine._resize_existing_position(
            self.day,
            self.state,
            "short",
            self.call_row,
            self.put_row,
            10,
            15,
        )

        self.assertTrue(resized)
        self.assertEqual(position["call_qty"], 11)
        self.assertEqual(position["put_qty"], 9)
        self.assertEqual(position["strategy_pair_qty"], 10)
        self.assertEqual(
            self.state.trades[-1]["type"],
            "dynamic_position_decrease_straddle",
        )


if __name__ == "__main__":
    unittest.main()
