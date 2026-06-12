import unittest
from unittest.mock import patch

import pandas as pd

from core import backtester


def _engine():
    engine = backtester.BacktestEngine.__new__(backtester.BacktestEngine)
    engine.config = {
        "initial_cash": 1_000_000.0,
        "min_cash_reserve": 50_000.0,
        "long_qty": 10,
        "short_qty": 80,
        "etf_fee_rate": 0.00005,
        "enable_delta_hedge": True,
        "delta_hedge_tolerance_ratio": 0.10,
        "allow_etf_short_hedge": False,
        "enable_option_delta_hedge": True,
        "option_delta_hedge_combination_enabled": True,
        "dynamic_position_control_enabled": False,
        "proportional_position_sizing_enabled": False,
        "position_sizing_base_nav": 1_000_000.0,
        "max_margin_to_nav_ratio": 0.80,
    }
    engine.hedge_by_date = None
    return engine


def _short_position(qty=80):
    return {
        "side": "short",
        "call_code": "CALL",
        "put_code": "PUT",
        "call_qty": qty,
        "put_qty": qty,
        "contract_multiplier": 10000,
        "option_margin": 50.0,
        "last_option_value": -20.0,
    }


class BacktesterLiveParityTest(unittest.TestCase):
    def test_fixed_live_quantities_when_dynamic_scaling_is_disabled(self):
        engine = _engine()
        engine.config["dynamic_position_control_enabled"] = False
        engine.config["proportional_position_sizing_enabled"] = False
        state = backtester.BacktestState(cash=10_000_000.0)
        day = engine._new_day(pd.Timestamp("2026-06-11"), 1.0, {}, None)

        self.assertEqual(engine._proportional_side_max_qty(day, state, "long"), 10)
        self.assertEqual(engine._proportional_side_max_qty(day, state, "short"), 80)
        self.assertEqual(
            engine._dynamic_target_qty(day, state, None, 80, "short"),
            80,
        )

    def test_dynamic_target_qty_respects_occupation_limit(self):
        engine = _engine()
        engine.config["dynamic_position_control_enabled"] = True
        state = backtester.BacktestState(cash=1_000_000.0)
        day = engine._new_day(pd.Timestamp("2026-06-11"), 1.0, {}, None)
        atm = {
            "call": {"mid": 0.01, "contract_multiplier": 10000},
            "put": {"mid": 0.01, "contract_multiplier": 10000},
        }
        greeks = {**backtester.empty_greeks(), "delta": 10_000.0}

        with (
            patch.object(engine, "_current_nav_and_margin", return_value=(1_000_000.0, 0.0)),
            patch.object(backtester.strategy, "calc_position_greeks", return_value=greeks),
            patch.object(backtester.opt_position, "calc_short_margin", side_effect=lambda *args: args[2] * 20_000.0),
        ):
            qty = engine._dynamic_target_qty(day, state, atm, 80, "short")

        self.assertEqual(qty, 39)

    def test_open_allows_same_close_delta_hedge_in_daily_backtest(self):
        engine = _engine()
        state = backtester.BacktestState(cash=1_000_000.0)
        day = engine._new_day(
            pd.Timestamp("2026-06-11"),
            1.0,
            {"short_open_signal": True},
            pd.DataFrame(),
        )
        atm = {
            "call": {"order_book_id": "CALL"},
            "put": {"order_book_id": "PUT"},
            "dte": 20,
            "underlying_order_book_id": "588000.XSHG",
        }
        position = _short_position()

        with (
            patch.object(backtester.vol_engine, "select_atm_from_chain", return_value=atm),
            patch.object(engine, "_entry_target_qty", return_value=80),
            patch.object(engine, "_project_cash_after_option_open", return_value=900_000.0),
            patch.object(backtester.strategy, "calc_position_greeks", return_value=backtester.empty_greeks()),
            patch.object(backtester.strategy, "get_short_open_regime", return_value="absolute"),
            patch.object(
                backtester.opt_position,
                "open_trade",
                return_value=(900_000.0, position, -20.0),
            ),
        ):
            engine._open_new_position(day, state, "open_short_straddle", "short")

        self.assertIs(state.positions["short"], position)
        self.assertFalse(day["defer_delta_hedge"])

    def test_missing_option_chain_preserves_position_and_hedge(self):
        engine = _engine()
        state = backtester.BacktestState(
            cash=1_000_000.0,
            positions={"long": None, "short": _short_position()},
            hedge_etf_qty=123.0,
        )
        captured = {}

        with patch.object(
            engine,
            "_record_day",
            side_effect=lambda day, current_state: captured.update(
                day=day,
                state=current_state,
            ),
        ):
            engine._handle_missing_option_day(
                pd.Timestamp("2026-06-11"),
                1.0,
                {},
                state,
            )

        self.assertIsNotNone(state.positions["short"])
        self.assertEqual(state.hedge_etf_qty, 123.0)
        self.assertTrue(captured["day"]["defer_delta_hedge"])
        self.assertEqual(
            captured["day"]["data_warnings"][0]["reason"],
            "missing_option_chain",
        )

    def test_capacity_reduction_only_reduces_short_once(self):
        engine = _engine()
        engine.config["min_cash_reserve"] = 0.0
        state = backtester.BacktestState(
            cash=100.0,
            positions={"long": None, "short": _short_position(qty=10)},
        )
        day = engine._new_day(pd.Timestamp("2026-06-11"), 1.0, {}, pd.DataFrame())
        rows = (pd.Series({"mid": 0.0}), pd.Series({"mid": 0.0}))

        with (
            patch.object(engine, "_update_day_aggregates"),
            patch.object(engine, "_current_nav_and_margin", return_value=(100.0, 90.0)),
            patch.object(engine, "_get_position_rows", return_value=rows),
            patch.object(engine, "_reduce_position_for_margin", return_value=True) as reduce,
        ):
            result = engine._enforce_margin_limit(day, state)

        self.assertTrue(result)
        reduce.assert_called_once_with(day, state, "short", 8)
        self.assertTrue(day["defer_delta_hedge"])

    def test_existing_short_margin_is_not_refreshed_from_market(self):
        engine = _engine()
        state = backtester.BacktestState(
            cash=100.0,
            positions={"long": None, "short": _short_position(qty=10)},
        )
        day = engine._new_day(pd.Timestamp("2026-06-11"), 1.0, {}, pd.DataFrame())

        with patch.object(
            backtester.opt_position,
            "calc_short_margin",
            side_effect=AssertionError("live margin must remain broker-imported"),
        ):
            engine._set_existing_position_eod(
                day,
                state,
                "short",
                pd.Series({"mid": 0.001}),
                pd.Series({"mid": 0.001}),
                backtester.empty_greeks(),
                10,
            )

        self.assertEqual(state.positions["short"]["option_margin"], 50.0)


if __name__ == "__main__":
    unittest.main()
