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


def _long_position(qty=10):
    return {
        "side": "long",
        "call_code": "CALL",
        "put_code": "PUT",
        "strike": 2.45,
        "expiry": pd.Timestamp("2024-04-24"),
        "call_qty": qty,
        "put_qty": qty,
        "contract_multiplier": 10000,
        "option_margin": 0.0,
        "last_option_value": 7200.0,
    }


class BacktesterLiveParityTest(unittest.TestCase):
    def test_day_aggregates_clear_stale_eod_marks_after_position_close(self):
        engine = _engine()
        state = backtester.BacktestState(cash=1_000_000.0)
        day = engine._new_day(pd.Timestamp("2026-06-12"), 1.8, {}, None)
        record = day["side_records"]["short"]
        record["option_value"] = -160_000.0
        record["greeks"]["delta"] = -200_000.0
        record["pnl_greeks"]["delta"] = -180_000.0
        record["eod_position_dte"] = 20

        engine._update_day_aggregates(day, state)

        self.assertEqual(day["option_value"], 0.0)
        self.assertEqual(day["greeks"]["delta"], 0.0)
        self.assertEqual(day["pnl_greeks"]["delta"], -180_000.0)
        self.assertIsNone(day["eod_position_dte"])

    def test_execute_delta_hedge_rounds_to_etf_board_lot(self):
        trades = []

        result = backtester.execute_delta_hedge(
            pd.Timestamp("2026-06-12"),
            1_000_000.0,
            {"delta": -259290.84776481945},
            53100.0,
            1.756,
            53100.0,
            1.8,
            trades,
            etf_fee_rate=0.0,
            underlying_order_book_id="588000.XSHG",
            current_underlying_order_book_id="588000.XSHG",
        )

        self.assertEqual(result[1], 259300)
        self.assertEqual(trades[0]["trade_etf_qty"], 206200)

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

    def test_dynamic_reduction_keeps_maximum_qty_that_allows_neutral_hedge(self):
        engine = _engine()
        state = backtester.BacktestState(
            cash=1_000_000.0,
            positions={"long": None, "short": _short_position(qty=80)},
        )
        position = state.positions["short"]
        position["option_margin"] = 431_680.0
        day = engine._new_day(pd.Timestamp("2026-06-12"), 1.8, {}, None)
        day["greeks"]["delta"] = -259290.84776481945
        day["side_records"]["short"]["greeks"]["delta"] = -259290.84776481945

        target = engine._dynamic_reduction_target_qty(
            day,
            state,
            "short",
            position,
            80,
            800_000.0,
        )

        self.assertEqual(target, 71)

    def test_dynamic_occupation_reduction_forces_neutral_hedge(self):
        engine = _engine()
        engine.config["dynamic_position_control_enabled"] = True
        state = backtester.BacktestState(
            cash=1_000_000.0,
            positions={"long": None, "short": _short_position(qty=80)},
        )
        day = engine._new_day(pd.Timestamp("2026-06-12"), 1.8, {}, None)
        day["greeks"]["delta"] = -259290.84776481945
        with (
            patch.object(
                engine,
                "_current_nav_and_occupation",
                side_effect=[
                    (1_000_000.0, 900_000.0),
                    (1_000_000.0, 790_000.0),
                ],
            ),
            patch.object(engine, "_update_day_aggregates"),
            patch.object(engine, "_dynamic_reduction_target_qty", return_value=71),
            patch.object(engine, "_reduce_position_for_margin", return_value=True),
            patch.object(engine, "_hedge_to") as hedge_to,
        ):
            engine._enforce_dynamic_occupation_limit(day, state)

        self.assertEqual(hedge_to.call_args.kwargs["target_qty"], 259300)

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

    def test_expired_missing_position_contracts_settle_at_intrinsic_value(self):
        engine = _engine()
        state = backtester.BacktestState(
            cash=1_000_000.0,
            positions={"long": _long_position(), "short": None},
        )
        day = engine._new_day(
            pd.Timestamp("2024-04-24"),
            2.46,
            {},
            pd.DataFrame({"order_book_id": ["OTHER"]}),
        )

        engine._mark_current_positions_for_capacity(day, state)

        self.assertIsNone(state.positions["long"])
        self.assertFalse(day["defer_delta_hedge"])
        close_trade = next(
            trade for trade in state.trades if trade.get("type") == "close_straddle"
        )
        self.assertEqual(close_trade["exit_reason"], "expired_missing_option_data_intrinsic")
        self.assertAlmostEqual(state.cash, 1_000_960.0)
        self.assertEqual(
            day["data_warnings"][-1]["reason"],
            "expired_missing_position_contracts_settled_intrinsic",
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

    def test_existing_short_margin_refresh_preserves_nav(self):
        engine = _engine()
        state = backtester.BacktestState(
            cash=100.0,
            positions={"long": None, "short": _short_position(qty=10)},
        )
        day = engine._new_day(pd.Timestamp("2026-06-11"), 1.0, {}, pd.DataFrame())

        with patch.object(
            backtester.opt_position,
            "calc_short_margin",
            return_value=80.0,
        ):
            engine._refresh_short_margin(
                day,
                state,
                "short",
                pd.Series({"mid": 0.001, "underlying_close": 1.0}),
                pd.Series({"mid": 0.001, "underlying_close": 1.0}),
            )

        self.assertEqual(state.cash, 70.0)
        self.assertEqual(state.positions["short"]["option_margin"], 80.0)
        self.assertEqual(state.cash + state.positions["short"]["option_margin"], 150.0)

    def test_etf_capital_occupation_uses_current_market_value(self):
        engine = _engine()
        state = backtester.BacktestState(
            cash=900.0,
            hedge_etf_qty=100.0,
            hedge_entry_price=1.0,
            hedge_margin=100.0,
            hedge_underlying_order_book_id="ETF",
        )
        day = engine._new_day(pd.Timestamp("2026-06-11"), 2.0, {}, pd.DataFrame())

        with patch.object(engine, "_get_hedge_price", return_value=2.0):
            nav, occupation = engine._current_nav_and_occupation(day, state)

        self.assertEqual(nav, 1_100.0)
        self.assertEqual(occupation, 200.0)


if __name__ == "__main__":
    unittest.main()
