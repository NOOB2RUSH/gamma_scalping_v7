import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core import strategy
from core.live import account, signal_engine


def _config():
    return SimpleNamespace(
        strategy=SimpleNamespace(
            enable_delta_hedge=True,
            delta_hedge_tolerance_ratio=0.10,
            allow_etf_short_hedge=True,
            enable_option_delta_hedge=False,
        ),
        backtest=SimpleNamespace(
            etf_fee_rate=0.0,
            min_cash_reserve=0.0,
            option_fee_per_contract=2.0,
            dynamic_position_control_enabled=False,
            max_margin_to_nav_ratio=0.80,
        ),
        vol=SimpleNamespace(contract_multiplier=10000),
    )


def _short_straddle(qty=80):
    return {
        "call_qty": qty,
        "put_qty": qty,
        "contract_multiplier": 10000,
    }


class DeltaHedgeToleranceTest(unittest.TestCase):
    def test_rounds_etf_hedge_target_to_nearest_board_lot(self):
        self.assertEqual(strategy.round_etf_hedge_target(259290.85), 259300)
        self.assertEqual(strategy.round_etf_hedge_target(-259249.99), -259200)
        self.assertEqual(strategy.round_etf_hedge_target(0), 0)

    def test_atm_rebalance_target_qty_priority(self):
        config = _config()
        self.assertEqual(signal_engine._atm_rebalance_target_pair_qty(config), 10)

        config.backtest.short_qty = 12
        self.assertEqual(signal_engine._atm_rebalance_target_pair_qty(config), 12)

        config.strategy.atm_rebalance_target_pair_qty = 8
        self.assertEqual(signal_engine._atm_rebalance_target_pair_qty(config), 8)

    def test_normalizes_straddle_by_pair_qty_not_both_legs(self):
        normalized, capacity = strategy.normalized_account_delta(
            49222.10725218244,
            {"long": None, "short": _short_straddle()},
        )

        self.assertEqual(capacity, 800000)
        self.assertAlmostEqual(normalized, 0.06152763406522805)

    def test_live_plan_does_not_hedge_inside_normalized_tolerance(self):
        live_account = account.AccountState(
            product="kc50etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle()},
            hedge=account.HedgeState(qty=53100),
        )

        plan = signal_engine._delta_hedge_plan(
            _config(),
            live_account,
            {"delta": -3877.892747817561},
            1.0,
            None,
            {"underlying_order_book_id": "588000.XSHG"},
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual(plan, [])

    def test_live_plan_hedges_outside_normalized_tolerance(self):
        live_account = account.AccountState(
            product="kc50etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle()},
            hedge=account.HedgeState(qty=100000),
        )

        plan = signal_engine._delta_hedge_plan(
            _config(),
            live_account,
            {"delta": -3877.892747817561},
            1.0,
            None,
            {"underlying_order_book_id": "588000.XSHG"},
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual(len(plan), 1)
        self.assertAlmostEqual(plan[0]["normalized_account_delta"], 0.12015263406522805)
        self.assertEqual(plan[0]["hedge_tolerance"], 80000)

    def test_live_plan_outputs_executable_etf_board_lot(self):
        live_account = account.AccountState(
            product="kc50etf",
            cash=10_000_000,
            positions={"long": None, "short": _short_straddle()},
            hedge=account.HedgeState(qty=53100),
        )

        plan = signal_engine._delta_hedge_plan(
            _config(),
            live_account,
            {"delta": -259290.84776481945},
            1.8,
            None,
            {"underlying_order_book_id": "588000.XSHG"},
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual(plan[0]["target_hedge_qty"], 259300)
        self.assertEqual(plan[0]["trade_etf_qty"], 206200)

    def test_positive_delta_after_etf_zero_uses_atm_straddle_leg_rebalance(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_option_delta_hedge = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="300etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    **_short_straddle(qty=10),
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "strike": 5.0,
                    "expiry": "2026-07-22",
                    "option_margin": 100_000.0,
                },
            },
            hedge=account.HedgeState(qty=5_000),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 5.0,
                    "mid": 0.11,
                    "delta": 0.55,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 5.0,
                    "mid": 0.10,
                    "delta": -0.45,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[0],
            "put": chain.iloc[1],
            "strike": 5.0,
            "expiry": expiry,
            "underlying_order_book_id": "510300.XSHG",
        }

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 12_000.0},
            5.0,
            chain,
            atm,
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual([item["action"] for item in plan], [
            "DELTA_HEDGE",
            "ATM_STRADDLE_DELTA_REBALANCE",
        ])
        self.assertEqual(plan[0]["target_hedge_qty"], 0.0)
        self.assertEqual(plan[0]["trade_etf_qty"], -5000.0)
        rebalance = plan[1]
        self.assertEqual(rebalance["close_put_code"], "PUT")
        self.assertEqual(rebalance["close_put_qty"], 2)
        self.assertEqual(rebalance["open_call_code"], "CALL")
        self.assertEqual(rebalance["open_call_qty"], 2)
        self.assertEqual(rebalance["target_call_qty"], 12)
        self.assertEqual(rebalance["target_put_qty"], 8)
        self.assertAlmostEqual(rebalance["estimated_delta_effect"], -20000.0)
        self.assertAlmostEqual(rebalance["estimated_gamma_effect"], -200.0)
        self.assertAlmostEqual(rebalance["estimated_vega_effect"], -40.0)
        self.assertAlmostEqual(rebalance["estimated_theta_effect"], 20.0)
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_option_hedge"],
            -8000.0,
        )
        self.assertAlmostEqual(rebalance["etf_delta_correction"], 8000.0)
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_combined_hedge"],
            0.0,
        )
        self.assertAlmostEqual(rebalance["normalized_combined_delta"], 0.0)
        self.assertAlmostEqual(rebalance["target_call_put_ratio_error"], 4 / 12)
        self.assertEqual(rebalance["target_pair_qty_deviation"], 4)
        self.assertEqual(rebalance["target_pair_qty_deviation_balance"], 0)
        self.assertAlmostEqual(rebalance["estimated_market_value_effect"], 200.0)

    def test_imbalanced_atm_straddle_rebalances_shape_before_etf(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_option_delta_hedge = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="50etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "call_qty": 14,
                    "put_qty": 6,
                    "contract_multiplier": 10000,
                    "strike": 3.1,
                    "expiry": "2026-07-22",
                    "option_margin": 80_000.0,
                },
            },
            hedge=account.HedgeState(qty=3_900),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.0283,
                    "delta": 0.352,
                    "gamma": 0.05,
                    "vega": 0.010,
                    "theta": -0.006,
                    "volume": 100000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.0850,
                    "delta": -0.625,
                    "gamma": 0.06,
                    "vega": 0.012,
                    "theta": -0.007,
                    "volume": 50000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[0],
            "put": chain.iloc[1],
            "strike": 3.1,
            "expiry": expiry,
            "underlying_order_book_id": "510050.XSHG",
        }

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": -11_780.0},
            3.051,
            chain,
            atm,
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual([item["action"] for item in plan], ["DELTA_HEDGE"])
        rebalance = plan[0]
        self.assertEqual(rebalance["target_hedge_qty"], 11800.0)
        self.assertEqual(rebalance["trade_etf_qty"], 7900.0)
        self.assertAlmostEqual(rebalance["projected_account_delta_after_hedge"], 20.0)

    def test_imbalanced_shape_rebalance_does_not_override_delta_control(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_option_delta_hedge = True
        config.strategy.delta_hedge_tolerance_ratio = 0.10
        live_account = account.AccountState(
            product="50etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "call_qty": 14,
                    "put_qty": 6,
                    "contract_multiplier": 10000,
                    "strike": 3.1,
                    "expiry": "2026-07-22",
                    "option_margin": 80_000.0,
                },
            },
            hedge=account.HedgeState(qty=11_800),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.0185,
                    "delta": 0.90,
                    "gamma": 0.05,
                    "vega": 0.010,
                    "theta": -0.006,
                    "volume": 100000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.1080,
                    "delta": -0.07,
                    "gamma": 0.06,
                    "vega": 0.012,
                    "theta": -0.007,
                    "volume": 50000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[0],
            "put": chain.iloc[1],
            "strike": 3.1,
            "expiry": expiry,
            "underlying_order_book_id": "510050.XSHG",
        }

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 6_638.0},
            3.017,
            chain,
            atm,
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual([item["action"] for item in plan], [
            "ATM_STRADDLE_SHAPE_REBALANCE",
        ])
        rebalance = plan[0]
        self.assertEqual(rebalance["target_hedge_qty"], 0.0)
        self.assertEqual(rebalance["trade_etf_qty"], -11_800.0)
        self.assertEqual(rebalance["open_put_qty"], 4)
        self.assertEqual(rebalance["target_call_qty"], 14)
        self.assertEqual(rebalance["target_put_qty"], 10)
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_combined_hedge"],
            9438.0,
        )
        self.assertTrue(rebalance["delta_not_worse"])
        self.assertTrue(rebalance["delta_tolerance_met"])

    def test_one_strike_from_atm_main_straddle_can_rebalance_delta(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_option_delta_hedge = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="300etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    **_short_straddle(qty=10),
                    "call_code": "HELD_CALL",
                    "put_code": "HELD_PUT",
                    "strike": 5.0,
                    "expiry": "2026-07-22",
                    "option_margin": 100_000.0,
                },
            },
            hedge=account.HedgeState(qty=5_000),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "ATM_CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 4.9,
                    "mid": 0.106,
                    "delta": 0.55,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "ATM_PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 4.9,
                    "mid": 0.095,
                    "delta": -0.45,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "HELD_CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 5.0,
                    "mid": 0.063,
                    "delta": 0.38,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "HELD_PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 5.0,
                    "mid": 0.152,
                    "delta": -0.60,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[0],
            "put": chain.iloc[1],
            "strike": 4.9,
            "expiry": expiry,
            "underlying_order_book_id": "510300.XSHG",
        }

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 9_500.0},
            4.922,
            chain,
            atm,
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual([item["action"] for item in plan], [
            "DELTA_HEDGE",
            "ATM_STRADDLE_DELTA_REBALANCE",
        ])
        rebalance = plan[1]
        self.assertEqual(rebalance["close_put_code"], "HELD_PUT")
        self.assertEqual(rebalance["open_call_code"], "HELD_CALL")
        self.assertEqual(rebalance["open_call_qty"], 1)
        self.assertEqual(rebalance["close_put_qty"], 1)
        self.assertEqual(rebalance["target_call_qty"], 11)
        self.assertEqual(rebalance["target_put_qty"], 9)
        self.assertAlmostEqual(rebalance["estimated_delta_effect"], -9800.0)
        self.assertAlmostEqual(rebalance["etf_delta_correction"], 300.0)
        self.assertAlmostEqual(rebalance["normalized_combined_delta"], 0.0)
        self.assertAlmostEqual(
            rebalance["target_call_put_ratio_error"],
            2 / 11,
        )
        self.assertEqual(rebalance["target_pair_qty_deviation"], 2)
        self.assertEqual(rebalance["target_pair_qty_deviation_balance"], 0)
        self.assertTrue(rebalance["delta_tolerance_met"])
        self.assertEqual(rebalance["target_pair_qty"], 10)
        self.assertAlmostEqual(rebalance["target_pair_market_value_error"], -890.0)

    def test_adjacent_uneven_strike_is_treated_as_atm_tolerated_main_straddle(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_option_delta_hedge = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="50etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    **_short_straddle(qty=10),
                    "call_code": "HELD_CALL",
                    "put_code": "HELD_PUT",
                    "strike": 3.1,
                    "expiry": "2026-07-22",
                    "option_margin": 50_000.0,
                },
            },
            hedge=account.HedgeState(qty=1_600),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "LOW_CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 2.95,
                    "mid": 0.12,
                    "delta": 0.70,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "ATM_CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 3.0,
                    "mid": 0.08,
                    "delta": 0.52,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "ATM_PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 3.0,
                    "mid": 0.06,
                    "delta": -0.48,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "HELD_CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.03,
                    "delta": 0.32,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "HELD_PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.11,
                    "delta": -0.66,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[1],
            "put": chain.iloc[2],
            "strike": 3.0,
            "expiry": expiry,
            "underlying_order_book_id": "510050.XSHG",
        }

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 63_000.0},
            3.025,
            chain,
            atm,
            action="DELTA_HEDGE",
            option_action="ATM_STRADDLE_DELTA_REBALANCE",
            reason="test",
        )

        self.assertEqual(plan[-1]["action"], "ATM_STRADDLE_DELTA_REBALANCE")
        self.assertEqual(plan[-1]["open_call_code"], "HELD_CALL")
        self.assertEqual(plan[-1]["close_put_code"], "HELD_PUT")
        self.assertTrue(plan[-1]["delta_tolerance_met"])

    def test_live_plan_reduces_short_before_neutral_hedge_breaches_occupation_limit(self):
        config = _config()
        config.backtest.dynamic_position_control_enabled = True
        live_account = account.AccountState(
            product="kc50etf",
            cash=573_471.73782,
            positions={
                "long": None,
                "short": {
                    **_short_straddle(),
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "option_margin": 431_680.0,
                },
            },
            hedge=account.HedgeState(
                qty=53100,
                entry_price=1.756,
                margin=93_243.6,
                latest_price=1.8,
            ),
        )
        chain = SimpleNamespace()

        with (
            mock.patch.object(
                signal_engine,
                "_live_account_capacity",
                return_value={
                    "nav": 1_000_000.0,
                    "total_margin": 524_923.6,
                    "capital_occupation": 524_923.6,
                    "hedge_capital_occupation": 93_243.6,
                },
            ),
            mock.patch.object(
                signal_engine.core.vol_engine,
                "resolve_position_pair",
                return_value=(
                    SimpleNamespace(
                        get=lambda key, default=None: 0.05 if key == "mid" else default
                    ),
                    SimpleNamespace(
                        get=lambda key, default=None: 0.05 if key == "mid" else default
                    ),
                ),
            ),
            mock.patch.object(
                signal_engine.core.strategy,
                "calc_position_greeks",
                return_value={"delta": -259290.84776481945},
            ),
            mock.patch.object(
                signal_engine,
                "_current_short_margin",
                return_value=431_680.0,
            ),
        ):
            plan = signal_engine._delta_hedge_plan(
                config,
                live_account,
                {"delta": -259290.84776481945},
                1.8,
                chain,
                {"underlying_order_book_id": "588000.XSHG"},
                action="DELTA_HEDGE",
                option_action="ATM_STRADDLE_DELTA_REBALANCE",
                reason="test",
            )

        self.assertEqual(plan[0]["action"], "REDUCE_SHORT_STRADDLE_FOR_CAPACITY")
        self.assertEqual(plan[0]["call_qty"], 9)
        self.assertEqual(plan[0]["target_call_qty"], 71)
        self.assertLessEqual(
            plan[0]["projected_capital_occupation_after_reduction_and_hedge"],
            plan[0]["capital_occupation_limit"],
        )

    def test_live_capacity_uses_current_margin_and_etf_market_value_for_occupation(self):
        config = _config()
        live_account = account.AccountState(
            product="kc50etf",
            cash=500_000.0,
            positions={
                "long": None,
                "short": {
                    **_short_straddle(qty=10),
                    "option_margin": 100_000.0,
                },
            },
            hedge=account.HedgeState(
                qty=50_000,
                entry_price=1.7,
                margin=85_000.0,
                latest_price=1.7,
            ),
        )
        chain = object()
        call_row = {"underlying_close": 1.8}
        put_row = {"underlying_close": 1.8}

        with (
            mock.patch.object(
                signal_engine.core.vol_engine,
                "resolve_position_pair",
                return_value=(call_row, put_row),
            ),
            mock.patch.object(
                signal_engine.core.position,
                "signed_value",
                return_value=-80_000.0,
            ),
            mock.patch.object(
                signal_engine.core.position,
                "calc_short_margin",
                return_value=120_000.0,
            ),
        ):
            capacity = signal_engine._live_account_capacity(
                config,
                live_account,
                chain,
                1.8,
            )

        self.assertEqual(capacity["nav"], 610_000.0)
        self.assertEqual(capacity["current_option_margin"], 120_000.0)
        self.assertEqual(capacity["hedge_capital_occupation"], 90_000.0)
        self.assertEqual(capacity["capital_occupation"], 210_000.0)


if __name__ == "__main__":
    unittest.main()
