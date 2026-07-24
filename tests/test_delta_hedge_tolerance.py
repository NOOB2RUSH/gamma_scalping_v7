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
            delta_residual_abs_tolerance=5_000.0,
            allow_etf_short_hedge=True,
            enable_atm_straddle_rebalance=False,
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
    def test_net_delta_within_five_thousand_does_not_buy_etf(self):
        config = _config()
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        config.strategy.delta_residual_abs_tolerance = 5_000.0
        config.strategy.allow_etf_short_hedge = False
        live_account = account.AccountState(
            product="kc50etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle(qty=10)},
            hedge=account.HedgeState(qty=3_100),
        )

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": -5_299.2708058599455},
            1.937,
            None,
            {"underlying_order_book_id": "588000.XSHG"},
            action="DELTA_HEDGE",
            reason="test",
        )

        self.assertEqual(plan, [])

    def test_absolute_delta_threshold_applies_with_etf_short_enabled(self):
        config = _config()
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        config.strategy.delta_residual_abs_tolerance = 5_000.0
        config.strategy.allow_etf_short_hedge = True
        live_account = account.AccountState(
            product="300etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle(qty=10)},
            hedge=account.HedgeState(qty=0),
        )

        for account_delta in (-5_000.0, -4_999.0, 4_999.0, 5_000.0):
            with self.subTest(account_delta=account_delta):
                plan = signal_engine._delta_hedge_plan(
                    config,
                    live_account,
                    {"delta": account_delta},
                    5.0,
                    None,
                    {"underlying_order_book_id": "510300.XSHG"},
                    action="DELTA_HEDGE",
                    reason="test",
                )
                self.assertEqual(plan, [])

        for account_delta in (-5_001.0, 5_001.0):
            with self.subTest(account_delta=account_delta):
                plan = signal_engine._delta_hedge_plan(
                    config,
                    live_account,
                    {"delta": account_delta},
                    5.0,
                    None,
                    {"underlying_order_book_id": "510300.XSHG"},
                    action="DELTA_HEDGE",
                    reason="test",
                )
                self.assertEqual([item["action"] for item in plan], ["DELTA_HEDGE"])

    def test_existing_option_plan_reports_non_blocking_residual_risk(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        config.strategy.delta_residual_abs_tolerance = 0.0
        live_account = account.AccountState(
            product="50etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle(qty=10)},
            hedge=account.HedgeState(qty=0),
        )

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 2_000.0},
            3.0,
            None,
            {"underlying_order_book_id": "510050.XSHG"},
            action="FINAL_DELTA_HEDGE",
            reason="test",
            after_actions=["ATM_STRADDLE_DELTA_REBALANCE"],
        )

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["action"], "RESIDUAL_RISK")
        self.assertEqual(plan[0]["code"], "DELTA_RESIDUAL_AFTER_OPTION_PLAN")
        self.assertFalse(plan[0]["blocking"])
        self.assertEqual(
            plan[0]["reason"],
            "Residual delta remains after the option rebalance; no further "
            "same-signal iteration is planned.",
        )

        metadata = signal_engine._plan_metadata(
            [
                {
                    "action": "DELTA_HEDGE",
                    "priority": "action",
                    "trade_etf_qty": -5_000,
                },
                plan[0],
            ]
        )
        self.assertEqual(metadata["plan_status"], "ACTIONABLE_WITH_RESIDUAL")
        self.assertTrue(metadata["execution_allowed"])
        self.assertEqual(
            metadata["residual_risks"][0]["code"],
            "DELTA_RESIDUAL_AFTER_OPTION_PLAN",
        )

    def test_positive_constrained_residual_uses_five_thousand_limit(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        config.strategy.delta_residual_abs_tolerance = 5_000.0
        live_account = account.AccountState(
            product="50etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle(qty=10)},
            hedge=account.HedgeState(qty=0),
        )

        accepted = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 5_000.0},
            3.0,
            None,
            {"underlying_order_book_id": "510050.XSHG"},
            action="FINAL_DELTA_HEDGE",
            reason="test",
            after_actions=["ATM_STRADDLE_DELTA_REBALANCE"],
        )
        rejected = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 5_001.0},
            3.0,
            None,
            {"underlying_order_book_id": "510050.XSHG"},
            action="FINAL_DELTA_HEDGE",
            reason="test",
            after_actions=["ATM_STRADDLE_DELTA_REBALANCE"],
        )

        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["action"], "RESIDUAL_RISK")

    def test_roll_plan_uses_threshold_only_to_trigger_zero_target(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        config.strategy.delta_residual_abs_tolerance = 5_000.0
        live_account = account.AccountState(
            product="kc50etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "call_code": "OLD_CALL",
                    "put_code": "OLD_PUT",
                    "call_qty": 10,
                    "put_qty": 10,
                    "contract_multiplier": 10000,
                    "option_margin": 72_000.0,
                },
            },
            hedge=account.HedgeState(qty=3_100),
        )
        expiry = pd.Timestamp("2026-08-26")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 1.95,
                    "mid": 0.14465,
                    "delta": 0.525221248137618,
                    "gamma": 0.10,
                    "vega": 0.026,
                    "theta": -0.026,
                    "iv": 0.575,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 1.95,
                    "mid": 0.16505,
                    "delta": -0.4722285400790186,
                    "gamma": 0.10,
                    "vega": 0.026,
                    "theta": -0.027,
                    "iv": 0.604,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[0],
            "put": chain.iloc[1],
            "strike": 1.95,
            "expiry": expiry,
            "underlying_order_book_id": "588000.XSHG",
        }
        roll_item = {
            "action": "ROLL_SHORT_STRADDLE",
            "priority": "action",
            "side": "short",
            "current_call_qty": 10,
            "current_put_qty": 10,
            "estimated_current_call_price": 0.10525,
            "estimated_current_put_price": 0.2249,
            "target_call_code": "CALL",
            "target_put_code": "PUT",
            "target_call_qty": 10,
            "target_put_qty": 10,
            "target_expiry": str(expiry.date()),
        }

        plan, planned_greeks = signal_engine._build_execution_plan(
            config,
            live_account,
            chain,
            1.937,
            atm,
            [roll_item],
            {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
            {},
        )

        self.assertAlmostEqual(planned_greeks["delta"], -5_299.270805859942)
        self.assertEqual(
            [item["action"] for item in plan],
            ["ROLL_SHORT_STRADDLE"],
        )
        final_state = signal_engine._project_final_account_state(
            config,
            live_account,
            chain,
            plan,
            planned_greeks,
        )
        self.assertEqual(final_state["hedge_qty"], 3_100.0)
        self.assertAlmostEqual(final_state["account_delta"], -2_199.270805859942)
        self.assertLessEqual(
            abs(final_state["account_delta"]),
            config.strategy.delta_residual_abs_tolerance,
        )

        # A pre-roll breach keeps delta control active even though the roll
        # itself brings projected delta back inside the 5,000 entry threshold.
        live_account.hedge.qty = 7_500
        triggered_plan, triggered_greeks = signal_engine._build_execution_plan(
            config,
            live_account,
            chain,
            1.937,
            atm,
            [roll_item],
            {"delta": 9_000.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
            {},
        )

        self.assertEqual(
            [item["action"] for item in triggered_plan],
            ["ROLL_SHORT_STRADDLE", "FINAL_DELTA_HEDGE"],
        )
        final_hedge = triggered_plan[-1]
        self.assertEqual(final_hedge["target_hedge_qty"], 5_300)
        self.assertEqual(final_hedge["trade_etf_qty"], -2_200)
        self.assertTrue(final_hedge["delta_hedge_triggered_before_option_plan"])
        self.assertFalse(final_hedge["delta_hedge_triggered_after_option_plan"])
        triggered_state = signal_engine._project_final_account_state(
            config,
            live_account,
            chain,
            triggered_plan,
            triggered_greeks,
        )
        self.assertEqual(triggered_state["hedge_qty"], 5_300.0)
        self.assertAlmostEqual(triggered_state["account_delta"], 0.729194140058)

    def test_pre_roll_close_is_included_in_final_account_projection(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        live_account = account.AccountState(
            product="300etf",
            cash=10_000_000,
            positions={"long": None, "short": _short_straddle(qty=10)},
            hedge=account.HedgeState(qty=12_600),
        )
        advice = [
            {
                "action": "CLOSE_HEDGE_BEFORE_ROLL",
                "priority": "action",
                "target_hedge_qty": 0,
                "trade_etf_qty": -12_600,
                "underlying_order_book_id": "510300.XSHG",
            }
        ]

        final_state = signal_engine._project_final_account_state(
            config,
            live_account,
            pd.DataFrame(),
            advice,
            {"delta": 1_170.1910488991125},
        )

        self.assertEqual(final_state["hedge_qty"], 0.0)
        self.assertAlmostEqual(final_state["account_delta"], 1_170.1910488991125)
        self.assertAlmostEqual(
            final_state["normalized_account_delta"],
            1_170.1910488991125 / 200_000,
        )

    def test_blocking_data_diagnostic_marks_plan_blocked(self):
        diagnostic = signal_engine._diagnostic_item(
            "DATA_WARNING",
            "POSITION_CONTRACT_MISSING",
            "missing contract",
            blocking=True,
        )

        metadata = signal_engine._plan_metadata([diagnostic])

        self.assertEqual(metadata["plan_status"], "BLOCKED")
        self.assertFalse(metadata["execution_allowed"])
        self.assertEqual(metadata["residual_risks"], [])

    def test_generic_position_target_is_projected_without_action_name_registration(self):
        config = _config()
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="kc50etf",
            cash=1_000_000,
            positions={"long": None, "short": None},
            hedge=account.HedgeState(qty=0),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 2.3,
                    "mid": 0.10,
                    "delta": 0.50,
                    "gamma": 0.08,
                    "vega": 0.02,
                    "theta": -0.01,
                    "iv": 0.20,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 2.3,
                    "mid": 0.10,
                    "delta": -0.20,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "iv": 0.21,
                    "contract_multiplier": 10000,
                },
            ]
        )
        custom_transition = {
            "action": "CUSTOM_OPTION_TRANSITION",
            "priority": "action",
            "side": "short",
            "position_target": {
                "call_code": "CALL",
                "put_code": "PUT",
                "call_qty": 10,
                "put_qty": 10,
                "strike": 2.3,
                "expiry": str(expiry.date()),
            },
        }

        plan, planned_greeks = signal_engine._build_execution_plan(
            config,
            live_account,
            chain,
            2.3,
            {"underlying_order_book_id": "588000.XSHG"},
            [custom_transition],
            {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
            {},
        )

        self.assertAlmostEqual(planned_greeks["delta"], -30000.0)
        self.assertEqual(
            [item["action"] for item in plan],
            ["CUSTOM_OPTION_TRANSITION", "FINAL_DELTA_HEDGE"],
        )
        self.assertEqual(plan[1]["target_hedge_qty"], 30000.0)

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

    def test_normalizes_straddle_by_total_option_qty(self):
        normalized, capacity = strategy.normalized_account_delta(
            49222.10725218244,
            {"long": None, "short": _short_straddle()},
        )

        self.assertEqual(capacity, 1_600_000)
        self.assertAlmostEqual(normalized, 0.030763817032614027)

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
            reason="test",
        )

        self.assertEqual(plan, [])

    def test_live_plan_does_not_emit_zero_quantity_delta_hedge(self):
        config = _config()
        live_account = account.AccountState(
            product="500etf",
            cash=1_000_000,
            positions={"long": None, "short": None},
            hedge=account.HedgeState(qty=0),
        )

        plan = signal_engine._delta_hedge_plan(
            config,
            live_account,
            {"delta": 0.0},
            8.1,
            None,
            {"underlying_order_book_id": "510500.XSHG"},
            action="DELTA_HEDGE",
            reason="Account delta exceeds tolerance.",
        )

        self.assertEqual(plan, [])

    def test_live_plan_hedges_outside_normalized_tolerance(self):
        live_account = account.AccountState(
            product="kc50etf",
            cash=1_000_000,
            positions={"long": None, "short": _short_straddle()},
            hedge=account.HedgeState(qty=200000),
        )

        plan = signal_engine._delta_hedge_plan(
            _config(),
            live_account,
            {"delta": -3877.892747817561},
            1.0,
            None,
            {"underlying_order_book_id": "588000.XSHG"},
            action="DELTA_HEDGE",
            reason="test",
        )

        self.assertEqual(len(plan), 1)
        self.assertAlmostEqual(plan[0]["normalized_account_delta"], 0.12257631703261403)
        self.assertEqual(plan[0]["hedge_tolerance"], 160000)

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
            reason="test",
        )

        self.assertEqual(plan[0]["target_hedge_qty"], 259300)
        self.assertEqual(plan[0]["trade_etf_qty"], 206200)

    def test_positive_delta_after_etf_zero_uses_atm_straddle_leg_rebalance(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
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
                    "iv": 0.50,
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
                    "iv": 0.55,
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
        self.assertEqual(rebalance["close_put_qty"], 1)
        self.assertEqual(rebalance["open_call_code"], "CALL")
        self.assertEqual(rebalance["open_call_qty"], 1)
        self.assertEqual(rebalance["target_call_qty"], 11)
        self.assertEqual(rebalance["target_put_qty"], 9)
        self.assertAlmostEqual(rebalance["estimated_delta_effect"], -10000.0)
        self.assertAlmostEqual(rebalance["estimated_gamma_effect"], -100.0)
        self.assertAlmostEqual(rebalance["estimated_vega_effect"], -20.0)
        self.assertAlmostEqual(rebalance["estimated_theta_effect"], 10.0)
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_option_rebalance"],
            2000.0,
        )
        self.assertAlmostEqual(rebalance["etf_delta_correction"], 0.0)
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_combined_hedge"],
            2000.0,
        )
        self.assertAlmostEqual(rebalance["normalized_combined_delta"], 0.01)
        self.assertAlmostEqual(rebalance["target_call_put_ratio_error"], 2 / 11)
        self.assertEqual(rebalance["target_pair_qty_deviation"], 2)
        self.assertEqual(rebalance["target_pair_qty_deviation_balance"], 0)
        self.assertTrue(rebalance["delta_tolerance_met"])
        self.assertAlmostEqual(rebalance["estimated_market_value_effect"], 100.0)

    def test_positive_delta_long_straddle_rebalances_long_legs_to_zero(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        config.strategy.delta_residual_abs_tolerance = 0.0
        expiry = pd.Timestamp("2026-07-22")
        long_position = {
            "side": "long",
            "call_code": "LONG_CALL",
            "put_code": "LONG_PUT",
            "call_qty": 10,
            "put_qty": 10,
            "strike": 5.0,
            "expiry": expiry,
            "contract_multiplier": 10_000,
            "option_margin": 0.0,
        }
        live_account = account.AccountState(
            product="300etf",
            cash=10_000_000,
            positions={"long": long_position, "short": None},
            hedge=account.HedgeState(qty=0),
        )
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "LONG_CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 5.0,
                    "mid": 0.10,
                    "delta": 0.55,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "volume": 10_000,
                    "contract_multiplier": 10_000,
                },
                {
                    "order_book_id": "LONG_PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 5.0,
                    "mid": 0.10,
                    "delta": -0.45,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "volume": 10_000,
                    "contract_multiplier": 10_000,
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
            reason="test",
        )

        self.assertEqual(
            [item["action"] for item in plan],
            ["ATM_STRADDLE_DELTA_REBALANCE"],
        )
        rebalance = plan[0]
        self.assertEqual(rebalance["side"], "long")
        self.assertGreater(rebalance["close_call_qty"], 0)
        self.assertGreater(rebalance["open_put_qty"], 0)
        self.assertEqual(rebalance["open_call_qty"], 0)
        self.assertEqual(rebalance["close_put_qty"], 0)
        self.assertGreaterEqual(rebalance["target_hedge_qty"], 0)
        self.assertLessEqual(
            abs(rebalance["projected_account_delta_after_combined_hedge"]),
            strategy.ETF_HEDGE_LOT_SIZE / 2,
        )
        self.assertTrue(rebalance["delta_tolerance_met"])

    def test_roll_plan_includes_follow_up_delta_rebalance(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="kc50etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "call_code": "OLD_CALL",
                    "put_code": "OLD_PUT",
                    "call_qty": 10,
                    "put_qty": 10,
                    "contract_multiplier": 10000,
                    "option_margin": 60_000.0,
                },
            },
            hedge=account.HedgeState(qty=11_200),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 2.3,
                    "mid": 0.06515,
                    "delta": 0.4317558674505776,
                    "gamma": 0.08,
                    "vega": 0.020,
                    "theta": -0.010,
                    "iv": 0.50,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "maturity_date": expiry,
                    "strike_price": 2.3,
                    "mid": 0.1242,
                    "delta": -0.5558421755001888,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "iv": 0.55,
                    "volume": 10000,
                    "contract_multiplier": 10000,
                },
            ]
        )
        atm = {
            "call": chain.iloc[0],
            "put": chain.iloc[1],
            "strike": 2.3,
            "expiry": expiry,
            "underlying_order_book_id": "588000.XSHG",
        }
        roll_item = {
            "action": "ROLL_SHORT_STRADDLE",
            "priority": "action",
            "side": "short",
            "current_call_qty": 10,
            "current_put_qty": 10,
            "estimated_current_call_price": 0.11,
            "estimated_current_put_price": 0.09,
            "target_call_code": "CALL",
            "target_put_code": "PUT",
            "target_call_qty": 10,
            "target_put_qty": 10,
            "target_expiry": str(expiry.date()),
        }

        plan, planned_greeks = signal_engine._build_execution_plan(
            config,
            live_account,
            chain,
            2.252,
            atm,
            [roll_item],
            {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0},
            {},
        )

        self.assertAlmostEqual(planned_greeks["delta"], 12408.63080496111)
        self.assertEqual(
            [item["action"] for item in plan],
            [
                "ROLL_SHORT_STRADDLE",
                "FINAL_DELTA_HEDGE",
                "FINAL_ATM_STRADDLE_DELTA_REBALANCE",
            ],
        )
        self.assertEqual(plan[1]["target_hedge_qty"], 0.0)
        self.assertEqual(plan[1]["trade_etf_qty"], -11200.0)
        rebalance = plan[2]
        self.assertEqual(rebalance["after_actions"], ["ROLL_SHORT_STRADDLE"])
        self.assertEqual(rebalance["current_call_qty"], 10)
        self.assertEqual(rebalance["current_put_qty"], 10)
        self.assertEqual(rebalance["open_call_qty"], 1)
        self.assertEqual(rebalance["close_put_qty"], 1)
        self.assertEqual(rebalance["target_call_qty"], 11)
        self.assertEqual(rebalance["target_put_qty"], 9)
        self.assertAlmostEqual(rebalance["trade_etf_qty"], 0.0)
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_combined_hedge"],
            2532.650375453446,
        )

    def test_imbalanced_atm_straddle_rebalances_shape_before_etf(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
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
            reason="test",
        )

        self.assertEqual(
            [item["action"] for item in plan],
            ["ATM_STRADDLE_SHAPE_REBALANCE"],
        )
        rebalance = plan[0]
        self.assertEqual(rebalance["close_call_qty"], 1)
        self.assertEqual(rebalance["open_put_qty"], 1)
        self.assertEqual(rebalance["target_call_qty"], 13)
        self.assertEqual(rebalance["target_put_qty"], 7)
        self.assertEqual(rebalance["target_hedge_qty"], 2000.0)
        self.assertEqual(rebalance["trade_etf_qty"], -1900.0)
        self.assertEqual(rebalance["position_total_qty_change"], 0)
        self.assertTrue(rebalance["delta_tolerance_met"])
        self.assertAlmostEqual(
            rebalance["projected_account_delta_after_combined_hedge"],
            -10.0,
        )

    def test_real_16_to_4_shape_balances_to_14_to_6_within_abs_delta_tolerance(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.0
        live_account = account.AccountState(
            product="50etf",
            cash=10_000_000,
            positions={
                "long": None,
                "short": {
                    "call_code": "CALL",
                    "put_code": "PUT",
                    "call_qty": 16,
                    "put_qty": 4,
                    "contract_multiplier": 10000,
                    "strike": 3.1,
                    "expiry": "2026-07-22",
                    "option_margin": 72_640.0,
                },
            },
            hedge=account.HedgeState(qty=4_800),
        )
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "maturity_date": expiry,
                    "strike_price": 3.1,
                    "mid": 0.01405,
                    "delta": 0.2661068992213909,
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
                    "mid": 0.07670,
                    "delta": -0.6986797187155358,
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
            {"delta": -14_629.915126801116},
            3.043,
            chain,
            atm,
            action="DELTA_HEDGE",
            reason="real 2026-07-14 regression",
        )

        self.assertEqual(
            [item["action"] for item in plan],
            ["ATM_STRADDLE_SHAPE_REBALANCE"],
        )
        rebalance = plan[0]
        self.assertEqual(rebalance["close_call_qty"], 2)
        self.assertEqual(rebalance["open_put_qty"], 2)
        self.assertEqual(rebalance["target_call_qty"], 14)
        self.assertEqual(rebalance["target_put_qty"], 6)
        self.assertEqual(rebalance["target_hedge_qty"], 0.0)
        self.assertEqual(rebalance["trade_etf_qty"], -4800.0)
        self.assertEqual(rebalance["position_total_qty_change"], 0)
        self.assertTrue(rebalance["delta_not_worse"])
        self.assertTrue(rebalance["delta_tolerance_met"])
        self.assertLessEqual(
            abs(rebalance["projected_account_delta_after_combined_hedge"]),
            config.strategy.delta_residual_abs_tolerance,
        )

    def test_imbalanced_shape_rebalance_does_not_override_delta_control(self):
        config = _config()
        config.strategy.allow_etf_short_hedge = False
        config.strategy.enable_atm_straddle_rebalance = True
        config.strategy.delta_hedge_tolerance_ratio = 0.05
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
        config.strategy.enable_atm_straddle_rebalance = True
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
        config.strategy.enable_atm_straddle_rebalance = True
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
            reason="test",
        )

        self.assertEqual(plan[-1]["action"], "ATM_STRADDLE_DELTA_REBALANCE")
        self.assertEqual(plan[-1]["open_call_code"], "HELD_CALL")
        self.assertEqual(plan[-1]["close_put_code"], "HELD_PUT")
        self.assertEqual(plan[-1]["open_call_qty"], 6)
        self.assertEqual(plan[-1]["close_put_qty"], 6)
        self.assertEqual(plan[-1]["target_call_qty"], 16)
        self.assertEqual(plan[-1]["target_put_qty"], 4)
        self.assertAlmostEqual(plan[-1]["trade_etf_qty"], 0.0)
        self.assertAlmostEqual(
            plan[-1]["projected_account_delta_after_combined_hedge"],
            4200.0,
        )
        self.assertLess(plan[-1]["normalized_combined_delta"], 0.05)
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
