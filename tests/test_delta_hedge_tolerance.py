import unittest
from types import SimpleNamespace
from unittest import mock

from core import strategy
from core.live import account, signal_engine


def _config():
    return SimpleNamespace(
        strategy=SimpleNamespace(
            enable_delta_hedge=True,
            delta_hedge_tolerance_ratio=0.10,
            allow_etf_short_hedge=True,
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
            option_action="OPTION_DELTA_HEDGE_SHORT_CALL",
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
            option_action="OPTION_DELTA_HEDGE_SHORT_CALL",
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
            option_action="OPTION_DELTA_HEDGE_SHORT_CALL",
            reason="test",
        )

        self.assertEqual(plan[0]["target_hedge_qty"], 259300)
        self.assertEqual(plan[0]["trade_etf_qty"], 206200)

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
                option_action="OPTION_DELTA_HEDGE_SHORT_CALL",
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
