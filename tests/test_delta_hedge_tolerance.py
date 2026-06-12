import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
