import unittest

import pandas as pd

from core import backtester, strategy


class BacktesterAtmRebalanceTest(unittest.TestCase):
    def test_short_disabled_hedge_rebalances_atm_short_straddle_legs(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.config = {
            "enable_delta_hedge": True,
            "delta_hedge_tolerance_ratio": 0.0,
            "allow_etf_short_hedge": False,
            "etf_fee_rate": 0.0,
            "min_cash_reserve": 0.0,
        }
        date = pd.Timestamp("2026-07-01")
        engine.hedge_by_date = {
            date: pd.DataFrame(
                [{"order_book_id": "ETF", "close": 5.0, "volume": 1_000_000}]
            )
        }
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "strike_price": 5.0,
                    "maturity_date": expiry,
                    "dte": 15,
                    "mid": 0.11,
                    "delta": 0.45,
                    "gamma": 0.08,
                    "vega": 0.02,
                    "theta": -0.01,
                    "iv": 0.30,
                    "volume": 10_000,
                    "contract_multiplier": 10_000,
                    "underlying_order_book_id": "ETF",
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "strike_price": 5.0,
                    "maturity_date": expiry,
                    "dte": 15,
                    "mid": 0.10,
                    "delta": -0.55,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "iv": 0.31,
                    "volume": 10_000,
                    "contract_multiplier": 10_000,
                    "underlying_order_book_id": "ETF",
                },
            ]
        )
        position = {
            "call_code": "CALL",
            "put_code": "PUT",
            "call_qty": 10,
            "put_qty": 10,
            "strike": 5.0,
            "expiry": expiry,
            "contract_multiplier": 10_000,
            "underlying_order_book_id": "ETF",
            "side": "short",
            "option_margin": backtester.opt_position.calc_short_margin(
                chain.iloc[0], chain.iloc[1], 10, 10, 5.0
            ),
            "entry_total_volume": 20_000,
        }
        state = backtester.BacktestState(
            cash=10_000_000,
            positions={"long": None, "short": position},
            hedge_etf_qty=5_000,
            hedge_entry_price=5.0,
            hedge_underlying_order_book_id="ETF",
        )
        greeks = strategy.calc_position_greeks(
            chain.iloc[0], chain.iloc[1], 10, 10, side="short"
        )
        day = {
            "date": date,
            "spot": 5.0,
            "chain_df": chain,
            "greeks": greeks.copy(),
            "side_records": {
                "long": backtester.empty_side_record(),
                "short": backtester.empty_side_record(),
            },
            "daily_etf_fee": 0.0,
            "daily_option_fee": 0.0,
            "option_value": 0.0,
        }
        engine._set_side_eod(
            day,
            state,
            "short",
            backtester.opt_position.signed_value(position, chain.iloc[0], chain.iloc[1]),
            greeks,
            15,
        )
        engine._update_day_aggregates(day, state)

        engine._hedge_to(date, 5.0, state, day, day["greeks"])

        self.assertEqual(state.hedge_etf_qty, 0.0)
        self.assertEqual(state.positions["short"]["call_qty"], 11)
        self.assertEqual(state.positions["short"]["put_qty"], 9)
        self.assertIsNone(state.positions["short"]["entry_total_volume"])
        self.assertEqual(
            [trade["type"] for trade in state.trades],
            [
                "reduce_hedge_before_atm_straddle_rebalance",
                "atm_straddle_delta_rebalance",
            ],
        )

    def test_short_disabled_hedge_rebalances_long_straddle_legs(self):
        engine = object.__new__(backtester.BacktestEngine)
        engine.config = {
            "enable_delta_hedge": True,
            "delta_hedge_tolerance_ratio": 0.0,
            "allow_etf_short_hedge": False,
            "etf_fee_rate": 0.0,
            "min_cash_reserve": 0.0,
        }
        date = pd.Timestamp("2026-07-01")
        engine.hedge_by_date = {
            date: pd.DataFrame(
                [{"order_book_id": "ETF", "close": 5.0, "volume": 1_000_000}]
            )
        }
        expiry = pd.Timestamp("2026-07-22")
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "strike_price": 5.0,
                    "maturity_date": expiry,
                    "dte": 15,
                    "mid": 0.10,
                    "delta": 0.55,
                    "gamma": 0.08,
                    "vega": 0.02,
                    "theta": -0.01,
                    "iv": 0.30,
                    "volume": 10_000,
                    "contract_multiplier": 10_000,
                    "underlying_order_book_id": "ETF",
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "strike_price": 5.0,
                    "maturity_date": expiry,
                    "dte": 15,
                    "mid": 0.10,
                    "delta": -0.45,
                    "gamma": 0.07,
                    "vega": 0.018,
                    "theta": -0.009,
                    "iv": 0.31,
                    "volume": 10_000,
                    "contract_multiplier": 10_000,
                    "underlying_order_book_id": "ETF",
                },
            ]
        )
        position = {
            "call_code": "CALL",
            "put_code": "PUT",
            "call_qty": 10,
            "put_qty": 10,
            "strike": 5.0,
            "expiry": expiry,
            "contract_multiplier": 10_000,
            "underlying_order_book_id": "ETF",
            "side": "long",
            "option_margin": 0.0,
        }
        state = backtester.BacktestState(
            cash=10_000_000,
            positions={"long": position, "short": None},
            hedge_etf_qty=0,
            hedge_entry_price=0.0,
            hedge_underlying_order_book_id=None,
        )
        greeks = strategy.calc_position_greeks(
            chain.iloc[0], chain.iloc[1], 10, 10, side="long"
        )
        day = {
            "date": date,
            "spot": 5.0,
            "chain_df": chain,
            "greeks": greeks.copy(),
            "side_records": {
                "long": backtester.empty_side_record(),
                "short": backtester.empty_side_record(),
            },
            "daily_etf_fee": 0.0,
            "daily_option_fee": 0.0,
            "option_value": 0.0,
        }
        engine._set_side_eod(
            day,
            state,
            "long",
            backtester.opt_position.signed_value(
                position, chain.iloc[0], chain.iloc[1]
            ),
            greeks,
            15,
        )
        engine._update_day_aggregates(day, state)

        engine._hedge_to(date, 5.0, state, day, day["greeks"])

        self.assertEqual(state.hedge_etf_qty, 0.0)
        self.assertEqual(state.positions["long"]["call_qty"], 9)
        self.assertEqual(state.positions["long"]["put_qty"], 11)
        self.assertAlmostEqual(day["greeks"]["delta"], 0.0)
        self.assertEqual(
            [trade["type"] for trade in state.trades],
            ["atm_straddle_delta_rebalance"],
        )


if __name__ == "__main__":
    unittest.main()
