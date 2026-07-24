import unittest

import pandas as pd

import run
from core.backtester import _black_scholes_price, add_full_revaluation_pnl


class BacktesterFullRevaluationTest(unittest.TestCase):
    def test_endpoint_revaluation_reconciles_observed_option_pnl(self):
        dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
        spot0, spot1 = 5.0, 5.2
        strike = 5.0
        iv0, iv1 = 0.24, 0.31
        dte0, dte1 = 10, 9
        multiplier = 10_000
        qty = 10

        def price(flag, spot, dte, iv):
            return _black_scholes_price(
                flag,
                spot,
                strike,
                dte / 252,
                0.0,
                iv,
            )

        start_call = price("c", spot0, dte0, iv0)
        start_put = price("p", spot0, dte0, iv0)
        end_call = price("c", spot1, dte1, iv1)
        end_put = price("p", spot1, dte1, iv1)
        exact_option_pnl = -qty * multiplier * (
            end_call + end_put - start_call - start_put
        )
        hedge_pnl = 123.0

        daily = pd.DataFrame(
            [
                {
                    "spot": spot0,
                    "daily_nav_pnl_before_fee": pd.NA,
                    "hedge_delta_pnl": 0.0,
                    "long_position_call_qty": 0,
                    "long_position_put_qty": 0,
                    "long_position_call_code": pd.NA,
                    "long_position_put_code": pd.NA,
                    "long_eod_call_delta": 0.0,
                    "long_eod_put_delta": 0.0,
                    "short_position_call_qty": qty,
                    "short_position_put_qty": qty,
                    "short_position_call_code": "CALL",
                    "short_position_put_code": "PUT",
                    "short_eod_call_delta": -55_000.0,
                    "short_eod_put_delta": 45_000.0,
                },
                {
                    "spot": spot1,
                    "daily_nav_pnl_before_fee": exact_option_pnl + hedge_pnl,
                    "hedge_delta_pnl": hedge_pnl,
                    "long_position_call_qty": 0,
                    "long_position_put_qty": 0,
                    "long_position_call_code": pd.NA,
                    "long_position_put_code": pd.NA,
                    "long_eod_call_delta": 0.0,
                    "long_eod_put_delta": 0.0,
                    "short_position_call_qty": 0,
                    "short_position_put_qty": 0,
                    "short_position_call_code": pd.NA,
                    "short_position_put_code": pd.NA,
                    "short_eod_call_delta": 0.0,
                    "short_eod_put_delta": 0.0,
                },
            ],
            index=dates,
        )

        def chain(date, spot, dte, iv, call_price, put_price):
            return pd.DataFrame(
                [
                    {
                        "date": date,
                        "order_book_id": "CALL",
                        "option_type": "c",
                        "strike_price": strike,
                        "dte": dte,
                        "iv": iv,
                        "mid": call_price,
                        "contract_multiplier": multiplier,
                        "pricing_spot": spot,
                    },
                    {
                        "date": date,
                        "order_book_id": "PUT",
                        "option_type": "p",
                        "strike_price": strike,
                        "dte": dte,
                        "iv": iv,
                        "mid": put_price,
                        "contract_multiplier": multiplier,
                        "pricing_spot": spot,
                    },
                ]
            )

        enriched = {
            dates[0]: chain(dates[0], spot0, dte0, iv0, start_call, start_put),
            dates[1]: chain(dates[1], spot1, dte1, iv1, end_call, end_put),
        }
        result = add_full_revaluation_pnl(daily, enriched)
        row = result.iloc[1]

        self.assertAlmostEqual(
            row["full_revaluation_greeks_pnl"],
            exact_option_pnl + hedge_pnl,
            places=7,
        )
        self.assertAlmostEqual(
            row["full_revaluation_unexplained_pnl_before_fee"],
            0.0,
            places=7,
        )

    def test_option_fee_mask_does_not_include_named_etf_hedge_rows(self):
        trades = pd.DataFrame(
            [
                {
                    "type": "atm_straddle_delta_rebalance",
                    "trade_call_qty": 1,
                    "trade_put_qty": -1,
                    "fee": 4.0,
                },
                {
                    "type": "delta_hedge_after_atm_straddle_rebalance",
                    "trade_etf_qty": 2_000,
                    "fee": 2.0,
                },
            ]
        )

        mask = run.option_trade_mask(trades)

        self.assertEqual(mask.tolist(), [True, False])
        self.assertEqual(trades.loc[mask, "fee"].sum(), 4.0)

    def test_missing_contract_pnl_is_reported_as_data_gap_not_unexplained(self):
        dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
        daily = pd.DataFrame(
            [
                {
                    "daily_nav_pnl_before_fee": pd.NA,
                    "hedge_delta_pnl": 0.0,
                    "data_warning_reasons": "",
                },
                {
                    "daily_nav_pnl_before_fee": 1234.0,
                    "hedge_delta_pnl": 0.0,
                    "data_warning_reasons": (
                        "expired_missing_position_contracts_settled_intrinsic"
                    ),
                },
            ],
            index=dates,
        )
        enriched = {dates[0]: pd.DataFrame(), dates[1]: pd.DataFrame()}

        result = add_full_revaluation_pnl(daily, enriched)
        row = result.iloc[1]

        self.assertAlmostEqual(row["full_revaluation_data_gap_pnl"], 1234.0)
        self.assertAlmostEqual(row["full_revaluation_accounted_pnl"], 1234.0)
        self.assertAlmostEqual(
            row["full_revaluation_unexplained_pnl_before_fee"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
