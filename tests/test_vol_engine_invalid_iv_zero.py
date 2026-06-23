import pandas as pd

from core import vol_engine


def test_add_iv_for_day_sets_unsolved_iv_to_zero():
    chain = pd.DataFrame(
        [
            {
                "date": "2026-06-17",
                "maturity_date": "2026-06-24",
                "strike_price": 1.75,
                "option_type": "C",
                "bid": 0.1853,
                "ask": 0.1853,
                "volume": 1,
                "contract_multiplier": 10000,
                "underlying_close": 1.939,
            }
        ]
    )

    result = vol_engine.add_iv_for_day(chain, 1.939)
    result = vol_engine.add_greeks_for_day(result, 1.939)

    assert result.loc[0, "iv"] == 0.0
    assert result.loc[0, "vega"] == 0.0
