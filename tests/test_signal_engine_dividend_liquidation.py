from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core import backtester
from core.live import account, infinitrader, signal_engine


def _adjusted_chain():
    common = {
        "strike_price": 8.104,
        "maturity_date": pd.Timestamp("2026-07-22"),
        "contract_multiplier": 10180,
        "underlying_order_book_id": "510500.XSHG",
        "bid": 0.1,
        "ask": 0.2,
        "dte": 7,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "order_book_id": "10011720",
                "option_type": "C",
                "contract_symbol": "XD500ETF购7月8104A",
                "mid": 0.1582,
            },
            {
                **common,
                "order_book_id": "10011729",
                "option_type": "P",
                "contract_symbol": "XD500ETF沽7月8104A",
                "mid": 0.1234,
            },
        ]
    )


def _adjusted_short_position():
    return {
        "side": "short",
        "entry_date": "2026-07-01",
        "call_code": "10011720",
        "put_code": "10011729",
        "call_qty": 10,
        "put_qty": 10,
        "strike": 8.104,
        "expiry": "2026-07-22",
        "contract_multiplier": 10180,
        "call_contract_symbol": "XD500ETF购7月8104A",
        "put_contract_symbol": "XD500ETF沽7月8104A",
        "entry_call_price": 0.181728880157,
        "entry_put_price": 0.156385068762,
        "entry_option_value": 34420.0,
        "option_margin": 100000.0,
        "underlying_order_book_id": "510500.XSHG",
    }


def _account():
    return account.AccountState(
        product="500etf",
        cash=500000.0,
        positions={"long": None, "short": _adjusted_short_position()},
        hedge=account.HedgeState(
            qty=1000.0,
            entry_price=8.0,
            margin=8000.0,
            underlying_order_book_id="510500.XSHG",
        ),
    )


def _empty_greeks(delta=0.0):
    result = backtester.empty_greeks()
    result["delta"] = delta
    return result


def test_detects_real_500etf_adjusted_contract_after_position_terms_are_updated():
    adjustments = signal_engine._dividend_adjustments_for_positions(
        {"long": None, "short": _adjusted_short_position()},
        _adjusted_chain(),
        default_multiplier=10000,
    )

    assert len(adjustments) == 1
    adjustment = adjustments[0]
    assert adjustment["side"] == "short"
    assert adjustment["current_call_strike"] == 8.104
    assert adjustment["current_call_contract_multiplier"] == 10180
    assert adjustment["evidence"]["adjusted_contract_symbol"] is True
    assert adjustment["evidence"]["nonstandard_contract_multiplier"] is True


def test_forced_plan_closes_options_then_etf_and_settles_with_adjusted_multiplier():
    live_account = _account()
    chain = _adjusted_chain()
    adjustments = signal_engine._dividend_adjustments_for_positions(
        live_account.positions,
        chain,
        default_multiplier=10000,
    )
    config = SimpleNamespace(vol=SimpleNamespace(contract_multiplier=10000))

    with mock.patch.object(
        signal_engine.core.strategy,
        "calc_position_greeks",
        return_value=_empty_greeks(delta=-2500.0),
    ):
        advice, _, _, ready = signal_engine._dividend_forced_liquidation_plan(
            config,
            live_account,
            chain,
            8.12,
            None,
            pd.Timestamp("2026-07-15"),
            adjustments,
        )

    assert ready is True
    assert [item["action"] for item in advice] == [
        "CLOSE_SHORT_STRADDLE",
        "FINAL_DELTA_HEDGE",
    ]
    assert advice[0]["contract_multiplier"] == 10180
    assert advice[1]["target_hedge_qty"] == 0
    assert advice[1]["trade_etf_qty"] == -1000
    assert all(item["dividend_forced_liquidation"] for item in advice)

    payload = {
        "product": "500etf",
        "account_id": "default",
        "date": "2026-07-15",
        "account": live_account.to_dict(),
        "advice": advice,
    }
    orders = infinitrader.compile_signal_orders(payload)
    assert [
        (order["instrument_id"], order["order_direction"], order["offset"])
        for order in orders
    ] == [
        ("10011720", "buy", "1"),
        ("10011729", "buy", "1"),
        ("510500", "sell", None),
    ]

    fills = infinitrader.build_fills_from_command(
        {"product": "500etf", "date": "2026-07-15", "signal": payload}
    )
    assert [fill["action"] for fill in fills] == [
        "close_short_straddle",
        "delta_hedge",
    ]
    assert fills[0]["contract_multiplier"] == 10180

    settlement = _account()
    for fill in fills:
        account._apply_fill(settlement, "500etf", fill)
    assert settlement.positions["short"] is None
    assert settlement.hedge.qty == 0


def test_forced_close_first_books_zero_cash_adjustment_when_account_has_old_terms():
    live_account = _account()
    position = live_account.positions["short"]
    position.update(
        {
            "strike": 8.25,
            "contract_multiplier": 10000,
            "call_contract_symbol": "500ETF购7月8250",
            "put_contract_symbol": "500ETF沽7月8250",
            "entry_call_price": 0.185,
            "entry_put_price": 0.1592,
        }
    )
    chain = _adjusted_chain()
    adjustments = signal_engine._dividend_adjustments_for_positions(
        live_account.positions,
        chain,
        default_multiplier=10000,
    )
    config = SimpleNamespace(vol=SimpleNamespace(contract_multiplier=10000))
    with mock.patch.object(
        signal_engine.core.strategy,
        "calc_position_greeks",
        return_value=_empty_greeks(delta=-2500.0),
    ):
        advice, _, _, ready = signal_engine._dividend_forced_liquidation_plan(
            config,
            live_account,
            chain,
            8.12,
            None,
            pd.Timestamp("2026-07-15"),
            adjustments,
        )

    assert ready is True
    pre_close = advice[0]["pre_close_contract_adjustment"]
    assert pre_close["old_strike"] == 8.25
    assert pre_close["new_strike"] == 8.104
    assert pre_close["old_contract_multiplier"] == 10000
    assert pre_close["new_contract_multiplier"] == 10180
    assert pre_close["cash_delta"] == 0.0
    assert abs(pre_close["entry_call_price"] * 10180 - 0.185 * 10000) < 1e-9

    payload = {
        "product": "500etf",
        "date": "2026-07-15",
        "account": live_account.to_dict(),
        "advice": advice,
    }
    fills = infinitrader.build_fills_from_command(
        {"product": "500etf", "date": "2026-07-15", "signal": payload}
    )
    assert [fill["action"] for fill in fills] == [
        "option_contract_adjustment",
        "close_short_straddle",
        "delta_hedge",
    ]
    settlement = live_account
    initial_cash = settlement.cash
    account._apply_fill(settlement, "500etf", fills[0])
    assert settlement.cash == initial_cash
    assert settlement.positions["short"]["contract_multiplier"] == 10180
    account._apply_fill(settlement, "500etf", fills[1])
    account._apply_fill(settlement, "500etf", fills[2])
    assert settlement.positions["short"] is None
    assert settlement.hedge.qty == 0


def test_generate_signal_gives_dividend_liquidation_priority_over_other_logic():
    live_account = _account()
    chain = _adjusted_chain()
    date = pd.Timestamp("2026-07-15")
    market = {
        "date": date,
        "signal_row": pd.Series(
            {
                "close": 8.12,
                "atm_strike": 8.104,
                "long_open_signal": True,
                "short_open_signal": True,
            }
        ),
        "signals": pd.DataFrame(index=pd.DatetimeIndex([date])),
        "chain_df": chain,
        "data_warning": {},
    }
    config = SimpleNamespace(
        data=SimpleNamespace(product="500etf"),
        vol=SimpleNamespace(contract_multiplier=10000),
        strategy=SimpleNamespace(
            short_stop_loss_enabled=False,
            delta_hedge_tolerance_ratio=0.05,
        ),
    )
    atm = {
        "strike": 8.104,
        "expiry": pd.Timestamp("2026-07-22"),
        "call": chain.iloc[0],
        "put": chain.iloc[1],
        "underlying_order_book_id": "510500.XSHG",
    }

    with (
        mock.patch.object(signal_engine, "load_product_config", return_value=config),
        mock.patch.object(signal_engine.account_store, "load_account", return_value=live_account),
        mock.patch.object(signal_engine.portfolio_account, "shared_cash", return_value=500000.0),
        mock.patch.object(signal_engine, "_load_market_context", return_value=market),
        mock.patch.object(signal_engine.core.vol_engine, "select_atm_from_chain", return_value=atm),
        mock.patch.object(
            signal_engine.core.strategy,
            "calc_position_greeks",
            return_value=_empty_greeks(delta=-2500.0),
        ),
        mock.patch.object(
            signal_engine.core.strategy,
            "normalized_account_delta",
            return_value=(-0.15, 10000.0),
        ),
        mock.patch.object(
            signal_engine,
            "_live_capacity_reduction_item",
            side_effect=AssertionError("capacity logic must not override dividend liquidation"),
        ),
        mock.patch.object(
            signal_engine,
            "_advice_for_existing_position",
            side_effect=AssertionError("roll/close strategy must not run"),
        ),
        mock.patch.object(
            signal_engine,
            "_entry_advice",
            side_effect=AssertionError("entry logic must not run"),
        ),
    ):
        payload = signal_engine.generate_signal(
            "500etf",
            quote_snapshot={"quote_date": "2026-07-15"},
        )

    assert payload["dividend_forced_liquidation"]["active"] is True
    assert payload["dividend_forced_liquidation"]["ready"] is True
    assert [item["action"] for item in payload["advice"]] == [
        "CLOSE_SHORT_STRADDLE",
        "FINAL_DELTA_HEDGE",
    ]
    assert payload["planned_account_greeks"]["delta"] == 0.0


def test_generate_signal_does_not_reopen_after_same_day_dividend_settlement():
    live_account = account.AccountState(product="500etf", cash=500000.0)
    date = pd.Timestamp("2026-07-15")
    chain = _adjusted_chain()
    market = {
        "date": date,
        "signal_row": pd.Series(
            {
                "close": 8.12,
                "atm_strike": 8.104,
                "long_open_signal": True,
                "short_open_signal": True,
            }
        ),
        "signals": pd.DataFrame(index=pd.DatetimeIndex([date])),
        "chain_df": chain,
        "data_warning": {},
    }
    config = SimpleNamespace(
        data=SimpleNamespace(product="500etf"),
        vol=SimpleNamespace(contract_multiplier=10000),
        strategy=SimpleNamespace(
            short_stop_loss_enabled=False,
            delta_hedge_tolerance_ratio=0.05,
        ),
    )
    atm = {
        "strike": 8.104,
        "expiry": pd.Timestamp("2026-07-22"),
        "call": chain.iloc[0],
        "put": chain.iloc[1],
        "underlying_order_book_id": "510500.XSHG",
    }
    recorded = [
        {
            "side": "short",
            "call_code": "10011720",
            "put_code": "10011729",
            "source": "account_fill",
        }
    ]

    with (
        mock.patch.object(signal_engine, "load_product_config", return_value=config),
        mock.patch.object(signal_engine.account_store, "load_account", return_value=live_account),
        mock.patch.object(signal_engine.portfolio_account, "shared_cash", return_value=500000.0),
        mock.patch.object(signal_engine, "_load_market_context", return_value=market),
        mock.patch.object(signal_engine.core.vol_engine, "select_atm_from_chain", return_value=atm),
        mock.patch.object(
            signal_engine,
            "_recorded_dividend_adjustments_on_date",
            return_value=recorded,
        ),
        mock.patch.object(
            signal_engine.core.strategy,
            "normalized_account_delta",
            return_value=(0.0, 0.0),
        ),
        mock.patch.object(
            signal_engine,
            "_entry_advice",
            side_effect=AssertionError("same-day reopening must be blocked"),
        ),
        mock.patch.object(
            signal_engine,
            "_live_capacity_reduction_item",
            side_effect=AssertionError("capacity logic must not run"),
        ),
    ):
        payload = signal_engine.generate_signal("500etf")

    assert payload["dividend_forced_liquidation"]["active"] is True
    assert payload["dividend_forced_liquidation"]["detected_from_account_fills"] is True
    assert [item["action"] for item in payload["advice"]] == [
        "DIVIDEND_LIQUIDATION_SETTLED"
    ]
