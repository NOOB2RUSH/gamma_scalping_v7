import pandas as pd

from . import hedge, position as opt_position, strategy, vol_engine
from .config import CONFIG


def empty_greeks():
    return {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "call_iv": None,
        "put_iv": None,
        "position_iv": None,
        "call_delta": 0.0,
        "put_delta": 0.0,
        "call_gamma": 0.0,
        "put_gamma": 0.0,
        "call_vega": 0.0,
        "put_vega": 0.0,
        "call_theta": 0.0,
        "put_theta": 0.0,
    }


def execute_delta_hedge(
    date,
    cash,
    greeks,
    hedge_etf_qty,
    hedge_entry_price,
    hedge_margin,
    spot,
    trades,
    target_qty=None,
    trade_type="delta_hedge",
    etf_fee_rate=None,
):
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate

    if target_qty is None:
        target_qty = -greeks["delta"]

    if hedge_etf_qty == target_qty:
        return cash, hedge_etf_qty, hedge_entry_price, hedge_margin

    old_qty = hedge_etf_qty
    cash, hedge_etf_qty, hedge_entry_price, hedge_margin, hedge_pnl = (
        hedge.rebalance_etf_hedge(
            cash,
            hedge_etf_qty,
            hedge_entry_price,
            hedge_margin,
            target_qty,
            spot,
        )
    )
    trade_qty = hedge_etf_qty - old_qty
    etf_fee = abs(trade_qty) * spot * etf_fee_rate
    cash -= etf_fee
    trades.append(
        {
            "date": date,
            "type": trade_type,
            "old_etf_qty": old_qty,
            "new_etf_qty": hedge_etf_qty,
            "trade_etf_qty": trade_qty,
            "etf_qty": trade_qty,
            "price": spot,
            "hedge_pnl": hedge_pnl,
            "fee": etf_fee,
        }
    )
    return cash, hedge_etf_qty, hedge_entry_price, hedge_margin


def build_daily_record(
    date,
    spot,
    cash,
    option_value,
    hedge_etf_qty,
    hedge_entry_price,
    hedge_margin,
    hedge_unrealized_pnl,
    nav,
    etf_fee,
    position,
    greeks,
    feature_row,
):
    eod_has_position = position is not None
    eod_position_dte = (
        int((position["expiry"] - date).days) if eod_has_position else None
    )
    account_delta = greeks["delta"] + hedge_etf_qty

    return {
        "date": date,
        "spot": spot,
        "cash": cash,
        "option_value": option_value,
        "hedge_etf_qty": hedge_etf_qty,
        "hedge_entry_price": hedge_entry_price,
        "hedge_margin": hedge_margin,
        "hedge_unrealized_pnl": hedge_unrealized_pnl,
        "nav": nav,
        "etf_fee": etf_fee,
        "eod_has_position": eod_has_position,
        "eod_position_call_code": (
            position["call_code"] if eod_has_position else None
        ),
        "eod_position_put_code": position["put_code"] if eod_has_position else None,
        "eod_position_strike": position["strike"] if eod_has_position else None,
        "eod_position_expiry": position["expiry"] if eod_has_position else None,
        "eod_position_dte": eod_position_dte,
        "eod_position_call_qty": position["call_qty"] if eod_has_position else 0,
        "eod_position_put_qty": position["put_qty"] if eod_has_position else 0,
        "account_delta": account_delta,
        "account_gamma": greeks["gamma"],
        "account_vega": greeks["vega"],
        "account_theta": greeks["theta"],
        "eod_position_iv": greeks["position_iv"],
        "atm_iv": feature_row["atm_iv"],
        "open_signal": feature_row["open_signal"],
    }


def add_greeks_pnl(daily_df):
    df = daily_df.copy()
    same_position = (
        df["eod_has_position"]
        & df["eod_has_position"].shift(1).fillna(False)
        & (df["eod_position_call_code"] == df["eod_position_call_code"].shift(1))
        & (df["eod_position_put_code"] == df["eod_position_put_code"].shift(1))
    )

    spot_chg = df["spot"].diff()
    df["delta_pnl"] = df["account_delta"].shift(1) * spot_chg
    df["gamma_pnl"] = 0.5 * df["account_gamma"].shift(1) * spot_chg**2
    df["vega_pnl"] = df["account_vega"].shift(1) * df["eod_position_iv"].diff() * 100
    df["theta_pnl"] = df["account_theta"].shift(1)

    pnl_cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]
    df.loc[~same_position, pnl_cols] = 0.0
    df["greeks_pnl"] = df[pnl_cols].sum(axis=1, min_count=len(pnl_cols))
    return df


def run_backtest(
    etf_by_date,
    opt_by_date,
    signals_df,
    initial_cash=None,
    call_qty=None,
    put_qty=None,
    etf_fee_rate=None,
):
    if initial_cash is None:
        initial_cash = CONFIG.backtest.initial_cash
    if call_qty is None:
        call_qty = CONFIG.backtest.call_qty
    if put_qty is None:
        put_qty = CONFIG.backtest.put_qty
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate

    daily_ohlc = vol_engine.build_daily_ohlc_df(etf_by_date)
    cash = initial_cash
    position = None
    hedge_etf_qty = 0.0
    hedge_entry_price = 0.0
    hedge_margin = 0.0
    trades = []
    daily_records = []

    for date in signals_df.index:
        if date not in daily_ohlc.index:
            continue

        spot = daily_ohlc.loc[date, "close"]
        feature_row = signals_df.loc[date]

        if date not in opt_by_date:
            if position is None:
                continue
            cash, _ = opt_position.close_at_last_value(
                date,
                cash,
                position,
                trades,
                exit_reason="missing_option_chain_last_price",
            )
            cash, hedge_etf_qty, hedge_entry_price, hedge_margin = (
                execute_delta_hedge(
                    date,
                    cash,
                    greeks={"delta": 0.0},
                    hedge_etf_qty=hedge_etf_qty,
                    hedge_entry_price=hedge_entry_price,
                    hedge_margin=hedge_margin,
                    spot=spot,
                    trades=trades,
                    target_qty=0.0,
                    trade_type="close_hedge",
                    etf_fee_rate=etf_fee_rate,
                )
            )
            position = None
            continue

        chain_df = vol_engine.add_iv_for_day(opt_by_date[date], spot)
        chain_df = vol_engine.add_greeks_for_day(chain_df, spot)

        option_value = 0.0
        greeks = empty_greeks()
        daily_etf_fee = 0.0

        def hedge_to(greeks, target_qty=None, trade_type="delta_hedge"):
            nonlocal cash, hedge_etf_qty, hedge_entry_price, hedge_margin
            nonlocal daily_etf_fee

            trade_count = len(trades)
            cash, hedge_etf_qty, hedge_entry_price, hedge_margin = (
                execute_delta_hedge(
                    date,
                    cash,
                    greeks,
                    hedge_etf_qty,
                    hedge_entry_price,
                    hedge_margin,
                    spot,
                    trades,
                    target_qty=target_qty,
                    trade_type=trade_type,
                    etf_fee_rate=etf_fee_rate,
                )
            )
            if len(trades) > trade_count:
                daily_etf_fee += trades[-1].get("fee", 0.0)

        def record_day():
            hedge_unrealized_pnl = hedge.calc_unrealized_pnl(
                hedge_etf_qty,
                hedge_entry_price,
                spot,
            )
            nav = cash + option_value + hedge_margin + hedge_unrealized_pnl
            daily_records.append(
                build_daily_record(
                    date,
                    spot,
                    cash,
                    option_value,
                    hedge_etf_qty,
                    hedge_entry_price,
                    hedge_margin,
                    hedge_unrealized_pnl,
                    nav,
                    daily_etf_fee,
                    position,
                    greeks,
                    feature_row,
                )
            )

        if pd.isna(feature_row["atm_iv"]) and position is None:
            continue

        if position is not None:
            try:
                call_row, put_row = opt_position.find_rows(position, chain_df)
            except IndexError:
                cash, _ = opt_position.close_at_last_value(
                    date,
                    cash,
                    position,
                    trades,
                )
                hedge_to(greeks, target_qty=0.0, trade_type="close_hedge")
                position = None
                option_value = 0.0
                greeks = empty_greeks()
                record_day()
                continue

            position_dte = int(call_row["dte"])
            greeks = strategy.calc_position_greeks(
                call_row,
                put_row,
                position["call_qty"],
                position["put_qty"],
            )
            if pd.isna(greeks["position_iv"]):
                cash, _ = opt_position.close_trade(
                    date,
                    cash,
                    position,
                    call_row,
                    put_row,
                    trades,
                    exit_reason="missing_position_iv",
                )
                hedge_to(empty_greeks(), target_qty=0.0, trade_type="close_hedge")
                position = None
                option_value = 0.0
                greeks = empty_greeks()
                record_day()
                continue

            close_reason = strategy.get_close_reason(
                greeks["position_iv"],
                position_dte,
            )
            roll_signal = (
                pd.notna(feature_row["atm_iv"])
                and strategy.should_roll_position(
                    feature_row,
                    position_dte,
                    position["strike"],
                    spot,
                )
            )

            if close_reason is not None:
                cash, _ = opt_position.close_trade(
                    date,
                    cash,
                    position,
                    call_row,
                    put_row,
                    trades,
                    exit_reason=close_reason,
                )
                hedge_to(greeks, target_qty=0.0, trade_type="close_hedge")
                position = None
                option_value = 0.0
                greeks = empty_greeks()

            elif roll_signal:
                atm = vol_engine.calc_atm_iv_for_day(opt_by_date[date], spot)
                cash, _ = opt_position.close_trade(
                    date,
                    cash,
                    position,
                    call_row,
                    put_row,
                    trades,
                    trade_type="roll_close_straddle",
                )
                cash, position, option_value = opt_position.open_trade(
                    date,
                    cash,
                    atm,
                    call_qty,
                    put_qty,
                    trades,
                    trade_type="roll_open_straddle",
                )
                greeks = strategy.calc_position_greeks(
                    atm["call"],
                    atm["put"],
                    call_qty,
                    put_qty,
                )
                hedge_to(greeks)

            else:
                option_value = opt_position.value(position, call_row, put_row)
                position["last_option_value"] = option_value
                hedge_to(greeks)

        if position is None and feature_row["open_signal"]:
            atm = vol_engine.calc_atm_iv_for_day(opt_by_date[date], spot)
            cash, position, option_value = opt_position.open_trade(
                date,
                cash,
                atm,
                call_qty,
                put_qty,
                trades,
                trade_type="open_straddle",
            )
            greeks = strategy.calc_position_greeks(
                atm["call"],
                atm["put"],
                call_qty,
                put_qty,
            )
            hedge_to(greeks)

        if position is not None:
            call_row, put_row = opt_position.find_rows(position, chain_df)
            option_value = opt_position.value(position, call_row, put_row)
            position["last_option_value"] = option_value
            greeks = strategy.calc_position_greeks(
                call_row,
                put_row,
                position["call_qty"],
                position["put_qty"],
            )

        record_day()

    daily_df = pd.DataFrame(daily_records).set_index("date")
    daily_df = add_greeks_pnl(daily_df)
    trades_df = pd.DataFrame(trades)
    return daily_df, trades_df
