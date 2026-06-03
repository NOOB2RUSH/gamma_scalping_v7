from __future__ import annotations

import math
import os
import time

import pandas as pd

import core
from . import account as account_store
from . import storage
from .runtime import load_product_config


def generate_signal(product, account_id="default", date=None):
    config = load_product_config(product)
    live_account = account_store.load_account(product, account_id=account_id)
    market = _load_market_context(config, date)

    feature_row = market["signal_row"]
    spot = float(feature_row["close"])
    chain_df = market["chain_df"]
    atm = core.vol_engine.select_atm_from_chain(chain_df, spot)
    state_changed = _refresh_strategy_state(
        config,
        live_account,
        market["signals"],
        market["date"],
    )

    advice = []
    position_greeks = []
    position_greeks_by_side = {}
    position_values = []

    for side, position in live_account.positions.items():
        if position is None:
            continue
        side_advice, greeks, option_value = _advice_for_existing_position(
            side,
            position,
            chain_df,
            feature_row,
            atm,
            spot,
            live_account.strategy_state,
            config,
        )
        advice.extend(side_advice)
        if greeks is not None:
            position_greeks.append(greeks)
            position_greeks_by_side[side] = greeks
        if option_value is not None:
            position_values.append(option_value)

    if all(value is None for value in live_account.positions.values()):
        advice.extend(
            _entry_advice(
                config,
                feature_row,
                atm,
                spot,
                live_account.strategy_state,
            )
        )

    account_greeks = core.backtester.combine_greeks(position_greeks)
    advice.extend(_hedge_advice(config, live_account, account_greeks, spot, atm))
    advice.extend(
        _projected_hedge_advices(
            config,
            live_account,
            chain_df,
            spot,
            advice,
            position_greeks_by_side,
        )
    )

    if not advice:
        advice.append(
            {
                "action": "NO_ACTION",
                "reason": "No open, close, roll, or hedge signal.",
                "priority": "info",
            }
        )

    if state_changed:
        account_store.save_account(live_account)

    return {
        "product": product,
        "account_id": account_id,
        "date": str(market["date"].date()),
        "spot": spot,
        "account": live_account.to_dict(),
        "feature": _feature_summary(feature_row),
        "account_greeks": account_greeks,
        "account_delta_after_hedge": account_greeks["delta"] + live_account.hedge.qty,
        "estimated_option_value": sum(position_values),
        "strategy_state": live_account.strategy_state.to_dict(),
        "advice": advice,
        "data_warning": market["data_warning"],
    }


def _load_market_context(config, date):
    start = config.backtest.start
    end = config.backtest.end if date is None else date
    etf_by_date = core.data_loader.load_etf_series(start, end)
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    latest_date = _resolve_latest_signal_date(config, etf_by_date, date)

    latest_opt_by_date = core.data_loader.load_opt_series(latest_date, latest_date)
    latest_hedge_by_date = core.data_loader.load_hedge_series(
        latest_date,
        latest_date,
    )
    latest_opt_by_date = core.data_loader.attach_underlying_prices(
        latest_opt_by_date,
        latest_hedge_by_date,
    )
    latest_enriched = _build_latest_enriched_chain(
        etf_by_date,
        latest_opt_by_date,
        trading_calendar,
        latest_date,
    )
    latest_features = core.vol_engine.build_vol_features(
        etf_by_date,
        latest_opt_by_date,
        trading_calendar=trading_calendar,
        enriched_opt_by_date={latest_date: latest_enriched},
    )

    seeded_history = False
    history = _load_feature_history(config.data.product)
    if history is None:
        history = _seed_feature_history(
            config,
            etf_by_date,
            trading_calendar,
            start,
            latest_date,
        )
        seeded_history = True

    features = _merge_latest_features(
        config.data.product,
        history,
        latest_features,
        latest_date,
    )
    signals = core.strategy.build_signals(features)
    if latest_date not in signals.index:
        raise ValueError(f"No signal row for {latest_date.date()}")

    return {
        "date": latest_date,
        "chain_df": latest_enriched,
        "signal_row": signals.loc[latest_date],
        "signals": signals,
        "data_warning": {
            "latest_signal_date": str(latest_date.date()),
            "mode": "daily_eod_reference_incremental",
            "seeded_feature_history": seeded_history,
        },
    }


def _resolve_latest_signal_date(config, etf_by_date, date):
    if date is not None:
        return pd.Timestamp(date).normalize()

    opt_dir = core.data_loader._resolve_data_dir(config.data.opt_dir)
    opt_dates = [
        core.data_loader._parse_date_from_file(path, "_chain")
        for path in sorted(opt_dir.glob("*_chain.parquet"))
    ]
    common_dates = sorted(set(etf_by_date) & set(opt_dates))
    if not common_dates:
        raise ValueError("No common ETF/option date available for live signal.")
    return common_dates[-1]


def _build_latest_enriched_chain(etf_by_date, latest_opt_by_date, trading_calendar, latest_date):
    daily_ohlc = core.vol_engine.build_daily_ohlc_df(etf_by_date)
    spot = daily_ohlc.loc[latest_date, "close"]
    chain = core.vol_engine.add_iv_for_day(
        latest_opt_by_date[latest_date],
        spot,
        trading_calendar=trading_calendar,
    )
    return core.vol_engine.add_greeks_for_day(chain, spot)


def _load_feature_history(product):
    path = storage.feature_history_path(product)
    if not path.exists():
        return None
    try:
        history = pd.read_parquet(path)
    except Exception:
        return None
    if "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"])
        history = history.set_index("date")
    history.index = pd.to_datetime(history.index)
    return history.sort_index()


def _save_feature_history(product, features):
    path = storage.feature_history_path(product)
    payload = features.reset_index()
    if "index" in payload.columns and "date" not in payload.columns:
        payload = payload.rename(columns={"index": "date"})
    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    payload.to_parquet(tmp_path, index=False)
    for attempt in range(5):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)


def _seed_feature_history(config, etf_by_date, trading_calendar, start, latest_date):
    """Build the rolling live feature store once.

    This may use the slower historical cache on the first live run. Subsequent
    ticks update only the latest row in `state/live/<product>/feature_history`.
    """
    opt_by_date = core.data_loader.load_opt_series(start, latest_date)
    hedge_by_date = core.data_loader.load_hedge_series(start, latest_date)
    opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        start,
        latest_date,
    )
    features = core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched,
        start,
        latest_date,
    )
    _save_feature_history(config.data.product, features)
    return features


def _merge_latest_features(product, history, latest_features, latest_date):
    latest_features = latest_features.copy()
    latest_features.index = pd.to_datetime(latest_features.index)
    latest_row = latest_features.loc[[latest_date]]
    history = history[history.index != latest_date]
    combined = pd.concat([history, latest_row], axis=0).sort_index()
    combined = _refresh_signal_columns(combined)
    _save_feature_history(product, combined)
    return combined


def _refresh_signal_columns(features):
    features = features.copy()
    features["atm_iv_percentile"] = core.vol_engine._calc_atm_iv_percentile(
        features["atm_iv"],
        core.config.CONFIG.vol.atm_iv_percentile_window,
    )
    observation_mode = core.vol_engine._iv_observation_mode()
    if core.vol_engine._surface_signal_enabled():
        features["surface_atm_iv_percentile"] = core.vol_engine._calc_atm_iv_percentile(
            features["surface_atm_iv"],
            core.config.CONFIG.vol.atm_iv_percentile_window,
        )
        features["signal_iv"] = features["surface_atm_iv"]
        features["signal_iv_percentile"] = features["surface_atm_iv_percentile"]
    elif observation_mode == "simple_atm_absolute":
        features["surface_atm_iv_percentile"] = pd.NA
        features["signal_iv"] = features["atm_iv"]
        features["signal_iv_percentile"] = pd.NA
    else:
        features["signal_iv"] = features["atm_iv"]
        features["signal_iv_percentile"] = features["atm_iv_percentile"]
    return features


def _refresh_strategy_state(config, live_account, features, latest_date):
    strategy_state = live_account.strategy_state
    before = strategy_state.to_dict()
    latest_date = pd.Timestamp(latest_date).normalize()
    last_signal_date = _date_or_none(strategy_state.last_signal_date)
    trading_dates = pd.DatetimeIndex(pd.to_datetime(features.index)).normalize()

    for side in account_store.POSITION_SIDES:
        cooldown_left = _cooldown_left_for_date(
            strategy_state,
            side,
            latest_date,
            trading_dates,
        )
        strategy_state.roll_cooldown_left[side] = cooldown_left
        if cooldown_left <= 0:
            strategy_state.cooldown_total_days[side] = 0
            strategy_state.cooldown_started_date[side] = None

        position = live_account.positions.get(side)
        if position is None:
            strategy_state.strike_mismatch_days[side] = 0
            continue

        _update_strike_mismatch_days(
            strategy_state,
            side,
            position,
            features,
            latest_date,
            last_signal_date,
        )

    strategy_state.last_signal_date = str(latest_date.date())
    return before != strategy_state.to_dict()


def _cooldown_left_for_date(strategy_state, side, latest_date, trading_dates):
    total_days = int(strategy_state.cooldown_total_days.get(side, 0) or 0)
    current_left = int(strategy_state.roll_cooldown_left.get(side, 0) or 0)
    started_date = _date_or_none(strategy_state.cooldown_started_date.get(side))
    if total_days <= 0:
        return max(current_left, 0) if started_date is None else 0
    if started_date is None:
        return max(current_left, 0)
    if latest_date <= started_date:
        return total_days

    completed_after_start = int(
        ((trading_dates > started_date) & (trading_dates < latest_date)).sum()
    )
    return max(total_days - completed_after_start, 0)


def _update_strike_mismatch_days(
    strategy_state,
    side,
    position,
    features,
    latest_date,
    last_signal_date,
):
    if last_signal_date is not None and latest_date <= last_signal_date:
        return

    start_date = last_signal_date
    if start_date is None:
        start_date = _date_or_none(position.get("entry_date"))

    indexed = features.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed.index)).normalize()
    if start_date is not None:
        rows = indexed[(indexed.index > start_date) & (indexed.index <= latest_date)]
    else:
        rows = indexed[indexed.index <= latest_date].tail(1)

    for _, row in rows.iterrows():
        atm_strike = row.get("atm_strike", pd.NA)
        if pd.notna(atm_strike) and _strike_differs(position.get("strike"), atm_strike):
            strategy_state.strike_mismatch_days[side] = (
                int(strategy_state.strike_mismatch_days.get(side, 0) or 0) + 1
            )
        else:
            strategy_state.strike_mismatch_days[side] = 0


def _date_or_none(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert(None)
        return timestamp.normalize()
    except (TypeError, ValueError):
        return None


def _strike_differs(left, right):
    try:
        return not math.isclose(float(left), float(right))
    except (TypeError, ValueError):
        return True


def _entry_advice(config, feature_row, atm, spot, strategy_state):
    if atm is None:
        return [
            {
                "action": "DATA_WARNING",
                "reason": "No valid ATM call/put pair found.",
                "priority": "warning",
            }
        ]

    advice = []
    for side, signal_col, action, qty in [
        ("long", "long_open_signal", "OPEN_LONG_STRADDLE", config.backtest.long_qty),
        ("short", "short_open_signal", "OPEN_SHORT_STRADDLE", config.backtest.short_qty),
    ]:
        if not bool(feature_row.get(signal_col, False)):
            continue
        cooldown_left = int(strategy_state.roll_cooldown_left.get(side, 0) or 0)
        if cooldown_left > 0:
            advice.append(
                {
                    "action": "COOLDOWN_BLOCK",
                    "priority": "info",
                    "side": side,
                    "reason": "Entry signal is active but side cooldown is still active.",
                    "cooldown_left": cooldown_left,
                }
            )
            continue
        item = _open_advice(action, side, qty, atm, spot)
        if side == "short":
            item["short_entry_regime"] = feature_row.get("short_open_regime")
        advice.append(item)
    return advice


def _open_advice(action, side, qty, atm, spot):
    position = core.position.open_straddle(
        pd.Timestamp.now(),
        atm,
        call_qty=qty,
        put_qty=qty,
        side=side,
        spot=spot,
        short_entry_regime=atm.get("short_entry_regime"),
    )
    trade_value = core.position.value(position, atm["call"], atm["put"])
    fee = core.position.calc_option_fee(qty, qty)
    cash_effect = trade_value - fee - position["option_margin"] if side == "short" else -trade_value - fee
    return {
        "action": action,
        "priority": "action",
        "side": side,
        "reason": "Entry signal is active.",
        "call_code": atm["call"]["order_book_id"],
        "put_code": atm["put"]["order_book_id"],
        "strike": float(atm["strike"]),
        "expiry": str(pd.Timestamp(atm["expiry"]).date()),
        "call_qty": qty,
        "put_qty": qty,
        "estimated_call_price": float(atm["call"]["mid"]),
        "estimated_put_price": float(atm["put"]["mid"]),
        "estimated_trade_value": float(trade_value),
        "estimated_fee": float(fee),
        "estimated_option_margin": float(position["option_margin"]),
        "estimated_cash_effect": float(cash_effect),
        "liquidity": _liquidity_summary(atm["call"], atm["put"], qty),
    }


def _advice_for_existing_position(
    side,
    position,
    chain_df,
    feature_row,
    atm,
    spot,
    strategy_state,
    config,
):
    try:
        call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
    except IndexError:
        return (
            [
                {
                    "action": "DATA_WARNING",
                    "priority": "warning",
                    "side": side,
                    "reason": "Current position contracts are missing from latest chain.",
                }
            ],
            None,
            None,
        )

    option_value = core.position.value(position, call_row, put_row)
    greeks = core.strategy.calc_position_greeks(
        call_row,
        put_row,
        position["call_qty"],
        position["put_qty"],
        side=side,
    )
    position_dte = int(call_row.get("dte", 0))
    advice = []

    if side == "short":
        close_reason = core.strategy.get_short_close_reason(feature_row, position_dte, position)
        if core.strategy.is_short_stop_loss(position, option_value):
            close_reason = "short_stop_loss"
        if core.position.has_short_volume_spike(position, call_row, put_row):
            close_reason = "short_volume_spike"
    else:
        close_reason = core.strategy.get_close_reason(feature_row, position_dte)

    if close_reason:
        advice.append(_close_advice(side, position, call_row, put_row, option_value, close_reason))

    roll_payload = _roll_payload(
        config,
        side,
        position,
        chain_df,
        feature_row,
        atm,
        spot,
        position_dte,
        strategy_state,
    )
    if roll_payload:
        advice.append(
            {
                "action": "ROLL_SHORT_STRADDLE" if side == "short" else "ROLL_LONG_STRADDLE",
                "priority": "action",
                "side": side,
                "current_call_code": position["call_code"],
                "current_put_code": position["put_code"],
                "current_strike": position["strike"],
                "current_expiry": str(position["expiry"]),
                "current_dte": position_dte,
                **roll_payload,
            }
        )

    return advice, greeks, core.position.signed_value(position, call_row, put_row)


def _close_advice(side, position, call_row, put_row, option_value, reason):
    fee = core.position.calc_option_fee(position["call_qty"], position["put_qty"])
    cash_effect = -option_value - fee + position.get("option_margin", 0.0) if side == "short" else option_value - fee
    return {
        "action": "CLOSE_SHORT_STRADDLE" if side == "short" else "CLOSE_LONG_STRADDLE",
        "priority": "action",
        "side": side,
        "reason": reason,
        "call_code": position["call_code"],
        "put_code": position["put_code"],
        "call_qty": position["call_qty"],
        "put_qty": position["put_qty"],
        "estimated_call_price": float(call_row["mid"]),
        "estimated_put_price": float(put_row["mid"]),
        "estimated_trade_value": float(option_value),
        "estimated_fee": float(fee),
        "estimated_cash_effect": float(cash_effect),
    }


def _roll_payload(
    config,
    side,
    position,
    chain_df,
    feature_row,
    atm,
    spot,
    position_dte,
    strategy_state,
):
    cooldown_left = int(strategy_state.roll_cooldown_left.get(side, 0) or 0)
    if cooldown_left > 0:
        return None

    feature_atm_strike = feature_row.get("atm_strike", pd.NA)
    if pd.isna(feature_atm_strike):
        return None

    dte_too_low = position_dte <= config.strategy.roll_dte_threshold
    mismatch_days = int(strategy_state.strike_mismatch_days.get(side, 0) or 0)
    strike_roll_ready = (
        _strike_differs(position.get("strike"), feature_atm_strike)
        and mismatch_days >= config.strategy.roll_strike_mismatch_days
    )
    if not (dte_too_low or strike_roll_ready):
        return None

    max_qty = config.backtest.short_qty if side == "short" else config.backtest.long_qty
    if _entry_target_qty(feature_row, max_qty, side) <= 0:
        return None

    target_atm = core.vol_engine.select_atm_from_chain(
        chain_df,
        spot,
        target_dte_min=config.strategy.roll_dte_threshold + 1,
    )
    if target_atm is None:
        return {
            "reason": "roll_condition_active_but_no_valid_target_atm",
            "roll_cooldown_left": cooldown_left,
            "strike_mismatch_days": mismatch_days,
            "target_strike": None,
            "target_expiry": None,
        }

    reasons = []
    if dte_too_low:
        reasons.append("dte_below_roll_threshold")
    if strike_roll_ready:
        reasons.append("held_strike_differs_from_current_atm")
    return {
        "reason": "+".join(reasons),
        "roll_cooldown_left": cooldown_left,
        "strike_mismatch_days": mismatch_days,
        "target_call_code": target_atm["call"]["order_book_id"],
        "target_put_code": target_atm["put"]["order_book_id"],
        "target_strike": float(target_atm["strike"]),
        "target_expiry": str(pd.Timestamp(target_atm["expiry"]).date()),
        "target_call_qty": max_qty,
        "target_put_qty": max_qty,
        "estimated_target_call_price": float(target_atm["call"]["mid"]),
        "estimated_target_put_price": float(target_atm["put"]["mid"]),
    }


def _entry_target_qty(feature_row, max_qty, side):
    if side == "short":
        return core.strategy.calc_short_entry_target_qty(feature_row, max_qty)
    return core.strategy.calc_entry_target_qty(feature_row, max_qty)


def _hedge_advice(config, live_account, greeks, spot, atm):
    if not config.strategy.enable_delta_hedge:
        return []
    option_delta = float(greeks["delta"])
    account_delta = option_delta + live_account.hedge.qty
    tolerance = max(1.0, abs(option_delta) * 0.05)
    if abs(account_delta) <= tolerance:
        return []

    target_qty = -option_delta
    trade_qty = target_qty - live_account.hedge.qty
    underlying_order_book_id = None
    if atm is not None:
        underlying_order_book_id = atm.get("underlying_order_book_id")
        if underlying_order_book_id is None:
            underlying_order_book_id = _projected_underlying_id(
                atm.get("call"),
                atm.get("put"),
            )
    return [
        {
            "action": "DELTA_HEDGE",
            "priority": "action",
            "reason": "Account delta exceeds tolerance.",
            "option_delta": option_delta,
            "current_hedge_qty": live_account.hedge.qty,
            "account_delta": account_delta,
            "target_hedge_qty": target_qty,
            "trade_etf_qty": trade_qty,
            "estimated_price": spot,
            "underlying_order_book_id": underlying_order_book_id,
        }
    ]


def _projected_hedge_advices(
    config,
    live_account,
    chain_df,
    spot,
    advice,
    current_greeks_by_side,
):
    if not config.strategy.enable_delta_hedge:
        return []

    result = []
    for item in advice:
        action = item.get("action")
        if action is None or action in {"DELTA_HEDGE", "PROJECTED_DELTA_HEDGE"}:
            continue
        projection = _project_greeks_after_advice(
            item,
            chain_df,
            current_greeks_by_side,
        )
        if projection is None:
            continue
        projected_greeks, target_call_row, target_put_row = projection
        projected = _projected_hedge_advice(
            item,
            live_account,
            projected_greeks,
            spot,
            target_call_row,
            target_put_row,
        )
        if projected is not None:
            result.append(projected)
    return result


def _project_greeks_after_advice(item, chain_df, current_greeks_by_side):
    action = item.get("action")
    side = item.get("side")
    if side not in account_store.POSITION_SIDES:
        return None

    if action.startswith("OPEN_"):
        call_code = item.get("call_code")
        put_code = item.get("put_code")
        call_qty = item.get("call_qty")
        put_qty = item.get("put_qty")
    elif action.startswith("ROLL_"):
        call_code = item.get("target_call_code")
        put_code = item.get("target_put_code")
        call_qty = item.get("target_call_qty")
        put_qty = item.get("target_put_qty")
    elif action.startswith("CLOSE_"):
        projected = [
            greeks
            for item_side, greeks in current_greeks_by_side.items()
            if item_side != side
        ]
        return core.backtester.combine_greeks(projected), None, None
    else:
        return None

    if call_code is None or put_code is None:
        return None
    try:
        call_row = _chain_row(chain_df, call_code)
        put_row = _chain_row(chain_df, put_code)
    except IndexError:
        return None

    projected = [
        greeks
        for item_side, greeks in current_greeks_by_side.items()
        if item_side != side
    ]
    projected.append(
        core.strategy.calc_position_greeks(
            call_row,
            put_row,
            int(call_qty or 0),
            int(put_qty or 0),
            side=side,
        )
    )
    return core.backtester.combine_greeks(projected), call_row, put_row


def _projected_hedge_advice(
    trigger_item,
    live_account,
    projected_greeks,
    spot,
    target_call_row=None,
    target_put_row=None,
):
    projected_option_delta = float(projected_greeks["delta"])
    projected_account_delta = projected_option_delta + live_account.hedge.qty
    tolerance = max(1.0, abs(projected_option_delta) * 0.05)
    if abs(projected_account_delta) <= tolerance:
        return None

    target_qty = -projected_option_delta
    trade_qty = target_qty - live_account.hedge.qty
    return {
        "action": "PROJECTED_DELTA_HEDGE",
        "priority": "action",
        "side": trigger_item.get("side"),
        "reason": (
            "If the preceding strategy action is executed, projected account "
            "delta exceeds tolerance."
        ),
        "trigger_action": trigger_item.get("action"),
        "trigger_side": trigger_item.get("side"),
        "projected_option_delta": projected_option_delta,
        "current_hedge_qty": live_account.hedge.qty,
        "projected_account_delta": projected_account_delta,
        "target_hedge_qty": target_qty,
        "trade_etf_qty": trade_qty,
        "estimated_price": spot,
        "underlying_order_book_id": _projected_underlying_id(
            target_call_row,
            target_put_row,
        ),
        "projected_account_gamma": projected_greeks["gamma"],
        "projected_account_vega": projected_greeks["vega"],
        "projected_account_theta": projected_greeks["theta"],
        "hedge_tolerance": tolerance,
    }


def _chain_row(chain_df, order_book_id):
    matches = chain_df[chain_df["order_book_id"].astype(str) == str(order_book_id)]
    if matches.empty:
        raise IndexError(order_book_id)
    return matches.iloc[0]


def _projected_underlying_id(call_row=None, put_row=None):
    for row in [call_row, put_row]:
        if row is None:
            continue
        value = row.get("underlying_order_book_id")
        if pd.notna(value):
            return value
        value = row.get("akshare_underlying_symbol")
        if pd.notna(value):
            return str(value)
    return None


def _feature_summary(row):
    keys = [
        "close",
        "atm_iv",
        "signal_iv",
        "atm_iv_percentile",
        "signal_iv_percentile",
        "atm_strike",
        "atm_expiry",
        "atm_dte",
        "long_open_signal",
        "short_open_signal",
        "short_open_regime",
    ]
    return {key: _jsonable(row.get(key)) for key in keys if key in row}


def _liquidity_summary(call_row, put_row, qty):
    ratio = core.config.CONFIG.backtest.liquidity_warning_volume_ratio
    result = {}
    for leg, row in {"call": call_row, "put": put_row}.items():
        volume = row.get("volume")
        limit = None if pd.isna(volume) else float(volume) * ratio
        result[f"{leg}_volume"] = _jsonable(volume)
        result[f"{leg}_limit_qty"] = limit
        result[f"{leg}_warning"] = False if limit is None else qty > limit
    return result


def _jsonable(value):
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
