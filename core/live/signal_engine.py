from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pandas as pd

import core
from . import account as account_store
from . import storage
from .runtime import load_product_config


def generate_signal(product, account_id="default", date=None, quote_snapshot=None, read_only=True):
    config = load_product_config(product)
    live_account = account_store.load_account(product, account_id=account_id)
    market = _load_market_context(
        config,
        date,
        quote_snapshot=quote_snapshot,
        persist_feature_history=False,
    )

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

    strategy_advice = []
    position_greeks = []
    position_greeks_by_side = {}
    position_values = []

    for side, position in live_account.positions.items():
        if position is None:
            continue
        side_advice, greeks, option_value = _advice_for_existing_position(
            product,
            side,
            position,
            chain_df,
            feature_row,
            market["signals"],
            market["date"],
            atm,
            spot,
            live_account.strategy_state,
            config,
        )
        strategy_advice.extend(side_advice)
        if greeks is not None:
            position_greeks.append(greeks)
            position_greeks_by_side[side] = greeks
        if option_value is not None:
            position_values.append(option_value)

    if all(value is None for value in live_account.positions.values()):
        strategy_advice.extend(
            _entry_advice(
                config,
                feature_row,
                atm,
                spot,
                live_account.strategy_state,
            )
        )

    account_greeks = core.backtester.combine_greeks(position_greeks)
    advice, planned_greeks = _build_execution_plan(
        config,
        live_account,
        chain_df,
        spot,
        atm,
        strategy_advice,
        account_greeks,
        position_greeks_by_side,
    )

    if not advice:
        advice.append(
            {
                "action": "NO_ACTION",
                "reason": "No open, close, roll, or hedge signal.",
                "priority": "info",
            }
        )

    return {
        "product": product,
        "account_id": account_id,
        "date": str(market["date"].date()),
        "spot": spot,
        "account": live_account.to_dict(),
        "feature": _feature_summary(feature_row),
        "account_greeks": account_greeks,
        "planned_account_greeks": planned_greeks,
        "account_delta_after_hedge": account_greeks["delta"] + live_account.hedge.qty,
        "estimated_option_value": sum(position_values),
        "strategy_state": live_account.strategy_state.to_dict(),
        "advice": advice,
        "data_warning": {
            **market["data_warning"],
            "read_only_signal": True,
            "state_changed_in_memory": state_changed,
        },
    }


def preview_signal(product, account_id="default", date=None, quote_snapshot=None):
    return generate_signal(
        product,
        account_id=account_id,
        date=date,
        quote_snapshot=quote_snapshot,
        read_only=True,
    )


def _load_market_context(config, date, quote_snapshot=None, persist_feature_history=True):
    start = config.backtest.start
    end = config.backtest.end if date is None else date
    etf_by_date = core.data_loader.load_etf_series(start, end)
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    data_mode = "daily_eod_reference_incremental"

    if quote_snapshot is not None:
        latest_date = pd.Timestamp(quote_snapshot["quote_date"]).normalize()
        etf_by_date[latest_date], latest_opt_by_date = _load_snapshot_quote_series(
            quote_snapshot,
            latest_date,
        )
        trading_calendar = _append_calendar_date(trading_calendar, latest_date)
        data_mode = "live_snapshot_incremental"
    else:
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
            persist=persist_feature_history,
        )
        seeded_history = True

    features = _merge_latest_features(
        config.data.product,
        history,
        latest_features,
        latest_date,
        persist=persist_feature_history,
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
            "mode": data_mode,
            "seeded_feature_history": seeded_history,
            "read_only": not persist_feature_history,
        },
    }


def _load_snapshot_quote_series(quote_snapshot, latest_date):
    etf_path = quote_snapshot.get("etf_snapshot")
    opt_path = quote_snapshot.get("option_snapshot")
    if not etf_path or not opt_path:
        raise ValueError("quote_snapshot must include etf_snapshot and option_snapshot.")

    etf_df = pd.read_parquet(etf_path)
    chain_df = pd.read_parquet(opt_path)
    if "date" not in chain_df.columns:
        chain_df.insert(0, "date", latest_date)
    else:
        chain_df["date"] = pd.to_datetime(chain_df["date"]).fillna(latest_date)
    chain_df = core.data_loader._ensure_option_underlying_id(chain_df)
    return etf_df, {latest_date: chain_df}


def _append_calendar_date(trading_calendar, latest_date):
    dates = pd.DatetimeIndex(pd.to_datetime(trading_calendar)).normalize()
    return pd.DatetimeIndex(sorted(set(dates) | {pd.Timestamp(latest_date).normalize()}))


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


def _seed_feature_history(config, etf_by_date, trading_calendar, start, latest_date, persist=True):
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
    if persist:
        _save_feature_history(config.data.product, features)
    return features


def _merge_latest_features(product, history, latest_features, latest_date, persist=True):
    latest_features = latest_features.copy()
    latest_features.index = pd.to_datetime(latest_features.index)
    latest_row = latest_features.loc[[latest_date]]
    history = history[history.index != latest_date]
    combined = pd.concat([history, latest_row], axis=0).sort_index()
    combined = _refresh_signal_columns(combined)
    if persist:
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


def _historical_strike_mismatch(product, side, position, signals, latest_date):
    latest_date = pd.Timestamp(latest_date).normalize()
    entry_date = _date_or_none(position.get("entry_date"))
    if entry_date is None:
        raise ValueError(
            f"Cannot evaluate roll mismatch for {side}: current position has no entry_date."
        )

    indexed = signals.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed.index)).normalize()
    trading_dates = pd.DatetimeIndex(sorted(set(indexed.index)))
    dates = [
        date
        for date in trading_dates
        if entry_date <= date <= latest_date
    ]
    if not dates:
        raise ValueError(
            f"Cannot evaluate roll mismatch for {side}: no signal rows between "
            f"{entry_date.date()} and {latest_date.date()}."
        )

    snapshots = _holding_snapshot_files_by_date()
    consecutive = 0
    trace = []
    for date in dates:
        date_text = str(date.date())
        path = snapshots.get(date_text)
        if path is None:
            raise ValueError(
                "Missing broker option holding snapshot for roll mismatch check: "
                f"date={date_text}, expected live_hold/实时持仓*.csv. "
                "No strategy_state fallback is used."
            )

        row = indexed.loc[date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        atm_strike = row.get("atm_strike", pd.NA)
        if pd.isna(atm_strike):
            raise ValueError(
                "Missing ATM strike for roll mismatch check: "
                f"date={date_text}. No strategy_state fallback is used."
            )

        if not _holding_snapshot_contains_position(path, position, side):
            raise ValueError(
                "Broker holding snapshot does not contain the current live position "
                "during roll mismatch check: "
                f"date={date_text}, side={side}, "
                f"call={position.get('call_code')}, put={position.get('put_code')}, "
                f"file={path}. No strategy_state fallback is used."
            )

        differs = _strike_differs(position.get("strike"), atm_strike)
        consecutive = consecutive + 1 if differs else 0
        trace.append(
            {
                "date": date_text,
                "holding_file": str(path),
                "position_strike": float(position.get("strike")),
                "atm_strike": float(atm_strike),
                "mismatch": bool(differs),
                "consecutive_mismatch_days": int(consecutive),
            }
        )

    return {"days": int(consecutive), "trace": trace}


def _holding_snapshot_files_by_date():
    live_hold_dir = storage.PROJECT_ROOT / "live_hold"
    files_by_date = {}
    for path in sorted(live_hold_dir.glob("实时持仓*.csv"), key=lambda item: item.stat().st_mtime):
        date_text = _date_from_filename(path)
        if date_text is None:
            continue
        files_by_date[date_text] = path
    return files_by_date


def _holding_snapshot_contains_position(path, position, side):
    df = _read_export_csv(path)
    required = {"合约代码", "总持仓"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Holding snapshot {path} missing columns: {sorted(missing)}")

    call_code = str(position.get("call_code"))
    put_code = str(position.get("put_code"))
    call_ok = False
    put_ok = False
    for _, row in df.iterrows():
        code = str(row.get("合约代码", "")).strip()
        qty = _number(row.get("总持仓"), 0.0) or 0.0
        if qty <= 0:
            continue
        row_side = _side_from_holding_snapshot_row(row)
        if row_side != side:
            continue
        if code == call_code:
            call_ok = True
        if code == put_code:
            put_ok = True
    return call_ok and put_ok


def _side_from_holding_snapshot_row(row):
    buy_sell = str(row.get("买卖", "")).strip()
    position_type = str(row.get("持仓类型", "")).strip()
    if "卖" in buy_sell or "义务" in position_type:
        return "short"
    return "long"


def _read_export_csv(path):
    for encoding in ["utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _number(value, default=None):
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "待设置", "全部"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _date_from_filename(path):
    match = Path(path).name
    parsed = pd.Series([match]).str.extract(r"(20\d{2})_(\d{2})_(\d{2})").iloc[0]
    if parsed.isna().any():
        return None
    return "-".join(str(item) for item in parsed)


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
    product,
    side,
    position,
    chain_df,
    feature_row,
    signals,
    latest_date,
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
    else:
        roll_payload = _roll_payload(
            config,
            product,
            side,
            position,
            chain_df,
            feature_row,
            signals,
            latest_date,
            atm,
            spot,
            position_dte,
            strategy_state,
        )
        if roll_payload:
            item = {
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
            if item.get("target_call_code") is None or item.get("target_put_code") is None:
                item["action"] = "DATA_WARNING"
                item["priority"] = "warning"
            advice.append(item)

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
    product,
    side,
    position,
    chain_df,
    feature_row,
    signals,
    latest_date,
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
        raise ValueError(
            "Missing current ATM strike for roll mismatch check. "
            "No strategy_state fallback is used."
        )

    dte_too_low = position_dte <= config.strategy.roll_dte_threshold
    mismatch = _historical_strike_mismatch(
        product,
        side,
        position,
        signals,
        latest_date,
    )
    mismatch_days = mismatch["days"]
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
            "strike_mismatch_days_source": "broker_holding_history",
            "strike_mismatch_trace": mismatch["trace"],
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
        "strike_mismatch_days_source": "broker_holding_history",
        "strike_mismatch_trace": mismatch["trace"],
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


def _build_execution_plan(
    config,
    live_account,
    chain_df,
    spot,
    atm,
    strategy_advice,
    account_greeks,
    current_greeks_by_side,
):
    plan = list(strategy_advice)
    option_actions = [
        item
        for item in plan
        if item.get("priority") == "action"
        and _is_option_execution_action(item.get("action"))
    ]
    if not option_actions:
        plan.extend(_hedge_advice(config, live_account, account_greeks, spot, atm))
        return plan, account_greeks

    planned_greeks = _project_greeks_after_plan(
        option_actions,
        chain_df,
        current_greeks_by_side,
    )
    if planned_greeks is None:
        return plan, account_greeks

    final_hedge = _final_hedge_advice(
        config,
        live_account,
        planned_greeks,
        spot,
        option_actions,
        chain_df,
        atm,
    )
    if final_hedge is not None:
        plan.append(final_hedge)
    return plan, planned_greeks


def _is_option_execution_action(action):
    if action is None:
        return False
    return (
        action.startswith("OPEN_")
        or action.startswith("ROLL_")
        or action.startswith("CLOSE_")
    )


def _project_greeks_after_plan(option_actions, chain_df, current_greeks_by_side):
    projected_by_side = dict(current_greeks_by_side)
    for item in option_actions:
        action = item.get("action")
        side = item.get("side")
        if side not in account_store.POSITION_SIDES:
            continue

        if action.startswith("CLOSE_"):
            projected_by_side.pop(side, None)
            continue

        leg_fields = _planned_leg_fields(item)
        if leg_fields is None:
            return None
        call_code, put_code, call_qty, put_qty = leg_fields
        try:
            call_row = _chain_row(chain_df, call_code)
            put_row = _chain_row(chain_df, put_code)
        except IndexError:
            return None

        projected_by_side[side] = core.strategy.calc_position_greeks(
            call_row,
            put_row,
            int(call_qty or 0),
            int(put_qty or 0),
            side=side,
        )
    return core.backtester.combine_greeks(projected_by_side.values())


def _planned_leg_fields(item):
    action = item.get("action")
    if action.startswith("OPEN_"):
        return (
            item.get("call_code"),
            item.get("put_code"),
            item.get("call_qty"),
            item.get("put_qty"),
        )
    if action.startswith("ROLL_"):
        return (
            item.get("target_call_code"),
            item.get("target_put_code"),
            item.get("target_call_qty"),
            item.get("target_put_qty"),
        )
    return None


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


def _final_hedge_advice(
    config,
    live_account,
    planned_greeks,
    spot,
    option_actions,
    chain_df,
    atm,
):
    if not config.strategy.enable_delta_hedge:
        return None

    planned_option_delta = float(planned_greeks["delta"])
    planned_account_delta = planned_option_delta + live_account.hedge.qty
    tolerance = max(1.0, abs(planned_option_delta) * 0.05)
    if abs(planned_account_delta) <= tolerance:
        return None

    target_qty = -planned_option_delta
    trade_qty = target_qty - live_account.hedge.qty
    return {
        "action": "FINAL_DELTA_HEDGE",
        "priority": "action",
        "reason": (
            "After executing the option plan, projected account delta exceeds "
            "tolerance."
        ),
        "after_actions": [item.get("action") for item in option_actions],
        "planned_option_delta": planned_option_delta,
        "current_hedge_qty": live_account.hedge.qty,
        "planned_account_delta": planned_account_delta,
        "target_hedge_qty": target_qty,
        "trade_etf_qty": trade_qty,
        "estimated_price": spot,
        "underlying_order_book_id": _underlying_id_after_plan(
            option_actions,
            chain_df,
            atm,
        ),
        "planned_account_gamma": planned_greeks["gamma"],
        "planned_account_vega": planned_greeks["vega"],
        "planned_account_theta": planned_greeks["theta"],
        "hedge_tolerance": tolerance,
    }


def _underlying_id_after_plan(option_actions, chain_df, atm):
    for item in reversed(option_actions):
        leg_fields = _planned_leg_fields(item)
        if leg_fields is None:
            continue
        call_code, put_code, _, _ = leg_fields
        if call_code is None or put_code is None:
            continue
        try:
            return _projected_underlying_id(
                _chain_row(chain_df, call_code),
                _chain_row(chain_df, put_code),
            )
        except IndexError:
            continue
    if atm is not None:
        return _projected_underlying_id(atm.get("call"), atm.get("put"))
    return None


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
