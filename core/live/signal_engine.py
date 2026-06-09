from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pandas as pd

import core
from . import account as account_store
from . import storage
from .runtime import load_product_config


def generate_signal(product, account_id="default", date=None, quote_snapshot=None):
    config = load_product_config(product)
    live_account = account_store.load_account(product, account_id=account_id)
    market = _load_market_context(
        config,
        date,
        quote_snapshot=quote_snapshot,
    )

    feature_row = market["signal_row"]
    spot = float(feature_row["close"])
    chain_df = market["chain_df"]
    atm = core.vol_engine.select_atm_from_chain(chain_df, spot)
    signal_state = _strategy_state_for_signal(
        live_account.strategy_state,
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
            signal_state,
            config,
        )
        strategy_advice.extend(side_advice)
        if greeks is not None:
            position_greeks.append(greeks)
            position_greeks_by_side[side] = greeks
        if option_value is not None:
            position_values.append(option_value)

    for index, hedge_position in enumerate(getattr(live_account, "option_hedges", []) or []):
        greeks, option_value = _option_hedge_greeks_and_value(
            hedge_position,
            chain_df,
        )
        if greeks is not None:
            position_greeks.append(greeks)
            position_greeks_by_side[f"option_hedge:{index}"] = greeks
        if option_value is not None:
            position_values.append(option_value)

    if all(value is None for value in live_account.positions.values()):
        strategy_advice.extend(
            _entry_advice(
                config,
                feature_row,
                atm,
                spot,
                signal_state,
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
        "strategy_state": signal_state.to_dict(),
        "advice": advice,
        "data_warning": {
            **market["data_warning"],
            "read_only_signal": True,
        },
    }


def preview_signal(product, account_id="default", date=None, quote_snapshot=None):
    return generate_signal(
        product,
        account_id=account_id,
        date=date,
        quote_snapshot=quote_snapshot,
    )


def _load_market_context(config, date, quote_snapshot=None):
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
            "mode": data_mode,
            "seeded_feature_history": seeded_history,
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


def _seed_feature_history(config, etf_by_date, trading_calendar, start, latest_date):
    """Build the rolling feature frame in memory when no live history exists."""
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
    return features


def _merge_latest_features(product, history, latest_features, latest_date):
    latest_features = latest_features.copy()
    latest_features.index = pd.to_datetime(latest_features.index)
    latest_row = latest_features.loc[[latest_date]]
    history = history[history.index != latest_date]
    combined = pd.concat([history, latest_row], axis=0).sort_index()
    combined = _refresh_signal_columns(combined)
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


def _strategy_state_for_signal(strategy_state, features, latest_date):
    strategy_state = deepcopy(strategy_state)
    latest_date = pd.Timestamp(latest_date).normalize()
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

    return strategy_state


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


def _historical_strike_mismatch(
    product,
    side,
    position,
    signals,
    latest_date,
    current_atm_strike=None,
):
    latest_date = pd.Timestamp(latest_date).normalize()
    entry_date = _date_or_none(position.get("entry_date"))
    if entry_date is None:
        raise ValueError(
            f"Cannot evaluate roll mismatch for {side}: current position has no entry_date."
        )

    snapshots = _holding_snapshot_files_by_date()
    snapshot_dates = sorted(
        pd.Timestamp(date_text).normalize()
        for date_text in snapshots
        if entry_date <= pd.Timestamp(date_text).normalize() <= latest_date
    )
    if not snapshot_dates:
        raise ValueError(
            "Cannot evaluate roll mismatch: no broker option holding snapshot "
            f"between entry_date={entry_date.date()} and signal_date={latest_date.date()}. "
            "Expected live_hold/实时持仓*.csv. No strategy_state fallback is used."
        )
    latest_snapshot_date = snapshot_dates[-1]

    dates = snapshot_dates
    if not dates:
        raise ValueError(
            f"Cannot evaluate roll mismatch for {side}: no holding snapshots between "
            f"{entry_date.date()} and {latest_snapshot_date.date()}."
        )

    consecutive = 0
    trace = []
    for date in dates:
        date_text = str(date.date())
        path = snapshots.get(date_text)
        atm_strike, atm_source = _atm_strike_for_roll_check(
            product,
            signals,
            date,
            latest_date,
            current_atm_strike,
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
                "atm_source": atm_source,
                "mismatch": bool(differs),
                "consecutive_mismatch_days": int(consecutive),
            }
        )

    return {
        "days": int(consecutive),
        "trace": trace,
        "latest_holding_snapshot_date": str(latest_snapshot_date.date()),
        "signal_date": str(latest_date.date()),
        "snapshot_lag_days": int((latest_date - latest_snapshot_date).days),
    }


def _atm_strike_for_roll_check(
    product,
    signals,
    date,
    latest_date,
    current_atm_strike=None,
):
    date = pd.Timestamp(date).normalize()
    latest_date = pd.Timestamp(latest_date).normalize()
    if date == latest_date and current_atm_strike is not None and not pd.isna(current_atm_strike):
        return float(current_atm_strike), "current_signal_row"

    indexed = signals.copy()
    indexed.index = pd.DatetimeIndex(pd.to_datetime(indexed.index)).normalize()
    if date in indexed.index:
        row = indexed.loc[date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        atm_strike = row.get("atm_strike", pd.NA)
        if not pd.isna(atm_strike):
            return float(atm_strike), "signal_history"

    try:
        etf_by_date = core.data_loader.load_etf_series(date, date)
        opt_by_date = core.data_loader.load_opt_series(date, date)
        try:
            hedge_by_date = core.data_loader.load_hedge_series(date, date)
        except Exception:
            hedge_by_date = None
        opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
        trading_calendar = core.data_loader.load_etf_trading_calendar()
        chain_df = _build_latest_enriched_chain(
            etf_by_date,
            opt_by_date,
            trading_calendar,
            date,
        )
        spot = float(etf_by_date[date]["close"].iloc[-1])
        atm = core.vol_engine.select_atm_from_chain(chain_df, spot)
    except Exception as exc:
        raise ValueError(
            "Missing ATM strike for roll mismatch check: "
            f"date={date.date()}, cannot build from signal history or daily market data: {exc}. "
            "No strategy_state fallback is used."
        ) from exc

    if atm is None or pd.isna(atm.get("strike")):
        raise ValueError(
            "Missing ATM strike for roll mismatch check: "
            f"date={date.date()}, no valid ATM pair. No strategy_state fallback is used."
        )
    return float(atm["strike"]), "daily_market_data"


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
                "current_call_qty": position["call_qty"],
                "current_put_qty": position["put_qty"],
                "estimated_current_call_price": float(call_row["mid"]),
                "estimated_current_put_price": float(put_row["mid"]),
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
        current_atm_strike=feature_atm_strike,
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
            "latest_holding_snapshot_date": mismatch["latest_holding_snapshot_date"],
            "snapshot_lag_days": mismatch["snapshot_lag_days"],
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
        "latest_holding_snapshot_date": mismatch["latest_holding_snapshot_date"],
        "snapshot_lag_days": mismatch["snapshot_lag_days"],
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


def _option_hedge_greeks_and_value(position, chain_df):
    code = position.get("order_book_id") or position.get("call_code") or position.get("put_code")
    if code is None:
        return None, None
    try:
        row = _chain_row(chain_df, code)
    except IndexError:
        return None, None

    qty = int(position.get("qty", position.get("call_qty", position.get("put_qty", 0))) or 0)
    if qty <= 0:
        return None, None
    multiplier = float(position.get("contract_multiplier", row.get("contract_multiplier", 10000)) or 10000)
    direction = -1.0 if str(position.get("side", "short")).lower() == "short" else 1.0
    scale = direction * qty * multiplier
    value = float(row.get("mid", 0.0) or 0.0) * qty * multiplier
    if direction < 0:
        value = -value
    option_type = str(position.get("option_type") or row.get("option_type") or "").lower()
    prefix = "call" if option_type == "c" else "put" if option_type == "p" else "leg"
    greeks = {
        "delta": float(row.get("delta", 0.0) or 0.0) * scale,
        "gamma": float(row.get("gamma", 0.0) or 0.0) * scale,
        "vega": float(row.get("vega", 0.0) or 0.0) * scale,
        "theta": float(row.get("theta", 0.0) or 0.0) * scale,
        "call_iv": row.get("iv") if prefix == "call" else 0.0,
        "put_iv": row.get("iv") if prefix == "put" else 0.0,
        "position_iv": row.get("iv"),
        "call_delta": float(row.get("delta", 0.0) or 0.0) * scale if prefix == "call" else 0.0,
        "put_delta": float(row.get("delta", 0.0) or 0.0) * scale if prefix == "put" else 0.0,
        "call_gamma": float(row.get("gamma", 0.0) or 0.0) * scale if prefix == "call" else 0.0,
        "put_gamma": float(row.get("gamma", 0.0) or 0.0) * scale if prefix == "put" else 0.0,
        "call_vega": float(row.get("vega", 0.0) or 0.0) * scale if prefix == "call" else 0.0,
        "put_vega": float(row.get("vega", 0.0) or 0.0) * scale if prefix == "put" else 0.0,
        "call_theta": float(row.get("theta", 0.0) or 0.0) * scale if prefix == "call" else 0.0,
        "put_theta": float(row.get("theta", 0.0) or 0.0) * scale if prefix == "put" else 0.0,
    }
    return greeks, value


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
        plan.extend(_hedge_advice(config, live_account, account_greeks, spot, chain_df, atm))
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
    plan.extend(final_hedge)
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


def _hedge_advice(config, live_account, greeks, spot, chain_df, atm):
    return _delta_hedge_plan(
        config,
        live_account,
        greeks,
        spot,
        chain_df,
        atm,
        action="DELTA_HEDGE",
        option_action="OPTION_DELTA_HEDGE_SHORT_CALL",
        reason="Account delta exceeds tolerance.",
    )


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
        return []

    planned_option_delta = float(planned_greeks["delta"])
    planned_account_delta = planned_option_delta + live_account.hedge.qty
    tolerance = max(
        1.0,
        abs(planned_option_delta) * config.strategy.delta_hedge_tolerance_ratio,
    )
    if abs(planned_account_delta) <= tolerance:
        return []

    plan = _delta_hedge_plan(
        config,
        live_account,
        planned_greeks,
        spot,
        chain_df,
        atm,
        action="FINAL_DELTA_HEDGE",
        option_action="FINAL_OPTION_DELTA_HEDGE_SHORT_CALL",
        reason=(
            "After executing the option plan, projected account delta exceeds "
            "tolerance."
        ),
        after_actions=[item.get("action") for item in option_actions],
        underlying_order_book_id=_underlying_id_after_plan(
            option_actions,
            chain_df,
            atm,
        ),
        exclude_call_codes=_planned_call_codes(option_actions),
    )
    for item in plan:
        item.setdefault("planned_account_gamma", planned_greeks["gamma"])
        item.setdefault("planned_account_vega", planned_greeks["vega"])
        item.setdefault("planned_account_theta", planned_greeks["theta"])
        item.setdefault("hedge_tolerance", tolerance)
    return plan


def _delta_hedge_plan(
    config,
    live_account,
    greeks,
    spot,
    chain_df,
    atm,
    action,
    option_action,
    reason,
    after_actions=None,
    underlying_order_book_id=None,
    exclude_call_codes=None,
):
    if not config.strategy.enable_delta_hedge:
        return []

    option_delta = float(greeks["delta"])
    current_hedge_qty = float(live_account.hedge.qty or 0.0)
    account_delta = option_delta + current_hedge_qty
    tolerance = max(
        1.0,
        abs(option_delta) * config.strategy.delta_hedge_tolerance_ratio,
    )
    if abs(account_delta) <= tolerance:
        return []

    target_qty = -option_delta
    if underlying_order_book_id is None:
        underlying_order_book_id = _underlying_id_from_atm(atm)

    if getattr(config.strategy, "allow_etf_short_hedge", True) or target_qty >= 0:
        return [
            _etf_hedge_item(
                action,
                reason,
                option_delta,
                current_hedge_qty,
                account_delta,
                target_qty,
                target_qty - current_hedge_qty,
                spot,
                underlying_order_book_id,
                after_actions=after_actions,
            )
        ]

    plan = []
    etf_target_qty = 0.0
    etf_trade_qty = etf_target_qty - current_hedge_qty
    if abs(etf_trade_qty) > 1e-9:
        plan.append(
            _etf_hedge_item(
                action,
                (
                    f"{reason} ETF short hedge is disabled; reduce ETF hedge "
                    "before using option delta hedge."
                ),
                option_delta,
                current_hedge_qty,
                account_delta,
                etf_target_qty,
                etf_trade_qty,
                spot,
                underlying_order_book_id,
                after_actions=after_actions,
            )
        )

    residual_delta = option_delta + etf_target_qty
    if abs(residual_delta) <= tolerance:
        return plan
    if residual_delta < 0:
        plan.append(
            {
                "action": "DATA_WARNING",
                "priority": "warning",
                "reason": (
                    "ETF short hedge is disabled and residual delta is negative; "
                    "short-call hedge cannot add positive delta."
                ),
                "residual_delta": residual_delta,
                "after_actions": after_actions,
            }
        )
        return plan
    if not getattr(config.strategy, "enable_option_delta_hedge", False):
        plan.append(
            {
                "action": "DATA_WARNING",
                "priority": "warning",
                "reason": (
                    "ETF short hedge is disabled and option delta hedge is not enabled."
                ),
                "residual_delta": residual_delta,
                "after_actions": after_actions,
            }
        )
        return plan

    if getattr(config.strategy, "option_delta_hedge_gamma_neutral", False):
        gamma_neutral_item = None
        if not after_actions:
            gamma_neutral_item = _gamma_neutral_option_delta_hedge_item(
                config,
                live_account,
                chain_df,
                residual_delta,
                spot,
                underlying_order_book_id,
                after_actions=after_actions,
                current_hedge_qty=etf_target_qty,
                option_delta=option_delta,
                final=action.startswith("FINAL_"),
            )
        if gamma_neutral_item is not None:
            plan.append(gamma_neutral_item)
        else:
            plan.append(
                {
                    "action": "DATA_WARNING",
                    "priority": "warning",
                    "reason": (
                        "Gamma-neutral option delta hedge is not feasible with "
                        "current short-call inventory and listed call contracts, "
                        "or the core option position is already changing in this plan."
                    ),
                    "residual_delta": residual_delta,
                    "after_actions": after_actions,
                }
            )
        return plan

    call_row = _select_itm_call_for_delta_hedge(
        config,
        chain_df,
        spot,
        atm,
        exclude_call_codes=_existing_call_codes(live_account) | set(exclude_call_codes or []),
    )
    if call_row is None:
        plan.append(
            {
                "action": "DATA_WARNING",
                "priority": "warning",
                "reason": "No valid lightly ITM call found for option delta hedge.",
                "residual_delta": residual_delta,
                "after_actions": after_actions,
            }
        )
        return plan

    plan.append(
        _short_call_delta_hedge_item(
            config,
            option_action,
            reason,
            call_row,
            residual_delta,
            spot,
            underlying_order_book_id,
            after_actions=after_actions,
            current_hedge_qty=etf_target_qty,
            option_delta=option_delta,
        )
    )
    return plan


def _gamma_neutral_option_delta_hedge_item(
    config,
    live_account,
    chain_df,
    residual_delta,
    spot,
    underlying_order_book_id,
    after_actions=None,
    current_hedge_qty=0.0,
    option_delta=0.0,
    final=False,
):
    target_delta_effect = -float(residual_delta)
    sources = _short_call_close_sources(live_account, chain_df)
    if not sources:
        return None

    candidates = chain_df.copy()
    candidates = candidates[candidates["option_type"].astype(str).str.lower().eq("c")]
    candidates = candidates[pd.to_numeric(candidates["mid"], errors="coerce") > 0]
    candidates = candidates[pd.to_numeric(candidates["delta"], errors="coerce") > 0]
    candidates = candidates[pd.to_numeric(candidates["gamma"], errors="coerce") > 0]
    existing_call_codes = _existing_call_codes(live_account)
    candidates = candidates[
        ~candidates["order_book_id"].astype(str).isin(existing_call_codes)
    ]
    if candidates.empty:
        return None

    best = None
    for source in sources:
        source_row = source["row"]
        source_code = str(source_row.get("order_book_id"))
        same_expiry = candidates[
            pd.to_datetime(candidates["maturity_date"]).dt.normalize().eq(
                pd.Timestamp(source_row.get("maturity_date")).normalize()
            )
        ]
        same_expiry = _light_itm_call_candidates(config, same_expiry, spot)
        for _, open_row in same_expiry.iterrows():
            if str(open_row.get("order_book_id")) == source_code:
                continue
            if float(open_row.get("contract_multiplier")) != float(
                source_row.get("contract_multiplier")
            ):
                continue
            solution = _integer_gamma_neutral_call_solution(
                source,
                open_row,
                target_delta_effect,
                max_etf_correction=max(1.0, abs(residual_delta) * 0.05),
            )
            if solution is None:
                continue
            score = (
                abs(solution["gamma_effect"]) * float(spot)
                + solution["etf_buy_qty"]
                + solution["close_qty"] * 1e-3
            )
            if best is None or score < best["score"]:
                best = {
                    **solution,
                    "score": score,
                    "source": source,
                    "open_row": open_row,
                }
    if best is None:
        return None

    source = best["source"]
    open_row = best["open_row"]
    open_qty = best["open_qty"]
    close_qty = best["close_qty"]
    multiplier = float(open_row.get("contract_multiplier", config.vol.contract_multiplier))
    open_price = float(open_row.get("mid"))
    close_price = float(source["row"].get("mid"))
    fee = core.position.calc_option_fee(open_qty + close_qty, 0, config.backtest.option_fee_per_contract)
    margin = core.position.margin_call(
        float(spot),
        float(open_row.get("strike_price")),
        open_price,
        multiplier,
    ) * open_qty
    item = {
        "action": (
            "FINAL_GAMMA_NEUTRAL_OPTION_DELTA_HEDGE"
            if final
            else "GAMMA_NEUTRAL_OPTION_DELTA_HEDGE"
        ),
        "priority": "action",
        "side": "short",
        "reason": "Offset account delta while keeping option gamma approximately unchanged.",
        "close_source": source["kind"],
        "close_call_code": source["row"].get("order_book_id"),
        "close_call_qty": close_qty,
        "estimated_close_call_price": close_price,
        "open_call_code": open_row.get("order_book_id"),
        "open_call_qty": open_qty,
        "estimated_open_call_price": open_price,
        "open_strike": float(open_row.get("strike_price")),
        "open_expiry": str(pd.Timestamp(open_row.get("maturity_date")).date()),
        "estimated_delta_effect": best["delta_effect"],
        "estimated_gamma_effect": best["gamma_effect"],
        "residual_delta_before_option_hedge": residual_delta,
        "projected_account_delta_after_option_hedge": best["projected_delta"],
        "etf_delta_correction": best["etf_buy_qty"],
        "projected_account_delta_after_combined_hedge": best["combined_delta"],
        "option_delta": option_delta,
        "current_hedge_qty": current_hedge_qty,
        "target_hedge_qty": current_hedge_qty + best["etf_buy_qty"],
        "trade_etf_qty": best["etf_buy_qty"],
        "estimated_price": float(spot),
        "estimated_fee": fee,
        "estimated_option_margin": margin,
        "underlying_order_book_id": underlying_order_book_id,
    }
    if after_actions is not None:
        item["after_actions"] = after_actions
    return item


def _short_call_close_sources(live_account, chain_df):
    sources = []
    short_position = live_account.positions.get("short")
    if short_position is not None and int(short_position.get("call_qty", 0) or 0) > 0:
        try:
            row = _chain_row(chain_df, short_position.get("call_code"))
            sources.append(
                {
                    "kind": "core_short_call",
                    "max_qty": int(short_position.get("call_qty", 0) or 0),
                    "row": row,
                }
            )
        except IndexError:
            pass
    for hedge in getattr(live_account, "option_hedges", []) or []:
        if str(hedge.get("side")).lower() != "short":
            continue
        if str(hedge.get("option_type")).lower() != "c":
            continue
        try:
            row = _chain_row(chain_df, hedge.get("order_book_id"))
            sources.append(
                {
                    "kind": "option_hedge_short_call",
                    "max_qty": int(hedge.get("qty", 0) or 0),
                    "row": row,
                }
            )
        except IndexError:
            pass
    return sources


def _integer_gamma_neutral_call_solution(
    source,
    open_row,
    target_delta_effect,
    max_etf_correction=None,
):
    close_delta = float(source["row"].get("delta", 0.0) or 0.0)
    close_gamma = float(source["row"].get("gamma", 0.0) or 0.0)
    open_delta = float(open_row.get("delta", 0.0) or 0.0)
    open_gamma = float(open_row.get("gamma", 0.0) or 0.0)
    multiplier = float(open_row.get("contract_multiplier", 10000) or 10000)
    if close_gamma <= 0 or open_gamma <= 0:
        return None
    denominator = close_delta * open_gamma / close_gamma - open_delta
    if abs(denominator) <= 1e-12:
        return None
    continuous_open_qty = (target_delta_effect / multiplier) / denominator
    if continuous_open_qty <= 0:
        return None

    best = None
    open_centers = {max(1, int(math.floor(continuous_open_qty))), max(1, int(math.ceil(continuous_open_qty)))}
    for center in list(open_centers):
        open_centers.update(
            {
                max(1, center - 4),
                max(1, center - 3),
                max(1, center - 2),
                max(1, center - 1),
                center + 1,
                center + 2,
                center + 3,
                center + 4,
            }
        )
    for open_qty in sorted(open_centers):
        continuous_close_qty = open_qty * open_gamma / close_gamma
        close_candidates = {
            max(1, int(math.floor(continuous_close_qty))),
            max(1, int(math.ceil(continuous_close_qty))),
        }
        for close_qty in close_candidates:
            # Keep both original straddle legs present so broker snapshots can
            # continue to identify and synchronize the core pair.
            if close_qty >= int(source["max_qty"]):
                continue
            delta_effect = multiplier * (close_qty * close_delta - open_qty * open_delta)
            gamma_effect = multiplier * (close_qty * close_gamma - open_qty * open_gamma)
            projected_delta = -target_delta_effect + delta_effect
            if max_etf_correction is not None:
                if projected_delta > 0 or abs(projected_delta) > max_etf_correction:
                    continue
                etf_buy_qty = float(math.ceil(-projected_delta))
            else:
                etf_buy_qty = 0.0
            gross_gamma = multiplier * max(
                close_qty * close_gamma,
                open_qty * open_gamma,
            )
            if abs(gamma_effect) > max(1.0, gross_gamma * 0.05):
                continue
            score = abs(gamma_effect) + etf_buy_qty
            if best is None or score < best["score"]:
                best = {
                    "open_qty": open_qty,
                    "close_qty": close_qty,
                    "delta_effect": delta_effect,
                    "gamma_effect": gamma_effect,
                    "projected_delta": projected_delta,
                    "etf_buy_qty": etf_buy_qty,
                    "combined_delta": projected_delta + etf_buy_qty,
                    "score": score,
                }
    return best


def _etf_hedge_item(
    action,
    reason,
    option_delta,
    current_hedge_qty,
    account_delta,
    target_qty,
    trade_qty,
    spot,
    underlying_order_book_id,
    after_actions=None,
):
    item = {
        "action": action,
        "priority": "action",
        "reason": reason,
        "option_delta": option_delta,
        "current_hedge_qty": current_hedge_qty,
        "account_delta": account_delta,
        "target_hedge_qty": target_qty,
        "trade_etf_qty": trade_qty,
        "estimated_price": spot,
        "underlying_order_book_id": underlying_order_book_id,
    }
    if action.startswith("FINAL_"):
        item["planned_option_delta"] = option_delta
        item["planned_account_delta"] = account_delta
    if after_actions is not None:
        item["after_actions"] = after_actions
    return item


def _short_call_delta_hedge_item(
    config,
    action,
    reason,
    call_row,
    residual_delta,
    spot,
    underlying_order_book_id,
    after_actions=None,
    current_hedge_qty=0.0,
    option_delta=0.0,
):
    raw_delta = float(call_row.get("delta"))
    multiplier = float(call_row.get("contract_multiplier", config.vol.contract_multiplier))
    hedge_delta_per_contract = -raw_delta * multiplier
    qty = int(math.ceil(residual_delta / abs(hedge_delta_per_contract)))
    hedge_delta = hedge_delta_per_contract * qty
    estimated_price = float(call_row.get("mid"))
    fee = core.position.calc_option_fee(qty, 0, config.backtest.option_fee_per_contract)
    margin = core.position.margin_call(
        float(spot),
        float(call_row.get("strike_price")),
        estimated_price,
        multiplier,
    ) * qty
    cash_effect = estimated_price * qty * multiplier - fee - margin
    item = {
        "action": action,
        "priority": "action",
        "side": "short",
        "reason": reason,
        "option_hedge_type": "short_otm_call",
        "call_code": call_row.get("order_book_id"),
        "put_code": None,
        "strike": float(call_row.get("strike_price")),
        "expiry": str(pd.Timestamp(call_row.get("maturity_date")).date()),
        "call_qty": qty,
        "put_qty": 0,
        "estimated_call_price": estimated_price,
        "estimated_put_price": None,
        "single_call_delta": -raw_delta,
        "hedge_delta_per_contract": hedge_delta_per_contract,
        "estimated_hedge_delta": hedge_delta,
        "residual_delta_before_option_hedge": residual_delta,
        "projected_account_delta_after_option_hedge": residual_delta + hedge_delta,
        "option_delta": option_delta,
        "current_hedge_qty": current_hedge_qty,
        "target_hedge_qty": current_hedge_qty,
        "trade_etf_qty": 0.0,
        "estimated_fee": fee,
        "estimated_option_margin": margin,
        "estimated_cash_effect": cash_effect,
        "underlying_order_book_id": underlying_order_book_id or call_row.get("underlying_order_book_id"),
        "contract_symbol": call_row.get("contract_symbol"),
        "contract_multiplier": multiplier,
    }
    if action.startswith("FINAL_"):
        item["planned_option_delta"] = option_delta
        item["planned_account_delta"] = option_delta + current_hedge_qty
    if after_actions is not None:
        item["after_actions"] = after_actions
    return item


def _light_itm_call_candidates(config, calls, spot):
    if calls.empty:
        return calls
    calls = calls.copy()
    calls["_strike"] = pd.to_numeric(calls["strike_price"], errors="coerce")
    calls["_volume"] = pd.to_numeric(calls.get("volume"), errors="coerce").fillna(-1.0)
    calls = calls[calls["_strike"] < float(spot)]
    strikes = sorted(calls["_strike"].dropna().unique(), reverse=True)
    steps = max(
        1,
        int(getattr(config.strategy, "option_delta_hedge_call_itm_steps", 1) or 1),
    )
    if len(strikes) < steps:
        return calls.iloc[0:0]
    target_strike = strikes[steps - 1]
    return calls[calls["_strike"] == target_strike].sort_values(
        "_volume",
        ascending=False,
    ).head(1)


def _select_itm_call_for_delta_hedge(config, chain_df, spot, atm, exclude_call_codes=None):
    if chain_df is None or chain_df.empty:
        return None
    calls = chain_df.copy()
    calls = calls[calls["option_type"].astype(str).str.lower().eq("c")]
    calls = calls[calls["contract_multiplier"] == config.vol.contract_multiplier]
    if exclude_call_codes:
        calls = calls[~calls["order_book_id"].astype(str).isin({str(code) for code in exclude_call_codes})]
    calls = calls[pd.to_numeric(calls["delta"], errors="coerce") > 0]
    calls = calls[pd.to_numeric(calls["mid"], errors="coerce") > 0]
    if calls.empty:
        return None

    if atm is not None and atm.get("expiry") is not None:
        expiry = pd.Timestamp(atm.get("expiry")).normalize()
        same_expiry = calls[pd.to_datetime(calls["maturity_date"]).dt.normalize().eq(expiry)]
        if not same_expiry.empty:
            calls = same_expiry

    calls = _light_itm_call_candidates(config, calls, spot)
    if calls.empty:
        return None
    return calls.iloc[0]


def _existing_call_codes(live_account):
    codes = set()
    for position in live_account.positions.values():
        if position is not None and position.get("call_code") is not None:
            codes.add(str(position.get("call_code")))
    for position in getattr(live_account, "option_hedges", []) or []:
        option_type = str(position.get("option_type") or "").lower()
        if option_type == "c" and position.get("order_book_id") is not None:
            codes.add(str(position.get("order_book_id")))
        if position.get("call_code") is not None:
            codes.add(str(position.get("call_code")))
    return codes


def _planned_call_codes(option_actions):
    codes = set()
    for item in option_actions or []:
        leg_fields = _planned_leg_fields(item)
        if leg_fields is None:
            continue
        call_code, _, _, _ = leg_fields
        if call_code is not None:
            codes.add(str(call_code))
    return codes


def _underlying_id_from_atm(atm):
    if atm is None:
        return None
    underlying_order_book_id = atm.get("underlying_order_book_id")
    if underlying_order_book_id is not None:
        return underlying_order_book_id
    return _projected_underlying_id(atm.get("call"), atm.get("put"))


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
