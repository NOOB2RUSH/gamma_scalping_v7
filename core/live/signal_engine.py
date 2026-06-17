from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd

import core
from . import account as account_store
from . import market_data
from . import portfolio_account
from . import storage
from .runtime import load_product_config


LIVE_ENTRY_QTY_PER_LEG = 10


def generate_signal(product, account_id="default", date=None, quote_snapshot=None):
    market_data.require_live_product(product)
    config = load_product_config(product)
    live_account = account_store.load_account(product, account_id=account_id)
    live_account.cash = portfolio_account.shared_cash(account_id=account_id)
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
    capacity_item = _live_capacity_reduction_item(
        config,
        live_account,
        chain_df,
        spot,
        reason="Current live account cash or margin capacity is too tight.",
    )
    if capacity_item is not None:
        advice = [capacity_item]
        planned_greeks = account_greeks
    else:
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

    account_delta = account_greeks["delta"] + live_account.hedge.qty
    normalized_account_delta, delta_hedge_capacity = (
        core.strategy.normalized_account_delta(
            account_delta,
            live_account.positions,
            option_hedges=live_account.option_hedges,
            default_multiplier=config.vol.contract_multiplier,
        )
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
        "account_delta_after_hedge": account_delta,
        "normalized_account_delta": normalized_account_delta,
        "delta_hedge_capacity": delta_hedge_capacity,
        "delta_hedge_tolerance_ratio": config.strategy.delta_hedge_tolerance_ratio,
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
    market_data.require_live_product(config.data.product)
    start = config.backtest.start
    end = pd.Timestamp.now().normalize() if date is None else date
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
    latest_enriched = market_data.attach_live_underlying_id(
        config.data.product,
        latest_enriched,
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
    _persist_feature_history(product, combined)
    return combined


def _persist_feature_history(product, features):
    path = storage.feature_history_path(product)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = features.reset_index(names="date")
    temp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    payload.to_parquet(temp_path, index=False)
    temp_path.replace(path)


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
    except Exception as local_exc:
        try:
            historical_atm = market_data.fetch_historical_atm_strike(product, date)
            return float(historical_atm["strike"]), historical_atm["source"]
        except Exception as akshare_exc:
            raise ValueError(
                "Missing ATM strike for roll mismatch check: "
                f"date={date.date()}, cannot build from signal history, daily market data "
                f"({local_exc}), or AKShare historical market data ({akshare_exc}). "
                "No strategy_state fallback is used."
            ) from akshare_exc

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
        ("long", "long_open_signal", "OPEN_LONG_STRADDLE", LIVE_ENTRY_QTY_PER_LEG),
        ("short", "short_open_signal", "OPEN_SHORT_STRADDLE", LIVE_ENTRY_QTY_PER_LEG),
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
    current_strike_differs = _strike_differs(
        position.get("strike"),
        feature_atm_strike,
    )
    if not dte_too_low and not current_strike_differs:
        return None

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
        current_strike_differs
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

    projected_cash = _projected_cash_after_option_actions(
        config,
        live_account,
        option_actions,
        chain_df,
        spot,
    )
    min_cash = float(config.backtest.min_cash_reserve)
    if projected_cash is not None and projected_cash < min_cash:
        reduction_item = _live_capacity_reduction_item(
            config,
            live_account,
            chain_df,
            spot,
            required_cash_relief=min_cash - projected_cash,
            reason=(
                "Reduce the short straddle before the planned main-position action "
                "would breach the live cash reserve."
            ),
        )
        if reduction_item is not None:
            return [reduction_item], account_greeks
        return [
            {
                "action": "DATA_WARNING",
                "priority": "warning",
                "reason": "Planned main-position action would breach the live cash reserve.",
                "projected_cash": projected_cash,
                "min_cash_reserve": min_cash,
            }
        ], account_greeks

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


def _projected_cash_after_option_actions(
    config,
    live_account,
    option_actions,
    chain_df,
    spot,
):
    cash = float(live_account.cash)
    for item in option_actions:
        action = str(item.get("action") or "")
        cash_effect = item.get("estimated_cash_effect")
        if cash_effect is not None:
            cash += float(cash_effect)
            continue
        if not action.startswith("ROLL_"):
            continue
        side = item.get("side")
        current = live_account.positions.get(side)
        if current is None:
            return None
        current_value = (
            float(item.get("estimated_current_call_price", 0.0) or 0.0)
            * int(item.get("current_call_qty", 0) or 0)
            + float(item.get("estimated_current_put_price", 0.0) or 0.0)
            * int(item.get("current_put_qty", 0) or 0)
        ) * float(current.get("contract_multiplier", config.vol.contract_multiplier))
        close_fee = core.position.calc_option_fee(
            int(item.get("current_call_qty", 0) or 0),
            int(item.get("current_put_qty", 0) or 0),
            config.backtest.option_fee_per_contract,
        )
        cash += (
            float(current.get("option_margin", 0.0) or 0.0)
            - current_value
            - close_fee
            if side == "short"
            else current_value - close_fee
        )
        try:
            call_row = _chain_row(chain_df, item.get("target_call_code"))
            put_row = _chain_row(chain_df, item.get("target_put_code"))
        except IndexError:
            return None
        target_atm = {
            "call": call_row,
            "put": put_row,
            "strike": float(call_row.get("strike_price")),
        }
        target_position = core.position.open_straddle(
            pd.Timestamp.now(),
            target_atm,
            int(item.get("target_call_qty", 0) or 0),
            int(item.get("target_put_qty", 0) or 0),
            side=side,
            spot=spot,
        )
        target_value = core.position.value(target_position, call_row, put_row)
        open_fee = core.position.calc_option_fee(
            target_position["call_qty"],
            target_position["put_qty"],
            config.backtest.option_fee_per_contract,
        )
        cash += (
            target_value - open_fee - target_position["option_margin"]
            if side == "short"
            else -target_value - open_fee
        )
    return cash


def _live_capacity_reduction_item(
    config,
    live_account,
    chain_df,
    spot,
    required_cash_relief=0.0,
    minimum_close_qty=0,
    reason=None,
):
    short_position = live_account.positions.get("short")
    if short_position is None:
        return None
    try:
        call_row, put_row = core.vol_engine.resolve_position_pair(
            short_position,
            chain_df,
        )
    except IndexError:
        return None

    current_qty = min(
        int(short_position.get("call_qty", 0) or 0),
        int(short_position.get("put_qty", 0) or 0),
    )
    if current_qty <= 0:
        return None

    capacity = _live_account_capacity(config, live_account, chain_df, spot)
    min_cash = float(config.backtest.min_cash_reserve)
    cash_shortfall = max(
        0.0,
        min_cash - float(live_account.cash),
        float(required_cash_relief),
    )
    margin_limit = max(
        0.0,
        capacity["nav"] * float(config.backtest.max_margin_to_nav_ratio),
    )
    capacity_usage = (
        capacity["capital_occupation"]
        if getattr(config.backtest, "dynamic_position_control_enabled", False)
        else capacity["total_margin"]
    )
    margin_excess = max(0.0, capacity_usage - margin_limit)
    if (
        cash_shortfall <= 1e-6
        and margin_excess <= 1e-6
        and int(minimum_close_qty or 0) <= 0
    ):
        return None

    option_margin = _current_short_margin(
        config,
        short_position,
        call_row,
        put_row,
        spot,
    )
    margin_relief_per_contract = option_margin / current_qty
    multiplier = float(
        short_position.get(
            "contract_multiplier",
            call_row.get("contract_multiplier", config.vol.contract_multiplier),
        )
        or config.vol.contract_multiplier
    )
    close_cost_per_contract = (
        float(call_row.get("mid", 0.0) or 0.0)
        + float(put_row.get("mid", 0.0) or 0.0)
    ) * multiplier + core.position.calc_option_fee(
        1,
        1,
        config.backtest.option_fee_per_contract,
    )
    cash_relief_per_contract = margin_relief_per_contract - close_cost_per_contract
    close_for_cash = (
        int(math.ceil(cash_shortfall / cash_relief_per_contract))
        if cash_shortfall > 0 and cash_relief_per_contract > 0
        else 0
    )
    close_for_margin = (
        int(math.ceil(margin_excess / margin_relief_per_contract))
        if margin_excess > 0 and margin_relief_per_contract > 0
        else 0
    )
    close_qty = min(
        current_qty,
        max(close_for_cash, close_for_margin, int(minimum_close_qty or 0)),
    )
    if close_qty <= 0:
        return {
            "action": "DATA_WARNING",
            "priority": "warning",
            "reason": (
                "Live account capacity is tight, but reducing the current short "
                "straddle would not release enough cash or margin."
            ),
            "cash": float(live_account.cash),
            "min_cash_reserve": min_cash,
            "total_margin": capacity["total_margin"],
            "capital_occupation": capacity["capital_occupation"],
            "margin_limit": margin_limit,
        }

    target_qty = current_qty - close_qty
    estimated_cash_effect = cash_relief_per_contract * close_qty
    return {
        "action": "REDUCE_SHORT_STRADDLE_FOR_CAPACITY",
        "priority": "action",
        "side": "short",
        "reason": reason or "Reduce the short straddle to restore live account capacity.",
        "call_code": short_position.get("call_code"),
        "put_code": short_position.get("put_code"),
        "call_qty": close_qty,
        "put_qty": close_qty,
        "current_call_qty": current_qty,
        "current_put_qty": current_qty,
        "target_call_qty": target_qty,
        "target_put_qty": target_qty,
        "estimated_call_price": float(call_row.get("mid")),
        "estimated_put_price": float(put_row.get("mid")),
        "estimated_fee": core.position.calc_option_fee(
            close_qty,
            close_qty,
            config.backtest.option_fee_per_contract,
        ),
        "estimated_cash_effect": estimated_cash_effect,
        "cash_before": float(live_account.cash),
        "projected_cash_after": float(live_account.cash) + estimated_cash_effect,
        "min_cash_reserve": min_cash,
        "margin_before": capacity_usage,
        "projected_margin_after": capacity_usage
        - margin_relief_per_contract * close_qty,
        "margin_limit": margin_limit,
        "requires_import_and_rerun": True,
    }


def _delta_hedge_capacity_reduction_item(
    config,
    live_account,
    chain_df,
    spot,
    option_delta,
    target_hedge_qty,
):
    if not getattr(config.backtest, "dynamic_position_control_enabled", False):
        return None

    short_position = live_account.positions.get("short")
    if short_position is None:
        return None
    try:
        call_row, put_row = core.vol_engine.resolve_position_pair(
            short_position,
            chain_df,
        )
    except IndexError:
        return None

    current_qty = min(
        int(short_position.get("call_qty", 0) or 0),
        int(short_position.get("put_qty", 0) or 0),
    )
    if current_qty <= 0:
        return None

    capacity = _live_account_capacity(config, live_account, chain_df, spot)
    occupation_limit = max(
        0.0,
        capacity["nav"] * float(config.backtest.max_margin_to_nav_ratio),
    )
    current_hedge_occupation = capacity["hedge_capital_occupation"]
    projected_occupation = (
        capacity["capital_occupation"]
        - current_hedge_occupation
        + abs(float(target_hedge_qty)) * float(spot)
    )
    if projected_occupation <= occupation_limit + 1e-6:
        return None

    short_margin = _current_short_margin(
        config,
        short_position,
        call_row,
        put_row,
        spot,
    )
    short_greeks = core.strategy.calc_position_greeks(
        call_row,
        put_row,
        current_qty,
        current_qty,
        side="short",
    )
    short_delta = float(short_greeks["delta"])
    other_occupation = (
        capacity["capital_occupation"] - current_hedge_occupation - short_margin
    )
    selected = None
    for close_qty in range(1, current_qty + 1):
        remaining_ratio = (current_qty - close_qty) / current_qty
        projected_option_delta = option_delta - short_delta * (close_qty / current_qty)
        projected_target = core.strategy.round_etf_hedge_target(-projected_option_delta)
        occupation_after = (
            other_occupation
            + short_margin * remaining_ratio
            + abs(projected_target) * float(spot)
        )
        if occupation_after <= occupation_limit + 1e-6:
            selected = {
                "close_qty": close_qty,
                "projected_target_hedge_qty": projected_target,
                "projected_capital_occupation": occupation_after,
            }
            break

    if selected is None:
        selected = {
            "close_qty": current_qty,
            "projected_target_hedge_qty": core.strategy.round_etf_hedge_target(
                -(option_delta - short_delta)
            ),
            "projected_capital_occupation": other_occupation
            + abs(
                core.strategy.round_etf_hedge_target(
                    -(option_delta - short_delta)
                )
            )
            * float(spot),
        }

    item = _live_capacity_reduction_item(
        config,
        live_account,
        chain_df,
        spot,
        minimum_close_qty=selected["close_qty"],
        reason=(
            "Reduce the short straddle before a delta-neutral ETF hedge would "
            "breach the live capital occupation limit."
        ),
    )
    if item is not None:
        item.update(
            {
                "projected_capital_occupation_before_reduction": projected_occupation,
                "projected_capital_occupation_after_reduction_and_hedge": selected[
                    "projected_capital_occupation"
                ],
                "capital_occupation_limit": occupation_limit,
                "projected_target_hedge_qty_after_reduction": selected[
                    "projected_target_hedge_qty"
                ],
            }
        )
    return item


def _live_account_capacity(config, live_account, chain_df, spot):
    option_value = 0.0
    stored_option_margin = 0.0
    current_option_margin = 0.0
    for side, position in live_account.positions.items():
        if position is None:
            continue
        stored_option_margin += float(position.get("option_margin", 0.0) or 0.0)
        try:
            call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
        except IndexError:
            continue
        option_value += core.position.signed_value(position, call_row, put_row)
        if side == "short":
            current_option_margin += _current_short_margin(
                config,
                position,
                call_row,
                put_row,
                spot,
            )
    for hedge_position in getattr(live_account, "option_hedges", []) or []:
        _, value = _option_hedge_greeks_and_value(hedge_position, chain_df)
        option_value += float(value or 0.0)
        stored_option_margin += float(hedge_position.get("option_margin", 0.0) or 0.0)
        current_option_margin += _current_option_hedge_margin(
            config,
            hedge_position,
            chain_df,
            spot,
        )

    stored_hedge_margin = float(live_account.hedge.margin or 0.0)
    hedge_price = float(spot)
    hedge_qty = float(live_account.hedge.qty or 0.0)
    hedge_pnl = core.hedge.calc_unrealized_pnl(
        hedge_qty,
        float(live_account.hedge.entry_price or 0.0),
        hedge_price,
    )
    total_margin = stored_option_margin + stored_hedge_margin
    hedge_capital_occupation = abs(hedge_qty) * hedge_price
    capital_occupation = current_option_margin + hedge_capital_occupation
    nav = (
        float(live_account.cash)
        + option_value
        + total_margin
        + hedge_pnl
    )
    return {
        "nav": nav,
        "total_margin": total_margin,
        "current_option_margin": current_option_margin,
        "hedge_capital_occupation": hedge_capital_occupation,
        "capital_occupation": capital_occupation,
    }


def _current_short_margin(config, position, call_row, put_row, spot):
    underlying_price = call_row.get("underlying_close")
    if pd.isna(underlying_price):
        underlying_price = spot
    return core.position.calc_short_margin(
        call_row,
        put_row,
        int(position.get("call_qty", 0) or 0),
        int(position.get("put_qty", 0) or 0),
        float(underlying_price),
    )


def _current_option_hedge_margin(config, hedge_position, chain_df, spot):
    if hedge_position.get("side", "long") != "short":
        return 0.0
    rows = chain_df[
        chain_df["order_book_id"].astype(str)
        == str(hedge_position.get("order_book_id"))
    ]
    if rows.empty:
        return float(hedge_position.get("option_margin", 0.0) or 0.0)
    row = rows.iloc[-1]
    underlying_price = row.get("underlying_close")
    if pd.isna(underlying_price):
        underlying_price = spot
    return core.position.margin_call(
        float(underlying_price),
        float(row.get("strike_price")),
        float(row.get("mid")),
        float(row.get("contract_multiplier", config.vol.contract_multiplier)),
    ) * int(hedge_position.get("qty", 0) or 0)


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
    planned_positions = _positions_after_option_actions(
        config,
        live_account.positions,
        option_actions,
        chain_df,
    )
    normalized_delta, delta_capacity = core.strategy.normalized_account_delta(
        planned_account_delta,
        planned_positions,
        option_hedges=live_account.option_hedges,
        default_multiplier=config.vol.contract_multiplier,
    )
    tolerance_ratio = float(config.strategy.delta_hedge_tolerance_ratio)
    tolerance = delta_capacity * tolerance_ratio
    if delta_capacity > 0 and abs(normalized_delta) <= tolerance_ratio:
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
        positions_for_tolerance=planned_positions,
    )
    for item in plan:
        item.setdefault("planned_account_gamma", planned_greeks["gamma"])
        item.setdefault("planned_account_vega", planned_greeks["vega"])
        item.setdefault("planned_account_theta", planned_greeks["theta"])
        item.setdefault("hedge_tolerance", tolerance)
        item.setdefault("normalized_account_delta", normalized_delta)
        item.setdefault("delta_hedge_tolerance_ratio", tolerance_ratio)
        item.setdefault("delta_hedge_capacity", delta_capacity)
    return plan


def _positions_after_option_actions(config, current_positions, option_actions, chain_df):
    planned_positions = dict(current_positions)
    for item in option_actions:
        action = str(item.get("action") or "")
        side = item.get("side")
        if side not in account_store.POSITION_SIDES:
            continue
        if action.startswith("CLOSE_"):
            planned_positions[side] = None
            continue

        leg_fields = _planned_leg_fields(item)
        if leg_fields is None:
            continue
        call_code, put_code, call_qty, put_qty = leg_fields
        try:
            call_row = _chain_row(chain_df, call_code)
        except IndexError:
            continue
        planned_positions[side] = {
            "call_code": call_code,
            "put_code": put_code,
            "call_qty": int(call_qty or 0),
            "put_qty": int(put_qty or 0),
            "contract_multiplier": float(
                call_row.get("contract_multiplier", config.vol.contract_multiplier)
                or config.vol.contract_multiplier
            ),
        }
    return planned_positions


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
    positions_for_tolerance=None,
):
    if not config.strategy.enable_delta_hedge:
        return []

    option_delta = float(greeks["delta"])
    current_hedge_qty = float(live_account.hedge.qty or 0.0)
    account_delta = option_delta + current_hedge_qty
    normalized_delta, delta_capacity = core.strategy.normalized_account_delta(
        account_delta,
        (
            positions_for_tolerance
            if positions_for_tolerance is not None
            else live_account.positions
        ),
        option_hedges=live_account.option_hedges,
        default_multiplier=config.vol.contract_multiplier,
    )
    tolerance_ratio = float(config.strategy.delta_hedge_tolerance_ratio)
    tolerance = delta_capacity * tolerance_ratio
    if delta_capacity > 0 and abs(normalized_delta) <= tolerance_ratio:
        return []

    if getattr(live_account, "option_hedges", None):
        close_items, core_greeks = _close_existing_option_hedge_plan(
            live_account,
            chain_df,
            greeks,
        )
        account_without_option_hedges = deepcopy(live_account)
        account_without_option_hedges.option_hedges = []
        next_after_actions = [
            *(after_actions or []),
            *[item["action"] for item in close_items],
        ]
        return [
            *close_items,
            *_delta_hedge_plan(
                config,
                account_without_option_hedges,
                core_greeks,
                spot,
                chain_df,
                atm,
                action=action,
                option_action=option_action,
                reason=reason,
                after_actions=next_after_actions,
                underlying_order_book_id=underlying_order_book_id,
                exclude_call_codes=exclude_call_codes,
                positions_for_tolerance=positions_for_tolerance,
            ),
        ]

    target_qty = core.strategy.round_etf_hedge_target(-option_delta)
    if underlying_order_book_id is None:
        underlying_order_book_id = _underlying_id_from_atm(atm)

    if getattr(config.strategy, "allow_etf_short_hedge", True) or target_qty >= 0:
        reduction_item = _delta_hedge_capacity_reduction_item(
            config,
            live_account,
            chain_df,
            spot,
            option_delta,
            target_qty,
        )
        if reduction_item is not None:
            return [reduction_item]

        trade_qty = target_qty - current_hedge_qty
        projected_cash = float(live_account.cash) - max(0.0, trade_qty) * float(
            spot
        ) * (1.0 + float(config.backtest.etf_fee_rate))
        min_cash = float(config.backtest.min_cash_reserve)
        if projected_cash < min_cash:
            reduction_item = _live_capacity_reduction_item(
                config,
                live_account,
                chain_df,
                spot,
                required_cash_relief=min_cash - projected_cash,
                reason=(
                    "Reduce the short straddle before the planned ETF delta hedge "
                    "would breach the live cash reserve."
                ),
            )
            if reduction_item is not None:
                return [reduction_item]
            return [
                {
                    "action": "DATA_WARNING",
                    "priority": "warning",
                    "reason": "Planned ETF delta hedge would breach the live cash reserve.",
                    "projected_cash": projected_cash,
                    "min_cash_reserve": min_cash,
                }
            ]
        item = _etf_hedge_item(
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
        item.update(
            {
                "normalized_account_delta": normalized_delta,
                "delta_hedge_tolerance_ratio": tolerance_ratio,
                "delta_hedge_capacity": delta_capacity,
                "hedge_tolerance": tolerance,
            }
        )
        return [item]

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

    if getattr(config.strategy, "option_delta_hedge_combination_enabled", False):
        combination_item = None
        if _can_plan_option_delta_hedge_after(after_actions):
            combination_item = _option_delta_hedge_combination_item(
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
        if combination_item is not None:
            if combination_item.get("action") == "REDUCE_SHORT_STRADDLE_FOR_CAPACITY":
                return [combination_item]
            plan.append(combination_item)
        else:
            plan.append(
                {
                    "action": "DATA_WARNING",
                    "priority": "warning",
                    "reason": (
                        "No liquid lightly-ITM option delta hedge is feasible with "
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


def _can_plan_option_delta_hedge_after(after_actions):
    return not after_actions or all(
        action == "CLOSE_OPTION_HEDGE" for action in after_actions
    )


def _close_existing_option_hedge_plan(live_account, chain_df, account_greeks):
    close_items = []
    core_greeks = deepcopy(account_greeks)
    for hedge_position in getattr(live_account, "option_hedges", []) or []:
        hedge_greeks, _ = _option_hedge_greeks_and_value(hedge_position, chain_df)
        if hedge_greeks is not None:
            for key in core.backtester.NUMERIC_GREEK_KEYS:
                core_greeks[key] = float(core_greeks.get(key, 0.0) or 0.0) - float(
                    hedge_greeks.get(key, 0.0) or 0.0
                )
        try:
            row = _chain_row(chain_df, hedge_position.get("order_book_id"))
            price = float(row.get("mid"))
        except (IndexError, TypeError, ValueError):
            price = float(hedge_position.get("last_price", 0.0) or 0.0)
        close_items.append(
            {
                "action": "CLOSE_OPTION_HEDGE",
                "priority": "action",
                "reason": (
                    "Close the existing option delta hedge before recalculating "
                    "the next account hedge."
                ),
                "order_book_id": hedge_position.get("order_book_id"),
                "side": hedge_position.get("side", "short"),
                "qty": int(hedge_position.get("qty", 0) or 0),
                "estimated_price": price,
            }
        )

    return close_items, core_greeks


def _option_delta_hedge_combination_item(
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
        same_expiry = same_expiry[
            ~same_expiry["order_book_id"].astype(str).eq(source_code)
            & (
                pd.to_numeric(same_expiry["contract_multiplier"], errors="coerce")
                == float(source_row.get("contract_multiplier"))
            )
        ]
        solution = core.position.solve_liquid_call_delta_hedge(
            source,
            [row for _, row in same_expiry.iterrows()],
            residual_delta,
        )
        if solution is not None and (best is None or solution["score"] < best["score"]):
            best = {
                **solution,
                "source": source,
            }
    if best is None:
        return None

    source = best["source"]
    open_legs = best["open_legs"]
    primary_open_row = open_legs[0]["row"]
    open_qty = int(best["open_qty"])
    close_qty = best["close_qty"]
    close_price = float(source["row"].get("mid"))
    fee = core.position.calc_option_fee(open_qty + close_qty, 0, config.backtest.option_fee_per_contract)
    margin = sum(
        core.position.margin_call(
            float(spot),
            float(leg["row"].get("strike_price")),
            float(leg["row"].get("mid")),
            float(leg["row"].get("contract_multiplier", config.vol.contract_multiplier)),
        ) * int(leg["qty"])
        for leg in open_legs
    )
    open_value = sum(
        float(leg["row"].get("mid"))
        * int(leg["qty"])
        * float(
            leg["row"].get(
                "contract_multiplier",
                config.vol.contract_multiplier,
            )
        )
        for leg in open_legs
    )
    close_value = (
        close_price
        * int(close_qty)
        * float(source["row"].get("contract_multiplier", config.vol.contract_multiplier))
    )
    close_margin_release = core.position.margin_call(
        float(spot),
        float(source["row"].get("strike_price")),
        close_price,
        float(source["row"].get("contract_multiplier", config.vol.contract_multiplier)),
    ) * int(close_qty)
    etf_correction_cost = float(best["etf_buy_qty"]) * float(spot) * (
        1.0 + float(config.backtest.etf_fee_rate)
    )
    estimated_cash_effect = (
        close_margin_release
        + open_value
        - close_value
        - fee
        - margin
        - etf_correction_cost
    )
    etf_reduction_cash_effect = max(0.0, float(live_account.hedge.qty or 0.0)) * float(
        spot
    ) * (1.0 - float(config.backtest.etf_fee_rate))
    projected_cash = (
        float(live_account.cash)
        + etf_reduction_cash_effect
        + estimated_cash_effect
    )
    min_cash = float(config.backtest.min_cash_reserve)
    if projected_cash < min_cash:
        reduction_item = _live_capacity_reduction_item(
            config,
            live_account,
            chain_df,
            spot,
            required_cash_relief=min_cash - projected_cash,
            reason=(
                "Reduce the short straddle before the planned option-combination "
                "delta hedge would breach the live cash reserve."
            ),
        )
        if reduction_item is not None:
            return reduction_item
        return {
            "action": "DATA_WARNING",
            "priority": "warning",
            "reason": (
                "Planned option-combination delta hedge would breach the live "
                "cash reserve."
            ),
            "projected_cash": projected_cash,
            "min_cash_reserve": min_cash,
        }
    item = {
        "action": (
            "FINAL_OPTION_DELTA_HEDGE_COMBINATION"
            if final
            else "OPTION_DELTA_HEDGE_COMBINATION"
        ),
        "priority": "action",
        "side": "short",
        "reason": (
            "Use liquid lightly-ITM calls to minimize account delta, then minimize "
            "the resulting gamma change."
        ),
        "close_source": source["kind"],
        "close_call_code": source["row"].get("order_book_id"),
        "close_call_qty": close_qty,
        "estimated_close_call_price": close_price,
        "open_call_code": primary_open_row.get("order_book_id"),
        "open_call_qty": open_qty,
        "estimated_open_call_price": float(primary_open_row.get("mid")),
        "open_strike": float(primary_open_row.get("strike_price")),
        "open_expiry": str(pd.Timestamp(primary_open_row.get("maturity_date")).date()),
        "open_legs": [
            {
                "order_book_id": leg["row"].get("order_book_id"),
                "qty": int(leg["qty"]),
                "estimated_price": float(leg["row"].get("mid")),
                "strike": float(leg["row"].get("strike_price")),
                "volume": float(leg["row"].get("volume")),
                "liquidity_capacity": int(leg["liquidity_capacity"]),
            }
            for leg in open_legs
        ],
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
        "estimated_close_margin_release": close_margin_release,
        "estimated_cash_effect": estimated_cash_effect,
        "preceding_etf_reduction_cash_effect": etf_reduction_cash_effect,
        "estimated_etf_correction_cost": etf_correction_cost,
        "projected_cash_after": projected_cash,
        "min_cash_reserve": min_cash,
        "open_liquidity_capacity": sum(
            int(leg["liquidity_capacity"]) for leg in open_legs
        ),
        "close_liquidity_capacity": best["close_liquidity_capacity"],
        "solver_priority": "liquidity_then_delta_then_gamma",
        "delta_neutral_achieved": best["delta_neutral_achieved"],
        "liquidity_capacity_exhausted": best["liquidity_capacity_exhausted"],
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
    calls = core.vol_engine.filter_standard_option_contracts(calls).copy()
    calls["_strike"] = pd.to_numeric(calls["strike_price"], errors="coerce")
    calls["_volume"] = pd.to_numeric(calls.get("volume"), errors="coerce").fillna(-1.0)
    max_itm_ratio = float(
        getattr(config.strategy, "option_delta_hedge_max_itm_ratio", 0.10)
    )
    calls = calls[
        (calls["_strike"] < float(spot))
        & (calls["_strike"] >= float(spot) * (1.0 - max_itm_ratio))
    ]
    calls["_liquidity_capacity"] = calls.apply(
        core.position.liquidity_capacity,
        axis=1,
    )
    return calls[calls["_liquidity_capacity"] > 0].sort_values(
        "_volume",
        ascending=False,
    )


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
