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


# Compatibility fallback for lightweight test/third-party configs that predate
# explicit backtest.long_qty/short_qty. Production live configs always provide
# both fields and therefore have a single parameter source.
LEGACY_ENTRY_QTY_PER_LEG = 10
POSITION_TARGET_KEY = "position_target"
DATA_WARNING_ACTION = "DATA_WARNING"
PLAN_BLOCKED_ACTION = "PLAN_BLOCKED"
RESIDUAL_RISK_ACTION = "RESIDUAL_RISK"
_USE_LIVE_SOURCE = object()


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

    return generate_signal_from_context(
        product=product,
        account_id=account_id,
        config=config,
        live_account=live_account,
        market=market,
    )


def generate_signal_from_context(
    *,
    product,
    config,
    live_account,
    market,
    account_id=None,
    recorded_dividend_adjustments=_USE_LIVE_SOURCE,
    previous_close_context=_USE_LIVE_SOURCE,
):
    """Build a read-only live plan from explicit market and account state.

    The production wrapper above remains responsible for reading SQLite and
    quote storage.  Replay and parity tests pass isolated in-memory state and
    explicit historical dependencies, so they cannot read or mutate the real
    live account while executing the exact same planner.
    """
    account_id = account_id or getattr(live_account, "account_id", "default")

    feature_row = market["signal_row"]
    spot = float(feature_row["close"])
    chain_df = market["chain_df"]
    atm = core.vol_engine.select_atm_from_chain(chain_df, spot)
    signal_state = _strategy_state_for_signal(
        live_account.strategy_state,
        market["signals"],
        market["date"],
    )

    dividend_adjustments = _dividend_adjustments_for_positions(
        live_account.positions,
        chain_df,
        default_multiplier=config.vol.contract_multiplier,
    )
    if dividend_adjustments:
        recorded_dividend_adjustments = []
    elif recorded_dividend_adjustments is _USE_LIVE_SOURCE:
        recorded_dividend_adjustments = _recorded_dividend_adjustments_on_date(
            product,
            account_id,
            market["date"],
        )
    else:
        recorded_dividend_adjustments = list(
            recorded_dividend_adjustments or []
        )
    dividend_liquidation_events = (
        dividend_adjustments or recorded_dividend_adjustments
    )

    strategy_advice = []
    position_greeks = []
    position_greeks_by_side = {}
    position_values = []
    if previous_close_context is _USE_LIVE_SOURCE:
        previous_close_date, previous_close_chain = _load_previous_close_chain(
            product,
            market["date"],
            enabled=(
                config.strategy.short_stop_loss_enabled
                and live_account.positions.get("short") is not None
            ),
        )
    else:
        previous_close_date, previous_close_chain = (
            previous_close_context or (None, None)
        )

    forced_liquidation_ready = False
    if dividend_liquidation_events:
        (
            strategy_advice,
            position_greeks_by_side,
            position_values,
            forced_liquidation_ready,
        ) = _dividend_forced_liquidation_plan(
            config,
            live_account,
            chain_df,
            spot,
            atm,
            market["date"],
            dividend_liquidation_events,
        )
        position_greeks = list(position_greeks_by_side.values())
    else:
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
                previous_close_chain,
                previous_close_date,
            )
            strategy_advice.extend(side_advice)
            if greeks is not None:
                position_greeks.append(greeks)
                position_greeks_by_side[side] = greeks
            if option_value is not None:
                position_values.append(option_value)

    if not dividend_liquidation_events and all(
        value is None for value in live_account.positions.values()
    ):
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
    if dividend_liquidation_events:
        advice = strategy_advice
        planned_greeks = (
            core.backtester.empty_greeks()
            if forced_liquidation_ready
            else account_greeks
        )
    else:
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

    plan_metadata = _plan_metadata(advice)

    current_account_delta = account_greeks["delta"] + live_account.hedge.qty
    current_normalized_account_delta, current_delta_hedge_capacity = (
        core.strategy.normalized_account_delta(
            current_account_delta,
            live_account.positions,
            default_multiplier=config.vol.contract_multiplier,
        )
    )
    final_state = _project_final_account_state(
        config,
        live_account,
        chain_df,
        advice,
        planned_greeks,
    )
    item = {
        "product": product,
        "account_id": account_id,
        "date": str(market["date"].date()),
        "spot": spot,
        "account": live_account.to_dict(),
        "feature": _feature_summary(feature_row),
        "account_greeks": account_greeks,
        "planned_account_greeks": planned_greeks,
        "current_account_delta": current_account_delta,
        "current_normalized_account_delta": current_normalized_account_delta,
        "current_delta_hedge_capacity": current_delta_hedge_capacity,
        "planned_hedge_qty": final_state["hedge_qty"],
        "planned_account_delta": final_state["account_delta"],
        "account_delta_after_hedge": final_state["account_delta"],
        "normalized_account_delta": final_state["normalized_account_delta"],
        "delta_hedge_capacity": final_state["delta_hedge_capacity"],
        "delta_hedge_tolerance_ratio": config.strategy.delta_hedge_tolerance_ratio,
        "delta_residual_abs_tolerance": _delta_residual_abs_tolerance(config),
        "estimated_option_value": sum(position_values),
        "strategy_state": signal_state.to_dict(),
        "advice": advice,
        **plan_metadata,
        "dividend_forced_liquidation": {
            "active": bool(dividend_liquidation_events),
            "ready": bool(forced_liquidation_ready),
            "adjustments": dividend_liquidation_events,
            "detected_from_current_chain": bool(dividend_adjustments),
            "detected_from_account_fills": bool(recorded_dividend_adjustments),
        },
        "data_warning": {
            **market["data_warning"],
            "read_only_signal": True,
        },
    }
    return item


def _diagnostic_item(action, code, reason, *, blocking, **details):
    return {
        "action": action,
        "priority": "warning" if blocking else "notice",
        "code": code,
        "reason": reason,
        "blocking": bool(blocking),
        **details,
    }


def _has_executable_action(item):
    if item.get("priority") != "action":
        return False
    if _has_position_target(item):
        return True
    trade_qty = _number(item.get("trade_etf_qty"))
    return trade_qty is not None and abs(trade_qty) > 1e-9


def _plan_metadata(advice):
    blocking_items = [item for item in advice if bool(item.get("blocking"))]
    residual_items = [
        item for item in advice if item.get("action") == RESIDUAL_RISK_ACTION
    ]
    has_actions = any(_has_executable_action(item) for item in advice)

    if blocking_items:
        status = "BLOCKED"
    elif has_actions and residual_items:
        status = "ACTIONABLE_WITH_RESIDUAL"
    elif has_actions:
        status = "ACTIONABLE"
    elif residual_items:
        status = "NO_ACTION_WITH_RESIDUAL"
    else:
        status = "NO_ACTION"

    residual_risks = [
        {
            key: item.get(key)
            for key in [
                "code",
                "reason",
                "residual_delta",
                "normalized_account_delta",
                "delta_hedge_tolerance_ratio",
                "delta_residual_abs_tolerance",
                "delta_hedge_capacity",
                "blocking",
            ]
            if item.get(key) is not None
        }
        for item in residual_items
    ]
    return {
        "plan_status": status,
        "execution_allowed": bool(has_actions and not blocking_items),
        "residual_risks": residual_risks,
    }


def _delta_residual_abs_tolerance(config):
    return max(
        0.0,
        float(
            getattr(config.strategy, "delta_residual_abs_tolerance", 0.0)
            or 0.0
        ),
    )


def _delta_hedge_triggered(config, account_delta, positions):
    """Return whether delta control should start for the supplied account state.

    The absolute/normalized tolerances are entry conditions only.  Once this
    returns true, the complete execution plan keeps delta control active and
    targets zero instead of treating the tolerance as a residual target band.
    """
    account_delta = float(account_delta)
    if not math.isfinite(account_delta):
        return True
    if abs(account_delta) <= _delta_residual_abs_tolerance(config) + 1e-9:
        return False
    normalized_delta, delta_capacity = core.strategy.normalized_account_delta(
        account_delta,
        positions,
        default_multiplier=config.vol.contract_multiplier,
    )
    tolerance_ratio = float(config.strategy.delta_hedge_tolerance_ratio)
    return not (
        delta_capacity > 0 and abs(normalized_delta) <= tolerance_ratio
    )


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
    trading_calendar = market_data.load_live_trading_calendar()
    history = _load_feature_history(config.data.product)
    seeded_history = False
    data_mode = "daily_eod_reference_incremental"

    if quote_snapshot is not None:
        latest_date = pd.Timestamp(quote_snapshot["quote_date"]).normalize()
        snapshot_etf, latest_opt_by_date = _load_snapshot_quote_series(
            quote_snapshot,
            latest_date,
        )
        trading_calendar = _append_calendar_date(trading_calendar, latest_date)
        data_mode = "live_snapshot_incremental"
    else:
        latest_date = _resolve_latest_signal_date(config, date=date)
        latest_opt_by_date = core.data_loader.load_opt_series(latest_date, latest_date)
        latest_hedge_by_date = core.data_loader.load_hedge_series(
            latest_date,
            latest_date,
        )
        latest_opt_by_date = core.data_loader.attach_underlying_prices(
            latest_opt_by_date,
            latest_hedge_by_date,
        )

    if history is None:
        etf_by_date = core.data_loader.load_etf_series(start, latest_date)
        if quote_snapshot is not None:
            etf_by_date[latest_date] = snapshot_etf
    else:
        incremental_start = _incremental_etf_start_date(
            trading_calendar,
            latest_date,
            config,
        )
        if quote_snapshot is not None:
            try:
                etf_by_date = core.data_loader.load_etf_series(
                    incremental_start,
                    latest_date,
                )
            except ValueError:
                etf_by_date = {}
            etf_by_date[latest_date] = snapshot_etf
        else:
            etf_by_date = core.data_loader.load_etf_series(
                incremental_start,
                latest_date,
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


def _resolve_latest_signal_date(config, date=None, etf_by_date=None):
    if date is not None:
        return pd.Timestamp(date).normalize()

    etf_dir = core.data_loader._resolve_data_dir(config.data.etf_dir)
    opt_dir = core.data_loader._resolve_data_dir(config.data.opt_dir)
    if etf_by_date is None:
        etf_dates = {
            core.data_loader._parse_date_from_file(path, "_price")
            for path in sorted(etf_dir.glob("*_price.parquet"))
        }
    else:
        etf_dates = set(etf_by_date)
    opt_dates = [
        core.data_loader._parse_date_from_file(path, "_chain")
        for path in sorted(opt_dir.glob("*_chain.parquet"))
    ]
    common_dates = sorted(etf_dates & set(opt_dates))
    if not common_dates:
        raise ValueError("No common ETF/option date available for live signal.")
    return common_dates[-1]


def _incremental_etf_start_date(trading_calendar, latest_date, config):
    dates = (
        pd.DatetimeIndex(pd.to_datetime(trading_calendar))
        .normalize()
        .drop_duplicates()
        .sort_values()
    )
    latest_date = pd.Timestamp(latest_date).normalize()
    dates = pd.DatetimeIndex(sorted(set(dates) | {latest_date}))
    dates = dates[dates <= latest_date]
    if len(dates) == 0:
        return latest_date
    hv_windows = getattr(config.vol, "hv_windows", (60,))
    lookback = max(int(window) for window in hv_windows) + 2
    return dates[max(0, len(dates) - lookback)]


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
    cooldown_left = _short_entry_cooldown_left_for_date(
        strategy_state,
        latest_date,
        trading_dates,
    )
    strategy_state.short_entry_cooldown_left = cooldown_left
    if cooldown_left <= 0:
        strategy_state.short_entry_cooldown_total_days = 0
        strategy_state.short_entry_cooldown_started_date = None

    return strategy_state


def _short_entry_cooldown_left_for_date(
    strategy_state,
    latest_date,
    trading_dates,
):
    total_days = int(strategy_state.short_entry_cooldown_total_days or 0)
    current_left = int(strategy_state.short_entry_cooldown_left or 0)
    started_date = _date_or_none(
        strategy_state.short_entry_cooldown_started_date
    )
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


def _dividend_adjustments_for_positions(
    positions,
    chain_df,
    default_multiplier=10000,
):
    """Return held sides whose contract terms show a dividend adjustment.

    The exchange keeps the contract code unchanged when an ETF option is
    adjusted.  Detection therefore combines the authoritative current terms
    with the persisted position terms and the adjusted-contract name suffix.
    """
    adjustments = []
    for side, position in (positions or {}).items():
        if position is None:
            continue
        try:
            call_row = _chain_row(chain_df, position.get("call_code"))
            put_row = _chain_row(chain_df, position.get("put_code"))
        except (IndexError, KeyError):
            continue

        stored_strike = _positive_float(position.get("strike"))
        stored_multiplier = _positive_float(position.get("contract_multiplier"))
        current_strikes = [
            _positive_float(call_row.get("strike_price")),
            _positive_float(put_row.get("strike_price")),
        ]
        current_multipliers = [
            _positive_float(call_row.get("contract_multiplier")),
            _positive_float(put_row.get("contract_multiplier")),
        ]
        symbols = [
            position.get("call_contract_symbol"),
            position.get("put_contract_symbol"),
            call_row.get("contract_symbol"),
            put_row.get("contract_symbol"),
        ]

        strike_changed = stored_strike is not None and any(
            value is not None
            and not math.isclose(value, stored_strike, rel_tol=0.0, abs_tol=1e-9)
            for value in current_strikes
        )
        multiplier_changed = stored_multiplier is not None and any(
            value is not None
            and not math.isclose(value, stored_multiplier, rel_tol=0.0, abs_tol=1e-9)
            for value in current_multipliers
        )
        adjusted_symbol = any(_is_adjusted_option_symbol(value) for value in symbols)
        nonstandard_multiplier = any(
            value is not None
            and not math.isclose(
                value,
                float(default_multiplier),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for value in current_multipliers
        )
        if not (
            strike_changed
            or multiplier_changed
            or adjusted_symbol
            or nonstandard_multiplier
        ):
            continue

        adjustments.append(
            {
                "side": side,
                "call_code": str(position.get("call_code")),
                "put_code": str(position.get("put_code")),
                "stored_strike": stored_strike,
                "current_call_strike": current_strikes[0],
                "current_put_strike": current_strikes[1],
                "stored_contract_multiplier": stored_multiplier,
                "current_call_contract_multiplier": current_multipliers[0],
                "current_put_contract_multiplier": current_multipliers[1],
                "current_call_contract_symbol": _optional_text(
                    call_row.get("contract_symbol")
                ),
                "current_put_contract_symbol": _optional_text(
                    put_row.get("contract_symbol")
                ),
                "evidence": {
                    "strike_changed": strike_changed,
                    "contract_multiplier_changed": multiplier_changed,
                    "adjusted_contract_symbol": adjusted_symbol,
                    "nonstandard_contract_multiplier": nonstandard_multiplier,
                },
            }
        )
    return adjustments


def _recorded_dividend_adjustments_on_date(product, account_id, signal_date):
    target_date = pd.Timestamp(signal_date).normalize()
    try:
        fills = account_store.list_fills(
            product,
            account_id=account_id,
            include_voided=False,
        )
    except (OSError, ValueError):
        return []
    result = []
    for row in fills:
        payload = row.get("payload") or {}
        if str(payload.get("action") or "").lower() != "option_contract_adjustment":
            continue
        fill_date = _date_or_none(payload.get("date"))
        if fill_date != target_date:
            continue
        result.append(
            {
                "side": payload.get("side"),
                "call_code": payload.get("call_code"),
                "put_code": payload.get("put_code"),
                "stored_strike": payload.get("old_strike"),
                "current_call_strike": payload.get("new_strike")
                or payload.get("strike"),
                "current_put_strike": payload.get("new_strike")
                or payload.get("strike"),
                "stored_contract_multiplier": payload.get(
                    "old_contract_multiplier"
                ),
                "current_call_contract_multiplier": payload.get(
                    "new_contract_multiplier",
                    payload.get("contract_multiplier"),
                ),
                "current_put_contract_multiplier": payload.get(
                    "new_contract_multiplier",
                    payload.get("contract_multiplier"),
                ),
                "source": "account_fill",
            }
        )
    return result


def _is_adjusted_option_symbol(value):
    text = _optional_text(value)
    if text is None:
        return False
    text = text.upper()
    return text.endswith("A") or "\u8c03\u6574" in text


def _optional_text(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _dividend_forced_liquidation_plan(
    config,
    live_account,
    chain_df,
    spot,
    atm,
    signal_date,
    adjustments,
):
    """Build an all-or-nothing option and ETF liquidation plan."""
    resolved = []
    missing = []
    greeks_by_side = {}
    position_values = []
    for side, position in live_account.positions.items():
        if position is None:
            continue
        try:
            call_row, put_row = core.vol_engine.resolve_position_pair(
                position,
                chain_df,
            )
        except IndexError:
            missing.append(
                {
                    "side": side,
                    "call_code": position.get("call_code"),
                    "put_code": position.get("put_code"),
                }
            )
            continue
        pricing_position, pre_close_adjustment = _position_with_current_contract_terms(
            position,
            call_row,
            put_row,
            signal_date,
        )
        option_value = core.position.value(pricing_position, call_row, put_row)
        greeks_by_side[side] = core.strategy.calc_position_greeks(
            call_row,
            put_row,
            pricing_position["call_qty"],
            pricing_position["put_qty"],
            side=side,
        )
        position_values.append(
            core.position.signed_value(pricing_position, call_row, put_row)
        )
        resolved.append(
            (
                side,
                pricing_position,
                call_row,
                put_row,
                option_value,
                pre_close_adjustment,
            )
        )

    if missing:
        return (
            [
                _diagnostic_item(
                    DATA_WARNING_ACTION,
                    "DIVIDEND_POSITION_PRICE_MISSING",
                    (
                        "Dividend-adjusted option position detected, but the latest "
                        "chain cannot price every held leg. No partial liquidation was "
                        "planned; close all option legs and the corresponding ETF "
                        "position manually, then import executions for settlement."
                    ),
                    blocking=True,
                    dividend_forced_liquidation=True,
                    missing_positions=missing,
                    adjustments=adjustments,
                )
            ],
            greeks_by_side,
            position_values,
            False,
        )

    reason = "dividend_adjusted_contract_forced_liquidation"
    actions = []
    adjusted_sides = [item["side"] for item in adjustments]
    for (
        side,
        position,
        call_row,
        put_row,
        option_value,
        pre_close_adjustment,
    ) in resolved:
        item = _close_advice(
            side,
            position,
            call_row,
            put_row,
            option_value,
            reason,
        )
        item.update(
            {
                "dividend_forced_liquidation": True,
                "liquidate_on_date": str(pd.Timestamp(signal_date).date()),
                "adjusted_sides": adjusted_sides,
                "adjustments": adjustments,
                "requires_execution_and_settlement": True,
            }
        )
        if pre_close_adjustment is not None:
            item["pre_close_contract_adjustment"] = pre_close_adjustment
        actions.append(item)

    current_hedge_qty = float(live_account.hedge.qty or 0.0)
    if abs(current_hedge_qty) > 1e-9:
        underlying_order_book_id = (
            live_account.hedge.underlying_order_book_id
            or next(
                (
                    position.get("underlying_order_book_id")
                    for _, position, _, _, _, _ in resolved
                    if position.get("underlying_order_book_id")
                ),
                None,
            )
            or _underlying_id_from_atm(atm)
        )
        if underlying_order_book_id is None:
            return (
                [
                    _diagnostic_item(
                        DATA_WARNING_ACTION,
                        "DIVIDEND_UNDERLYING_CODE_MISSING",
                        (
                            "Dividend-adjusted option position detected, but the "
                            "corresponding ETF code is unavailable. No partial "
                            "liquidation was planned; close options and ETF manually, "
                            "then import executions for settlement."
                        ),
                        blocking=True,
                        dividend_forced_liquidation=True,
                        adjustments=adjustments,
                    )
                ],
                greeks_by_side,
                position_values,
                False,
            )
        hedge_item = _etf_hedge_item(
            "FINAL_DELTA_HEDGE",
            reason,
            0.0,
            current_hedge_qty,
            current_hedge_qty,
            0.0,
            -current_hedge_qty,
            spot,
            underlying_order_book_id,
            after_actions=[item["action"] for item in actions],
        )
        hedge_item.update(
            {
                "dividend_forced_liquidation": True,
                "liquidate_on_date": str(pd.Timestamp(signal_date).date()),
                "requires_execution_and_settlement": True,
            }
        )
        actions.append(hedge_item)

    if not actions:
        actions.append(
            {
                "action": "DIVIDEND_LIQUIDATION_SETTLED",
                "priority": "info",
                "reason": (
                    "Dividend-adjusted option liquidation has been settled for "
                    "the signal date; no new position may be opened today."
                ),
                "dividend_forced_liquidation": True,
                "liquidate_on_date": str(pd.Timestamp(signal_date).date()),
                "adjustments": adjustments,
            }
        )

    return actions, greeks_by_side, position_values, True


def _position_with_current_contract_terms(position, call_row, put_row, signal_date):
    old_multiplier = _positive_float(position.get("contract_multiplier"))
    call_multiplier = _positive_float(call_row.get("contract_multiplier"))
    put_multiplier = _positive_float(put_row.get("contract_multiplier"))
    new_multiplier = call_multiplier or put_multiplier or old_multiplier
    if new_multiplier is None:
        return dict(position), None
    if (
        call_multiplier is not None
        and put_multiplier is not None
        and not math.isclose(
            call_multiplier,
            put_multiplier,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            "Adjusted call and put contract multipliers do not match: "
            f"{call_multiplier} != {put_multiplier}"
        )

    old_strike = _positive_float(position.get("strike"))
    call_strike = _positive_float(call_row.get("strike_price"))
    put_strike = _positive_float(put_row.get("strike_price"))
    new_strike = call_strike or put_strike or old_strike
    multiplier_changed = old_multiplier is not None and not math.isclose(
        old_multiplier,
        new_multiplier,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    strike_changed = (
        old_strike is not None
        and new_strike is not None
        and not math.isclose(
            old_strike,
            new_strike,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    if not multiplier_changed and not strike_changed:
        return dict(position), None
    if old_multiplier is None or old_multiplier <= 0:
        raise ValueError("Held adjusted option position has no valid old contract multiplier.")

    price_ratio = old_multiplier / new_multiplier
    adjusted = dict(position)
    adjusted.update(
        {
            "strike": new_strike,
            "contract_multiplier": new_multiplier,
            "call_contract_symbol": _optional_text(call_row.get("contract_symbol")),
            "put_contract_symbol": _optional_text(put_row.get("contract_symbol")),
            "entry_call_price": float(position.get("entry_call_price", 0.0) or 0.0)
            * price_ratio,
            "entry_put_price": float(position.get("entry_put_price", 0.0) or 0.0)
            * price_ratio,
        }
    )
    adjustment_fill = {
        **adjusted,
        "action": "option_contract_adjustment",
        "date": str(pd.Timestamp(signal_date).date()),
        "cash_delta": 0.0,
        "old_strike": old_strike,
        "new_strike": new_strike,
        "old_contract_multiplier": old_multiplier,
        "new_contract_multiplier": new_multiplier,
        "adjustment_price_ratio": price_ratio,
        "last_call_price": float(call_row.get("mid", 0.0) or 0.0),
        "last_put_price": float(put_row.get("mid", 0.0) or 0.0),
        "import_source": "signal_pre_close_contract_adjustment",
        "source_limitations": [
            "authoritative live contract terms are applied before forced close",
            "entry premiums are inversely adjusted to preserve historical cost notional",
            "corporate action has no trade cash flow or realized pnl",
        ],
    }
    adjustment_fill["last_option_value"] = core.position.signed_value(
        adjusted,
        call_row,
        put_row,
    )
    return adjusted, adjustment_fill


def _date_from_filename(path):
    match = Path(path).name
    parsed = pd.Series([match]).str.extract(r"(20\d{2})_(\d{2})_(\d{2})").iloc[0]
    if parsed.isna().any():
        return None
    return "-".join(str(item) for item in parsed)


def _entry_advice(config, feature_row, atm, spot, strategy_state):
    if atm is None:
        return [
            _diagnostic_item(
                DATA_WARNING_ACTION,
                "ATM_PAIR_MISSING",
                "No valid ATM call/put pair found.",
                blocking=True,
            )
        ]

    advice = []
    for side, signal_col, action, qty in [
        ("long", "long_open_signal", "OPEN_LONG_STRADDLE", config.backtest.long_qty),
        ("short", "short_open_signal", "OPEN_SHORT_STRADDLE", config.backtest.short_qty),
    ]:
        if not bool(feature_row.get(signal_col, False)):
            continue
        cooldown_left = int(strategy_state.short_entry_cooldown_left or 0)
        if side == "short" and cooldown_left > 0:
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
    item = {
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
    _set_position_target(
        item,
        call_code=item["call_code"],
        put_code=item["put_code"],
        call_qty=qty,
        put_qty=qty,
        strike=item["strike"],
        expiry=item["expiry"],
    )
    return item


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
    previous_close_chain=None,
    previous_close_date=None,
):
    try:
        call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
    except IndexError:
        return (
            [
                _diagnostic_item(
                    DATA_WARNING_ACTION,
                    "POSITION_CONTRACT_MISSING",
                    "Current position contracts are missing from latest chain.",
                    blocking=True,
                    side=side,
                )
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
    stop_loss_metrics = None

    if side == "short":
        close_reason = core.strategy.get_short_close_reason(
            feature_row,
            position_dte,
            position,
            config=config,
        )
        stop_loss_metrics = _short_daily_loss_aum_metrics(
            position,
            option_value,
            previous_close_chain,
            previous_close_date,
            spot,
            config,
        )
        if stop_loss_metrics and core.strategy.is_short_daily_loss_aum_stop(
            stop_loss_metrics["short_daily_pnl"],
            stop_loss_metrics["short_stop_loss_aum"],
            config=config,
        ):
            close_reason = "short_daily_loss_aum_stop"
        if core.position.has_short_volume_spike(
            position,
            call_row,
            put_row,
            config=config,
        ):
            close_reason = "short_volume_spike"
    else:
        close_reason = core.strategy.get_close_reason(
            feature_row,
            position_dte,
            config=config,
        )

    if close_reason:
        item = _close_advice(
            side,
            position,
            call_row,
            put_row,
            option_value,
            close_reason,
        )
        if close_reason == "short_daily_loss_aum_stop":
            item.update(stop_loss_metrics)
        advice.append(item)
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
        )
        if roll_payload:
            if roll_payload.get("close_current_position"):
                item = _close_advice(
                    side,
                    position,
                    call_row,
                    put_row,
                    option_value,
                    roll_payload["reason"],
                )
                item.update(roll_payload)
                advice.append(item)
                return advice, greeks, core.position.signed_value(
                    position,
                    call_row,
                    put_row,
                )
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
            _set_position_target(
                item,
                call_code=item["target_call_code"],
                put_code=item["target_put_code"],
                call_qty=item["target_call_qty"],
                put_qty=item["target_put_qty"],
                strike=item.get("target_strike"),
                expiry=item.get("target_expiry"),
            )
            advice.append(item)

    return advice, greeks, core.position.signed_value(position, call_row, put_row)


def _load_previous_close_chain(product, current_date, enabled=True):
    if not enabled:
        return None, None
    try:
        snapshot = market_data.load_previous_quote_snapshot(product, current_date)
        snapshot_time = pd.to_datetime(
            snapshot.get("snapshot_stamp"),
            format="%Y%m%d_%H%M%S",
            errors="coerce",
        )
        if pd.isna(snapshot_time) or snapshot_time.time() < pd.Timestamp(
            "15:00"
        ).time():
            return None, None
        chain = pd.read_parquet(snapshot["option_snapshot"]).copy()
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None, None
    if not {"order_book_id", "bid", "ask"}.issubset(chain.columns):
        return None, None
    chain["order_book_id"] = chain["order_book_id"].astype(str)
    bid = pd.to_numeric(chain["bid"], errors="coerce")
    ask = pd.to_numeric(chain["ask"], errors="coerce")
    midpoint = (bid + ask) / 2.0
    valid_midpoint = bid.gt(0) & ask.gt(0)
    close = pd.to_numeric(chain.get("close"), errors="coerce")
    chain["mid"] = midpoint.where(valid_midpoint, close)
    return pd.Timestamp(snapshot["quote_date"]).normalize(), chain


def _short_daily_loss_aum_metrics(
    position,
    current_market_value,
    previous_close_chain,
    previous_close_date,
    spot,
    config,
):
    if previous_close_chain is None or previous_close_date is None:
        return None
    try:
        previous_call, previous_put = core.vol_engine.resolve_position_pair(
            position,
            previous_close_chain,
        )
        previous_market_value = core.position.value(
            position,
            previous_call,
            previous_put,
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if pd.isna(previous_market_value) or pd.isna(current_market_value):
        return None

    call_qty = abs(float(position.get("call_qty", 0.0) or 0.0))
    put_qty = abs(float(position.get("put_qty", 0.0) or 0.0))
    multiplier = float(
        position.get("contract_multiplier") or config.vol.contract_multiplier
    )
    aum = max(call_qty, put_qty) * multiplier * float(spot)
    if aum <= 0:
        return None
    daily_pnl = float(previous_market_value) - float(current_market_value)
    item = {
        "short_stop_loss_previous_date": str(previous_close_date.date()),
        "short_previous_close_value": float(previous_market_value),
        "short_current_market_value": float(current_market_value),
        "short_daily_pnl": daily_pnl,
        "short_stop_loss_aum": aum,
        "short_daily_pnl_aum_ratio": daily_pnl / aum,
        "short_daily_loss_aum_threshold": (
            config.strategy.short_daily_loss_aum_threshold
        ),
    }


def _close_advice(side, position, call_row, put_row, option_value, reason):
    fee = core.position.calc_option_fee(position["call_qty"], position["put_qty"])
    cash_effect = -option_value - fee + position.get("option_margin", 0.0) if side == "short" else option_value - fee
    item = {
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
        "contract_multiplier": float(
            position.get(
                "contract_multiplier",
                call_row.get("contract_multiplier", 10000),
            )
            or call_row.get("contract_multiplier", 10000)
        ),
        "underlying_order_book_id": position.get("underlying_order_book_id")
        or _projected_underlying_id(call_row, put_row),
    }
    _set_position_target(item, close=True)
    return item


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
):
    feature_atm_strike = feature_row.get("atm_strike", pd.NA)
    atm_strike_source = "current_signal_row"
    if pd.isna(feature_atm_strike):
        feature_atm_strike = atm.get("strike") if atm is not None else pd.NA
        atm_strike_source = "current_atm_selection"
    if pd.isna(feature_atm_strike):
        try:
            feature_atm_strike, atm_strike_source = _atm_strike_for_roll_check(
                product,
                signals,
                latest_date,
                latest_date,
            )
        except ValueError:
            return None

    dte_too_low = position_dte <= config.strategy.roll_dte_threshold
    strike_roll_ready = core.vol_engine.strike_differs_by_at_least_one_step(
        position.get("strike"),
        feature_atm_strike,
        chain_df,
    )
    if not dte_too_low and not strike_roll_ready:
        return None

    mismatch_days = 1 if strike_roll_ready else 0
    if not (dte_too_low or strike_roll_ready):
        return None

    roll_trigger = "dte" if dte_too_low else "strike"
    if roll_trigger == "strike":
        target_atm = core.vol_engine.select_atm_from_chain_for_expiry(
            chain_df,
            spot,
            position.get("expiry"),
        )
    else:
        target_atm = core.vol_engine.select_atm_from_chain(
            chain_df,
            spot,
            target_dte_min=config.strategy.roll_dte_threshold + 1,
        )
    if target_atm is None:
        return {
            "close_current_position": True,
            "reason": (
                "Roll is required, but no eligible replacement ATM straddle is "
                "available. Close the current position; do not substitute a "
                "delta hedge for this roll."
            ),
            "roll_failure_reason": "roll_condition_active_but_no_valid_target_atm",
            "strike_mismatch_days": mismatch_days,
            "strike_mismatch_days_source": atm_strike_source,
            "strike_mismatch_trace": [],
            "latest_holding_snapshot_date": None,
            "snapshot_lag_days": None,
            "target_strike": None,
            "target_expiry": None,
            "roll_trigger": roll_trigger,
        }

    reasons = []
    if dte_too_low:
        reasons.append("dte_below_roll_threshold")
    if strike_roll_ready:
        reasons.append("held_strike_differs_from_current_atm")
    max_qty = config.backtest.short_qty if side == "short" else config.backtest.long_qty
    item = {
        "reason": "+".join(reasons),
        "strike_mismatch_days": mismatch_days,
        "strike_mismatch_days_source": atm_strike_source,
        "strike_mismatch_trace": [],
        "latest_holding_snapshot_date": None,
        "snapshot_lag_days": None,
        "target_call_code": target_atm["call"]["order_book_id"],
        "target_put_code": target_atm["put"]["order_book_id"],
        "target_strike": float(target_atm["strike"]),
        "target_expiry": str(pd.Timestamp(target_atm["expiry"]).date()),
        "target_call_qty": max_qty,
        "target_put_qty": max_qty,
        "estimated_target_call_price": float(target_atm["call"]["mid"]),
        "estimated_target_put_price": float(target_atm["put"]["mid"]),
        "roll_trigger": roll_trigger,
    }
    return item


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
        and _has_position_target(item)
    ]
    if not option_actions:
        plan.extend(_hedge_advice(config, live_account, account_greeks, spot, chain_df, atm))
        option_actions = [
            item
            for item in plan
            if item.get("priority") == "action" and _has_position_target(item)
        ]
        if not option_actions:
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
            _diagnostic_item(
                PLAN_BLOCKED_ACTION,
                "MAIN_POSITION_CASH_RESERVE_BREACH",
                "Planned main-position action would breach the live cash reserve.",
                blocking=True,
                projected_cash=projected_cash,
                min_cash_reserve=min_cash,
            )
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
        account_greeks,
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
            "expiry": item.get("target_expiry") or call_row.get("maturity_date"),
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
        return _diagnostic_item(
            PLAN_BLOCKED_ACTION,
            "CAPACITY_REDUCTION_INSUFFICIENT",
            (
                "Live account capacity is tight, but reducing the current short "
                "straddle would not release enough cash or margin."
            ),
            blocking=True,
            cash=float(live_account.cash),
            min_cash_reserve=min_cash,
            total_margin=capacity["total_margin"],
            capital_occupation=capacity["capital_occupation"],
            margin_limit=margin_limit,
        )

    target_qty = current_qty - close_qty
    estimated_cash_effect = cash_relief_per_contract * close_qty
    item = {
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
    _set_position_target(
        item,
        call_code=item["call_code"],
        put_code=item["put_code"],
        call_qty=target_qty,
        put_qty=target_qty,
    )
    return item


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


def _set_position_target(
    item,
    *,
    close=False,
    call_code=None,
    put_code=None,
    call_qty=None,
    put_qty=None,
    strike=None,
    expiry=None,
):
    """Declare the post-trade option position for generic plan projection.

    Action names are presentation/execution labels.  Planning relies only on this
    explicit state transition, so a new option action does not need registration
    in a central action-name allowlist.
    """
    if close:
        item[POSITION_TARGET_KEY] = None
        return item
    if None in {call_code, put_code, call_qty, put_qty}:
        raise ValueError("A position target requires both legs and quantities.")
    item[POSITION_TARGET_KEY] = {
        "call_code": str(call_code),
        "put_code": str(put_code),
        "call_qty": int(call_qty),
        "put_qty": int(put_qty),
        "strike": strike,
        "expiry": expiry,
    }
    return item


def _has_position_target(item):
    if POSITION_TARGET_KEY in item:
        return True
    # Compatibility for persisted/third-party plans created before the explicit
    # position-target contract. New actions should always use position_target.
    action = str(item.get("action") or "")
    return action.startswith(("OPEN_", "ROLL_", "CLOSE_"))


def _position_target_is_close(item):
    if POSITION_TARGET_KEY in item:
        return item[POSITION_TARGET_KEY] is None
    return str(item.get("action") or "").startswith("CLOSE_")


def _project_greeks_after_plan(option_actions, chain_df, current_greeks_by_side):
    projected_by_side = dict(current_greeks_by_side)
    for item in option_actions:
        side = item.get("side")
        if side not in account_store.POSITION_SIDES:
            continue

        if _position_target_is_close(item):
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
    target = item.get(POSITION_TARGET_KEY)
    if isinstance(target, dict):
        return (
            target.get("call_code"),
            target.get("put_code"),
            target.get("call_qty"),
            target.get("put_qty"),
        )
    action = str(item.get("action") or "")
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
        reason="Account delta exceeds tolerance.",
    )


def _final_hedge_advice(
    config,
    live_account,
    current_greeks,
    planned_greeks,
    spot,
    option_actions,
    chain_df,
    atm,
):
    if not config.strategy.enable_delta_hedge:
        return []

    planned_positions = _positions_after_option_actions(
        config,
        live_account.positions,
        option_actions,
        chain_df,
    )
    projected_account = deepcopy(live_account)
    projected_account.positions = planned_positions
    projected_account.hedge.qty = _projected_hedge_qty(live_account, option_actions)
    projected_cash = _projected_cash_after_option_actions(
        config,
        live_account,
        option_actions,
        chain_df,
        spot,
    )
    if projected_cash is not None:
        projected_account.cash = projected_cash
    planned_option_delta = float(planned_greeks["delta"])
    planned_account_delta = planned_option_delta + projected_account.hedge.qty
    current_account_delta = (
        float(current_greeks["delta"]) + float(live_account.hedge.qty or 0.0)
    )
    triggered_before_option_plan = _delta_hedge_triggered(
        config,
        current_account_delta,
        live_account.positions,
    )
    triggered_after_option_plan = _delta_hedge_triggered(
        config,
        planned_account_delta,
        planned_positions,
    )
    if not (triggered_before_option_plan or triggered_after_option_plan):
        return []

    normalized_delta, delta_capacity = core.strategy.normalized_account_delta(
        planned_account_delta,
        planned_positions,
        default_multiplier=config.vol.contract_multiplier,
    )
    tolerance_ratio = float(config.strategy.delta_hedge_tolerance_ratio)
    tolerance = delta_capacity * tolerance_ratio

    plan = _delta_hedge_plan(
        config,
        projected_account,
        planned_greeks,
        spot,
        chain_df,
        atm,
        action="FINAL_DELTA_HEDGE",
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
        positions_for_tolerance=planned_positions,
        force_to_zero=True,
    )
    for item in plan:
        item.setdefault("planned_account_gamma", planned_greeks["gamma"])
        item.setdefault("planned_account_vega", planned_greeks["vega"])
        item.setdefault("planned_account_theta", planned_greeks["theta"])
        item.setdefault("hedge_tolerance", tolerance)
        item.setdefault("normalized_account_delta", normalized_delta)
        item.setdefault("delta_hedge_tolerance_ratio", tolerance_ratio)
        item.setdefault(
            "delta_residual_abs_tolerance",
            _delta_residual_abs_tolerance(config),
        )
        item.setdefault("delta_hedge_capacity", delta_capacity)
        item.setdefault(
            "delta_hedge_triggered_before_option_plan",
            triggered_before_option_plan,
        )
        item.setdefault(
            "delta_hedge_triggered_after_option_plan",
            triggered_after_option_plan,
        )
        item.setdefault("delta_hedge_target", 0.0)
    return plan


def _positions_after_option_actions(config, current_positions, option_actions, chain_df):
    planned_positions = dict(current_positions)
    for item in option_actions:
        side = item.get("side")
        if side not in account_store.POSITION_SIDES:
            continue
        if _position_target_is_close(item):
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


def _projected_hedge_qty(live_account, actions):
    qty = float(live_account.hedge.qty or 0.0)
    for item in actions:
        target_qty = item.get("target_hedge_qty")
        if target_qty is not None:
            qty = float(target_qty)
    return qty


def _project_final_account_state(
    config,
    live_account,
    chain_df,
    advice,
    planned_greeks,
):
    """Project the one authoritative account state after every planned action."""
    option_actions = [
        item
        for item in advice
        if item.get("priority") == "action" and _has_position_target(item)
    ]
    positions = _positions_after_option_actions(
        config,
        live_account.positions,
        option_actions,
        chain_df,
    )
    hedge_qty = _projected_hedge_qty(live_account, advice)
    account_delta = float(planned_greeks.get("delta", 0.0) or 0.0) + hedge_qty
    normalized_delta, delta_capacity = core.strategy.normalized_account_delta(
        account_delta,
        positions,
        default_multiplier=config.vol.contract_multiplier,
    )
    return {
        "positions": positions,
        "hedge_qty": hedge_qty,
        "account_delta": account_delta,
        "normalized_account_delta": normalized_delta,
        "delta_hedge_capacity": delta_capacity,
    }


def _delta_hedge_plan(
    config,
    live_account,
    greeks,
    spot,
    chain_df,
    atm,
    action,
    reason,
    after_actions=None,
    underlying_order_book_id=None,
    positions_for_tolerance=None,
    force_to_zero=False,
):
    if not config.strategy.enable_delta_hedge:
        return []

    option_delta = float(greeks["delta"])
    current_hedge_qty = float(live_account.hedge.qty or 0.0)
    spot = float(spot)
    if (
        not math.isfinite(option_delta)
        or not math.isfinite(current_hedge_qty)
        or not math.isfinite(spot)
    ):
        return [
            _diagnostic_item(
                DATA_WARNING_ACTION,
                "DELTA_HEDGE_INPUT_NOT_FINITE",
                "Cannot evaluate delta hedge because account delta, ETF hedge quantity, or spot is not finite.",
                blocking=True,
                option_delta=option_delta if math.isfinite(option_delta) else None,
                current_hedge_qty=(
                    current_hedge_qty if math.isfinite(current_hedge_qty) else None
                ),
                estimated_price=spot if math.isfinite(spot) else None,
            )
        ]
    account_delta = option_delta + current_hedge_qty
    normalized_delta, delta_capacity = core.strategy.normalized_account_delta(
        account_delta,
        (
            positions_for_tolerance
            if positions_for_tolerance is not None
            else live_account.positions
        ),
        default_multiplier=config.vol.contract_multiplier,
    )
    tolerance_ratio = float(config.strategy.delta_hedge_tolerance_ratio)
    residual_abs_tolerance = _delta_residual_abs_tolerance(config)
    tolerance = delta_capacity * tolerance_ratio
    if not force_to_zero and not _delta_hedge_triggered(
        config,
        account_delta,
        (
            positions_for_tolerance
            if positions_for_tolerance is not None
            else live_account.positions
        ),
    ):
        return []

    target_qty = core.strategy.round_etf_hedge_target(-option_delta)
    if underlying_order_book_id is None:
        underlying_order_book_id = _underlying_id_from_atm(atm)

    shape_rebalance_item = None
    if (
        getattr(config.strategy, "enable_atm_straddle_rebalance", False)
        and _can_plan_option_delta_hedge_after(after_actions)
    ):
        shape_rebalance_item = _atm_straddle_shape_rebalance_item(
            config,
            live_account,
            chain_df,
            spot,
            atm,
            underlying_order_book_id,
            option_delta=option_delta,
            account_delta=account_delta,
            after_actions=after_actions,
            final=action.startswith("FINAL_"),
        )
    if shape_rebalance_item is not None:
        if shape_rebalance_item.get("delta_tolerance_met"):
            return [shape_rebalance_item]

    if getattr(config.strategy, "allow_etf_short_hedge", True) or target_qty >= 0:
        trade_qty = target_qty - current_hedge_qty
        if abs(trade_qty) <= 1e-9:
            return [shape_rebalance_item] if shape_rebalance_item is not None else []

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
                _diagnostic_item(
                    PLAN_BLOCKED_ACTION,
                    "ETF_HEDGE_CASH_RESERVE_BREACH",
                    "Planned ETF delta hedge would breach the live cash reserve.",
                    blocking=True,
                    projected_cash=projected_cash,
                    min_cash_reserve=min_cash,
                )
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
                "delta_residual_abs_tolerance": residual_abs_tolerance,
                "delta_hedge_capacity": delta_capacity,
                "hedge_tolerance": tolerance,
                "delta_hedge_target": 0.0,
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
    constrained_residual_tolerance = max(tolerance, residual_abs_tolerance)
    if abs(residual_delta) <= constrained_residual_tolerance:
        return plan
    if residual_delta < 0:
        plan.append(
            _diagnostic_item(
                RESIDUAL_RISK_ACTION,
                "NEGATIVE_DELTA_WITH_ETF_SHORT_DISABLED",
                (
                    "ETF short hedge is disabled and residual delta is negative; "
                    "the available option-leg adjustment cannot remove the remaining "
                    "exposure. This is a residual risk, not a market-data failure."
                ),
                blocking=False,
                residual_delta=residual_delta,
                normalized_account_delta=(
                    abs(residual_delta) / delta_capacity if delta_capacity > 0 else None
                ),
                delta_hedge_tolerance_ratio=tolerance_ratio,
                delta_residual_abs_tolerance=residual_abs_tolerance,
                delta_hedge_capacity=delta_capacity,
                after_actions=after_actions,
            )
        )
        return plan
    if not getattr(config.strategy, "enable_atm_straddle_rebalance", False):
        plan.append(
            _diagnostic_item(
                RESIDUAL_RISK_ACTION,
                "OPTION_DELTA_REBALANCE_DISABLED",
                (
                    "ETF short hedge is disabled and option delta rebalance is not "
                    "enabled, so residual delta remains. This is a strategy constraint, "
                    "not a market-data failure."
                ),
                blocking=False,
                residual_delta=residual_delta,
                normalized_account_delta=(
                    abs(residual_delta) / delta_capacity if delta_capacity > 0 else None
                ),
                delta_hedge_tolerance_ratio=tolerance_ratio,
                delta_residual_abs_tolerance=residual_abs_tolerance,
                delta_hedge_capacity=delta_capacity,
                after_actions=after_actions,
            )
        )
        return plan

    can_plan_rebalance = _can_plan_option_delta_hedge_after(after_actions)
    rebalance_item = None
    if can_plan_rebalance:
        rebalance_item = _atm_straddle_delta_rebalance_item(
            config,
            live_account,
            chain_df,
            residual_delta,
            spot,
            atm,
            underlying_order_book_id,
            after_actions=after_actions,
            current_hedge_qty=etf_target_qty,
            option_delta=option_delta,
            final=action.startswith("FINAL_"),
        )
    if rebalance_item is not None and rebalance_item.get("blocking"):
        return [rebalance_item]
    if rebalance_item is not None:
        plan.append(rebalance_item)
    else:
        code = (
            "DELTA_RESIDUAL_AFTER_OPTION_PLAN"
            if not can_plan_rebalance
            else "DELTA_REBALANCE_CONSTRAINT_RESIDUAL"
        )
        detail = (
            "Residual delta remains after the option rebalance; no further "
            "same-signal iteration is planned"
            if not can_plan_rebalance
            else "No option-leg candidate satisfies all planning constraints"
        )
        plan.append(
            _diagnostic_item(
                RESIDUAL_RISK_ACTION,
                code,
                f"{detail}.",
                blocking=False,
                residual_delta=residual_delta,
                normalized_account_delta=(
                    abs(residual_delta) / delta_capacity if delta_capacity > 0 else None
                ),
                delta_hedge_tolerance_ratio=tolerance_ratio,
                delta_residual_abs_tolerance=residual_abs_tolerance,
                delta_hedge_capacity=delta_capacity,
                after_actions=after_actions,
            )
        )
    return plan


def _can_plan_option_delta_hedge_after(after_actions):
    if not after_actions:
        return True
    return all(
        str(action or "").startswith(("OPEN_", "ROLL_"))
        for action in after_actions
    )


def _atm_rebalance_target_pair_qty(config, side="short"):
    explicit = getattr(config.strategy, "atm_rebalance_target_pair_qty", None)
    if explicit is not None:
        return int(explicit)
    if side == "short":
        configured_qty = getattr(config.backtest, "short_qty", None)
    else:
        configured_qty = getattr(config.backtest, "long_qty", None)
    if configured_qty is not None:
        return int(configured_qty)
    return LEGACY_ENTRY_QTY_PER_LEG


def _atm_rebalance_context(
    config,
    live_account,
    chain_df,
    spot,
    atm,
    require_imbalanced=False,
    side="short",
):
    position = (live_account.positions or {}).get(side)
    if position is None:
        return None
    if side == "short" and not _position_is_atm_tolerated_main_straddle(
        position, chain_df, spot, atm
    ):
        return None
    try:
        call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
    except IndexError:
        return None

    call_qty = int(position.get("call_qty", 0) or 0)
    put_qty = int(position.get("put_qty", 0) or 0)
    if put_qty <= 0 or (require_imbalanced and call_qty <= 0):
        return None
    if require_imbalanced and call_qty == put_qty:
        return None

    call_price = _positive_float(call_row.get("mid"))
    put_price = _positive_float(put_row.get("mid"))
    multiplier = float(
        call_row.get("contract_multiplier", config.vol.contract_multiplier)
        or config.vol.contract_multiplier
    )
    target_pair_qty = _atm_rebalance_target_pair_qty(config, side=side)
    if (
        call_price is None
        or put_price is None
        or multiplier <= 0
        or target_pair_qty <= 0
    ):
        return None
    return {
        "position": position,
        "side": side,
        "call_row": call_row,
        "put_row": put_row,
        "call_qty": call_qty,
        "put_qty": put_qty,
        "call_price": call_price,
        "put_price": put_price,
        "multiplier": multiplier,
        "target_pair_qty": target_pair_qty,
        "target_pair_value": (
            target_pair_qty * call_price + target_pair_qty * put_price
        )
        * multiplier,
    }


def _atm_rebalance_candidate(
    context,
    *,
    open_call_qty=0,
    close_call_qty=0,
    open_put_qty=0,
    close_put_qty=0,
    base_delta,
    current_hedge_qty,
    allow_etf_short,
    delta_tolerance,
    absolute_delta_tolerance=0.0,
    delta_not_worse_than=None,
):
    call_row = context["call_row"]
    put_row = context["put_row"]
    multiplier = context["multiplier"]
    target_call_qty = (
        context["call_qty"] - close_call_qty + open_call_qty
    )
    target_put_qty = context["put_qty"] - close_put_qty + open_put_qty
    if target_call_qty <= 0 or target_put_qty <= 0:
        return None

    delta_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        open_call_qty,
        close_call_qty,
        open_put_qty,
        close_put_qty,
        multiplier,
        "delta",
        side=context.get("side", "short"),
    )
    projected_delta = float(base_delta) + delta_effect
    raw_target_hedge_qty = core.strategy.round_etf_hedge_target(-projected_delta)
    target_hedge_qty = (
        raw_target_hedge_qty
        if allow_etf_short or raw_target_hedge_qty >= 0
        else 0.0
    )
    combined_delta = projected_delta + target_hedge_qty
    projected_capacity = (target_call_qty + target_put_qty) * multiplier
    normalized_projected_delta = abs(projected_delta) / projected_capacity
    normalized_combined_delta = abs(combined_delta) / projected_capacity
    # ETF hedges are rounded to board lots. A configured zero tolerance must not
    # make an otherwise valid option-shape improvement impossible merely because
    # the final ETF correction leaves less than half a board lot of delta.
    rounding_delta_tolerance = (
        float(core.strategy.ETF_HEDGE_LOT_SIZE) / 2.0 / projected_capacity
    )
    absolute_normalized_delta_tolerance = (
        max(0.0, float(absolute_delta_tolerance)) / projected_capacity
    )
    effective_delta_tolerance = max(
        float(delta_tolerance),
        rounding_delta_tolerance,
        absolute_normalized_delta_tolerance,
    )
    delta_not_worse = (
        True
        if delta_not_worse_than is None
        else abs(combined_delta) <= abs(delta_not_worse_than) + 1e-9
    )

    call_price = context["call_price"]
    put_price = context["put_price"]
    premium_effect = (
        open_call_qty * call_price
        + open_put_qty * put_price
        - close_call_qty * call_price
        - close_put_qty * put_price
    ) * multiplier
    target_straddle_value = (
        target_call_qty * call_price + target_put_qty * put_price
    ) * multiplier
    call_deviation = abs(target_call_qty - context["target_pair_qty"])
    put_deviation = abs(target_put_qty - context["target_pair_qty"])
    shape_error = abs(target_call_qty - target_put_qty)
    target_total_qty = 2 * context["target_pair_qty"]
    target_total_qty_deviation = abs(
        target_call_qty + target_put_qty - target_total_qty
    )
    position_total_qty_change = abs(
        target_call_qty
        + target_put_qty
        - context["call_qty"]
        - context["put_qty"]
    )
    return {
        "open_call_qty": open_call_qty,
        "close_call_qty": close_call_qty,
        "open_put_qty": open_put_qty,
        "close_put_qty": close_put_qty,
        "target_call_qty": target_call_qty,
        "target_put_qty": target_put_qty,
        "delta_effect": delta_effect,
        "projected_delta": projected_delta,
        "projected_option_delta": projected_delta,
        "target_hedge_qty": target_hedge_qty,
        "trade_etf_qty": target_hedge_qty - current_hedge_qty,
        "etf_delta_correction": target_hedge_qty,
        "combined_delta": combined_delta,
        "normalized_projected_delta": normalized_projected_delta,
        "normalized_combined_delta": normalized_combined_delta,
        "configured_delta_tolerance": float(delta_tolerance),
        "absolute_delta_tolerance": max(0.0, float(absolute_delta_tolerance)),
        "absolute_normalized_delta_tolerance": absolute_normalized_delta_tolerance,
        "rounding_delta_tolerance": rounding_delta_tolerance,
        "effective_delta_tolerance": effective_delta_tolerance,
        "delta_not_worse": delta_not_worse,
        "delta_tolerance_met": (
            normalized_combined_delta <= effective_delta_tolerance + 1e-12
            and delta_not_worse
        ),
        "premium_effect": premium_effect,
        "shape_error": shape_error,
        "ratio_error": shape_error / float(max(target_call_qty, target_put_qty)),
        "target_pair_qty": context["target_pair_qty"],
        "target_pair_qty_deviation": call_deviation + put_deviation,
        "target_pair_qty_deviation_balance": abs(call_deviation - put_deviation),
        "target_total_qty": target_total_qty,
        "target_total_qty_deviation": target_total_qty_deviation,
        "position_total_qty_change": position_total_qty_change,
        "target_pair_value": context["target_pair_value"],
        "target_straddle_value": target_straddle_value,
        "target_pair_value_error": (
            target_straddle_value - context["target_pair_value"]
        ),
    }


def _atm_straddle_shape_rebalance_item(
    config,
    live_account,
    chain_df,
    spot,
    atm,
    underlying_order_book_id,
    option_delta=0.0,
    account_delta=0.0,
    after_actions=None,
    final=False,
):
    context = _atm_rebalance_context(
        config,
        live_account,
        chain_df,
        spot,
        atm,
        require_imbalanced=True,
    )
    if context is None:
        return None
    short_position = context["position"]
    call_row = context["call_row"]
    put_row = context["put_row"]
    current_call_qty = context["call_qty"]
    current_put_qty = context["put_qty"]
    call_price = context["call_price"]
    put_price = context["put_price"]
    multiplier = context["multiplier"]
    target_pair_qty = context["target_pair_qty"]

    close_call_max = max(current_call_qty - target_pair_qty, 0)
    close_put_max = max(current_put_qty - target_pair_qty, 0)
    open_call_max = max(target_pair_qty - current_call_qty, 0)
    open_put_max = max(target_pair_qty - current_put_qty, 0)
    call_capacity = core.position.liquidity_capacity(call_row)
    put_capacity = core.position.liquidity_capacity(put_row)
    if call_capacity > 0:
        open_call_max = min(open_call_max, call_capacity)
    if put_capacity > 0:
        open_put_max = min(open_put_max, put_capacity)
    if close_call_max + close_put_max + open_call_max + open_put_max <= 0:
        return None

    current_shape_error = abs(current_call_qty - current_put_qty)
    current_pair_qty_deviation = (
        abs(current_call_qty - target_pair_qty)
        + abs(current_put_qty - target_pair_qty)
    )
    current_hedge_qty = float(live_account.hedge.qty or 0.0)
    allow_etf_short_hedge = getattr(config.strategy, "allow_etf_short_hedge", True)
    normalized_delta_tolerance = max(
        0.0, float(config.strategy.delta_hedge_tolerance_ratio)
    )
    absolute_delta_tolerance = _delta_residual_abs_tolerance(config)
    target_pair_value = context["target_pair_value"]

    best = None
    fallback_best = None
    for close_call_qty in range(0, close_call_max + 1):
        for open_call_qty in range(0, open_call_max + 1):
            for close_put_qty in range(0, close_put_max + 1):
                for open_put_qty in range(0, open_put_max + 1):
                    option_trade_qty = (
                        close_call_qty
                        + open_call_qty
                        + close_put_qty
                        + open_put_qty
                    )
                    if option_trade_qty <= 0:
                        continue
                    candidate = _atm_rebalance_candidate(
                        context,
                        open_call_qty=open_call_qty,
                        close_call_qty=close_call_qty,
                        open_put_qty=open_put_qty,
                        close_put_qty=close_put_qty,
                        base_delta=option_delta,
                        current_hedge_qty=current_hedge_qty,
                        allow_etf_short=allow_etf_short_hedge,
                        delta_tolerance=normalized_delta_tolerance,
                        absolute_delta_tolerance=absolute_delta_tolerance,
                        delta_not_worse_than=account_delta,
                    )
                    if candidate is None:
                        continue

                    shape_error = candidate["shape_error"]
                    target_pair_qty_deviation = candidate[
                        "target_pair_qty_deviation"
                    ]
                    if (
                        shape_error >= current_shape_error
                        and target_pair_qty_deviation >= current_pair_qty_deviation
                    ):
                        continue

                    crosses_to_short_etf = (
                        current_hedge_qty >= 0
                        and candidate["target_hedge_qty"] < 0
                    )
                    score = (
                        0 if candidate["delta_tolerance_met"] else 1,
                        1 if crosses_to_short_etf else 0,
                        0 if candidate["position_total_qty_change"] == 0 else 1,
                        shape_error,
                        candidate["position_total_qty_change"],
                        target_pair_qty_deviation,
                        candidate["target_pair_qty_deviation_balance"],
                        abs(candidate["trade_etf_qty"]),
                        option_trade_qty,
                        abs(candidate["target_pair_value_error"]),
                    )
                    fallback_score = (
                        1 if crosses_to_short_etf else 0,
                        abs(candidate["combined_delta"]),
                        candidate["normalized_combined_delta"],
                        target_pair_qty_deviation,
                        shape_error,
                        option_trade_qty,
                    )
                    candidate["score"] = score
                    candidate["fallback_score"] = fallback_score
                    if best is None or score < best["score"]:
                        best = candidate
                    if (
                        fallback_best is None
                        or fallback_score < fallback_best["fallback_score"]
                    ):
                        fallback_best = candidate

    if best is None:
        best = fallback_best
    if best is None:
        return None

    current_margin = _current_short_margin(
        config,
        short_position,
        call_row,
        put_row,
        spot,
    )
    target_margin = core.position.calc_short_margin(
        call_row,
        put_row,
        best["target_call_qty"],
        best["target_put_qty"],
        spot,
    )
    fee = core.position.calc_option_fee(
        best["open_call_qty"] + best["close_call_qty"],
        best["open_put_qty"] + best["close_put_qty"],
        config.backtest.option_fee_per_contract,
    )
    cash_delta = best["premium_effect"] - fee
    etf_correction_cost = float(best["trade_etf_qty"]) * float(spot) * (
        1.0 + float(config.backtest.etf_fee_rate)
    )
    projected_cash = float(live_account.cash) + cash_delta - etf_correction_cost
    min_cash = float(config.backtest.min_cash_reserve)
    if projected_cash < min_cash:
        return None

    gamma_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        best["open_call_qty"],
        best["close_call_qty"],
        best["open_put_qty"],
        best["close_put_qty"],
        multiplier,
        "gamma",
    )
    vega_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        best["open_call_qty"],
        best["close_call_qty"],
        best["open_put_qty"],
        best["close_put_qty"],
        multiplier,
        "vega",
    )
    theta_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        best["open_call_qty"],
        best["close_call_qty"],
        best["open_put_qty"],
        best["close_put_qty"],
        multiplier,
        "theta",
    )

    item = {
        "action": (
            "FINAL_ATM_STRADDLE_SHAPE_REBALANCE"
            if final
            else "ATM_STRADDLE_SHAPE_REBALANCE"
        ),
        "priority": "action",
        "side": "short",
        "reason": (
            "Current ATM short straddle legs are imbalanced; rebalance option legs "
            "toward the target straddle shape first, then use ETF only for the "
            "remaining delta gap."
        ),
        "close_source": "core_short_straddle",
        "close_call_code": call_row.get("order_book_id"),
        "close_call_qty": best["close_call_qty"],
        "estimated_close_call_price": call_price,
        "open_call_code": call_row.get("order_book_id"),
        "open_call_qty": best["open_call_qty"],
        "estimated_open_call_price": call_price,
        "close_put_code": put_row.get("order_book_id"),
        "close_put_qty": best["close_put_qty"],
        "estimated_close_put_price": put_price,
        "open_put_code": put_row.get("order_book_id"),
        "open_put_qty": best["open_put_qty"],
        "estimated_open_put_price": put_price,
        "current_call_code": call_row.get("order_book_id"),
        "current_put_code": put_row.get("order_book_id"),
        "current_call_qty": current_call_qty,
        "current_put_qty": current_put_qty,
        "target_call_code": call_row.get("order_book_id"),
        "target_put_code": put_row.get("order_book_id"),
        "target_call_qty": best["target_call_qty"],
        "target_put_qty": best["target_put_qty"],
        "strike": float(short_position.get("strike", call_row.get("strike_price"))),
        "expiry": short_position.get("expiry")
        or str(pd.Timestamp(call_row.get("maturity_date")).date()),
        "estimated_delta_effect": best["delta_effect"],
        "estimated_gamma_effect": gamma_effect,
        "estimated_vega_effect": vega_effect,
        "estimated_theta_effect": theta_effect,
        "account_delta_before_shape_rebalance": account_delta,
        "projected_account_delta_after_option_rebalance": (
            best["projected_option_delta"] + current_hedge_qty
        ),
        "projected_option_delta_after_shape_rebalance": best["projected_option_delta"],
        "etf_delta_correction": best["trade_etf_qty"],
        "projected_account_delta_after_combined_hedge": best["combined_delta"],
        "option_delta": option_delta,
        "current_hedge_qty": current_hedge_qty,
        "target_hedge_qty": best["target_hedge_qty"],
        "trade_etf_qty": best["trade_etf_qty"],
        "estimated_price": float(spot),
        "estimated_fee": fee,
        "estimated_cash_effect": cash_delta,
        "estimated_etf_correction_cost": etf_correction_cost,
        "estimated_option_margin": target_margin,
        "estimated_margin_change": target_margin - current_margin,
        "estimated_market_value_effect": best["premium_effect"],
        "market_value_preservation_error": abs(best["premium_effect"]),
        "target_call_put_ratio_error": best["ratio_error"],
        "target_pair_qty": target_pair_qty,
        "target_pair_qty_deviation": best["target_pair_qty_deviation"],
        "target_pair_qty_deviation_balance": (
            best["target_pair_qty_deviation_balance"]
        ),
        "target_pair_market_value": best["target_pair_value"],
        "target_straddle_market_value": best["target_straddle_value"],
        "target_pair_market_value_error": best["target_pair_value_error"],
        "normalized_combined_delta": best["normalized_combined_delta"],
        "normalized_delta_tolerance": best["effective_delta_tolerance"],
        "configured_delta_tolerance": best["configured_delta_tolerance"],
        "absolute_delta_tolerance": best["absolute_delta_tolerance"],
        "rounding_delta_tolerance": best["rounding_delta_tolerance"],
        "delta_not_worse": best["delta_not_worse"],
        "delta_tolerance_met": best["delta_tolerance_met"],
        "target_total_qty": best["target_total_qty"],
        "target_total_qty_deviation": best["target_total_qty_deviation"],
        "position_total_qty_change": best["position_total_qty_change"],
        "projected_cash_after": projected_cash,
        "min_cash_reserve": min_cash,
        "solver_priority": (
            "effective_delta_tolerance_then_preserve_total_option_qty_then_balance_legs_then_etf"
        ),
        "underlying_order_book_id": underlying_order_book_id,
    }
    _set_position_target(
        item,
        call_code=item["target_call_code"],
        put_code=item["target_put_code"],
        call_qty=item["target_call_qty"],
        put_qty=item["target_put_qty"],
        strike=item.get("strike"),
        expiry=item.get("expiry"),
    )
    if after_actions is not None:
        item["after_actions"] = after_actions
    return item


def _atm_straddle_delta_rebalance_item(
    config,
    live_account,
    chain_df,
    residual_delta,
    spot,
    atm,
    underlying_order_book_id,
    after_actions=None,
    current_hedge_qty=0.0,
    option_delta=0.0,
    final=False,
):
    positions = live_account.positions or {}
    side = "short" if positions.get("short") is not None else "long"
    if positions.get(side) is None:
        return None
    context = _atm_rebalance_context(
        config,
        live_account,
        chain_df,
        spot,
        atm,
        side=side,
    )
    if context is None:
        return None
    position = context["position"]
    call_row = context["call_row"]
    put_row = context["put_row"]
    current_call_qty = context["call_qty"]
    current_put_qty = context["put_qty"]
    call_price = context["call_price"]
    put_price = context["put_price"]
    multiplier = context["multiplier"]

    if side == "short":
        open_row = call_row
        one_open_delta_effect = (
            -float(call_row.get("delta", 0.0) or 0.0) * multiplier
        )
        current_opposite_qty = current_put_qty
    else:
        open_row = put_row
        one_open_delta_effect = (
            float(put_row.get("delta", 0.0) or 0.0) * multiplier
        )
        current_opposite_qty = current_call_qty
    if one_open_delta_effect >= 0:
        return None
    needed_open_only = int(
        math.ceil(float(residual_delta) / abs(one_open_delta_effect))
    )
    max_open_qty = max(
        current_opposite_qty * 4,
        needed_open_only + current_opposite_qty,
        1,
    )
    open_capacity = core.position.liquidity_capacity(open_row)
    if open_capacity > 0:
        max_open_qty = min(max_open_qty, open_capacity)

    best = None
    fallback_best = None
    target_pair_qty = context["target_pair_qty"]
    target_pair_value = context["target_pair_value"]
    normalized_delta_tolerance = max(
        0.0, float(config.strategy.delta_hedge_tolerance_ratio)
    )
    absolute_delta_tolerance = _delta_residual_abs_tolerance(config)
    max_close_qty = current_put_qty if side == "short" else current_call_qty
    for close_qty in range(0, max_close_qty + 1):
        for open_qty in range(0, max_open_qty + 1):
            if close_qty <= 0 and open_qty <= 0:
                continue
            adjustments = (
                {"open_call_qty": open_qty, "close_put_qty": close_qty}
                if side == "short"
                else {"close_call_qty": close_qty, "open_put_qty": open_qty}
            )
            candidate = _atm_rebalance_candidate(
                context,
                **adjustments,
                base_delta=residual_delta,
                current_hedge_qty=0.0,
                allow_etf_short=False,
                delta_tolerance=normalized_delta_tolerance,
                absolute_delta_tolerance=absolute_delta_tolerance,
            )
            if candidate is None:
                continue
            score = (
                0 if candidate["delta_tolerance_met"] else 1,
                candidate["target_pair_qty_deviation_balance"],
                candidate["target_pair_qty_deviation"],
                abs(candidate["target_pair_value_error"]),
                open_qty + close_qty,
            )
            fallback_score = (
                abs(candidate["combined_delta"]),
                candidate["target_pair_qty_deviation_balance"],
                candidate["target_pair_qty_deviation"],
                abs(candidate["target_pair_value_error"]),
                open_qty + close_qty,
            )
            candidate["score"] = score
            candidate["fallback_score"] = fallback_score
            if candidate["delta_tolerance_met"] and (
                best is None or score < best["score"]
            ):
                best = candidate
            if fallback_best is None or fallback_score < fallback_best["fallback_score"]:
                fallback_best = candidate
    if best is None:
        best = fallback_best
    if best is None:
        return None

    target_call_qty = best["target_call_qty"]
    target_put_qty = best["target_put_qty"]
    if side == "short":
        current_margin = _current_short_margin(
            config,
            position,
            call_row,
            put_row,
            spot,
        )
        target_margin = core.position.calc_short_margin(
            call_row,
            put_row,
            target_call_qty,
            target_put_qty,
            spot,
        )
    else:
        current_margin = float(position.get("option_margin", 0.0) or 0.0)
        target_margin = 0.0
    fee = core.position.calc_option_fee(
        best["open_call_qty"] + best["close_call_qty"],
        best["open_put_qty"] + best["close_put_qty"],
        config.backtest.option_fee_per_contract,
    )
    cash_delta = (1.0 if side == "short" else -1.0) * best["premium_effect"] - fee
    etf_correction_cost = float(best["etf_delta_correction"]) * float(spot) * (
        1.0 + float(config.backtest.etf_fee_rate)
    )
    projected_cash = float(live_account.cash) + cash_delta - etf_correction_cost
    min_cash = float(config.backtest.min_cash_reserve)
    if projected_cash < min_cash:
        return _diagnostic_item(
            PLAN_BLOCKED_ACTION,
            "OPTION_REBALANCE_CASH_RESERVE_BREACH",
            "Planned ATM straddle delta rebalance would breach the live cash reserve.",
            blocking=True,
            projected_cash=projected_cash,
            min_cash_reserve=min_cash,
        )

    gamma_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        best["open_call_qty"],
        best["close_call_qty"],
        best["open_put_qty"],
        best["close_put_qty"],
        multiplier,
        "gamma",
        side=side,
    )
    vega_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        best["open_call_qty"],
        best["close_call_qty"],
        best["open_put_qty"],
        best["close_put_qty"],
        multiplier,
        "vega",
        side=side,
    )
    theta_effect = _atm_shape_rebalance_greek_effect(
        call_row,
        put_row,
        best["open_call_qty"],
        best["close_call_qty"],
        best["open_put_qty"],
        best["close_put_qty"],
        multiplier,
        "theta",
        side=side,
    )
    combined_delta = best["combined_delta"]
    item = {
        "action": (
            "FINAL_ATM_STRADDLE_DELTA_REBALANCE"
            if final
            else "ATM_STRADDLE_DELTA_REBALANCE"
        ),
        "priority": "action",
        "side": side,
        "reason": (
            "ETF hedge is reduced to zero and residual positive delta remains; "
            f"rebalance only the current {side} straddle legs to minimize delta "
            "while preserving total option market value."
        ),
        "close_source": f"core_{side}_straddle",
        "close_call_code": call_row.get("order_book_id"),
        "close_call_qty": best["close_call_qty"],
        "estimated_close_call_price": call_price,
        "close_put_code": put_row.get("order_book_id"),
        "close_put_qty": best["close_put_qty"],
        "estimated_close_put_price": put_price,
        "open_call_code": call_row.get("order_book_id"),
        "open_call_qty": best["open_call_qty"],
        "estimated_open_call_price": call_price,
        "open_put_code": put_row.get("order_book_id"),
        "open_put_qty": best["open_put_qty"],
        "estimated_open_put_price": put_price,
        "current_call_code": call_row.get("order_book_id"),
        "current_put_code": put_row.get("order_book_id"),
        "current_call_qty": current_call_qty,
        "current_put_qty": current_put_qty,
        "target_call_code": call_row.get("order_book_id"),
        "target_put_code": put_row.get("order_book_id"),
        "target_call_qty": target_call_qty,
        "target_put_qty": target_put_qty,
        "strike": float(position.get("strike", call_row.get("strike_price"))),
        "expiry": position.get("expiry")
        or str(pd.Timestamp(call_row.get("maturity_date")).date()),
        "estimated_delta_effect": best["delta_effect"],
        "estimated_gamma_effect": gamma_effect,
        "estimated_vega_effect": vega_effect,
        "estimated_theta_effect": theta_effect,
        "residual_delta_before_option_rebalance": residual_delta,
        "projected_account_delta_after_option_rebalance": best["projected_delta"],
        "etf_delta_correction": best["etf_delta_correction"],
        "projected_account_delta_after_combined_hedge": combined_delta,
        "option_delta": option_delta,
        "current_hedge_qty": current_hedge_qty,
        "target_hedge_qty": current_hedge_qty + best["etf_delta_correction"],
        "trade_etf_qty": best["etf_delta_correction"],
        "estimated_price": float(spot),
        "estimated_fee": fee,
        "estimated_cash_effect": cash_delta,
        "estimated_etf_correction_cost": etf_correction_cost,
        "estimated_option_margin": target_margin,
        "estimated_margin_change": target_margin - current_margin,
        "estimated_market_value_effect": best["premium_effect"],
        "market_value_preservation_error": abs(best["premium_effect"]),
        "target_call_put_ratio_error": best["ratio_error"],
        "target_pair_qty": best["target_pair_qty"],
        "target_pair_qty_deviation": best["target_pair_qty_deviation"],
        "target_pair_qty_deviation_balance": best["target_pair_qty_deviation_balance"],
        "target_pair_market_value": best["target_pair_value"],
        "target_straddle_market_value": best["target_straddle_value"],
        "target_pair_market_value_error": best["target_pair_value_error"],
        "normalized_projected_delta": best["normalized_projected_delta"],
        "normalized_combined_delta": best["normalized_combined_delta"],
        "normalized_delta_tolerance": normalized_delta_tolerance,
        "absolute_delta_tolerance": best["absolute_delta_tolerance"],
        "delta_tolerance_met": best["delta_tolerance_met"],
        "projected_cash_after": projected_cash,
        "min_cash_reserve": min_cash,
        "solver_priority": (
            "configured_delta_tolerance_then_target_pair_qty_deviation_balance_then_deviation_then_market_value"
        ),
        "underlying_order_book_id": underlying_order_book_id,
        "open_legs": [
            leg
            for leg in [
                {
                    "order_book_id": call_row.get("order_book_id"),
                    "option_type": "c",
                    "qty": best["open_call_qty"],
                    "estimated_price": call_price,
                    "strike": float(call_row.get("strike_price")),
                    "volume": call_row.get("volume"),
                    "liquidity_capacity": core.position.liquidity_capacity(call_row),
                },
                {
                    "order_book_id": put_row.get("order_book_id"),
                    "option_type": "p",
                    "qty": best["open_put_qty"],
                    "estimated_price": put_price,
                    "strike": float(put_row.get("strike_price")),
                    "volume": put_row.get("volume"),
                    "liquidity_capacity": core.position.liquidity_capacity(put_row),
                },
            ]
            if leg["qty"] > 0
        ],
    }
    _set_position_target(
        item,
        call_code=item["target_call_code"],
        put_code=item["target_put_code"],
        call_qty=item["target_call_qty"],
        put_qty=item["target_put_qty"],
        strike=item.get("strike"),
        expiry=item.get("expiry"),
    )
    if after_actions is not None:
        item["after_actions"] = after_actions
    return item


def _position_is_atm_tolerated_main_straddle(position, chain_df, spot, atm):
    if atm is None:
        return False
    call = atm.get("call")
    put = atm.get("put")
    if call is None or put is None:
        return False
    if str(position.get("call_code")) == str(call.get("order_book_id")) and str(
        position.get("put_code")
    ) == str(put.get("order_book_id")):
        return True

    try:
        position_strike = float(position.get("strike"))
        atm_strike = float(atm.get("strike"))
    except (TypeError, ValueError):
        return False
    if math.isnan(position_strike) or math.isnan(atm_strike):
        return False

    within_one_strike = core.vol_engine.strikes_within_chain_steps(
        chain_df,
        position_strike,
        atm_strike,
        max_steps=1,
    )
    if within_one_strike is not None:
        return within_one_strike

    step = core.vol_engine.strike_step_from_chain(chain_df, atm_strike)
    if step is None or step <= 0:
        try:
            return math.isclose(position_strike, float(spot)) or math.isclose(
                position_strike,
                atm_strike,
            )
        except (TypeError, ValueError):
            return math.isclose(position_strike, atm_strike)
    deviation = abs(position_strike - atm_strike)
    return deviation < float(step) or math.isclose(deviation, float(step))


def _positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or number <= 0:
        return None
    return number


def _atm_rebalance_greek_effect(
    call_row,
    put_row,
    open_call_qty,
    close_put_qty,
    multiplier,
    greek_key,
):
    call_value = float(call_row.get(greek_key, 0.0) or 0.0)
    put_value = float(put_row.get(greek_key, 0.0) or 0.0)
    return (
        -float(open_call_qty) * call_value
        + float(close_put_qty) * put_value
    ) * float(multiplier)


def _atm_shape_rebalance_greek_effect(
    call_row,
    put_row,
    open_call_qty,
    close_call_qty,
    open_put_qty,
    close_put_qty,
    multiplier,
    greek_key,
    side="short",
):
    call_value = float(call_row.get(greek_key, 0.0) or 0.0)
    put_value = float(put_row.get(greek_key, 0.0) or 0.0)
    short_effect = (
        -float(open_call_qty) * call_value
        + float(close_call_qty) * call_value
        - float(open_put_qty) * put_value
        + float(close_put_qty) * put_value
    ) * float(multiplier)
    return short_effect if side == "short" else -short_effect


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
        "estimated_delta_effect": trade_qty,
        "projected_account_delta_after_hedge": option_delta + target_qty,
        "estimated_price": spot,
        "underlying_order_book_id": underlying_order_book_id,
    }
    if action.startswith("FINAL_"):
        item["planned_option_delta"] = option_delta
        item["planned_account_delta"] = account_delta
    if after_actions is not None:
        item["after_actions"] = after_actions
    return item


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
