from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

import core
from core.backtest_strategies import create_strategy

from . import account as account_store
from . import etf_netting
from . import market_data
from . import signal_engine


@dataclass(frozen=True)
class SnapshotReplayEvent:
    product: str
    signal_path: Path
    signal_timestamp: pd.Timestamp
    quote_date: pd.Timestamp
    snapshot_stamp: str
    etf_snapshot: Path
    option_snapshot: Path
    feature: dict
    embedded_account: dict


def discover_signal_events(
    product: str,
    *,
    start=None,
    end=None,
    output_root="output/live",
    quote_root="data/live",
) -> list[SnapshotReplayEvent]:
    """Return immutable quote checkpoints referenced by saved live signals.

    Paths embedded in old JSON files can contain replacement characters after a
    machine/user-name migration.  Snapshot files are therefore resolved from
    the product, quote date and immutable snapshot stamp instead of trusting an
    absolute path serialized by another environment.
    """
    start = pd.Timestamp(start).normalize() if start is not None else None
    end = pd.Timestamp(end).normalize() if end is not None else None
    events = []
    seen = set()
    signal_dir = Path(output_root) / product
    for path in sorted(signal_dir.glob("*_signal.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        snapshot = payload.get("quote_snapshot") or {}
        stamp = str(snapshot.get("snapshot_stamp") or "")
        quote_date = pd.to_datetime(
            snapshot.get("quote_date") or payload.get("date"),
            errors="coerce",
        )
        if not stamp or pd.isna(quote_date):
            continue
        quote_date = pd.Timestamp(quote_date).normalize()
        if start is not None and quote_date < start:
            continue
        if end is not None and quote_date > end:
            continue

        # Multiple reports can be rendered from exactly the same immutable
        # quote. Replaying it twice would invent a second decision event.
        key = (quote_date, stamp)
        if key in seen:
            continue
        seen.add(key)

        day_dir = Path(quote_root) / product / "quotes" / quote_date.strftime("%Y%m%d")
        time_part = stamp.split("_", 1)[-1]
        etf_path = day_dir / f"{time_part}_etf.parquet"
        option_path = day_dir / f"{time_part}_option_chain.parquet"
        if not etf_path.exists() or not option_path.exists():
            continue
        signal_timestamp = pd.to_datetime(
            path.stem.split("_signal", 1)[0],
            format="%Y%m%d_%H%M%S",
            errors="coerce",
        )
        if pd.isna(signal_timestamp):
            signal_timestamp = pd.Timestamp(
                f"{quote_date.date()} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
            )
        events.append(
            SnapshotReplayEvent(
                product=product,
                signal_path=path,
                signal_timestamp=pd.Timestamp(signal_timestamp),
                quote_date=quote_date,
                snapshot_stamp=stamp,
                etf_snapshot=etf_path,
                option_snapshot=option_path,
                feature=dict(payload.get("feature") or {}),
                embedded_account=dict(payload.get("account") or {}),
            )
        )
    return sorted(events, key=lambda item: (item.signal_timestamp, item.snapshot_stamp))


def account_from_dict(payload: dict, *, product: str | None = None) -> account_store.AccountState:
    payload = deepcopy(payload or {})
    hedge = payload.get("hedge") or {}
    strategy_state = payload.get("strategy_state") or {}
    return account_store.AccountState(
        product=product or payload.get("product"),
        account_id=str(payload.get("account_id") or "replay"),
        cash=float(payload.get("cash", 0.0) or 0.0),
        positions={
            side: deepcopy((payload.get("positions") or {}).get(side))
            for side in account_store.POSITION_SIDES
        },
        hedge=account_store.HedgeState(
            qty=float(hedge.get("qty", 0.0) or 0.0),
            entry_price=float(hedge.get("entry_price", 0.0) or 0.0),
            margin=float(hedge.get("margin", 0.0) or 0.0),
            underlying_order_book_id=hedge.get("underlying_order_book_id"),
            latest_price=_optional_float(hedge.get("latest_price")),
            last_market_value=_optional_float(hedge.get("last_market_value")),
            last_unrealized_pnl=_optional_float(hedge.get("last_unrealized_pnl")),
            last_mark_date=hedge.get("last_mark_date"),
        ),
        strategy_state=account_store.StrategyState(
            short_entry_cooldown_left=int(
                strategy_state.get("short_entry_cooldown_left", 0) or 0
            ),
            short_entry_cooldown_total_days=int(
                strategy_state.get("short_entry_cooldown_total_days", 0) or 0
            ),
            short_entry_cooldown_started_date=(
                strategy_state.get("short_entry_cooldown_started_date")
            ),
        ),
        reset_at=payload.get("reset_at"),
        updated_at=payload.get("updated_at"),
    )


def run_snapshot_replay(
    product: str,
    *,
    config,
    start=None,
    end=None,
    initial_account: dict | account_store.AccountState | None = None,
    output_root="output/live",
    quote_root="data/live",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Replay current live policy over saved immutable signal snapshots.

    The function is intentionally storage-isolated: it never calls
    ``load_account``, ``save_account`` or ``record_fill``.  Every state change is
    applied to a deep-copied in-memory account.
    """
    all_events = discover_signal_events(
        product,
        end=end,
        output_root=output_root,
        quote_root=quote_root,
    )
    start_date = pd.Timestamp(start).normalize() if start is not None else None
    events = [
        event
        for event in all_events
        if start_date is None or event.quote_date >= start_date
    ]
    if not events:
        raise ValueError(f"No replayable live signal snapshots found for {product}.")

    if initial_account is None:
        if not events[0].embedded_account:
            raise ValueError("First replay event has no embedded initial account state.")
        replay_account = account_from_dict(events[0].embedded_account, product=product)
    elif isinstance(initial_account, account_store.AccountState):
        replay_account = deepcopy(initial_account)
    else:
        replay_account = account_from_dict(initial_account, product=product)
    replay_account.account_id = "snapshot_replay"

    strategy_plugin = create_strategy("live_straddle", config)
    # Saved pre-interval features are immutable signal-time observations. They
    # seed lagged predicates such as prev_signal_iv without executing a trade or
    # importing any old account state.
    feature_rows: dict[pd.Timestamp, dict] = {
        event.quote_date: event.feature
        for event in all_events
        if event.quote_date < events[0].quote_date
    }
    previous_events = [
        event for event in all_events if event.quote_date < events[0].quote_date
    ]
    if previous_events:
        previous_event = previous_events[-1]
        previous_close_context = (
            previous_event.quote_date,
            _chain_for_event(previous_event, config),
        )
    else:
        previous_close_context = (None, None)
    current_event_date = None
    last_chain_for_date = None
    event_records = []
    trade_records = []
    plans = []

    for event in events:
        if current_event_date is not None and event.quote_date != current_event_date:
            previous_close_context = (current_event_date, last_chain_for_date)
        current_event_date = event.quote_date
        feature_rows[event.quote_date] = event.feature
        features = pd.DataFrame.from_dict(feature_rows, orient="index").sort_index()
        features.index = pd.to_datetime(features.index)
        signals = strategy_plugin.build_signals(features)
        market, chain_df = _market_for_event(event, config, signals)
        nav_before = account_nav(replay_account, chain_df, float(market["signal_row"]["close"]))
        plan = signal_engine.generate_signal_from_context(
            product=product,
            account_id="snapshot_replay",
            config=config,
            live_account=replay_account,
            market=market,
            recorded_dividend_adjustments=[],
            previous_close_context=previous_close_context,
        )
        _apply_effective_strategy_state(replay_account, plan.get("strategy_state"))
        plans.append(plan)
        executed = []
        if plan.get("execution_allowed"):
            executed = execute_plan_in_memory(
                replay_account,
                plan,
                chain_df,
                config,
                event.signal_timestamp,
            )
            trade_records.extend(executed)
        nav_after = account_nav(replay_account, chain_df, float(market["signal_row"]["close"]))
        event_records.append(
            {
                "product": product,
                "signal_timestamp": event.signal_timestamp,
                "date": event.quote_date,
                "snapshot_stamp": event.snapshot_stamp,
                "signal_path": str(event.signal_path),
                "spot": float(market["signal_row"]["close"]),
                "nav_before": nav_before,
                "nav_after": nav_after,
                "event_pnl": nav_after - nav_before,
                "plan_status": plan.get("plan_status"),
                "execution_allowed": bool(plan.get("execution_allowed")),
                "actions": ",".join(
                    str(item.get("action")) for item in plan.get("advice", [])
                ),
                "executed_trade_count": len(executed),
                "planned_account_delta": plan.get("planned_account_delta"),
                "planned_hedge_qty": plan.get("planned_hedge_qty"),
                "ending_cash": float(replay_account.cash),
                "ending_position_fingerprint": json.dumps(
                    position_fingerprint(replay_account),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "ending_hedge_qty": float(replay_account.hedge.qty or 0.0),
            }
        )
        last_chain_for_date = chain_df

    events_df = pd.DataFrame(event_records)
    daily_df = _daily_from_events(events_df)
    trades_df = pd.DataFrame(trade_records)
    return daily_df, trades_df, plans


def _market_for_event(event, config, signals):
    chain_df = _chain_for_event(event, config)
    signal_row = signals.loc[event.quote_date].copy()
    # Preserve the exact captured underlying reference. Recomputed features use
    # the same snapshot but can otherwise inherit an end-of-day canonical close.
    signal_row["close"] = float(event.feature["close"])
    return (
        {
            "date": event.quote_date,
            "chain_df": chain_df,
            "signal_row": signal_row,
            "signals": signals,
            "data_warning": {
                "latest_signal_date": str(event.quote_date.date()),
                "mode": "immutable_signal_snapshot_replay",
                "seeded_feature_history": bool(
                    (pd.DatetimeIndex(signals.index) < event.quote_date).any()
                ),
            },
        },
        chain_df,
    )


def _chain_for_event(event, config):
    etf_df = pd.read_parquet(event.etf_snapshot)
    raw_chain = pd.read_parquet(event.option_snapshot)
    if "date" not in raw_chain:
        raw_chain.insert(0, "date", event.quote_date)
    raw_chain = core.data_loader._ensure_option_underlying_id(raw_chain)
    calendar = market_data.load_live_trading_calendar()
    calendar = signal_engine._append_calendar_date(calendar, event.quote_date)
    chain_df = signal_engine._build_latest_enriched_chain(
        {event.quote_date: etf_df},
        {event.quote_date: raw_chain},
        calendar,
        event.quote_date,
    )
    chain_df = market_data.attach_live_underlying_id(event.product, chain_df)
    return chain_df


def _apply_effective_strategy_state(account, payload):
    if not payload:
        return
    account.strategy_state.short_entry_cooldown_left = int(
        payload.get("short_entry_cooldown_left", 0) or 0
    )
    account.strategy_state.short_entry_cooldown_total_days = int(
        payload.get("short_entry_cooldown_total_days", 0) or 0
    )
    account.strategy_state.short_entry_cooldown_started_date = payload.get(
        "short_entry_cooldown_started_date"
    )


def execute_plan_in_memory(account, plan, chain_df, config, timestamp):
    """Fill one generated plan at its own reference mids without persistence."""
    advice = plan.get("advice") or []
    spot = float(plan.get("spot", 0.0) or 0.0)
    executed = []
    for item in advice:
        if item.get("priority") != "action":
            continue
        if signal_engine._has_position_target(item):
            executed.extend(
                _execute_option_target(
                    account,
                    item,
                    chain_df,
                    config,
                    timestamp,
                    spot,
                )
            )

    # ETF advice is a target-state contract. Net every intermediate instruction
    # into one final trade, exactly as the live report and broker adapter do.
    for item in etf_netting.netted_etf_advice_items(advice):
        trade = _execute_etf_target(account, item, config, timestamp)
        if trade is not None:
            executed.append(trade)
    return executed


def _execute_option_target(account, item, chain_df, config, timestamp, spot):
    side = str(item.get("side") or "")
    if side not in account_store.POSITION_SIDES:
        return []
    target = item.get(signal_engine.POSITION_TARGET_KEY)
    current = account.positions.get(side)
    timestamp_text = pd.Timestamp(timestamp).isoformat()
    trades = []

    if target is None:
        if current is None:
            return []
        cash_delta, fee = _close_position_cash(current, side, chain_df, config)
        account.cash += cash_delta
        account.positions[side] = None
        trades.append(
            _option_trade_record(
                timestamp,
                item,
                current,
                -int(current.get("call_qty", 0) or 0),
                -int(current.get("put_qty", 0) or 0),
                chain_df,
                fee,
                cash_delta,
            )
        )
        if side == "long" and item.get("reason") == "iv_high":
            account.strategy_state.short_entry_cooldown_left = int(
                config.strategy.short_cooldown_after_long_iv_high_exit_days
            )
            account.strategy_state.short_entry_cooldown_total_days = (
                account.strategy_state.short_entry_cooldown_left
            )
            account.strategy_state.short_entry_cooldown_started_date = str(
                pd.Timestamp(timestamp).date()
            )
        return trades

    target = dict(target)
    same_contracts = current is not None and (
        str(current.get("call_code")) == str(target.get("call_code"))
        and str(current.get("put_code")) == str(target.get("put_code"))
    )
    if same_contracts:
        call_row = signal_engine._chain_row(chain_df, target["call_code"])
        put_row = signal_engine._chain_row(chain_df, target["put_code"])
        old_call_qty = int(current.get("call_qty", 0) or 0)
        old_put_qty = int(current.get("put_qty", 0) or 0)
        new_call_qty = int(target.get("call_qty", 0) or 0)
        new_put_qty = int(target.get("put_qty", 0) or 0)
        call_change = new_call_qty - old_call_qty
        put_change = new_put_qty - old_put_qty
        multiplier = float(
            call_row.get("contract_multiplier", config.vol.contract_multiplier)
            or config.vol.contract_multiplier
        )
        direction = 1.0 if side == "short" else -1.0
        premium_cash = direction * multiplier * (
            call_change * float(call_row["mid"])
            + put_change * float(put_row["mid"])
        )
        fee = core.position.calc_option_fee(
            abs(call_change),
            abs(put_change),
            option_fee_per_contract=config.backtest.option_fee_per_contract,
        )
        old_margin = float(current.get("option_margin", 0.0) or 0.0)
        new_margin = (
            core.position.calc_short_margin(
                call_row,
                put_row,
                new_call_qty,
                new_put_qty,
                float(spot),
            )
            if side == "short"
            else 0.0
        )
        cash_delta = premium_cash - fee - (new_margin - old_margin)
        account.cash += cash_delta
        current.update(
            {
                "call_qty": new_call_qty,
                "put_qty": new_put_qty,
                "option_margin": new_margin,
                "last_option_value": core.position.signed_value(
                    {**current, "call_qty": new_call_qty, "put_qty": new_put_qty},
                    call_row,
                    put_row,
                ),
                "last_call_price": float(call_row["mid"]),
                "last_put_price": float(put_row["mid"]),
                "last_mark_date": str(pd.Timestamp(timestamp).date()),
                "last_mark_source_timestamp": timestamp_text,
            }
        )
        trades.append(
            _option_trade_record(
                timestamp,
                item,
                current,
                call_change,
                put_change,
                chain_df,
                fee,
                cash_delta,
            )
        )
        return trades

    if current is not None:
        cash_delta, fee = _close_position_cash(current, side, chain_df, config)
        account.cash += cash_delta
        trades.append(
            _option_trade_record(
                timestamp,
                item,
                current,
                -int(current.get("call_qty", 0) or 0),
                -int(current.get("put_qty", 0) or 0),
                chain_df,
                fee,
                cash_delta,
                phase="close",
            )
        )

    new_position, cash_delta, fee = _open_target_position(
        target,
        side,
        chain_df,
        config,
        timestamp,
        item,
        spot,
    )
    account.cash += cash_delta
    account.positions[side] = new_position
    if side == "short":
        account.strategy_state.short_entry_cooldown_left = 0
        account.strategy_state.short_entry_cooldown_total_days = 0
        account.strategy_state.short_entry_cooldown_started_date = None
    trades.append(
        _option_trade_record(
            timestamp,
            item,
            new_position,
            int(new_position["call_qty"]),
            int(new_position["put_qty"]),
            chain_df,
            fee,
            cash_delta,
            phase="open",
        )
    )
    return trades


def _close_position_cash(position, side, chain_df, config):
    call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
    value = core.position.value(position, call_row, put_row)
    fee = core.position.calc_option_fee(
        position.get("call_qty", 0),
        position.get("put_qty", 0),
        option_fee_per_contract=config.backtest.option_fee_per_contract,
    )
    if side == "short":
        return (
            -float(value) - fee + float(position.get("option_margin", 0.0) or 0.0),
            fee,
        )
    return float(value) - fee, fee


def _open_target_position(target, side, chain_df, config, timestamp, item, spot):
    call_row = signal_engine._chain_row(chain_df, target["call_code"])
    put_row = signal_engine._chain_row(chain_df, target["put_code"])
    atm = {
        "call": call_row,
        "put": put_row,
        "strike": float(target.get("strike", call_row.get("strike_price"))),
        "expiry": target.get("expiry") or call_row.get("maturity_date"),
        "dte": call_row.get("dte"),
        "underlying_order_book_id": signal_engine._projected_underlying_id(
            call_row, put_row
        ),
    }
    spot = float(spot)
    position = core.position.open_straddle(
        pd.Timestamp(timestamp),
        atm,
        call_qty=int(target["call_qty"]),
        put_qty=int(target["put_qty"]),
        side=side,
        spot=spot,
        short_entry_regime=item.get("short_entry_regime"),
    )
    # Broker-confirmed live opens currently do not persist a model-side entry
    # volume baseline. Keeping one here would create a replay-only volume-spike
    # exit that production live cannot generate from the same imported state.
    position["entry_call_volume"] = None
    position["entry_put_volume"] = None
    position["entry_total_volume"] = None
    value = core.position.value(position, call_row, put_row)
    fee = core.position.calc_option_fee(
        target["call_qty"],
        target["put_qty"],
        option_fee_per_contract=config.backtest.option_fee_per_contract,
    )
    cash_delta = (
        float(value) - fee - float(position.get("option_margin", 0.0) or 0.0)
        if side == "short"
        else -float(value) - fee
    )
    return position, cash_delta, fee


def _execute_etf_target(account, item, config, timestamp):
    current_qty = float(account.hedge.qty or 0.0)
    target_qty = float(item.get("target_hedge_qty", current_qty) or 0.0)
    trade_qty = target_qty - current_qty
    if abs(trade_qty) <= 1e-9:
        return None
    price = float(item.get("estimated_price", 0.0) or 0.0)
    fee = abs(trade_qty) * price * float(config.backtest.etf_fee_rate)
    cash_delta = -trade_qty * price - fee
    account.cash += cash_delta
    if target_qty <= 0:
        entry_price = 0.0
        margin = 0.0
    elif current_qty <= 0 or trade_qty >= 0:
        old_cost = max(0.0, current_qty) * float(account.hedge.entry_price or 0.0)
        entry_price = (old_cost + max(0.0, trade_qty) * price) / target_qty
        margin = target_qty * entry_price
    else:
        entry_price = float(account.hedge.entry_price or price)
        margin = target_qty * entry_price
    account.hedge = account_store.HedgeState(
        qty=target_qty,
        entry_price=entry_price,
        margin=margin,
        underlying_order_book_id=item.get("underlying_order_book_id"),
        latest_price=price,
        last_market_value=target_qty * price,
        last_unrealized_pnl=target_qty * (price - entry_price),
        last_mark_date=str(pd.Timestamp(timestamp).date()),
        last_mark_source_timestamp=pd.Timestamp(timestamp).isoformat(),
    )
    return {
        "timestamp": pd.Timestamp(timestamp),
        "date": pd.Timestamp(timestamp).normalize(),
        "asset": "ETF",
        "action": item.get("action"),
        "underlying_order_book_id": item.get("underlying_order_book_id"),
        "trade_qty": trade_qty,
        "target_qty": target_qty,
        "price": price,
        "fee": fee,
        "cash_delta": cash_delta,
    }


def account_nav(account, chain_df, spot):
    option_value = 0.0
    option_margin = 0.0
    for position in account.positions.values():
        if position is None:
            continue
        option_margin += float(position.get("option_margin", 0.0) or 0.0)
        try:
            call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
            option_value += core.position.signed_value(position, call_row, put_row)
        except IndexError:
            option_value += float(position.get("last_option_value", 0.0) or 0.0)
    hedge_value = float(account.hedge.qty or 0.0) * float(spot)
    return float(account.cash) + option_value + option_margin + hedge_value


def _option_trade_record(
    timestamp,
    item,
    position,
    call_change,
    put_change,
    chain_df,
    fee,
    cash_delta,
    phase=None,
):
    call_row = signal_engine._chain_row(chain_df, position["call_code"])
    put_row = signal_engine._chain_row(chain_df, position["put_code"])
    return {
        "timestamp": pd.Timestamp(timestamp),
        "date": pd.Timestamp(timestamp).normalize(),
        "asset": "OPTION",
        "action": item.get("action"),
        "phase": phase,
        "side": item.get("side"),
        "call_code": position["call_code"],
        "put_code": position["put_code"],
        "trade_call_qty": call_change,
        "trade_put_qty": put_change,
        "call_price": float(call_row["mid"]),
        "put_price": float(put_row["mid"]),
        "contract_multiplier": float(position.get("contract_multiplier", 10000)),
        "fee": float(fee),
        "cash_delta": float(cash_delta),
    }


def _daily_from_events(events_df):
    if events_df.empty:
        return events_df
    rows = []
    previous_nav = None
    for date, frame in events_df.sort_values("signal_timestamp").groupby("date"):
        first = frame.iloc[0]
        last = frame.iloc[-1]
        opening_nav = float(first["nav_before"] if previous_nav is None else previous_nav)
        ending_nav = float(last["nav_after"])
        rows.append(
            {
                "date": pd.Timestamp(date),
                "product": last["product"],
                "opening_nav": opening_nav,
                "ending_nav": ending_nav,
                "theoretical_daily_pnl": ending_nav - opening_nav,
                "first_signal_timestamp": first["signal_timestamp"],
                "last_signal_timestamp": last["signal_timestamp"],
                "snapshot_count": len(frame),
                "executed_trade_count": int(frame["executed_trade_count"].sum()),
                "last_plan_status": last["plan_status"],
                "last_planned_account_delta": last["planned_account_delta"],
                "last_planned_hedge_qty": last["planned_hedge_qty"],
                "ending_position_fingerprint": last[
                    "ending_position_fingerprint"
                ],
                "ending_hedge_qty": last["ending_hedge_qty"],
            }
        )
        previous_nav = ending_nav
    return pd.DataFrame(rows)


def plan_spot_from_chain(call_row, put_row):
    for row in (call_row, put_row):
        value = pd.to_numeric(row.get("underlying_close"), errors="coerce")
        if pd.notna(value) and float(value) > 0:
            return float(value)
    raise ValueError("Option chain has no valid underlying price for replay execution.")


def position_fingerprint(account):
    return {
        side: (
            None
            if position is None
            else {
                "call_code": str(position.get("call_code")),
                "put_code": str(position.get("put_code")),
                "call_qty": int(position.get("call_qty", 0) or 0),
                "put_qty": int(position.get("put_qty", 0) or 0),
            }
        )
        for side, position in account.positions.items()
    }


def _optional_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


__all__ = [
    "SnapshotReplayEvent",
    "account_from_dict",
    "account_nav",
    "discover_signal_events",
    "execute_plan_in_memory",
    "run_snapshot_replay",
]
