from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import account as account_store
from . import market_data
from .runtime import load_product_config, project_path


OPTION_TYPE_MAP = {"C": "call", "P": "put"}
PRODUCT_CONTRACT_NAME_MARKERS = {
    "50etf": "50ETF",
    "300etf": "300ETF",
    "500etf": "500ETF",
    "kc50etf": "科创50",
}


def import_holding_file(
    product,
    file_path=None,
    account_id="default",
    date=None,
    include_existing=False,
    dry_run=False,
):
    market_data.require_live_product(product)
    config = load_product_config(product)
    path = _resolve_holding_file(file_path, date)
    trade_date = date or _parse_date_from_filename(path) or pd.Timestamp.today().strftime(
        "%Y-%m-%d"
    )
    source_timestamp = _parse_timestamp_from_filename(path)
    raw = _read_holding_csv(path)
    rows = _rows_for_product(_normalize_rows(raw, include_existing), product)
    snapshot_rows = _rows_for_product(_normalize_rows(raw, True), product)
    trade_summary_path = _resolve_trade_detail_file(trade_date)
    trade_summary = (
        _read_holding_csv(trade_summary_path)
        if trade_summary_path is not None
        else pd.DataFrame()
    )
    if source_timestamp is not None:
        snapshot_rows = _remove_rows_fully_closed_after_snapshot(
            snapshot_rows,
            trade_summary,
            source_timestamp,
        )
        rows = _remove_rows_fully_closed_after_snapshot(
            rows,
            trade_summary,
            source_timestamp,
        )
    codes = list(
        {
            row["order_book_id"]
            for row in [*rows, *snapshot_rows]
        }
    )
    metadata = _load_contract_metadata(config, codes, trade_date=trade_date)

    warnings = []
    candidates = _build_straddle_candidates(
        rows,
        metadata,
        config,
        trade_date,
        str(path),
        warnings,
    )
    snapshot_candidates = _build_straddle_candidates(
        snapshot_rows,
        metadata,
        config,
        trade_date,
        str(path),
        [],
    )
    local = account_store.load_account(product, account_id=account_id)
    if not rows and not include_existing and not _local_contains_snapshot_rows(
        local,
        snapshot_rows,
    ):
        existing_qty = _total_positive_holding_qty(raw)
        if existing_qty > 0:
            warnings.append(
                {
                    "reason": (
                        "holding_snapshot_has_positions_but_today_open_qty_is_zero; "
                        "use include_existing/导入总持仓 when seeding or repairing "
                        "the local shadow account"
                    ),
                    "total_positive_holding_qty": existing_qty,
                }
            )
    applied = []
    skipped = []
    local = _apply_existing_position_rebalances(
        product,
        account_id,
        local,
        snapshot_candidates,
        trade_summary,
        trade_summary_path,
        source_timestamp,
        dry_run,
        applied,
        warnings,
        config,
    )
    local = _apply_missing_straddle_closes(
        product,
        account_id,
        local,
        snapshot_rows,
        trade_summary,
        trade_summary_path,
        trade_date,
        config,
        dry_run,
        applied,
        warnings,
    )

    for candidate in candidates:
        side = candidate["side"]
        fill = candidate["fill"]
        existing = local.positions.get(side)
        if existing is not None:
            if _same_position(existing, fill):
                skipped.append(
                    {
                        "side": side,
                        "reason": "local_position_already_matches_snapshot",
                        "fill": fill,
                    }
                )
                if _is_newer_mark(fill, existing):
                    mark_fill = _mark_update_fill(fill, source_timestamp)
                    local = _record_or_preview_fill(
                        product,
                        account_id,
                        local,
                        mark_fill,
                        dry_run,
                        applied,
                    )
                continue
            warnings.append(
                {
                    "side": side,
                    "reason": "local_position_differs_from_snapshot; manual amend/roll/close is required",
                    "local_position": existing,
                    "snapshot_fill": fill,
                }
            )
            continue

        local = _record_or_preview_fill(
            product,
            account_id,
            local,
            fill,
            dry_run,
            applied,
        )
    _warn_missing_local_positions(local, snapshot_rows, warnings, applied)
    return {
        "product": product,
        "account_id": account_id,
        "file": str(path),
        "trade_file": str(trade_summary_path) if trade_summary_path is not None else None,
        "trade_date": trade_date,
        "include_existing": include_existing,
        "dry_run": dry_run,
        "input_rows": len(raw),
        "usable_rows": len(rows),
        "applied": applied,
        "skipped": skipped,
        "warnings": warnings,
    }


def _record_or_preview_fill(product, account_id, local, fill, dry_run, applied):
    if dry_run:
        account_store._apply_fill(local, product, fill)
        applied.append({"dry_run": True, "fill": account_store.normalize_fill(fill)})
        return local

    local = account_store.record_fill(product, fill, account_id=account_id)
    applied.append({"dry_run": False, "fill": fill})
    return local


def _apply_existing_position_rebalances(
    product,
    account_id,
    local,
    snapshot_candidates,
    trade_summary,
    trade_summary_path,
    source_timestamp,
    dry_run,
    applied,
    warnings,
    config,
):
    summary_by_code = _trade_detail_by_code(trade_summary)
    for candidate in snapshot_candidates:
        if candidate.get("kind") != "straddle":
            continue
        side = candidate["side"]
        snapshot_fill = candidate["fill"]
        existing = local.positions.get(side)
        if existing is None or _same_position(existing, snapshot_fill):
            continue
        if (
            str(existing.get("call_code")) != str(snapshot_fill.get("call_code"))
            or str(existing.get("put_code")) != str(snapshot_fill.get("put_code"))
        ):
            continue
        fill = _straddle_leg_rebalance_fill(
            existing,
            snapshot_fill,
            summary_by_code,
            trade_summary_path,
            source_timestamp,
            config,
        )
        if fill is None:
            warnings.append(
                {
                    "side": side,
                    "reason": (
                        "broker snapshot changed straddle leg quantities, but matching "
                        "trade detail does not prove the quantity adjustment"
                    ),
                    "local_position": existing,
                    "snapshot_fill": snapshot_fill,
                }
            )
            continue
        local = _record_or_preview_fill(product, account_id, local, fill, dry_run, applied)
    return local


def _straddle_leg_rebalance_fill(
    existing,
    snapshot_fill,
    summary_by_code,
    trade_summary_path,
    source_timestamp,
    config,
):
    side = str(snapshot_fill.get("side"))
    multiplier = int(snapshot_fill.get("contract_multiplier", config.vol.contract_multiplier))
    cash_delta = 0.0
    total_fee = 0.0
    adjustments = []
    for leg in ["call", "put"]:
        code = str(snapshot_fill.get(f"{leg}_code"))
        old_qty = int(existing.get(f"{leg}_qty", 0) or 0)
        new_qty = int(snapshot_fill.get(f"{leg}_qty", 0) or 0)
        qty_change = new_qty - old_qty
        if qty_change == 0:
            continue
        summary = summary_by_code.get(code)
        trade = _straddle_leg_trade_from_summary(side, qty_change, summary)
        if trade is None:
            return None
        fee = abs(qty_change) * float(config.backtest.option_fee_per_contract)
        cash_delta += trade["cash_delta"] * multiplier - fee
        total_fee += fee
        adjustments.append(
            {
                "leg": leg,
                "order_book_id": code,
                "qty_change": qty_change,
                **trade,
                "fee": fee,
            }
        )
    if not adjustments:
        return None
    if side == "short":
        cash_delta += (
            float(existing.get("option_margin", 0.0) or 0.0)
            - float(snapshot_fill.get("option_margin", 0.0) or 0.0)
        )
    return {
        **snapshot_fill,
        "action": "rebalance_straddle_legs",
        "entry_date": existing.get("entry_date"),
        "cash_delta": cash_delta,
        "estimated_fee": total_fee,
        "leg_adjustments": adjustments,
        "source_timestamp": source_timestamp,
        "source_file": snapshot_fill.get("source_file"),
        "trade_source_file": str(trade_summary_path) if trade_summary_path is not None else None,
        "import_source": "broker_holding_and_trade_detail_leg_rebalance",
    }


def _straddle_leg_trade_from_summary(side, qty_change, summary):
    if summary is None:
        return None
    if side == "short" and qty_change < 0:
        qty_column, price_column, direction = "买平", "买平均价", "买入平仓"
        cash_sign = -1.0
    elif side == "short" and qty_change > 0:
        qty_column, price_column, direction = "卖开", "卖开均价", "卖出开仓"
        cash_sign = 1.0
    elif side == "long" and qty_change < 0:
        qty_column, price_column, direction = "卖平", "卖平均价", "卖出平仓"
        cash_sign = 1.0
    else:
        qty_column, price_column, direction = "买开", "买开均价", "买入开仓"
        cash_sign = -1.0
    qty = int(_number(summary.get(qty_column), 0) or 0)
    price = _number(summary.get(price_column))
    if qty < abs(qty_change) or price is None:
        return None
    return {
        "direction": direction,
        "price": price,
        "qty": abs(qty_change),
        "cash_delta": cash_sign * price * abs(qty_change),
    }


def _trade_detail_by_code(df):
    if not _is_trade_detail_export(df):
        return {}
    df = _aggregate_trade_detail(df)
    result = {}
    if df is None or df.empty or "合约代码" not in df.columns:
        return result
    for _, row in df.iterrows():
        code = str(row.get("合约代码", "")).strip()
        if not code or code == "全部":
            continue
        result[code] = row.to_dict()
    return result


def _is_trade_detail_export(df):
    required = {"合约代码", "开平", "买卖", "成交数量", "成交价格"}
    return df is not None and required.issubset(df.columns)


def _aggregate_trade_detail(df):
    rows = []
    for code, trades in df.groupby(df["合约代码"].astype(str).str.strip()):
        summary = {"合约代码": code}
        for direction in ("买开", "买平", "卖开", "卖平"):
            matched = trades[
                trades["买卖"].astype(str).str.strip().str.cat(
                    trades["开平"].astype(str).str.strip().str.replace(
                        "仓", "", regex=False
                    )
                ).eq(direction)
            ]
            quantities = matched["成交数量"].map(lambda value: _number(value, 0.0) or 0.0)
            prices = matched["成交价格"].map(lambda value: _number(value))
            valid = prices.notna() & quantities.gt(0)
            qty = float(quantities[valid].sum())
            summary[direction] = qty
            summary[f"{direction}均价"] = (
                float((prices[valid] * quantities[valid]).sum() / qty)
                if qty > 0
                else None
            )
        rows.append(summary)
    return pd.DataFrame(rows)


def _remove_rows_fully_closed_after_snapshot(rows, trade_summary, source_timestamp):
    summary_by_code = _trade_detail_by_code(
        _trade_rows_after_snapshot(trade_summary, source_timestamp)
    )
    result = []
    for row in rows:
        summary = summary_by_code.get(str(row.get("order_book_id")))
        side = str(row.get("side") or "")
        close_column = "买平" if side == "short" else "卖平"
        close_qty = int(_number(summary.get(close_column), 0) or 0) if summary else 0
        holding_qty = int(row.get("total_qty", 0) or 0)
        if holding_qty > 0 and close_qty >= holding_qty:
            continue
        result.append(row)
    return result


def _trade_rows_after_snapshot(trade_summary, source_timestamp):
    if trade_summary is None or trade_summary.empty or source_timestamp is None:
        return pd.DataFrame()
    snapshot_time = pd.Timestamp(source_timestamp)
    keep = []
    for _, row in trade_summary.iterrows():
        trade_time = _trade_execution_timestamp(row)
        keep.append(trade_time is not None and trade_time > snapshot_time)
    return trade_summary.loc[keep].copy()


def _trade_execution_timestamp(row):
    value = row.get("成交时间(日)")
    if value is not None and not pd.isna(value):
        try:
            return pd.Timestamp(str(value))
        except (TypeError, ValueError):
            pass

    date_value = row.get("日期")
    time_value = row.get("成交时间")
    if date_value is None or pd.isna(date_value) or time_value is None or pd.isna(time_value):
        return None
    date_text = str(date_value).strip()
    time_text = str(time_value).strip()
    if not date_text or not time_text:
        return None
    try:
        return pd.Timestamp(f"{date_text} {time_text}")
    except (TypeError, ValueError):
        return None


def _apply_missing_straddle_closes(
    product,
    account_id,
    local,
    snapshot_rows,
    trade_summary,
    trade_summary_path,
    trade_date,
    config,
    dry_run,
    applied,
    warnings,
):
    snapshot_codes = {str(row["order_book_id"]) for row in snapshot_rows}
    summary_by_code = _trade_detail_by_code(trade_summary)
    for side, position in list(local.positions.items()):
        if position is None:
            continue
        call_code = str(position.get("call_code") or "")
        put_code = str(position.get("put_code") or "")
        if call_code in snapshot_codes or put_code in snapshot_codes:
            continue
        fill = _straddle_close_fill(
            position,
            summary_by_code,
            trade_date,
            trade_summary_path,
            config,
        )
        if fill is None:
            warnings.append(
                {
                    "side": side,
                    "reason": (
                        "local straddle is absent from effective holding snapshot, "
                        "but matching trade detail does not prove both legs were "
                        "fully closed"
                    ),
                    "local_position": position,
                }
            )
            continue
        local = _record_or_preview_fill(product, account_id, local, fill, dry_run, applied)
    return local


def _straddle_close_fill(position, summary_by_code, trade_date, source_file, config):
    side = str(position.get("side") or "short")
    close_qty_column = "买平" if side == "short" else "卖平"
    close_price_column = "买平均价" if side == "short" else "卖平均价"
    multiplier = int(
        _number(position.get("contract_multiplier"), config.vol.contract_multiplier)
        or config.vol.contract_multiplier
    )
    legs = []
    for leg in ["call", "put"]:
        code = str(position.get(f"{leg}_code") or "")
        qty = int(position.get(f"{leg}_qty", 0) or 0)
        summary = summary_by_code.get(code)
        close_qty = int(_number(summary.get(close_qty_column), 0) or 0) if summary else 0
        close_price = _number(summary.get(close_price_column)) if summary else None
        if qty <= 0 or close_qty < qty or close_price is None:
            return None
        legs.append(
            {
                "leg": leg,
                "order_book_id": code,
                "qty": qty,
                "price": close_price,
            }
        )

    fee = sum(item["qty"] for item in legs) * float(config.backtest.option_fee_per_contract)
    close_value = sum(item["qty"] * item["price"] * multiplier for item in legs)
    margin_release = float(position.get("option_margin", 0.0) or 0.0) if side == "short" else 0.0
    cash_delta = close_value - fee if side == "long" else -close_value - fee + margin_release
    return {
        "action": f"close_{side}_straddle",
        "side": side,
        "date": trade_date,
        "call_code": position.get("call_code"),
        "put_code": position.get("put_code"),
        "call_qty": int(position.get("call_qty", 0) or 0),
        "put_qty": int(position.get("put_qty", 0) or 0),
        "call_price": legs[0]["price"],
        "put_price": legs[1]["price"],
        "contract_multiplier": multiplier,
        "fee": fee,
        "option_margin_release": margin_release,
        "cash_delta": cash_delta,
        "leg_closes": legs,
        "source_timestamp": _parse_timestamp_from_filename(source_file),
        "import_source": "broker_option_trade_detail",
        "source_file": str(source_file),
        "source_limitations": [
            "option trade detail is aggregated by contract for close confirmation",
            "configured local option fee is used because broker export fee is unavailable",
        ],
    }


def _resolve_holding_file(file_path, report_date=None):
    if file_path is not None:
        path = Path(file_path)
        if not path.name.startswith("实时持仓"):
            raise ValueError(f"Option import requires 实时持仓*.csv: {path}")
        return path
    files = sorted(
        Path("live_hold").glob("实时持仓*.csv"),
        key=lambda item: item.stat().st_mtime,
    )
    if report_date is not None:
        files = [item for item in files if _parse_date_from_filename(item) == report_date]
    if not files:
        suffix = f" for date {report_date}" if report_date is not None else ""
        raise FileNotFoundError(f"No holding CSV found under live_hold/实时持仓*.csv{suffix}.")
    return files[-1]


def _resolve_trade_detail_file(report_date=None):
    files = sorted(
        Path("live_hold").glob("成交明细*.csv"),
        key=lambda item: item.stat().st_mtime,
    )
    if report_date is not None:
        files = [item for item in files if _parse_date_from_filename(item) == report_date]
    return files[-1] if files else None


def _read_holding_csv(path):
    for encoding in ["utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _normalize_rows(df, include_existing):
    rows = []
    required = ["合约代码", "合约名称", "买卖", "总持仓", "开仓均价", "最新价", "占用保证金"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        if "成交编号" in df.columns or "成交价格" in df.columns:
            raise ValueError(
                "This looks like a trade-detail export, not a holding export. "
                "Use a live_hold/实时持仓*.csv file for import_holdings.py."
            )
        raise ValueError(f"Holding file missing columns: {missing}")

    for _, row in df.iterrows():
        total_qty = int(_number(row.get("总持仓"), 0) or 0)
        today_open_qty = int(_number(row.get("今开仓"), 0) or 0)
        if total_qty <= 0:
            continue
        if not include_existing and today_open_qty <= 0:
            continue

        import_qty = total_qty if include_existing else min(today_open_qty, total_qty)
        rows.append(
            {
                "order_book_id": str(row["合约代码"]).strip(),
                "contract_name": str(row["合约名称"]).strip(),
                "side": _side_from_row(row),
                "qty": import_qty,
                "total_qty": total_qty,
                "today_open_qty": today_open_qty,
                "entry_price": _number(row.get("开仓均价"), 0.0),
                "latest_price": _number(row.get("最新价"), 0.0),
                "margin": _number(row.get("占用保证金"), 0.0),
                "option_market_value": _number(row.get("期权市值"), 0.0),
                "broker_account": row.get("投资者账号"),
            }
        )
    return rows


def _total_positive_holding_qty(df):
    if "总持仓" not in df.columns:
        return 0
    total = 0
    for _, row in df.iterrows():
        total += int(_number(row.get("总持仓"), 0) or 0)
    return total


def _rows_for_product(rows, product):
    marker = PRODUCT_CONTRACT_NAME_MARKERS.get(product)
    if marker is None:
        return rows
    return [row for row in rows if marker in row["contract_name"]]


def _side_from_row(row):
    buy_sell = str(row.get("买卖", "")).strip()
    position_type = str(row.get("持仓类型", "")).strip()
    if "卖" in buy_sell or "义务" in position_type:
        return "short"
    return "long"


def _load_contract_metadata(config, codes, trade_date=None):
    opt_dir = project_path(config.data.opt_dir)
    live_quote_dir = project_path(f"data/live/{config.data.product}/quotes")
    metadata = {}
    remaining = {str(code) for code in codes}
    paths = {
        *opt_dir.glob("*_chain.parquet"),
        *live_quote_dir.rglob("*_option_chain.parquet"),
    }
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
        if not remaining:
            break
        df = pd.read_parquet(path)
        if "order_book_id" not in df.columns:
            continue
        df["order_book_id"] = df["order_book_id"].astype(str)
        hit = df[df["order_book_id"].isin(remaining)]
        for _, row in hit.iterrows():
            code = str(row["order_book_id"])
            metadata[code] = {
                "strike": float(row["strike_price"]),
                "expiry": str(pd.Timestamp(row["maturity_date"]).date()),
                "option_type": str(row["option_type"]).upper(),
                "contract_multiplier": int(row.get("contract_multiplier", config.vol.contract_multiplier)),
                "contract_symbol": row.get("contract_symbol"),
                "underlying_order_book_id": market_data.option_underlying_order_book_id(
                    config.data.product
                ),
                "metadata_source": "local_option_chain",
            }
            remaining.discard(code)
    if remaining and trade_date is not None:
        try:
            metadata.update(
                market_data.fetch_historical_option_metadata(
                    config.data.product,
                    trade_date,
                    codes=remaining,
                )
            )
        except Exception:
            # Missing metadata is handled as a per-contract warning by the caller.
            pass
    return metadata


def _build_straddle_candidates(rows, metadata, config, trade_date, source_file, warnings):
    grouped = {}
    for row in rows:
        meta = metadata.get(row["order_book_id"])
        if meta is None:
            warnings.append(
                {
                    "order_book_id": row["order_book_id"],
                    "reason": "contract metadata not found in local option chain; cannot infer expiry/strike safely",
                }
            )
            continue
        key = (row["side"], meta["strike"], meta["expiry"])
        grouped.setdefault(key, {})[OPTION_TYPE_MAP.get(meta["option_type"])] = {
            "row": row,
            "meta": meta,
        }

    by_side = {}
    for key, legs in grouped.items():
        side, strike, expiry = key
        if "call" not in legs or "put" not in legs:
            leg_name, leg_payload = next(iter(legs.items()))
            warnings.append(
                {
                    "side": side,
                    "order_book_id": leg_payload["row"]["order_book_id"],
                    "reason": (
                        "unpaired option holding is not imported; live account only "
                        "supports paired straddle positions"
                    ),
                    "leg": leg_name,
                    "strike": strike,
                    "expiry": expiry,
                }
            )
            continue
        by_side.setdefault(side, []).append((strike, expiry, legs))

    candidates = []
    for side, pairs in by_side.items():
        if len(pairs) > 1:
            warnings.append(
                {
                    "side": side,
                    "reason": "multiple straddle pairs on same side; live account currently supports one position per side",
                    "pairs": [(strike, expiry) for strike, expiry, _ in pairs],
                }
            )
            continue
        strike, expiry, legs = pairs[0]
        candidates.append(
            {
                "kind": "straddle",
                "side": side,
                "fill": _candidate_to_fill(
                    side,
                    strike,
                    expiry,
                    legs,
                    config,
                    trade_date,
                    source_file,
                ),
            }
        )
    return candidates


def _candidate_to_fill(side, strike, expiry, legs, config, trade_date, source_file):
    call = legs["call"]["row"]
    put = legs["put"]["row"]
    multiplier = legs["call"]["meta"]["contract_multiplier"]
    call_qty = int(call["qty"])
    put_qty = int(put["qty"])
    entry_call_price = float(call["entry_price"])
    entry_put_price = float(put["entry_price"])
    entry_value = (
        entry_call_price * call_qty + entry_put_price * put_qty
    ) * multiplier
    latest_value = (
        float(call["latest_price"]) * call_qty + float(put["latest_price"]) * put_qty
    ) * multiplier
    fee = (call_qty + put_qty) * config.backtest.option_fee_per_contract
    margin = float(call["margin"] or 0.0) + float(put["margin"] or 0.0)
    if side == "short":
        cash_delta = entry_value - fee - margin
        action = "open_short_straddle"
    else:
        cash_delta = -entry_value - fee
        action = "open_long_straddle"

    return {
        "action": action,
        "side": side,
        "date": trade_date,
        "call_code": call["order_book_id"],
        "put_code": put["order_book_id"],
        "strike": strike,
        "expiry": expiry,
        "call_qty": call_qty,
        "put_qty": put_qty,
        "entry_call_price": entry_call_price,
        "entry_put_price": entry_put_price,
        "last_call_price": float(call["latest_price"]),
        "last_put_price": float(put["latest_price"]),
        "entry_call_volume": None,
        "entry_put_volume": None,
        "entry_total_volume": None,
        "contract_multiplier": multiplier,
        "short_entry_regime": config.strategy.short_signal_mode if side == "short" else None,
        "entry_option_value": entry_value,
        "option_margin": margin if side == "short" else 0.0,
        "last_option_value": latest_value,
        "cash_delta": cash_delta,
        "source_timestamp": _parse_timestamp_from_filename(source_file),
        "import_source": "broker_holding_snapshot",
        "source_file": source_file,
        "source_broker_account": call.get("broker_account") or put.get("broker_account"),
        "source_limitations": [
            "holding snapshot has no per-fill execution id/time",
            "cash_delta is estimated from open average, configured fee, and occupied margin",
            "commission/settlement differences should be reconciled against broker account",
        ],
    }


def _same_position(position, fill):
    return (
        str(position.get("call_code")) == str(fill.get("call_code"))
        and str(position.get("put_code")) == str(fill.get("put_code"))
        and int(position.get("call_qty", 0) or 0) == int(fill.get("call_qty", 0) or 0)
        and int(position.get("put_qty", 0) or 0) == int(fill.get("put_qty", 0) or 0)
    )


def _local_contains_snapshot_rows(local, rows):
    local_qty = {}
    for position in local.positions.values():
        if position is None:
            continue
        local_qty[str(position.get("call_code"))] = int(position.get("call_qty", 0) or 0)
        local_qty[str(position.get("put_code"))] = int(position.get("put_qty", 0) or 0)
    return all(
        local_qty.get(str(row.get("order_book_id"))) == int(row.get("total_qty", 0) or 0)
        for row in rows
    )


def _mark_update_fill(fill, source_timestamp):
    return {
        "action": "option_mark_update",
        "side": fill["side"],
        "date": fill["date"],
        "call_code": fill["call_code"],
        "put_code": fill["put_code"],
        "call_qty": fill["call_qty"],
        "put_qty": fill["put_qty"],
        "last_call_price": fill.get("last_call_price"),
        "last_put_price": fill.get("last_put_price"),
        "last_option_value": fill.get("last_option_value"),
        "option_margin": fill.get("option_margin"),
        "cash_delta": 0.0,
        "source_file": fill.get("source_file"),
        "source_timestamp": source_timestamp or fill.get("source_timestamp"),
        "import_source": "broker_holding_mark_snapshot",
    }


def _is_newer_mark(fill, position):
    source_timestamp = fill.get("source_timestamp")
    if source_timestamp is None:
        return True
    existing_timestamp = position.get("last_mark_source_timestamp")
    if existing_timestamp is None:
        return True
    return str(source_timestamp) > str(existing_timestamp)


def _warn_missing_local_positions(local, rows, warnings, applied=None):
    snapshot_codes = {row["order_book_id"] for row in rows}
    closing_sides = {
        item.get("fill", {}).get("side")
        for item in (applied or [])
        if str(item.get("fill", {}).get("action", "")).startswith("close_")
    }
    for side, position in local.positions.items():
        if position is None:
            continue
        if side in closing_sides:
            continue
        if position.get("call_code") not in snapshot_codes and position.get("put_code") not in snapshot_codes:
            warnings.append(
                {
                    "side": side,
                    "reason": "local position is absent from imported holding snapshot; close cannot be auto-confirmed without close price/cash_delta",
                    "local_position": position,
                }
            )


def _parse_date_from_filename(path):
    match = re.search(r"(20\d{2})_(\d{2})_(\d{2})", path.name)
    if match:
        return "-".join(match.groups())
    return None


def _parse_timestamp_from_filename(path):
    match = re.search(
        r"(20\d{2})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})",
        str(path),
    )
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"


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
