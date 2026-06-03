from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import account as account_store
from .runtime import load_product_config, project_path


OPTION_TYPE_MAP = {"C": "call", "P": "put"}


def import_holding_file(
    product,
    file_path=None,
    account_id="default",
    date=None,
    include_existing=False,
    dry_run=False,
):
    config = load_product_config(product)
    path = _resolve_holding_file(file_path)
    trade_date = date or _parse_date_from_filename(path) or pd.Timestamp.today().strftime(
        "%Y-%m-%d"
    )
    raw = _read_holding_csv(path)
    rows = _normalize_rows(raw, include_existing)
    codes = [row["order_book_id"] for row in rows]
    metadata = _load_contract_metadata(config, codes)

    warnings = []
    candidates = _build_straddle_candidates(
        rows,
        metadata,
        config,
        trade_date,
        str(path),
        warnings,
    )
    local = account_store.load_account(product, account_id=account_id)
    applied = []
    skipped = []

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

        if dry_run:
            applied.append({"dry_run": True, "fill": fill})
            continue

        local = account_store.record_fill(product, fill, account_id=account_id)
        applied.append({"dry_run": False, "fill": fill})

    _warn_missing_local_positions(local, rows, warnings)
    return {
        "product": product,
        "account_id": account_id,
        "file": str(path),
        "trade_date": trade_date,
        "include_existing": include_existing,
        "dry_run": dry_run,
        "input_rows": len(raw),
        "usable_rows": len(rows),
        "applied": applied,
        "skipped": skipped,
        "warnings": warnings,
    }


def _resolve_holding_file(file_path):
    if file_path is not None:
        return Path(file_path)
    files = sorted(
        Path("live_hold").glob("实时持仓*.csv"),
        key=lambda item: item.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError("No holding CSV found under live_hold/实时持仓*.csv.")
    return files[-1]


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


def _side_from_row(row):
    buy_sell = str(row.get("买卖", "")).strip()
    position_type = str(row.get("持仓类型", "")).strip()
    if "卖" in buy_sell or "义务" in position_type:
        return "short"
    return "long"


def _load_contract_metadata(config, codes):
    opt_dir = project_path(config.data.opt_dir)
    metadata = {}
    remaining = {str(code) for code in codes}
    for path in sorted(opt_dir.glob("*_chain.parquet"), reverse=True):
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
            }
            remaining.discard(code)
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
            warnings.append(
                {
                    "side": side,
                    "strike": strike,
                    "expiry": expiry,
                    "reason": "unpaired option holding; live account currently supports straddle pairs only",
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
        "entry_call_volume": None,
        "entry_put_volume": None,
        "entry_total_volume": None,
        "contract_multiplier": multiplier,
        "short_entry_regime": config.strategy.short_signal_mode if side == "short" else None,
        "entry_option_value": entry_value,
        "option_margin": margin if side == "short" else 0.0,
        "last_option_value": latest_value,
        "cash_delta": cash_delta,
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


def _warn_missing_local_positions(local, rows, warnings):
    snapshot_codes = {row["order_book_id"] for row in rows}
    for side, position in local.positions.items():
        if position is None:
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
