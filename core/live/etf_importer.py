from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import account as account_store
from .market_data import SSE_ETF_OPTION_SPECS, require_live_product


HOLDING_PREFIX = "证券持仓查询(信息导出)"
TRADE_PREFIX = "证券委托查询_实时成交(信息导出)"


def import_etf_files(
    product,
    holding_file=None,
    trade_file=None,
    account_id="default",
    date=None,
    dry_run=False,
):
    require_live_product(product)
    holding_path = _resolve_file(holding_file, HOLDING_PREFIX, date)
    snapshot_date = date or _parse_date_from_filename(holding_path)
    trade_path = _resolve_optional_file(trade_file, TRADE_PREFIX, snapshot_date)

    holding_raw = _read_export_csv(holding_path)
    trade_raw = _read_export_csv(trade_path) if trade_path is not None else pd.DataFrame()
    target = _target_from_holding(product, holding_raw, holding_path)
    trades = _trade_rows_for_target(product, trade_raw, snapshot_date)
    local = account_store.load_account(product, account_id=account_id)
    fill = _build_fill(target, trades, snapshot_date, holding_path, trade_path)

    applied = []
    skipped = []
    warnings = []
    local_hedge = local.hedge.to_dict()
    if _same_target(local_hedge, fill):
        skipped.append({"reason": "local_etf_hedge_already_matches_snapshot", "fill": fill})
        if abs(float(fill["qty"])) > 1e-6 and _is_newer_mark(fill, local_hedge):
            mark_fill = _mark_update_fill(fill)
            if dry_run:
                applied.append({"dry_run": True, "fill": mark_fill})
            else:
                account_store.record_fill(product, mark_fill, account_id=account_id)
                applied.append({"dry_run": False, "fill": mark_fill})
    elif dry_run:
        applied.append({"dry_run": True, "fill": fill})
    else:
        account_store.record_fill(product, fill, account_id=account_id)
        applied.append({"dry_run": False, "fill": fill})

    has_relevant_hedge = (
        abs(float(target["qty"])) > 1e-6
        or abs(float(local_hedge.get("qty", 0.0) or 0.0)) > 1e-6
    )
    if trade_path is None and has_relevant_hedge:
        warnings.append(
            {
                "reason": "no_etf_trade_detail_for_snapshot_date",
                "snapshot_date": snapshot_date,
                "holding_file": str(holding_path),
            }
        )
    elif not trades and has_relevant_hedge:
        warnings.append(
            {
                "reason": "etf_trade_detail_has_no_matching_product_rows",
                "product": product,
                "trade_file": str(trade_path),
            }
        )

    return {
        "product": product,
        "account_id": account_id,
        "holding_file": str(holding_path),
        "trade_file": str(trade_path) if trade_path is not None else None,
        "trade_date": snapshot_date,
        "dry_run": dry_run,
        "holding_rows": len(holding_raw),
        "trade_rows": len(trade_raw),
        "matched_trade_rows": len(trades),
        "applied": applied,
        "skipped": skipped,
        "warnings": warnings,
    }


def _resolve_file(file_path, prefix, report_date=None):
    if file_path is not None:
        path = Path(file_path)
        if not path.name.startswith(prefix):
            raise ValueError(f"ETF import requires {prefix}*.csv: {path}")
        return path
    files = sorted(
        Path("live_hold").glob(f"{prefix}*.csv"),
        key=lambda item: item.stat().st_mtime,
    )
    if report_date is not None:
        files = [item for item in files if _parse_date_from_filename(item) == report_date]
    if not files:
        suffix = f" for date {report_date}" if report_date is not None else ""
        raise FileNotFoundError(f"No CSV found under live_hold/{prefix}*.csv{suffix}.")
    return files[-1]


def _resolve_optional_file(file_path, prefix, report_date):
    try:
        return _resolve_file(file_path, prefix, report_date)
    except FileNotFoundError:
        return None


def _read_export_csv(path):
    for encoding in ("utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _target_from_holding(product, df, path):
    required = {"证券代码", "证券名称", "持有数量", "成本价", "最新价", "市值"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ETF holding file missing columns: {sorted(missing)}")
    target_code = SSE_ETF_OPTION_SPECS[product].etf_symbol
    rows = [
        row
        for _, row in df.iterrows()
        if _security_code(row.get("证券代码")) == target_code
    ]
    if not rows:
        return {
            "qty": 0.0,
            "entry_price": 0.0,
            "latest_price": None,
            "market_value": 0.0,
            "unrealized_pnl": 0.0,
            "margin": 0.0,
            "security_code": target_code,
            "security_name": None,
            "broker_account": None,
            "underlying_order_book_id": SSE_ETF_OPTION_SPECS[product].etf_file_prefix,
        }
    if len(rows) > 1:
        raise ValueError(f"Multiple ETF holding rows found for {target_code} in {path}.")
    row = rows[0]
    qty = float(_number(row.get("持有数量"), 0.0) or 0.0)
    entry_price = float(_number(row.get("成本价"), 0.0) or 0.0)
    latest_price = float(_number(row.get("最新价"), entry_price) or entry_price)
    market_value = float(_number(row.get("市值"), qty * latest_price) or 0.0)
    margin = qty * entry_price
    return {
        "qty": qty,
        "entry_price": entry_price,
        "latest_price": latest_price,
        "market_value": market_value,
        "unrealized_pnl": market_value - margin,
        "margin": margin,
        "security_code": target_code,
        "security_name": _clean_text(row.get("证券名称")),
        "broker_account": row.get("投资者账号"),
        "underlying_order_book_id": SSE_ETF_OPTION_SPECS[product].etf_file_prefix,
    }


def _trade_rows_for_target(product, df, trade_date):
    if df.empty:
        return []
    required = {"证券代码", "买卖", "成交价格", "成交数量"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ETF trade-detail file missing columns: {sorted(missing)}")
    target_code = SSE_ETF_OPTION_SPECS[product].etf_symbol
    rows = []
    for _, row in df.iterrows():
        if _security_code(row.get("证券代码")) != target_code:
            continue
        if _date8_to_iso(row.get("日期")) != trade_date:
            continue
        qty = float(_number(row.get("成交数量"), 0.0) or 0.0)
        price = float(_number(row.get("成交价格"), 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            continue
        direction = _clean_text(row.get("买卖")) or ""
        signed_qty = -qty if "卖" in direction else qty
        rows.append(
            {
                "trade_id": row.get("成交编号") or row.get("报单编号"),
                "order_id": row.get("报单编号"),
                "security_code": target_code,
                "security_name": _clean_text(row.get("证券名称")),
                "direction": direction,
                "price": price,
                "qty": qty,
                "signed_qty": signed_qty,
                "cash_delta": -price * signed_qty,
                "trade_time": row.get("成交时间(日)") or row.get("成交时间"),
            }
        )
    return rows


def _build_fill(target, trades, trade_date, holding_path, trade_path):
    target_qty = target["qty"]
    trade_qty = sum(row["signed_qty"] for row in trades)
    trade_notional = sum(row["price"] * abs(row["signed_qty"]) for row in trades)
    trade_price = trade_notional / sum(abs(row["signed_qty"]) for row in trades) if trades else target["entry_price"]
    return {
        "action": "close_hedge" if abs(target_qty) < 1e-6 else "delta_hedge",
        "date": trade_date,
        "qty": target_qty,
        "new_etf_qty": target_qty,
        "target_hedge_qty": target_qty,
        "trade_etf_qty": trade_qty,
        "entry_price": target["entry_price"],
        "price": trade_price,
        "latest_price": target["latest_price"],
        "market_value": target["market_value"],
        "unrealized_pnl": target["unrealized_pnl"],
        "margin": target["margin"],
        "cash_delta": sum(row["cash_delta"] for row in trades),
        "underlying_order_book_id": target["underlying_order_book_id"],
        "security_code": target["security_code"],
        "security_name": target["security_name"],
        "source_broker_account": target["broker_account"],
        "import_source": "broker_etf_holding_and_trade_detail",
        "holding_source_file": str(holding_path),
        "trade_source_file": str(trade_path) if trade_path is not None else None,
        "security_trades": trades,
        "source_timestamp": _parse_timestamp_from_filename(holding_path),
    }


def _same_target(hedge, fill):
    hedge_qty = float(hedge.get("qty", 0.0) or 0.0)
    fill_qty = float(fill["qty"])
    if abs(hedge_qty) < 1e-6 and abs(fill_qty) < 1e-6:
        return True
    return (
        abs(hedge_qty - fill_qty) < 1e-6
        and str(hedge.get("underlying_order_book_id")) == str(fill["underlying_order_book_id"])
    )


def _is_newer_mark(fill, hedge):
    fill_date = fill.get("date")
    local_date = hedge.get("last_mark_date")
    if fill_date is not None and local_date is not None:
        if pd.Timestamp(fill_date).normalize() < pd.Timestamp(local_date).normalize():
            return False
    existing = hedge.get("last_mark_source_timestamp")
    return existing is None or str(fill.get("source_timestamp") or "") > str(existing)


def _mark_update_fill(fill):
    return {
        **fill,
        "action": "hedge_mark_update",
        "cash_delta": 0.0,
        "source_file": fill.get("holding_source_file"),
    }


def _security_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def _number(value, default=None):
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().replace(",", "")
    if text in {"", "--", "待设置"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _clean_text(value):
    return None if value is None or pd.isna(value) else str(value).strip()


def _date8_to_iso(value):
    text = str(value).strip()
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if re.fullmatch(r"20\d{6}", text) else text


def _parse_date_from_filename(path):
    match = re.search(r"(20\d{2})_(\d{2})_(\d{2})", Path(path).name)
    return "-".join(match.groups()) if match else None


def _parse_timestamp_from_filename(path):
    match = re.search(
        r"(20\d{2})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})",
        Path(path).name,
    )
    return "-".join(match.groups()[:3]) + "T" + ":".join(match.groups()[3:]) if match else None
