from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import account as account_store
from .market_data import SSE_ETF_OPTION_SPECS


def import_hedge_files(
    product,
    holding_file=None,
    trade_file=None,
    account_id="default",
    date=None,
    dry_run=False,
):
    trade_path = _resolve_live_hold_file(trade_file, "证券委托查询", date)
    trade_date = (
        date
        or _parse_date_from_filename(trade_path)
        or pd.Timestamp.today().strftime("%Y-%m-%d")
    )
    holding_path = _resolve_optional_holding_file(holding_file, trade_date)

    trade_raw = _read_export_csv(trade_path)
    trades = _trade_rows_for_target(product, trade_raw, trade_date)
    local = account_store.load_account(product, account_id=account_id)

    holding_raw = pd.DataFrame()
    target = None
    holding_error = None
    if holding_path is not None:
        holding_raw = _read_export_csv(holding_path)
        try:
            target = _target_hedge_from_holding(product, holding_raw, holding_path)
        except ValueError as exc:
            holding_error = exc
            if not trades:
                raise
    if target is None:
        target = _target_hedge_from_trades(product, local.hedge, trades, trade_date)
    fill = _build_hedge_fill(target, trades, trade_date, holding_path, trade_path)
    warnings = []
    skipped = []
    applied = []
    if _same_hedge(local.hedge.to_dict(), fill):
        skipped.append(
            {
                "reason": "local_hedge_already_matches_snapshot",
                "fill": fill,
            }
        )
        if abs(float(fill.get("qty", 0.0) or 0.0)) > 1e-6 and _is_newer_mark(fill, local.hedge.to_dict()):
            mark_fill = _mark_update_fill(fill)
            if dry_run:
                applied.append({"dry_run": True, "fill": mark_fill})
            else:
                local = account_store.record_fill(product, mark_fill, account_id=account_id)
                applied.append({"dry_run": False, "fill": mark_fill})
    elif dry_run:
        applied.append({"dry_run": True, "fill": fill})
    else:
        local = account_store.record_fill(product, fill, account_id=account_id)
        applied.append({"dry_run": False, "fill": fill})

    if holding_path is None:
        warnings.append(
            {
                "reason": "no_security_holding_file_for_trade_date; target inferred from local hedge plus matched trades",
                "trade_date": trade_date,
                "trade_file": str(trade_path),
            }
        )
    elif holding_error is not None:
        warnings.append(
            {
                "reason": "security_holding_snapshot_has_no_positive_target; target inferred from local hedge plus matched trades",
                "holding_file": str(holding_path),
                "detail": str(holding_error),
            }
        )

    if not trades:
        warnings.append(
            {
                "reason": "no_matching_security_trade_rows; cash_delta estimated from holding cost",
                "holding_file": str(holding_path) if holding_path is not None else None,
                "trade_file": str(trade_path),
            }
        )
    elif abs(sum(row["signed_qty"] for row in trades) - fill["trade_etf_qty"]) > 1e-6:
        warnings.append(
            {
                "reason": "security_trade_qty_differs_from_target_change",
                "trade_qty": sum(row["signed_qty"] for row in trades),
                "target_change": fill["trade_etf_qty"],
            }
        )

    return {
        "product": product,
        "account_id": account_id,
        "holding_file": str(holding_path) if holding_path is not None else None,
        "trade_file": str(trade_path),
        "trade_date": trade_date,
        "dry_run": dry_run,
        "holding_rows": len(holding_raw),
        "trade_rows": len(trade_raw),
        "matched_trade_rows": len(trades),
        "applied": applied,
        "skipped": skipped,
        "warnings": warnings,
    }


def _resolve_live_hold_file(file_path, prefix, report_date=None):
    if file_path is not None:
        return Path(file_path)
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


def _resolve_optional_holding_file(file_path, trade_date):
    if file_path is not None:
        return Path(file_path)
    try:
        return _resolve_live_hold_file(None, "证券持仓查询", trade_date)
    except FileNotFoundError:
        return None


def _read_export_csv(path):
    for encoding in ["utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _target_hedge_from_holding(product, df, path):
    required = ["证券代码", "证券名称", "持有数量", "成本价", "最新价", "市值"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Security holding file missing columns: {missing}")

    target_code = _product_etf_symbol(product)
    rows = []
    for _, row in df.iterrows():
        code = _security_code(row.get("证券代码"))
        if target_code is not None and code != target_code:
            continue
        qty = _number(row.get("持有数量"), 0.0) or 0.0
        if qty <= 0:
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No positive ETF holding found in {path}.")
    if len(rows) > 1:
        raise ValueError(f"Multiple ETF holding rows found in {path}; cannot infer one hedge.")

    row = rows[0]
    qty = float(_number(row.get("持有数量"), 0.0) or 0.0)
    entry_price = float(_number(row.get("成本价"), 0.0) or 0.0)
    latest_price = float(_number(row.get("最新价"), entry_price) or entry_price)
    market_value = float(_number(row.get("市值"), qty * latest_price) or 0.0)
    margin = qty * entry_price if entry_price > 0 else market_value
    return {
        "action": "delta_hedge",
        "qty": qty,
        "entry_price": entry_price,
        "latest_price": latest_price,
        "unrealized_pnl": market_value - margin,
        "margin": margin,
        "market_value": market_value,
        "underlying_order_book_id": _underlying_order_book_id(product, row),
        "security_code": _security_code(row.get("证券代码")),
        "security_name": _clean_text(row.get("证券名称")),
        "broker_account": row.get("投资者账号"),
    }


def _target_hedge_from_trades(product, hedge, trades, trade_date):
    if not trades:
        raise ValueError(
            "Cannot infer ETF hedge target without a security holding snapshot "
            "or matched ETF trade rows."
        )

    previous_qty = float(hedge.qty or 0.0)
    trade_qty = sum(row["signed_qty"] for row in trades)
    target_qty = previous_qty + trade_qty
    clamped_to_zero = False
    if target_qty < 0 and trade_qty < 0:
        target_qty = 0.0
        clamped_to_zero = True
    if abs(target_qty) < 1e-6:
        target_qty = 0.0

    trade_abs_qty = sum(abs(row["signed_qty"]) for row in trades)
    trade_notional = sum(row["price"] * abs(row["signed_qty"]) for row in trades)
    trade_price = trade_notional / trade_abs_qty if trade_abs_qty > 1e-9 else 0.0
    entry_price = _trade_only_entry_price(
        previous_qty,
        hedge.entry_price,
        trades,
        target_qty,
    )
    latest_price = trade_price or float(hedge.latest_price or hedge.entry_price or 0.0)
    margin = target_qty * entry_price if target_qty > 0 and entry_price > 0 else 0.0
    market_value = target_qty * latest_price if target_qty > 0 else 0.0
    first_trade = trades[0]

    return {
        "action": "close_hedge" if abs(target_qty) <= 1e-6 else "delta_hedge",
        "qty": float(target_qty),
        "entry_price": float(entry_price if target_qty > 0 else 0.0),
        "latest_price": float(latest_price) if latest_price else None,
        "unrealized_pnl": market_value - margin if target_qty > 0 else 0.0,
        "margin": float(margin),
        "market_value": float(market_value),
        "underlying_order_book_id": hedge.underlying_order_book_id
        or _default_underlying_order_book_id(product),
        "security_code": _product_etf_symbol(product) or first_trade.get("security_code"),
        "security_name": first_trade.get("security_name"),
        "broker_account": None,
        "trade_only": True,
        "trade_only_clamped_to_zero": clamped_to_zero,
    }


def _trade_only_entry_price(previous_qty, previous_entry_price, trades, target_qty):
    previous_qty = float(previous_qty or 0.0)
    previous_entry_price = float(previous_entry_price or 0.0)
    if target_qty <= 0:
        return 0.0

    buy_qty = sum(row["signed_qty"] for row in trades if row["signed_qty"] > 0)
    buy_notional = sum(
        row["price"] * row["signed_qty"]
        for row in trades
        if row["signed_qty"] > 0
    )
    if buy_qty <= 0:
        return previous_entry_price
    if previous_qty <= 0:
        return buy_notional / buy_qty
    return ((previous_qty * previous_entry_price) + buy_notional) / (previous_qty + buy_qty)


def _trade_rows_for_target(product, df, trade_date):
    required = ["证券代码", "买卖", "成交数量"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Security trade file missing columns: {missing}")

    target_code = _product_etf_symbol(product)
    rows = []
    for _, row in df.iterrows():
        if not _is_executed_security_trade_row(row):
            continue
        if _security_trade_date(row) != trade_date:
            continue
        code = _security_code(row.get("证券代码"))
        if target_code is not None and code != target_code:
            continue
        qty = float(_number(row.get("成交数量"), 0.0) or 0.0)
        price = float(_security_trade_price(row) or 0.0)
        if qty <= 0 or price <= 0:
            continue
        direction = _clean_text(row.get("买卖")) or ""
        signed_qty = -qty if "卖" in direction else qty
        cash_delta = price * qty if signed_qty < 0 else -price * qty
        rows.append(
            {
                "trade_id": row.get("成交编号") or row.get("报单编号"),
                "order_id": row.get("报单编号"),
                "security_code": code,
                "security_name": _clean_text(row.get("证券名称")),
                "direction": direction,
                "price": price,
                "qty": qty,
                "signed_qty": signed_qty,
                "cash_delta": cash_delta,
                "trade_time": row.get("成交时间(日)") or row.get("成交时间"),
            }
        )
    return rows


def _is_executed_security_trade_row(row):
    status = _clean_text(row.get("报单状态"))
    if status is not None and "成交" not in status:
        return False
    qty = _number(row.get("成交数量"), 0.0) or 0.0
    price = _security_trade_price(row) or 0.0
    return qty > 0 and price > 0


def _security_trade_price(row):
    return _number(row.get("成交价格"), None) or _number(row.get("成交均价"), None)


def _security_trade_date(row):
    date_value = row.get("日期")
    if date_value is not None and not pd.isna(date_value):
        return _date8_to_iso(date_value)
    trade_time_day = row.get("成交时间(日)")
    if trade_time_day is None or pd.isna(trade_time_day):
        return None
    match = re.search(r"(20\d{6})", str(trade_time_day))
    return _date8_to_iso(match.group(1)) if match else None


def _build_hedge_fill(target, trades, trade_date, holding_path, trade_path):
    trade_qty = sum(row["signed_qty"] for row in trades)
    trade_notional = sum(row["price"] * abs(row["signed_qty"]) for row in trades)
    trade_price = None if abs(trade_qty) <= 1e-9 else trade_notional / abs(trade_qty)
    cash_delta = sum(row["cash_delta"] for row in trades)
    if not trades:
        trade_qty = target["qty"]
        trade_price = target["entry_price"]
        cash_delta = -target["margin"]

    return {
        "action": target.get("action", "delta_hedge"),
        "date": trade_date,
        "qty": target["qty"],
        "new_etf_qty": target["qty"],
        "target_hedge_qty": target["qty"],
        "trade_etf_qty": trade_qty,
        "entry_price": target["entry_price"],
        "price": trade_price if trade_price is not None else target["entry_price"],
        "latest_price": target["latest_price"],
        "market_value": target["market_value"],
        "unrealized_pnl": target["unrealized_pnl"],
        "margin": target["margin"],
        "cash_delta": cash_delta,
        "underlying_order_book_id": target["underlying_order_book_id"],
        "security_code": target["security_code"],
        "security_name": target["security_name"],
        "source_broker_account": target.get("broker_account"),
        "import_source": (
            "broker_security_trade_only_snapshot"
            if target.get("trade_only")
            else "broker_security_holding_and_trade_snapshot"
        ),
        "holding_source_file": str(holding_path) if holding_path is not None else None,
        "trade_source_file": str(trade_path),
        "security_trades": trades,
        "source_timestamp": _parse_timestamp_from_filename(holding_path or trade_path),
        "source_limitations": [
            "security trade export does not expose commission in the observed file",
            (
                "target hedge is inferred from local hedge plus matched ETF executions"
                if target.get("trade_only")
                else "cash_delta is estimated from matched ETF executions only"
            ),
            (
                "hedge entry_price is inferred locally because no broker holding row exists"
                if target.get("trade_only")
                else "hedge entry_price is taken from broker holding cost price"
            ),
            *(
                ["trade-only target qty was clamped to zero because net sell exceeds local hedge qty"]
                if target.get("trade_only_clamped_to_zero")
                else []
            ),
        ],
    }


def _same_hedge(hedge, fill):
    hedge_qty = float(hedge.get("qty", 0.0) or 0.0)
    fill_qty = float(fill.get("qty", 0.0) or 0.0)
    if abs(hedge_qty) < 1e-6 and abs(fill_qty) < 1e-6:
        return True
    return (
        abs(hedge_qty - fill_qty) < 1e-6
        and abs(float(hedge.get("entry_price", 0.0) or 0.0) - float(fill.get("entry_price", 0.0) or 0.0)) < 1e-6
        and str(hedge.get("underlying_order_book_id")) == str(fill.get("underlying_order_book_id"))
    )


def _mark_update_fill(fill):
    return {
        "action": "hedge_mark_update",
        "date": fill["date"],
        "qty": fill["qty"],
        "new_etf_qty": fill["new_etf_qty"],
        "target_hedge_qty": fill["target_hedge_qty"],
        "entry_price": fill["entry_price"],
        "latest_price": fill.get("latest_price"),
        "market_value": fill.get("market_value"),
        "unrealized_pnl": fill.get("unrealized_pnl"),
        "cash_delta": 0.0,
        "underlying_order_book_id": fill.get("underlying_order_book_id"),
        "security_code": fill.get("security_code"),
        "security_name": fill.get("security_name"),
        "holding_source_file": fill.get("holding_source_file"),
        "trade_source_file": fill.get("trade_source_file"),
        "source_timestamp": fill.get("source_timestamp"),
        "import_source": "broker_security_holding_mark_snapshot",
    }


def _is_newer_mark(fill, hedge):
    source_timestamp = fill.get("source_timestamp")
    if source_timestamp is None:
        return True
    existing_timestamp = hedge.get("last_mark_source_timestamp")
    if existing_timestamp is None:
        return True
    return str(source_timestamp) > str(existing_timestamp)


def _product_etf_symbol(product):
    spec = SSE_ETF_OPTION_SPECS.get(product)
    return spec.etf_symbol if spec is not None else None


def _underlying_order_book_id(product, row):
    spec = SSE_ETF_OPTION_SPECS.get(product)
    if spec is not None:
        return spec.etf_file_prefix
    code = _security_code(row.get("证券代码"))
    exchange = str(row.get("交易所", ""))
    suffix = ".XSHG" if "上" in exchange else ".XSHE" if "深" in exchange else ""
    return f"{code}{suffix}" if code else None


def _default_underlying_order_book_id(product):
    spec = SSE_ETF_OPTION_SPECS.get(product)
    return spec.etf_file_prefix if spec is not None else None


def _security_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def _parse_date_from_filename(path):
    match = re.search(r"(20\d{2})_(\d{2})_(\d{2})", Path(path).name)
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


def _date8_to_iso(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    try:
        return str(pd.Timestamp(text).date())
    except Exception:
        return text


def _clean_text(value):
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


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
