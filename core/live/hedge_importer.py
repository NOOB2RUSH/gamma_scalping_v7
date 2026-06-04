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
    holding_path = _resolve_live_hold_file(holding_file, "证券持仓查询")
    trade_path = _resolve_live_hold_file(trade_file, "证券委托查询_实时成交")
    trade_date = (
        date
        or _parse_date_from_filename(trade_path)
        or _parse_date_from_filename(holding_path)
        or pd.Timestamp.today().strftime("%Y-%m-%d")
    )

    holding_raw = _read_export_csv(holding_path)
    trade_raw = _read_export_csv(trade_path)
    target = _target_hedge_from_holding(product, holding_raw, holding_path)
    trades = _trade_rows_for_target(product, trade_raw, trade_date)
    fill = _build_hedge_fill(target, trades, trade_date, holding_path, trade_path)

    local = account_store.load_account(product, account_id=account_id)
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
    elif dry_run:
        applied.append({"dry_run": True, "fill": fill})
    else:
        local = account_store.record_fill(product, fill, account_id=account_id)
        applied.append({"dry_run": False, "fill": fill})

    if not trades:
        warnings.append(
            {
                "reason": "no_matching_security_trade_rows; cash_delta estimated from holding cost",
                "holding_file": str(holding_path),
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
        "holding_file": str(holding_path),
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


def _resolve_live_hold_file(file_path, prefix):
    if file_path is not None:
        return Path(file_path)
    files = sorted(
        Path("live_hold").glob(f"{prefix}*.csv"),
        key=lambda item: item.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No CSV found under live_hold/{prefix}*.csv.")
    return files[-1]


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
        "qty": qty,
        "entry_price": entry_price,
        "latest_price": latest_price,
        "margin": margin,
        "market_value": market_value,
        "underlying_order_book_id": _underlying_order_book_id(product, row),
        "security_code": _security_code(row.get("证券代码")),
        "security_name": _clean_text(row.get("证券名称")),
        "broker_account": row.get("投资者账号"),
    }


def _trade_rows_for_target(product, df, trade_date):
    required = ["证券代码", "买卖", "成交价格", "成交数量", "日期", "成交编号"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Security trade file missing columns: {missing}")

    target_code = _product_etf_symbol(product)
    rows = []
    for _, row in df.iterrows():
        if _date8_to_iso(row.get("日期")) != trade_date:
            continue
        code = _security_code(row.get("证券代码"))
        if target_code is not None and code != target_code:
            continue
        qty = float(_number(row.get("成交数量"), 0.0) or 0.0)
        price = float(_number(row.get("成交价格"), 0.0) or 0.0)
        if qty <= 0 or price <= 0:
            continue
        direction = _clean_text(row.get("买卖")) or ""
        signed_qty = -qty if "卖" in direction else qty
        cash_delta = price * qty if signed_qty < 0 else -price * qty
        rows.append(
            {
                "trade_id": row.get("成交编号"),
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
        "action": "delta_hedge",
        "date": trade_date,
        "qty": target["qty"],
        "new_etf_qty": target["qty"],
        "target_hedge_qty": target["qty"],
        "trade_etf_qty": trade_qty,
        "entry_price": target["entry_price"],
        "price": trade_price if trade_price is not None else target["entry_price"],
        "latest_price": target["latest_price"],
        "margin": target["margin"],
        "cash_delta": cash_delta,
        "underlying_order_book_id": target["underlying_order_book_id"],
        "security_code": target["security_code"],
        "security_name": target["security_name"],
        "source_broker_account": target.get("broker_account"),
        "import_source": "broker_security_holding_and_trade_snapshot",
        "holding_source_file": str(holding_path),
        "trade_source_file": str(trade_path),
        "security_trades": trades,
        "source_limitations": [
            "security trade export does not expose commission in the observed file",
            "cash_delta is estimated from matched ETF executions only",
            "hedge entry_price is taken from broker holding cost price",
        ],
    }


def _same_hedge(hedge, fill):
    return (
        abs(float(hedge.get("qty", 0.0) or 0.0) - float(fill.get("qty", 0.0) or 0.0)) < 1e-6
        and abs(float(hedge.get("entry_price", 0.0) or 0.0) - float(fill.get("entry_price", 0.0) or 0.0)) < 1e-6
        and str(hedge.get("underlying_order_book_id")) == str(fill.get("underlying_order_book_id"))
    )


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
