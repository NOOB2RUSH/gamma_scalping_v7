from __future__ import annotations

from . import account as account_store
from . import etf_importer
from . import holding_importer
from .market_data import SSE_ETF_OPTION_SPECS, require_live_product


def check_account_positions(product, account_id="default", date=None):
    """Compare local shadow-account positions with latest broker exports."""
    require_live_product(product)
    local = account_store.load_account(product, account_id=account_id)

    option_path = holding_importer._resolve_holding_file(None, date)
    option_raw = holding_importer._read_holding_csv(option_path)
    option_rows = holding_importer._rows_for_product(
        holding_importer._normalize_rows(option_raw, include_existing=True),
        product,
    )

    etf_path = etf_importer._resolve_file(
        None,
        etf_importer.HOLDING_PREFIX,
        date,
    )
    etf_raw = etf_importer._read_export_csv(etf_path)
    etf_target = etf_importer._target_from_holding(product, etf_raw, etf_path)

    rows = [
        *_compare_option_positions(local, option_rows),
        _compare_etf_position(local, etf_target),
    ]
    active_rows = [row for row in rows if not row.get("skipped")]
    return {
        "product": product,
        "account_id": account_id,
        "date": date or holding_importer._parse_date_from_filename(option_path),
        "ok": all(row["ok"] for row in active_rows),
        "option_holding_file": str(option_path),
        "etf_holding_file": str(etf_path),
        "rows": rows,
    }


def format_position_check(payload):
    lines = [
        (
            f"账户持仓检查 {payload['product']} "
            f"date={payload.get('date')} "
            f"ok={'Y' if payload.get('ok') else 'N'}"
        ),
        f"期权文件={payload.get('option_holding_file')}",
        f"ETF文件={payload.get('etf_holding_file')}",
    ]
    for row in payload.get("rows", []):
        status = "OK" if row.get("ok") else "FAIL"
        if row.get("skipped"):
            status = "SKIP"
        lines.append(
            f"{status} {row['类型']} {row['合约代码']} {row.get('方向') or ''} "
            f"本地={_fmt_qty(row.get('本地数量'))} "
            f"券商={_fmt_qty(row.get('券商数量'))} "
            f"差异={_fmt_qty(row.get('数量差异'))}"
        )
    return lines


def _compare_option_positions(local, broker_rows):
    local_by_key = _local_option_positions(local)
    broker_by_key = _broker_option_positions(broker_rows)
    rows = []
    for key in sorted(set(local_by_key) | set(broker_by_key)):
        code, side = key
        local_qty = float(local_by_key.get(key, 0.0) or 0.0)
        broker_qty = float(broker_by_key.get(key, 0.0) or 0.0)
        diff = local_qty - broker_qty
        rows.append(
            {
                "类型": "期权",
                "合约代码": code,
                "方向": side,
                "本地数量": local_qty,
                "券商数量": broker_qty,
                "数量差异": diff,
                "ok": abs(diff) <= 1e-9,
                "skipped": False,
            }
        )
    return rows


def _local_option_positions(local):
    result = {}
    for position in (local.positions or {}).values():
        if not position:
            continue
        side = str(position.get("side") or "short").lower()
        _add_qty(result, position.get("call_code"), side, position.get("call_qty"))
        _add_qty(result, position.get("put_code"), side, position.get("put_qty"))
    return result


def _broker_option_positions(rows):
    result = {}
    for row in rows:
        _add_qty(
            result,
            row.get("order_book_id"),
            str(row.get("side") or "").lower(),
            row.get("total_qty"),
        )
    return result


def _compare_etf_position(local, broker_target):
    spec = SSE_ETF_OPTION_SPECS[local.product]
    local_qty = float(getattr(local.hedge, "qty", 0.0) or 0.0)
    broker_qty = float(broker_target.get("qty", 0.0) or 0.0)
    diff = local_qty - broker_qty
    return {
        "类型": "ETF",
        "合约代码": spec.etf_symbol,
        "方向": "long",
        "本地数量": local_qty,
        "券商数量": broker_qty,
        "数量差异": diff,
        "ok": abs(diff) <= 1e-9,
        "skipped": False,
    }


def _add_qty(result, code, side, qty):
    code = _display_code(code)
    qty = float(qty or 0.0)
    if not code or abs(qty) <= 1e-9:
        return
    key = (code, side)
    result[key] = result.get(key, 0.0) + qty


def _display_code(code):
    if code is None:
        return None
    return str(code).split(".", 1)[0].strip()


def _fmt_qty(value):
    if value is None:
        return "nan"
    number = float(value)
    return str(int(number)) if number.is_integer() else f"{number:.6f}"
