from __future__ import annotations

from . import etf_netting, storage


def write_signal_report(product, signal_payload):
    stamp = storage.local_now_stamp()
    path = storage.output_dir(product) / f"{stamp}_signal.md"
    rows, notices = _execution_rows(signal_payload)
    reasons = _advice_reason_lines(signal_payload)
    lines = [
        f"# Live Signal: {product}",
    ]
    if reasons:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(
        [
            "",
            "## Execution Plan",
            "",
            "| 执行顺序 | 合约代码 | 方向 | 数量 | 执行价格 |",
            "| ---: | --- | --- | ---: | ---: |",
        ]
    )
    if rows:
        for row in rows:
            lines.append(
                f"| {row['执行顺序']} | {row['合约代码']} | {row['方向']} | "
                f"{_fmt_qty(row['数量'])} | {_fmt(row['执行价格'])} |"
            )
    else:
        lines.append("| - | - | 无操作 | 0 | - |")
    if notices:
        lines.extend(["", "## Notices", ""])
        lines.extend(f"- {notice}" for notice in notices)

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_signal_summary(signal_payload):
    """Return terminal-friendly live advice lines for manual execution."""
    rows, notices = _execution_rows(signal_payload)
    lines = _advice_reason_lines(signal_payload)
    lines.append("执行顺序 | 合约代码 | 方向 | 数量 | 执行价格")
    if not rows:
        lines.append("- | - | 无操作 | 0 | -")
    for row in rows:
        lines.append(
            f"{row['执行顺序']} | {row['合约代码']} | {row['方向']} | "
            f"{_fmt_qty(row['数量'])} | {_fmt(row['执行价格'])}"
        )
    lines.extend(f"提示: {notice}" for notice in notices)
    return lines


def _advice_reason_lines(signal_payload):
    return [
        f"reason: {item.get('action')}={item.get('reason')}"
        for item in signal_payload.get("advice", [])
        if item.get("reason")
    ]


def _execution_rows(signal_payload):
    rows = []
    notices = []
    advice = signal_payload["advice"]
    netted_etf_items = etf_netting.netted_etf_advice_items(advice)
    for item in advice:
        item_rows = _advice_execution_rows(item, include_etf=False)
        if item_rows:
            rows.extend(item_rows)
        elif item.get("reason") and etf_netting.extract_etf_trade(item) is None:
            notices.append(f"{item.get('action')}: {item.get('reason')}")
    for item in netted_etf_items:
        rows.extend(_advice_execution_rows(item, include_etf=True))
    for index, row in enumerate(rows, start=1):
        row["执行顺序"] = index
    return rows, notices


def _advice_execution_rows(item, include_etf=True):
    action = item.get("action", "")
    side = item.get("side")

    if action == "CLOSE_OPTION_HEDGE":
        return [_execution_row(
            item.get("order_book_id"),
            "买入平仓" if side == "short" else "卖出平仓",
            item.get("qty"),
            item.get("estimated_price"),
        )]

    if action == "REDUCE_SHORT_STRADDLE_FOR_CAPACITY":
        return _option_pair_rows(
            item,
            "买入平仓",
            "call_code",
            "put_code",
            "call_qty",
            "put_qty",
            "estimated_call_price",
            "estimated_put_price",
        )

    if action.startswith("OPEN_"):
        return _option_pair_rows(
            item,
            "买入开仓" if side == "long" else "卖出开仓",
            "call_code",
            "put_code",
            "call_qty",
            "put_qty",
            "estimated_call_price",
            "estimated_put_price",
        )

    if action.startswith("CLOSE_"):
        return _option_pair_rows(
            item,
            "卖出平仓" if side == "long" else "买入平仓",
            "call_code",
            "put_code",
            "call_qty",
            "put_qty",
            "estimated_call_price",
            "estimated_put_price",
        )

    if action.startswith("ROLL_"):
        close_direction = "卖出平仓" if side == "long" else "买入平仓"
        open_direction = "买入开仓" if side == "long" else "卖出开仓"
        return _option_pair_rows(
            item,
            close_direction,
            "current_call_code",
            "current_put_code",
            "current_call_qty",
            "current_put_qty",
            "estimated_current_call_price",
            "estimated_current_put_price",
        ) + _option_pair_rows(
            item,
            open_direction,
            "target_call_code",
            "target_put_code",
            "target_call_qty",
            "target_put_qty",
            "estimated_target_call_price",
            "estimated_target_put_price",
        )

    if action in {
        "DELTA_HEDGE",
        "FINAL_DELTA_HEDGE",
        etf_netting.NETTED_ETF_HEDGE_ACTION,
    }:
        if not include_etf:
            return []
        trade_qty = item.get("trade_etf_qty")
        return [_execution_row(
            item.get("underlying_order_book_id"),
            _trade_direction(trade_qty),
            abs(float(trade_qty or 0.0)),
            item.get("estimated_price"),
        )]

    if action in {"OPTION_DELTA_HEDGE_SHORT_CALL", "FINAL_OPTION_DELTA_HEDGE_SHORT_CALL"}:
        return [_execution_row(
            item.get("call_code"),
            "卖出开仓",
            item.get("call_qty"),
            item.get("estimated_call_price"),
        )]
    if action in {
        "OPTION_DELTA_HEDGE_COMBINATION",
        "FINAL_OPTION_DELTA_HEDGE_COMBINATION",
        "GAMMA_NEUTRAL_OPTION_DELTA_HEDGE",
        "FINAL_GAMMA_NEUTRAL_OPTION_DELTA_HEDGE",
    }:
        rows = []
        if float(item.get("close_call_qty", 0.0) or 0.0) > 0:
            rows.append(_execution_row(
                item.get("close_call_code"),
                "买入平仓",
                item.get("close_call_qty"),
                item.get("estimated_close_call_price"),
            ))
        open_legs = item.get("open_legs") or [
            {
                "order_book_id": item.get("open_call_code"),
                "qty": item.get("open_call_qty"),
                "estimated_price": item.get("estimated_open_call_price"),
            }
        ]
        rows.extend(
            _execution_row(
                leg.get("order_book_id"),
                "卖出开仓",
                leg.get("qty"),
                leg.get("estimated_price"),
            )
            for leg in open_legs
        )
        if include_etf and float(item.get("trade_etf_qty", 0.0) or 0.0) > 0:
            rows.append(
                _execution_row(
                    item.get("underlying_order_book_id"),
                    "买入",
                    item.get("trade_etf_qty"),
                    item.get("estimated_price"),
                )
            )
        return rows
    return []


def _option_pair_rows(
    item,
    direction,
    call_code_key,
    put_code_key,
    call_qty_key,
    put_qty_key,
    call_price_key,
    put_price_key,
):
    rows = [
        _execution_row(
            item.get(call_code_key),
            direction,
            item.get(call_qty_key),
            item.get(call_price_key),
        ),
        _execution_row(
            item.get(put_code_key),
            direction,
            item.get(put_qty_key),
            item.get(put_price_key),
        ),
    ]
    return [
        row
        for row in rows
        if row["合约代码"] not in {None, ""}
        and row["数量"] is not None
        and float(row["数量"]) > 0
    ]


def _execution_row(code, direction, qty, price):
    return {
        "合约代码": _display_code(code),
        "方向": direction,
        "数量": qty,
        "执行价格": price,
    }


def _display_code(code):
    if code is None:
        return None
    return str(code).split(".", 1)[0]


def _trade_direction(qty):
    if qty is None:
        return "NONE"
    qty = float(qty)
    if qty > 0:
        return "买入"
    if qty < 0:
        return "卖出"
    return "无操作"


def _fmt_qty(value):
    if value is None:
        return "-"
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt(value):
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
