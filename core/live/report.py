from __future__ import annotations

import pandas as pd

from . import etf_netting, option_netting, storage


def write_signal_report(product, signal_payload):
    stamp = storage.local_now_stamp()
    path = storage.output_dir(product) / f"{stamp}_signal.md"
    rows, notices = _execution_rows(signal_payload)
    reasons = _advice_reason_lines(signal_payload)
    greek_target_rows = _expected_greek_target_rows(signal_payload)
    lines = [
        f"# Live Signal: {product}",
    ]
    plan_status = signal_payload.get("plan_status")
    if plan_status:
        lines.extend(
            [
                "",
                "## Plan Status",
                "",
                f"- status: {plan_status}",
                "- execution_allowed: "
                f"{str(bool(signal_payload.get('execution_allowed'))).lower()}",
            ]
        )
    if reasons:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    if greek_target_rows:
        lines.extend(
            [
                "",
                "## Expected Greeks Target",
                "",
                "| Greek | 调整前 | 信号影响 | 影响/原值 | 调整目标 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in greek_target_rows:
            lines.append(
                f"| {row['Greek']} | {_fmt(row['调整前'])} | "
                f"{_fmt(row['信号影响'])} | {_fmt_pct(row['影响/原值'])} | "
                f"{_fmt(row['调整目标'])} |"
            )
    lines.extend(
        [
            "",
            "## Execution Plan",
            "",
            "| 执行顺序 | 合约代码 | 方向 | 数量 | 执行后预计数量 | 执行价格 |",
            "| ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    if rows:
        for row in rows:
            lines.append(
                f"| {row['执行顺序']} | {_contract_display(row)} | {row['方向']} | "
                f"{_fmt_qty(row['数量'])} | {_fmt_qty(row['执行后预计数量'])} | "
                f"{_fmt(row['执行价格'])} |"
            )
    else:
        lines.append("| - | - | 无操作 | 0 | - | - |")
    if notices:
        lines.extend(["", "## Notices", ""])
        lines.extend(f"- {notice}" for notice in notices)

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_signal_summary(signal_payload):
    """Return terminal-friendly live advice lines for manual execution."""
    rows, notices = _execution_rows(signal_payload)
    lines = _advice_reason_lines(signal_payload)
    plan_status = signal_payload.get("plan_status")
    if plan_status:
        lines.insert(
            0,
            "plan_status="
            f"{plan_status} | execution_allowed="
            f"{str(bool(signal_payload.get('execution_allowed'))).lower()}",
        )
    greek_target_rows = _expected_greek_target_rows(signal_payload)
    if greek_target_rows:
        lines.append("预期Greeks目标 | Greek | 调整前 | 信号影响 | 影响/原值 | 调整目标")
        for row in greek_target_rows:
            lines.append(
                f"预期Greeks目标 | {row['Greek']} | {_fmt(row['调整前'])} | "
                f"{_fmt(row['信号影响'])} | {_fmt_pct(row['影响/原值'])} | "
                f"{_fmt(row['调整目标'])}"
            )
    lines.append("执行顺序 | 合约代码 | 方向 | 数量 | 执行后预计数量 | 执行价格")
    if not rows:
        lines.append("- | - | 无操作 | 0 | - | -")
    for row in rows:
        lines.append(
            f"{row['执行顺序']} | {_contract_display(row)} | {row['方向']} | "
            f"{_fmt_qty(row['数量'])} | {_fmt_qty(row['执行后预计数量'])} | "
            f"{_fmt(row['执行价格'])}"
        )
    lines.extend(f"提示: {notice}" for notice in notices)
    return lines


def _advice_reason_lines(signal_payload):
    return [
        f"reason: {item.get('action')}={item.get('reason')}"
        for item in signal_payload.get("advice", [])
        if item.get("reason") and not _is_zero_quantity_etf_advice(item)
    ]


def _expected_greek_target_rows(signal_payload):
    advice = signal_payload.get("advice", []) or []
    if not advice:
        return []

    account_greeks = signal_payload.get("account_greeks") or {}
    planned_greeks = signal_payload.get("planned_account_greeks") or account_greeks
    available_greeks = account_greeks or planned_greeks
    current_hedge_qty = _current_hedge_qty(signal_payload)

    delta_before = _number(signal_payload.get("current_account_delta"))
    if delta_before is None:
        delta_before = _number(signal_payload.get("account_delta_after_hedge"))
    if delta_before is None:
        delta_before = _first_number(
            advice,
            "residual_delta_before_option_rebalance",
            "planned_account_delta",
            "account_delta",
        )
    if delta_before is None:
        option_delta = _number(available_greeks.get("delta"))
        delta_before = (
            None
            if option_delta is None
            else option_delta + current_hedge_qty
        )
    delta_after = _number(signal_payload.get("planned_account_delta"))
    if delta_after is None:
        delta_after = _target_delta_after_signal(
            advice,
            planned_greeks,
            current_hedge_qty,
        )
    delta_effect = _difference(delta_after, delta_before)

    rows = [
        _greek_target_row("Delta", delta_before, delta_effect, delta_after),
    ]
    for label, key in [("Gamma", "gamma"), ("Vega", "vega"), ("Theta", "theta")]:
        before = _number(account_greeks.get(key))
        target = _number(planned_greeks.get(key))
        effect = _difference(target, before)
        rows.append(_greek_target_row(label, before, effect, target))

    return [
        row
        for row in rows
        if row["调整前"] is not None or row["调整目标"] is not None
    ]


def _target_delta_after_signal(
    advice,
    planned_greeks,
    current_hedge_qty,
):
    delta_after = _last_number(
        advice,
        "projected_account_delta_after_hedge",
        "projected_account_delta_after_combined_hedge",
        "projected_account_delta_after_option_rebalance",
    )
    if delta_after is not None:
        return delta_after

    option_delta = _last_number(advice, "planned_option_delta", "option_delta")
    if option_delta is None:
        option_delta = _number(planned_greeks.get("delta"))
    target_hedge_qty = _last_number(advice, "target_hedge_qty")
    if target_hedge_qty is None:
        target_hedge_qty = current_hedge_qty
    if option_delta is None:
        return None
    return option_delta + target_hedge_qty


def _current_hedge_qty(signal_payload):
    account = signal_payload.get("account") or {}
    hedge = account.get("hedge") or {}
    return _number(hedge.get("qty")) or 0.0


def _difference(after, before):
    after = _number(after)
    before = _number(before)
    if after is None or before is None:
        return None
    return after - before


def _greek_target_row(label, before, effect, target):
    return {
        "Greek": label,
        "调整前": before,
        "信号影响": effect,
        "影响/原值": _safe_ratio(effect, before),
        "调整目标": target,
    }


def _first_number(items, *keys):
    for item in items:
        for key in keys:
            value = _number(item.get(key))
            if value is not None:
                return value
    return None


def _last_number(items, *keys):
    for item in reversed(items):
        for key in keys:
            value = _number(item.get(key))
            if value is not None:
                return value
    return None


def _safe_ratio(numerator, denominator):
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def _execution_rows(signal_payload):
    rows = []
    notices = []
    advice = signal_payload["advice"]
    netted_etf_items = etf_netting.netted_etf_advice_items(advice)
    for item in advice:
        item_rows = _advice_execution_rows(item, include_etf=False)
        if item_rows:
            rows.extend(item_rows)
        elif (
            item.get("reason")
            and etf_netting.extract_etf_trade(item) is None
            and not _is_zero_quantity_etf_advice(item)
        ):
            notices.append(f"{item.get('action')}: {item.get('reason')}")
    for item in netted_etf_items:
        rows.extend(_advice_execution_rows(item, include_etf=True))
    rows = option_netting.net_exact_report_rows(rows)
    _annotate_projected_quantities(rows, signal_payload)
    _annotate_contract_symbols(rows, signal_payload)
    for index, row in enumerate(rows, start=1):
        row["执行顺序"] = index
    return rows, notices


def _is_zero_quantity_etf_advice(item):
    action = str(item.get("action") or "")
    if action not in etf_netting.ETF_HEDGE_ACTIONS | {
        etf_netting.NETTED_ETF_HEDGE_ACTION,
    }:
        return False
    if item.get("trade_etf_qty") is None:
        return False
    trade_qty = _number(item.get("trade_etf_qty"))
    return trade_qty is not None and abs(trade_qty) <= 1e-9


def _annotate_projected_quantities(rows, signal_payload):
    quantities = _initial_display_quantities(signal_payload)
    for row in rows:
        code = str(row.get("合约代码") or "")
        if not code:
            row["执行后预计数量"] = None
            continue
        current_qty = float(quantities.get(code, 0.0) or 0.0)
        trade_qty = _number(row.get("数量")) or 0.0
        direction = row.get("方向")
        if direction in {"买入开仓", "卖出开仓", "买入"}:
            current_qty += trade_qty
        elif direction in {"买入平仓", "卖出平仓", "卖出"}:
            current_qty -= trade_qty
        quantities[code] = current_qty
        row["执行后预计数量"] = _display_quantity(current_qty)


def _annotate_contract_symbols(rows, signal_payload):
    symbol_by_code = _contract_symbol_map(signal_payload)
    if not symbol_by_code:
        return
    for row in rows:
        code = _display_code(row.get("合约代码"))
        if code is None:
            continue
        symbol = symbol_by_code.get(str(code))
        if symbol:
            row["contract_symbol"] = symbol


def _contract_symbol_map(signal_payload):
    mapping = {}
    account = signal_payload.get("account") or {}
    snapshot = signal_payload.get("quote_snapshot") or {}
    option_snapshot = snapshot.get("option_snapshot")
    if option_snapshot:
        try:
            frame = pd.read_parquet(option_snapshot, columns=["order_book_id", "contract_symbol"])
        except Exception:
            frame = None
        if frame is not None and not frame.empty:
            for _, row in frame.iterrows():
                _add_contract_symbol(
                    mapping,
                    row.get("order_book_id"),
                    row.get("contract_symbol"),
                )
    return mapping


def _add_contract_symbol(mapping, code, symbol):
    display_code = _display_code(code)
    if display_code is None or symbol is None:
        return
    symbol_text = str(symbol).strip()
    if not symbol_text or symbol_text.lower() == "nan":
        return
    mapping[str(display_code)] = symbol_text


def _contract_display(row):
    code = row.get("合约代码")
    symbol = row.get("contract_symbol")
    if symbol and str(symbol) != str(code):
        return f"{code} ({symbol})"
    return code


def _initial_display_quantities(signal_payload):
    account = signal_payload.get("account") or {}
    quantities = {}
    positions = account.get("positions") or {}
    for position in positions.values():
        if not position:
            continue
        _add_quantity(quantities, position.get("call_code"), position.get("call_qty"))
        _add_quantity(quantities, position.get("put_code"), position.get("put_qty"))
    hedge_state = account.get("hedge") or {}
    hedge_code = _display_code(hedge_state.get("underlying_order_book_id"))
    _add_quantity(quantities, hedge_code, hedge_state.get("qty"))
    return quantities


def _add_quantity(quantities, code, qty):
    display_code = _display_code(code)
    if display_code is None:
        return
    number = _number(qty)
    if number is None:
        return
    quantities[str(display_code)] = float(quantities.get(str(display_code), 0.0)) + number


def _display_quantity(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if abs(number) <= 1e-9:
        number = 0.0
    return int(number) if number.is_integer() else number


def _advice_execution_rows(item, include_etf=True):
    action = item.get("action", "")
    side = item.get("side")

    if action == etf_netting.PRE_ROLL_HEDGE_CLOSE_ACTION:
        if not include_etf:
            return []
        trade_qty = item.get("trade_etf_qty")
        return [_execution_row(
            item.get("underlying_order_book_id"),
            _trade_direction(trade_qty),
            abs(float(trade_qty or 0.0)),
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
        "ATM_STRADDLE_DELTA_REBALANCE",
        "FINAL_ATM_STRADDLE_DELTA_REBALANCE",
        "ATM_STRADDLE_SHAPE_REBALANCE",
        "FINAL_ATM_STRADDLE_SHAPE_REBALANCE",
    }:
        return [
            row
            for row in [
            _execution_row(
                item.get("close_call_code"),
                "买入平仓",
                item.get("close_call_qty"),
                item.get("estimated_close_call_price"),
            ),
            _execution_row(
                item.get("close_put_code"),
                "买入平仓",
                item.get("close_put_qty"),
                item.get("estimated_close_put_price"),
            ),
            _execution_row(
                item.get("open_call_code"),
                "卖出开仓",
                item.get("open_call_qty"),
                item.get("estimated_open_call_price"),
            ),
            _execution_row(
                item.get("open_put_code"),
                "卖出开仓",
                item.get("open_put_qty"),
                item.get("estimated_open_put_price"),
            ),
        ]
            if (_number(row.get("数量")) or 0.0) > 0
        ]

    if action in etf_netting.ETF_HEDGE_ACTIONS | {
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


def _fmt_pct(value):
    if value is None:
        return "None"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number
