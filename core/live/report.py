from __future__ import annotations

from . import storage


def write_signal_report(product, signal_payload):
    stamp = storage.local_now_stamp()
    path = storage.output_dir(product) / f"{stamp}_signal.md"
    lines = [
        f"# Live Signal: {product}",
        "",
        f"- date: {signal_payload['date']}",
        f"- spot: {signal_payload['spot']:.6f}",
        f"- account_delta_after_hedge: {signal_payload['account_delta_after_hedge']:.2f}",
        f"- estimated_option_value: {signal_payload['estimated_option_value']:.2f}",
        "",
        "## Feature",
        "",
    ]
    for key, value in signal_payload["feature"].items():
        lines.append(f"- {key}: {value}")

    strategy_state = signal_payload.get("strategy_state")
    if strategy_state:
        lines.extend(["", "## Strategy State", ""])
        for key, value in strategy_state.items():
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## Advice", ""])
    for item in signal_payload["advice"]:
        lines.append(f"### {item['action']}")
        for key, value in item.items():
            if key == "action":
                continue
            lines.append(f"- {key}: {value}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_signal_summary(signal_payload):
    """Return terminal-friendly live advice lines for manual execution."""
    lines = [
        f"date={signal_payload['date']} spot={signal_payload['spot']:.6f}",
        f"account_delta_after_hedge={signal_payload['account_delta_after_hedge']:.2f}",
    ]
    for item in signal_payload["advice"]:
        lines.extend(_format_advice_item(item))
    return lines


def _format_advice_item(item):
    action = item["action"]
    reason = item.get("reason")
    side = item.get("side")
    prefix = action if side is None else f"{action} side={side}"

    if action.startswith("OPEN_"):
        return [
            (
                f"{prefix} qty={item.get('call_qty')}/{item.get('put_qty')} "
                f"strike={item.get('strike')} expiry={item.get('expiry')} "
                f"call={item.get('call_code')} put={item.get('put_code')} "
                f"call_px={_fmt(item.get('estimated_call_price'))} "
                f"put_px={_fmt(item.get('estimated_put_price'))} "
                f"cash_effect={_fmt(item.get('estimated_cash_effect'))} "
                f"margin={_fmt(item.get('estimated_option_margin'))} "
                f"reason={reason}"
            )
        ]

    if action.startswith("CLOSE_"):
        return [
            (
                f"{prefix} qty={item.get('call_qty')}/{item.get('put_qty')} "
                f"call={item.get('call_code')} put={item.get('put_code')} "
                f"call_px={_fmt(item.get('estimated_call_price'))} "
                f"put_px={_fmt(item.get('estimated_put_price'))} "
                f"cash_effect={_fmt(item.get('estimated_cash_effect'))} "
                f"reason={reason}"
            )
        ]

    if action.startswith("ROLL_"):
        return [
            (
                f"{prefix} current_strike={item.get('current_strike')} "
                f"current_expiry={item.get('current_expiry')} "
                f"current_dte={item.get('current_dte')} "
                f"target_qty={item.get('target_call_qty')}/{item.get('target_put_qty')} "
                f"target_strike={item.get('target_strike')} "
                f"target_expiry={item.get('target_expiry')} "
                f"target_call={item.get('target_call_code')} "
                f"target_put={item.get('target_put_code')} "
                f"reason={reason}"
            )
        ]

    if action == "DELTA_HEDGE":
        trade_qty = item.get("trade_etf_qty")
        direction = _trade_direction(trade_qty)
        return [
            (
                f"DELTA_HEDGE direction={direction} qty={_fmt(trade_qty)} "
                f"target_hedge_qty={_fmt(item.get('target_hedge_qty'))} "
                f"current_hedge_qty={_fmt(item.get('current_hedge_qty'))} "
                f"price={_fmt(item.get('estimated_price'))} "
                f"underlying={item.get('underlying_order_book_id')} "
                f"reason={reason}"
            )
        ]

    if action == "PROJECTED_DELTA_HEDGE":
        trade_qty = item.get("trade_etf_qty")
        direction = _trade_direction(trade_qty)
        return [
            (
                f"PROJECTED_DELTA_HEDGE trigger={item.get('trigger_action')} "
                f"side={side} direction={direction} qty={_fmt(trade_qty)} "
                f"projected_option_delta={_fmt(item.get('projected_option_delta'))} "
                f"target_hedge_qty={_fmt(item.get('target_hedge_qty'))} "
                f"current_hedge_qty={_fmt(item.get('current_hedge_qty'))} "
                f"price={_fmt(item.get('estimated_price'))} "
                f"underlying={item.get('underlying_order_book_id')} "
                f"reason={reason}"
            )
        ]

    if action == "COOLDOWN_BLOCK":
        return [
            (
                f"COOLDOWN_BLOCK side={side} "
                f"cooldown_left={item.get('cooldown_left')} reason={reason}"
            )
        ]

    return [f"{prefix}: {reason}"]


def _trade_direction(qty):
    if qty is None:
        return "NONE"
    qty = float(qty)
    if qty > 0:
        return "BUY"
    if qty < 0:
        return "SELL"
    return "NONE"


def _fmt(value):
    if value is None:
        return "None"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)
