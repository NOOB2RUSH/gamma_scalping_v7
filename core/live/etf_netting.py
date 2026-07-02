from __future__ import annotations


ETF_HEDGE_ACTIONS = {
    "DELTA_HEDGE",
    "FINAL_DELTA_HEDGE",
    "OPTION_DELTA_HEDGE_COMBINATION",
    "FINAL_OPTION_DELTA_HEDGE_COMBINATION",
    "GAMMA_NEUTRAL_OPTION_DELTA_HEDGE",
    "FINAL_GAMMA_NEUTRAL_OPTION_DELTA_HEDGE",
}
NETTED_ETF_HEDGE_ACTION = "NETTED_ETF_HEDGE"


def extract_etf_trade(item):
    action = str(item.get("action") or "")
    if item.get("priority") != "action" or action not in ETF_HEDGE_ACTIONS:
        return None

    qty = _float(item.get("trade_etf_qty"))
    if abs(qty) <= 1e-9:
        return None

    underlying = item.get("underlying_order_book_id")
    if underlying in {None, ""}:
        return None

    return {
        "action": action,
        "underlying_order_book_id": underlying,
        "trade_etf_qty": qty,
        "current_hedge_qty": _optional_float(item.get("current_hedge_qty")),
        "target_hedge_qty": _optional_float(item.get("target_hedge_qty")),
        "estimated_price": item.get("estimated_price"),
    }


def netted_etf_advice_items(advice):
    by_underlying = {}
    for index, item in enumerate(advice or []):
        trade = extract_etf_trade(item)
        if trade is None:
            continue

        key = str(trade["underlying_order_book_id"])
        existing = by_underlying.setdefault(
            key,
            {
                "action": NETTED_ETF_HEDGE_ACTION,
                "priority": "action",
                "reason": "Netted ETF hedge trade from generated advice.",
                "underlying_order_book_id": trade["underlying_order_book_id"],
                "trade_etf_qty": 0.0,
                "current_hedge_qty": trade.get("current_hedge_qty"),
                "target_hedge_qty": None,
                "estimated_price": None,
                "source_actions": [],
                "_last_index": index,
            },
        )
        existing["trade_etf_qty"] += trade["trade_etf_qty"]
        existing["_last_index"] = index
        existing["source_actions"].append(trade["action"])
        if trade.get("estimated_price") is not None:
            existing["estimated_price"] = trade["estimated_price"]
        if trade.get("target_hedge_qty") is not None:
            existing["target_hedge_qty"] = trade["target_hedge_qty"]

    result = []
    for item in sorted(by_underlying.values(), key=lambda value: value["_last_index"]):
        net_qty = float(item["trade_etf_qty"])
        if abs(net_qty) <= 1e-9:
            continue
        if item.get("target_hedge_qty") is None:
            current_qty = _optional_float(item.get("current_hedge_qty")) or 0.0
            item["target_hedge_qty"] = current_qty + net_qty
        item.pop("_last_index", None)
        result.append(item)
    return result


def _float(value):
    return float(value or 0.0)


def _optional_float(value):
    if value is None:
        return None
    return float(value)
