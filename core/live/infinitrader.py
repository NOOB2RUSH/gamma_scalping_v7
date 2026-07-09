from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from .. import strategy as core_strategy
from . import account as account_store
from . import etf_netting
from . import storage
from .runtime import load_product_config


BUY = "buy"
SELL = "sell"
BUY_TYPE = "0"
SELL_TYPE = "1"
OFFSET_OPEN = "0"
OFFSET_CLOSE = "1"
ORDER_TYPE_GFD = "0"
HEDGEFLAG_SPECULATION = "1"
PENDING_COMMAND_FILE = "pending_command.json"


@dataclass(frozen=True)
class InfiniOrder:
    sequence: int
    action: str
    leg: str
    exchange: str
    instrument_id: str
    order_direction: str
    order_direction_type: str
    offset: str | None
    volume: int
    price: float
    order_type: str = ORDER_TYPE_GFD
    hedgeflag: str = HEDGEFLAG_SPECULATION
    market: bool = False
    memo: str | None = None

    def to_dict(self):
        return asdict(self)


def compile_signal_orders(signal_payload: dict[str, Any]) -> list[dict[str, Any]]:
    orders: list[InfiniOrder] = []
    advice = signal_payload.get("advice", []) or []
    for item in advice:
        if item.get("priority") != "action":
            continue
        orders.extend(_compile_advice_item(item, len(orders) + 1, include_etf=False))
    for item in etf_netting.netted_etf_advice_items(advice):
        orders.extend(_compile_advice_item(item, len(orders) + 1, include_etf=True))
    orders = _renumber_orders(_net_exact_option_hedge_orders(orders))
    return [order.to_dict() for order in orders if order.volume > 0]


def write_command(
    signal_payload: dict[str, Any],
    orders: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    product = str(signal_payload.get("product"))
    run_id = run_id or storage.local_now_stamp()
    orders = orders if orders is not None else compile_signal_orders(signal_payload)
    command = {
        "run_id": run_id,
        "product": product,
        "account_id": signal_payload.get("account_id", "default"),
        "date": signal_payload.get("date"),
        "created_at": pd.Timestamp.now().isoformat(),
        "signal": signal_payload,
        "orders": orders,
    }
    runtime = runtime_dir(product)
    command_path = runtime / "commands" / f"{run_id}_command.json"
    command_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(command_path, command)
    _write_json(runtime / PENDING_COMMAND_FILE, command)
    command["command_path"] = str(command_path)
    command["pending_path"] = str(runtime / PENDING_COMMAND_FILE)
    return command


def runtime_dir(product: str) -> Path:
    return storage.output_dir(product) / "infinitrader"


def sync_command_to_account(
    command_path,
    account_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    command = storage.read_json(command_path)
    product = command["product"]
    account_id = account_id or command.get("account_id", "default")
    fills = build_fills_from_command(command)
    applied = []
    warnings = []
    local = account_store.load_account(product, account_id=account_id)
    for fill in fills:
        try:
            normalized = account_store.normalize_fill(fill)
            if dry_run:
                account_store._apply_fill(local, product, normalized)
            else:
                local = account_store.record_fill(product, normalized, account_id=account_id)
            applied.append({"dry_run": dry_run, "fill": normalized})
        except Exception as exc:
            warnings.append(
                {
                    "fill": fill,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "product": product,
        "account_id": account_id,
        "command_path": str(command_path),
        "dry_run": dry_run,
        "applied": applied,
        "warnings": warnings,
    }


def build_fills_from_command(command: dict[str, Any]) -> list[dict[str, Any]]:
    signal = command.get("signal") or {}
    product = command.get("product") or signal.get("product")
    config = load_product_config(product)
    date = command.get("date") or signal.get("date")
    source_file = command.get("command_path")
    fills: list[dict[str, Any]] = []
    advice = signal.get("advice", []) or []
    for item in advice:
        if item.get("priority") != "action":
            continue
        fills.extend(
            _fills_from_advice(
                item,
                signal,
                date,
                source_file,
                float(config.vol.contract_multiplier),
                float(config.backtest.option_fee_per_contract),
                include_etf=False,
            )
        )
    for item in etf_netting.netted_etf_advice_items(advice):
        fills.extend(
            _fills_from_advice(
                item,
                signal,
                date,
                source_file,
                float(config.vol.contract_multiplier),
                float(config.backtest.option_fee_per_contract),
                include_etf=True,
            )
        )
    return _net_exact_option_hedge_fills(fills)


def _compile_advice_item(
    item: dict[str, Any],
    start_sequence: int,
    include_etf: bool = True,
) -> list[InfiniOrder]:
    action = str(item.get("action") or "")
    side = str(item.get("side") or "").lower()

    if action == "CLOSE_OPTION_HEDGE":
        close_direction = BUY if side == "short" else SELL
        return [
            _option_order(
                start_sequence,
                action,
                "option_hedge_close",
                item.get("order_book_id"),
                close_direction,
                OFFSET_CLOSE,
                item.get("qty"),
                item.get("estimated_price"),
            )
        ]

    if action == "REDUCE_SHORT_STRADDLE_FOR_CAPACITY":
        return _option_pair_orders(
            item,
            start_sequence,
            action,
            BUY,
            OFFSET_CLOSE,
            "call_code",
            "put_code",
            "call_qty",
            "put_qty",
            "estimated_call_price",
            "estimated_put_price",
            leg_prefix="reduce",
        )

    if action.startswith("OPEN_"):
        direction = BUY if side == "long" else SELL
        return _option_pair_orders(
            item,
            start_sequence,
            action,
            direction,
            OFFSET_OPEN,
            "call_code",
            "put_code",
            "call_qty",
            "put_qty",
            "estimated_call_price",
            "estimated_put_price",
            leg_prefix="open",
        )

    if action.startswith("CLOSE_"):
        direction = SELL if side == "long" else BUY
        return _option_pair_orders(
            item,
            start_sequence,
            action,
            direction,
            OFFSET_CLOSE,
            "call_code",
            "put_code",
            "call_qty",
            "put_qty",
            "estimated_call_price",
            "estimated_put_price",
            leg_prefix="close",
        )

    if action.startswith("ROLL_"):
        close_direction = SELL if side == "long" else BUY
        open_direction = BUY if side == "long" else SELL
        close_orders = _option_pair_orders(
            item,
            start_sequence,
            action,
            close_direction,
            OFFSET_CLOSE,
            "current_call_code",
            "current_put_code",
            "current_call_qty",
            "current_put_qty",
            "estimated_current_call_price",
            "estimated_current_put_price",
            leg_prefix="roll_close",
        )
        open_orders = _option_pair_orders(
            item,
            start_sequence + len(close_orders),
            action,
            open_direction,
            OFFSET_OPEN,
            "target_call_code",
            "target_put_code",
            "target_call_qty",
            "target_put_qty",
            "estimated_target_call_price",
            "estimated_target_put_price",
            leg_prefix="roll_open",
        )
        return close_orders + open_orders

    if action in {
        "DELTA_HEDGE",
        "FINAL_DELTA_HEDGE",
        etf_netting.NETTED_ETF_HEDGE_ACTION,
    }:
        if not include_etf:
            return []
        return [_etf_order(start_sequence, action, item)]

    if action in {
        "ATM_STRADDLE_DELTA_REBALANCE",
        "FINAL_ATM_STRADDLE_DELTA_REBALANCE",
        "ATM_STRADDLE_SHAPE_REBALANCE",
        "FINAL_ATM_STRADDLE_SHAPE_REBALANCE",
    }:
        return _atm_straddle_delta_rebalance_orders(
            item,
            start_sequence,
            action,
        )

    return []


def _fills_from_advice(
    item: dict[str, Any],
    signal: dict[str, Any],
    date,
    source_file,
    multiplier: float,
    option_fee_per_contract: float,
    include_etf: bool = True,
) -> list[dict[str, Any]]:
    action = str(item.get("action") or "")
    side = str(item.get("side") or "").lower()
    if action in {
        "DELTA_HEDGE",
        "FINAL_DELTA_HEDGE",
        etf_netting.NETTED_ETF_HEDGE_ACTION,
    }:
        if not include_etf:
            return []
        return [_delta_hedge_fill(item, date, source_file)]

    if action == "CLOSE_OPTION_HEDGE":
        qty = _int_qty(item.get("qty"))
        price = float(item.get("estimated_price", 0.0) or 0.0)
        direction = -1.0 if side == "short" else 1.0
        return [
            {
                "action": "close_option_hedge",
                "date": date,
                "side": side or "short",
                "order_book_id": item.get("order_book_id"),
                "qty": qty,
                "price": price,
                "cash_delta": direction * qty * price * multiplier,
                "source_file": source_file,
                "import_source": "infinitrader_command",
            }
        ]

    if action.startswith("OPEN_"):
        return [_straddle_fill(item, action.lower(), date, source_file, multiplier)]

    if action.startswith("ROLL_"):
        return [_straddle_fill(item, action.lower(), date, source_file, multiplier)]

    if action.startswith("CLOSE_"):
        return [_close_straddle_fill(item, date, source_file, multiplier)]

    if action == "REDUCE_SHORT_STRADDLE_FOR_CAPACITY":
        fill = _rebalance_short_core_call_fill(
            item,
            signal,
            date,
            source_file,
            multiplier,
            option_fee_per_contract,
        )
        return [fill] if fill is not None else []

    if action in {
        "ATM_STRADDLE_DELTA_REBALANCE",
        "FINAL_ATM_STRADDLE_DELTA_REBALANCE",
        "ATM_STRADDLE_SHAPE_REBALANCE",
        "FINAL_ATM_STRADDLE_SHAPE_REBALANCE",
    }:
        fill = _atm_straddle_delta_rebalance_fill(
            item,
            signal,
            date,
            source_file,
            multiplier,
            option_fee_per_contract,
        )
        return [fill] if fill is not None else []

    return []


def _net_exact_option_hedge_orders(orders: list[InfiniOrder]) -> list[InfiniOrder]:
    result: list[InfiniOrder] = []
    for order in orders:
        if not (_is_option_hedge_close_order(order) or _is_option_hedge_open_order(order)):
            result.append(order)
            continue

        residual = order
        residual_volume = int(order.volume or 0)
        index = 0
        while index < len(result) and residual_volume > 0:
            other = result[index]
            if not _opposite_option_hedge_orders(other, residual):
                index += 1
                continue
            matched_volume = min(int(other.volume or 0), residual_volume)
            other_volume = int(other.volume or 0) - matched_volume
            residual_volume -= matched_volume
            if other_volume <= 0:
                result.pop(index)
            else:
                result[index] = replace(other, volume=other_volume)
                index += 1
        if residual_volume > 0:
            result.append(replace(residual, volume=residual_volume))
    return result


def _is_option_hedge_close_order(order: InfiniOrder) -> bool:
    return order.leg == "option_hedge_close"


def _opposite_option_hedge_orders(left: InfiniOrder, right: InfiniOrder) -> bool:
    if left.exchange != right.exchange or left.instrument_id != right.instrument_id:
        return False
    if _is_option_hedge_close_order(left) and _is_option_hedge_open_order(right):
        close_order, open_order = left, right
    elif _is_option_hedge_close_order(right) and _is_option_hedge_open_order(left):
        close_order, open_order = right, left
    else:
        return False
    return _opposite_close_open(close_order, open_order)


def _is_option_hedge_open_order(order: InfiniOrder) -> bool:
    return False


def _opposite_close_open(close_order: InfiniOrder, open_order: InfiniOrder) -> bool:
    return (
        close_order.offset == OFFSET_CLOSE
        and open_order.offset == OFFSET_OPEN
        and (
            (close_order.order_direction == BUY and open_order.order_direction == SELL)
            or (close_order.order_direction == SELL and open_order.order_direction == BUY)
        )
    )


def _renumber_orders(orders: list[InfiniOrder]) -> list[InfiniOrder]:
    return [
        replace(order, sequence=index, memo=f"{order.action}:{order.leg}:{index}")
        for index, order in enumerate(orders, start=1)
    ]


def _net_exact_option_hedge_fills(fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fill in fills:
        if fill.get("action") not in {"close_option_hedge", "open_option_hedge"}:
            result.append(fill)
            continue

        residual = dict(fill)
        residual_qty = _int_qty(residual.get("qty"))
        index = 0
        while index < len(result) and residual_qty > 0:
            other = result[index]
            if not _opposite_option_hedge_fills(other, residual):
                index += 1
                continue
            other_qty = _int_qty(other.get("qty"))
            matched_qty = min(other_qty, residual_qty)
            other_qty -= matched_qty
            residual_qty -= matched_qty
            if other_qty <= 0:
                result.pop(index)
            else:
                result[index] = _scale_option_hedge_fill(other, other_qty)
                index += 1
        if residual_qty > 0:
            result.append(_scale_option_hedge_fill(residual, residual_qty))
    return result


def _opposite_option_hedge_fills(left: dict[str, Any], right: dict[str, Any]) -> bool:
    actions = {left.get("action"), right.get("action")}
    if actions != {"close_option_hedge", "open_option_hedge"}:
        return False
    return (
        _same_code(left.get("order_book_id"), right.get("order_book_id"))
        and str(left.get("side", "short")).lower()
        == str(right.get("side", "short")).lower()
    )


def _scale_option_hedge_fill(fill: dict[str, Any], qty: int) -> dict[str, Any]:
    original_qty = max(_int_qty(fill.get("qty")), 1)
    if qty == original_qty:
        return dict(fill)
    scale = qty / original_qty
    result = dict(fill)
    result["qty"] = qty
    for key in ("option_margin", "cash_delta"):
        if key in result:
            result[key] = float(result.get(key, 0.0) or 0.0) * scale
    return result


def _same_code(left, right) -> bool:
    return str(left or "").split(".", 1)[0] == str(right or "").split(".", 1)[0]


def _delta_hedge_fill(item, date, source_file):
    trade_qty = _etf_board_lot_qty(item.get("trade_etf_qty", 0.0))
    price = float(item.get("estimated_price", 0.0) or 0.0)
    target_qty = float(item.get("target_hedge_qty", 0.0) or 0.0)
    target_qty += trade_qty - float(item.get("trade_etf_qty", 0.0) or 0.0)
    return {
        "action": "delta_hedge",
        "date": date,
        "underlying_order_book_id": item.get("underlying_order_book_id"),
        "trade_etf_qty": trade_qty,
        "qty": target_qty,
        "target_hedge_qty": target_qty,
        "price": price,
        "entry_price": price,
        "margin": abs(target_qty) * price,
        "cash_delta": -trade_qty * price,
        "source_file": source_file,
        "import_source": "infinitrader_command",
        "security_trades": [
            {
                "signed_qty": trade_qty,
                "qty": abs(trade_qty),
                "price": price,
            }
        ],
    }


def _straddle_fill(item, action, date, source_file, multiplier):
    side = item.get("side")
    target_prefix = "target_" if action.startswith("roll_") else ""
    return {
        "action": action,
        "date": date,
        "side": side,
        "call_code": item.get(f"{target_prefix}call_code") or item.get("call_code"),
        "put_code": item.get(f"{target_prefix}put_code") or item.get("put_code"),
        "call_qty": item.get(f"{target_prefix}call_qty") or item.get("call_qty"),
        "put_qty": item.get(f"{target_prefix}put_qty") or item.get("put_qty"),
        "strike": item.get("target_strike") or item.get("strike"),
        "expiry": item.get("target_expiry") or item.get("expiry"),
        "entry_call_price": item.get(f"estimated_{target_prefix}call_price")
        or item.get("estimated_call_price"),
        "entry_put_price": item.get(f"estimated_{target_prefix}put_price")
        or item.get("estimated_put_price"),
        "entry_option_value": item.get("estimated_option_value", 0.0),
        "option_margin": item.get("estimated_option_margin", 0.0),
        "contract_multiplier": multiplier,
        "underlying_order_book_id": item.get("underlying_order_book_id"),
        "cash_delta": item.get("estimated_cash_effect", 0.0),
        "source_file": source_file,
        "import_source": "infinitrader_command",
    }


def _close_straddle_fill(item, date, source_file, multiplier):
    return {
        "action": str(item.get("action") or "").lower(),
        "date": date,
        "side": item.get("side"),
        "call_code": item.get("call_code"),
        "put_code": item.get("put_code"),
        "call_qty": item.get("call_qty"),
        "put_qty": item.get("put_qty"),
        "call_price": item.get("estimated_call_price"),
        "put_price": item.get("estimated_put_price"),
        "contract_multiplier": multiplier,
        "cash_delta": item.get("estimated_cash_effect", 0.0),
        "source_file": source_file,
        "import_source": "infinitrader_command",
    }


def _rebalance_short_core_call_fill(
    item,
    signal,
    date,
    source_file,
    multiplier,
    option_fee_per_contract,
):
    close_source = item.get("close_source")
    close_qty = _int_qty(item.get("close_call_qty"))
    if close_source != "core_short_call" or close_qty <= 0:
        return None
    position = ((signal.get("account") or {}).get("positions") or {}).get("short")
    if not position:
        return None
    new_call_qty = max(0, int(position.get("call_qty", 0) or 0) - close_qty)
    close_price = float(item.get("estimated_close_call_price", 0.0) or 0.0)
    return {
        "action": "rebalance_straddle_legs",
        "date": date,
        "side": "short",
        "call_code": position.get("call_code"),
        "put_code": position.get("put_code"),
        "call_qty": new_call_qty,
        "put_qty": position.get("put_qty"),
        "strike": position.get("strike"),
        "expiry": position.get("expiry"),
        "entry_call_price": position.get("entry_call_price"),
        "entry_put_price": position.get("entry_put_price"),
        "entry_option_value": position.get("entry_option_value", 0.0),
        "option_margin": max(
            0.0,
            float(position.get("option_margin", 0.0) or 0.0)
            - float(item.get("estimated_close_margin_release", 0.0) or 0.0),
        ),
        "contract_multiplier": position.get("contract_multiplier", multiplier),
        "underlying_order_book_id": position.get("underlying_order_book_id")
        or item.get("underlying_order_book_id"),
        "cash_delta": -close_qty * close_price * multiplier - close_qty * option_fee_per_contract,
        "leg_adjustments": [
            {
                "leg": "call",
                "qty_change": -close_qty,
                "price": close_price,
                "order_book_id": position.get("call_code"),
            }
        ],
        "source_file": source_file,
        "import_source": "infinitrader_command",
    }


def _atm_straddle_delta_rebalance_fill(
    item,
    signal,
    date,
    source_file,
    multiplier,
    option_fee_per_contract,
):
    position = ((signal.get("account") or {}).get("positions") or {}).get("short")
    if not position:
        return None
    close_call_qty = _int_qty(item.get("close_call_qty"))
    close_put_qty = _int_qty(item.get("close_put_qty"))
    open_put_qty = _int_qty(item.get("open_put_qty"))
    open_call_qty = _int_qty(item.get("open_call_qty"))
    if close_call_qty <= 0 and close_put_qty <= 0 and open_call_qty <= 0 and open_put_qty <= 0:
        return None
    close_call_price = float(item.get("estimated_close_call_price", 0.0) or 0.0)
    close_put_price = float(item.get("estimated_close_put_price", 0.0) or 0.0)
    open_call_price = float(item.get("estimated_open_call_price", 0.0) or 0.0)
    open_put_price = float(item.get("estimated_open_put_price", 0.0) or 0.0)
    target_call_qty = _int_qty(
        item.get(
            "target_call_qty",
            int(position.get("call_qty", 0) or 0) - close_call_qty + open_call_qty,
        )
    )
    target_put_qty = _int_qty(
        item.get(
            "target_put_qty",
            int(position.get("put_qty", 0) or 0) - close_put_qty + open_put_qty,
        )
    )
    fee = (
        close_call_qty + close_put_qty + open_call_qty + open_put_qty
    ) * float(option_fee_per_contract)
    cash_delta = (
        open_call_qty * open_call_price * multiplier
        + open_put_qty * open_put_price * multiplier
        - close_call_qty * close_call_price * multiplier
        - close_put_qty * close_put_price * multiplier
        - fee
    )
    leg_adjustments = [
        {
            "leg": "call",
            "qty_change": -close_call_qty,
            "price": close_call_price,
            "order_book_id": position.get("call_code"),
        },
        {
            "leg": "put",
            "qty_change": -close_put_qty,
            "price": close_put_price,
            "order_book_id": position.get("put_code"),
        },
        {
            "leg": "call",
            "qty_change": open_call_qty,
            "price": open_call_price,
            "order_book_id": position.get("call_code"),
        },
        {
            "leg": "put",
            "qty_change": open_put_qty,
            "price": open_put_price,
            "order_book_id": position.get("put_code"),
        },
    ]
    leg_adjustments = [
        adjustment
        for adjustment in leg_adjustments
        if _int_qty(abs(adjustment["qty_change"])) > 0
    ]
    return {
        "action": "rebalance_straddle_legs",
        "date": date,
        "side": "short",
        "call_code": position.get("call_code"),
        "put_code": position.get("put_code"),
        "call_qty": target_call_qty,
        "put_qty": max(0, target_put_qty),
        "strike": position.get("strike"),
        "expiry": position.get("expiry"),
        "entry_call_price": position.get("entry_call_price"),
        "entry_put_price": position.get("entry_put_price"),
        "entry_option_value": position.get("entry_option_value", 0.0),
        "option_margin": float(
            item.get("estimated_option_margin", position.get("option_margin", 0.0))
            or 0.0
        ),
        "contract_multiplier": position.get("contract_multiplier", multiplier),
        "underlying_order_book_id": position.get("underlying_order_book_id")
        or item.get("underlying_order_book_id"),
        "cash_delta": cash_delta,
        "leg_adjustments": leg_adjustments,
        "source_file": source_file,
        "import_source": "infinitrader_command",
    }


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _atm_straddle_delta_rebalance_orders(
    item: dict[str, Any],
    start_sequence: int,
    action: str,
) -> list[InfiniOrder]:
    orders: list[InfiniOrder] = []
    close_call_qty = _int_qty(item.get("close_call_qty"))
    if close_call_qty > 0:
        orders.append(
            _option_order(
                start_sequence,
                action,
                "atm_rebalance_close_call",
                item.get("close_call_code"),
                BUY,
                OFFSET_CLOSE,
                close_call_qty,
                item.get("estimated_close_call_price"),
            )
        )
    close_put_qty = _int_qty(item.get("close_put_qty"))
    if close_put_qty > 0:
        orders.append(
            _option_order(
                start_sequence + len(orders),
                action,
                "atm_rebalance_close_put",
                item.get("close_put_code"),
                BUY,
                OFFSET_CLOSE,
                close_put_qty,
                item.get("estimated_close_put_price"),
            )
        )
    open_call_qty = _int_qty(item.get("open_call_qty"))
    if open_call_qty > 0:
        orders.append(
            _option_order(
                start_sequence + len(orders),
                action,
                "atm_rebalance_open_call",
                item.get("open_call_code"),
                SELL,
                OFFSET_OPEN,
                open_call_qty,
                item.get("estimated_open_call_price"),
            )
        )
    open_put_qty = _int_qty(item.get("open_put_qty"))
    if open_put_qty > 0:
        orders.append(
            _option_order(
                start_sequence + len(orders),
                action,
                "atm_rebalance_open_put",
                item.get("open_put_code"),
                SELL,
                OFFSET_OPEN,
                open_put_qty,
                item.get("estimated_open_put_price"),
            )
        )
    return orders


def _option_pair_orders(
    item: dict[str, Any],
    start_sequence: int,
    action: str,
    direction: str,
    offset: str,
    call_code_key: str,
    put_code_key: str,
    call_qty_key: str,
    put_qty_key: str,
    call_price_key: str,
    put_price_key: str,
    leg_prefix: str,
) -> list[InfiniOrder]:
    return [
        _option_order(
            start_sequence,
            action,
            f"{leg_prefix}_call",
            item.get(call_code_key),
            direction,
            offset,
            item.get(call_qty_key),
            item.get(call_price_key),
        ),
        _option_order(
            start_sequence + 1,
            action,
            f"{leg_prefix}_put",
            item.get(put_code_key),
            direction,
            offset,
            item.get(put_qty_key),
            item.get(put_price_key),
        ),
    ]


def _option_order(
    sequence: int,
    action: str,
    leg: str,
    order_book_id,
    direction: str,
    offset: str,
    qty,
    price,
) -> InfiniOrder:
    instrument_id, exchange = split_instrument(order_book_id)
    return InfiniOrder(
        sequence=sequence,
        action=action,
        leg=leg,
        exchange=exchange,
        instrument_id=instrument_id,
        order_direction=direction,
        order_direction_type=_direction_type(direction),
        offset=offset,
        volume=_int_qty(qty),
        price=float(price or 0.0),
        memo=f"{action}:{leg}:{sequence}",
    )


def _etf_order(sequence: int, action: str, item: dict[str, Any]) -> InfiniOrder:
    qty = _etf_board_lot_qty(item.get("trade_etf_qty", 0.0))
    instrument_id, exchange = split_instrument(item.get("underlying_order_book_id"))
    direction = BUY if qty > 0 else SELL
    return InfiniOrder(
        sequence=sequence,
        action=action,
        leg="etf",
        exchange=exchange,
        instrument_id=instrument_id,
        order_direction=direction,
        order_direction_type=_direction_type(direction),
        offset=None,
        volume=_int_qty(abs(qty)),
        price=float(item.get("estimated_price", 0.0) or 0.0),
        memo=f"{action}:etf:{sequence}",
    )


def _etf_board_lot_qty(qty) -> float:
    return float(core_strategy.round_etf_hedge_target(float(qty or 0.0)))


def split_instrument(order_book_id) -> tuple[str, str]:
    text = str(order_book_id or "").strip()
    if not text:
        raise ValueError("missing order_book_id")
    if "." not in text:
        return text, "SSE"
    instrument_id, suffix = text.split(".", 1)
    suffix = suffix.upper()
    exchange_map = {
        "XSHG": "SSE",
        "SH": "SSE",
        "XSHE": "SZSE",
        "SZ": "SZSE",
    }
    return instrument_id, exchange_map.get(suffix, suffix)


def _direction_type(direction: str) -> str:
    if direction == BUY:
        return BUY_TYPE
    if direction == SELL:
        return SELL_TYPE
    raise ValueError(f"Unsupported order direction: {direction}")


def _int_qty(value) -> int:
    qty = int(round(float(value or 0.0)))
    return max(qty, 0)
