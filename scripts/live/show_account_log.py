from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import core
from core.live import account


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show live account state, confirmed fills, and reconciliation history."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Hide voided fills.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Show fills as a unified flat trade table.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    state = account.load_account(args.product, account_id=args.account_id)
    fills = account.list_fills(
        args.product,
        account_id=args.account_id,
        include_voided=not args.active_only,
        limit=args.limit,
    )
    reconciliations = account.list_reconciliations(
        args.product,
        account_id=args.account_id,
        limit=args.limit,
    )

    payload = {
        "product": args.product,
        "account_id": args.account_id,
        "account": state.to_dict(),
        "fills": fills,
        "reconciliations": reconciliations,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return

    _print_account(state)
    if args.table:
        fill_table = account.list_fill_table(
            args.product,
            account_id=args.account_id,
            include_voided=not args.active_only,
            limit=args.limit,
        )
        _print_fill_table(fill_table)
    else:
        _print_fills(fills)
    _print_reconciliations(reconciliations)


def _print_account(state):
    print(f"account={state.product}/{state.account_id}")
    print(f"cash={state.cash:.2f}")
    print(
        "hedge="
        f"qty={state.hedge.qty} "
        f"entry_price={state.hedge.entry_price} "
        f"margin={state.hedge.margin} "
        f"underlying={state.hedge.underlying_order_book_id}"
    )
    for side, position in state.positions.items():
        if position is None:
            print(f"position.{side}=None")
            continue
        print(
            f"position.{side}={position['call_code']}/{position['put_code']} "
            f"qty={position['call_qty']}/{position['put_qty']} "
            f"strike={position['strike']} expiry={position['expiry']} "
            f"entry={position.get('entry_date')}"
        )
    print(f"strategy_state={state.strategy_state.to_dict()}")


def _print_fills(fills):
    print("")
    print("fills")
    if not fills:
        print("(none)")
        return
    for row in fills:
        payload = row["payload"]
        status = "VOID" if row["voided_at"] else "ACTIVE"
        details = _fill_details(payload)
        replacement = ""
        if row.get("replaces_fill_id") is not None:
            replacement = f" replaces={row['replaces_fill_id']}"
        print(
            f"#{row['id']} {status} {row['created_at']} "
            f"action={row['action']} date={payload.get('date')} "
            f"side={payload.get('side')} cash_delta={payload.get('cash_delta', 0)}"
            f"{replacement} {details}"
        )
        if row["voided_at"]:
            print(
                f"  voided_at={row['voided_at']} "
                f"void_reason={row['void_reason']}"
            )


def _fill_details(payload):
    action = payload.get("action", "")
    if "straddle" in action and "close" not in action:
        return (
            f"call={payload.get('call_code')} put={payload.get('put_code')} "
            f"qty={payload.get('call_qty', payload.get('qty'))}/"
            f"{payload.get('put_qty', payload.get('qty'))} "
            f"strike={payload.get('strike')} expiry={payload.get('expiry')} "
            f"call_px={payload.get('entry_call_price', payload.get('call_price'))} "
            f"put_px={payload.get('entry_put_price', payload.get('put_price'))}"
        )
    if "close" in action:
        return f"exit_reason={payload.get('exit_reason')}"
    if "hedge" in action:
        return (
            f"qty={payload.get('qty', payload.get('new_etf_qty'))} "
            f"price={payload.get('entry_price', payload.get('price'))} "
            f"underlying={payload.get('underlying_order_book_id')}"
        )
    return ""


def _print_reconciliations(reconciliations):
    print("")
    print("reconciliations")
    if not reconciliations:
        print("(none)")
        return
    for row in reconciliations:
        payload = row["payload"]
        print(
            f"#{row['id']} {row['created_at']} "
            f"ok={payload.get('ok')} diffs={len(payload.get('diffs', []))}"
        )


def _print_fill_table(rows):
    print("")
    print("fill_table")
    if not rows:
        print("(none)")
        return

    columns = [
        "id",
        "status",
        "trade_date",
        "action",
        "side",
        "cash_delta",
        "call_code",
        "put_code",
        "call_qty",
        "put_qty",
        "strike",
        "expiry",
        "call_price",
        "put_price",
        "exit_reason",
        "replaces_fill_id",
    ]
    widths = {
        column: min(
            22,
            max(
                len(column),
                *[len(_cell(row.get(column))) for row in rows],
            ),
        )
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            " | ".join(
                _cell(row.get(column))[: widths[column]].ljust(widths[column])
                for column in columns
            )
        )


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
