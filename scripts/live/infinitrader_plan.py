from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import core
from core.live import infinitrader, market_data, signal_engine, storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an InfiniTrader/PythonGO order plan from the live signal."
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--source",
        choices=["snapshot", "local", "akshare"],
        default="snapshot",
    )
    parser.add_argument("--date", default="latest")
    parser.add_argument(
        "--signal-json",
        default=None,
        help="Use an existing signal JSON instead of generating a fresh signal.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.signal_json:
        payload = storage.read_json(args.signal_json)
    else:
        snapshot = market_data.fetch_quote_snapshot(args.product, args.source, args.date)
        payload = signal_engine.generate_signal(
            args.product,
            args.account_id,
            snapshot["quote_date"],
            quote_snapshot=snapshot,
        )
        payload["quote_snapshot"] = snapshot

    orders = infinitrader.compile_signal_orders(payload)
    plan = {
        "product": payload.get("product", args.product),
        "account_id": payload.get("account_id", args.account_id),
        "date": payload.get("date"),
        "source_signal": args.signal_json,
        "orders": orders,
    }

    output = Path(args.output) if args.output else _default_output_path(plan["product"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"infinitrader_plan={output}")
    if not orders:
        print("orders=0")
        return
    print("seq | exchange | instrument | dir | offset | qty | price | action | leg")
    for order in orders:
        print(
            "{sequence} | {exchange} | {instrument_id} | {order_direction} | "
            "{offset} | {volume} | {price:.6f} | {action} | {leg}".format(
                **order
            )
        )


def _default_output_path(product):
    stamp = storage.local_now_stamp()
    return storage.output_dir(product) / f"{stamp}_infinitrader_plan.json"


if __name__ == "__main__":
    main()
