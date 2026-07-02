from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import core
from core.live import (
    account_report,
    infinitrader,
    market_data,
    report,
    signal_engine,
    storage,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the local live pipeline and dispatch an InfiniTrader command."
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--source",
        choices=["snapshot", "local", "akshare"],
        default="akshare",
        help="Quote source for the fresh snapshot step.",
    )
    parser.add_argument("--date", default="latest")
    parser.add_argument(
        "--no-command",
        action="store_true",
        help="Generate signal and plan only; do not write pending_command.json.",
    )
    parser.add_argument(
        "--wait-executed",
        action="store_true",
        help="Wait until PythonGO archives the pending command as executed_<run_id>.json.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--sync-account",
        action="store_true",
        help="Apply generated fills to the local shadow account after dispatch/execution.",
    )
    parser.add_argument(
        "--sync-dry-run",
        action="store_true",
        help="Preview local account sync without writing fills.",
    )
    parser.add_argument(
        "--write-account-report",
        action="store_true",
        help="Build a local account report after syncing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload, signal_path = _generate_signal(args)
    orders = infinitrader.compile_signal_orders(payload)
    plan_path = _write_plan(payload, orders)
    command = None
    if not args.no_command:
        command = infinitrader.write_command(payload, orders)

    print(f"quote_date={payload['date']}")
    print(f"signal_report={signal_path}")
    print(f"signal_json={signal_path.with_suffix('.json')}")
    print(f"infinitrader_plan={plan_path}")
    print(f"orders={len(orders)}")
    if command is not None:
        print(f"command_path={command['command_path']}")
        print(f"pending_command={command['pending_path']}")

    executed_path = None
    if command is not None and args.wait_executed:
        executed_path = _wait_executed(args.product, command["run_id"], args.timeout_seconds)
        print(f"executed_command={executed_path}")

    if args.sync_account:
        if command is None:
            raise ValueError("--sync-account requires a command; remove --no-command.")
        sync_path = executed_path or command["command_path"]
        sync = infinitrader.sync_command_to_account(
            sync_path,
            account_id=args.account_id,
            dry_run=args.sync_dry_run,
        )
        sync_out = _write_sync_result(args.product, sync)
        print(f"sync_result={sync_out}")
        print(f"sync_applied={len(sync['applied'])} warnings={len(sync['warnings'])}")

    if args.write_account_report:
        report_payload = account_report.build_live_account_report(
            args.product,
            account_id=args.account_id,
            source="snapshot",
            date=payload["date"],
            persist_history=True,
        )
        paths = account_report.write_live_account_report(args.product, report_payload)
        print(f"account_report_total_excel={paths['total_excel']}")
        print(f"account_report_json={paths['json']}")


def _generate_signal(args):
    snapshot = market_data.fetch_quote_snapshot(args.product, args.source, args.date)
    payload = signal_engine.generate_signal(
        args.product,
        args.account_id,
        snapshot["quote_date"],
        quote_snapshot=snapshot,
    )
    payload["quote_snapshot"] = snapshot
    signal_path = report.write_signal_report(args.product, payload)
    storage.write_json(signal_path.with_suffix(".json"), payload)
    return payload, signal_path


def _write_plan(payload, orders):
    plan = {
        "product": payload.get("product"),
        "account_id": payload.get("account_id"),
        "date": payload.get("date"),
        "orders": orders,
    }
    path = storage.output_dir(payload["product"]) / f"{storage.local_now_stamp()}_infinitrader_plan.json"
    storage.write_json(path, plan)
    return path


def _wait_executed(product, run_id, timeout_seconds):
    runtime = infinitrader.runtime_dir(product)
    target = runtime / f"executed_{run_id}.json"
    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        if target.exists():
            return target
        time.sleep(1.0)
    raise TimeoutError(f"InfiniTrader command was not executed within {timeout_seconds}s: {target}")


def _write_sync_result(product, sync):
    path = infinitrader.runtime_dir(product) / f"sync_{storage.local_now_stamp()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sync, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    main()
