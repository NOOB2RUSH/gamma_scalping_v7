from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import reconciler, storage


def parse_args():
    parser = argparse.ArgumentParser(description="Reconcile shadow account with broker snapshot.")
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--snapshot", required=True, help="Path to broker snapshot JSON.")
    parser.add_argument("--account-id", default="default")
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot = storage.read_json(args.snapshot)
    payload = reconciler.reconcile(args.product, snapshot, args.account_id)
    report_path = reconciler.write_reconcile_report(args.product, payload)
    storage.write_json(report_path.with_suffix(".json"), payload)
    print(f"reconcile_report={report_path}")
    print(f"ok={payload['ok']}")
    if payload["diffs"]:
        for diff in payload["diffs"]:
            print(f"{diff['field']}: local={diff.get('local')} broker={diff.get('broker')}")


if __name__ == "__main__":
    main()
