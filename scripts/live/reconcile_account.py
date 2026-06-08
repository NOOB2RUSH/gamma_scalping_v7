from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import reconciler, storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Greeks PnL explainability against fee-adjusted NAV changes."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Deprecated and ignored; kept for compatibility with old commands.",
    )
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--abs-tolerance", type=float, default=reconciler.DEFAULT_ABS_TOLERANCE)
    parser.add_argument("--rel-tolerance", type=float, default=reconciler.DEFAULT_REL_TOLERANCE)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = reconciler.reconcile(
        args.product,
        account_id=args.account_id,
        start_date=args.from_date,
        end_date=args.to_date,
        abs_tolerance=args.abs_tolerance,
        rel_tolerance=args.rel_tolerance,
    )
    report_path = reconciler.write_reconcile_report(args.product, payload)
    storage.write_json(report_path.with_suffix(".json"), payload)
    print(f"reconcile_report={report_path}")
    for line in reconciler.format_terminal_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
