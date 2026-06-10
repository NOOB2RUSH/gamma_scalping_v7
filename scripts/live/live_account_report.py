from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import core
from core.live import account_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report live account status with real-time mark-to-market positions and trade records."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--source",
        choices=["akshare", "local", "none"],
        default="akshare",
        help="akshare fetches latest quotes; local snapshots existing data; none uses existing canonical data.",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument(
        "--all-trades",
        action="store_true",
        help="Deprecated; the trade sheet always uses the report-date broker trade export.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write report files.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Do not update cumulative account/position history CSV files.",
    )
    parser.add_argument(
        "--format",
        choices=["excel", "csv", "both"],
        default="excel",
        help="Report file format. CSV writes one file per section.",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "diagnose"],
        default="default",
        help="default shows operator essentials; diagnose includes internal reconciliation fields.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = account_report.build_live_account_report(
        args.product,
        account_id=args.account_id,
        source=args.source,
        date=args.date,
        all_trades=args.all_trades,
        persist_history=not args.no_history,
    )
    if args.json:
        print(
            json.dumps(
                account_report._json_payload(payload, mode=args.mode),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return

    if not args.no_write:
        paths = account_report.write_live_account_report(
            args.product,
            payload,
            output_format=args.format,
            mode=args.mode,
        )
        if "excel" in paths:
            print(f"account_report_excel={paths['excel']}")
        if "total_excel" in paths:
            print(f"account_report_total_excel={paths['total_excel']}")
        if "csv" in paths:
            for sheet_name, path in paths["csv"].items():
                print(f"account_report_csv[{sheet_name}]={path}")
        print(f"account_report_json={paths['json']}")
        if "diagnostics" in paths:
            print(f"account_report_diagnostics={paths['diagnostics']}")

    for line in account_report.format_terminal_summary(payload, mode=args.mode):
        print(line)


if __name__ == "__main__":
    main()
