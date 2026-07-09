from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import _bootstrap  # noqa: F401
import core
from core.live import account_report
import capture_intraday_quotes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Report live account status with real-time mark-to-market positions and trade records."
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--source",
        choices=["snapshot", "akshare", "local", "none"],
        default="snapshot",
        help="snapshot uses the latest saved AKShare quote without network I/O.",
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
        "--mode",
        choices=["default", "diagnose"],
        default="default",
        help="default shows operator essentials; diagnose includes internal reconciliation fields.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    intraday_capture = _capture_intraday_for_report(
        args.product,
        args.account_id,
    )
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
            mode=args.mode,
        )
        if "total_excel" in paths:
            print(f"account_report_total_excel={paths['total_excel']}")
        print(f"account_report_json={paths['json']}")

    if not args.json:
        for line in account_report.format_intraday_data_usage(
            payload,
            capture_result=intraday_capture,
        ):
            print(line)

    for line in account_report.format_terminal_summary(payload, mode=args.mode):
        print(line)


def _capture_intraday_for_report(product, account_id):
    capture_args = SimpleNamespace(
        product=product,
        account_id=account_id,
        output_dir=None,
        option_code=[],
        no_account_positions=False,
        save_option_greeks_snapshot=False,
    )
    try:
        return capture_intraday_quotes.capture_once(capture_args)
    except Exception as exc:
        return {
            "captured_at": None,
            "etf_rows": 0,
            "option_minute_rows": {},
            "errors": [f"{type(exc).__name__}:{exc}"],
        }


if __name__ == "__main__":
    main()
