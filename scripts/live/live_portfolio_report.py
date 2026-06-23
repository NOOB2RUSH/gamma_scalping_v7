from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from core.live import portfolio_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one combined report from all live product subaccounts."
    )
    parser.add_argument(
        "--source",
        choices=["snapshot", "akshare", "local", "none"],
        default="snapshot",
    )
    parser.add_argument("--date", default=None)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = portfolio_report.build_portfolio_report(
        source=args.source,
        date=args.date,
        persist_history=not args.no_history,
    )
    if not args.no_write:
        paths = portfolio_report.write_portfolio_report(payload)
        print(f"portfolio_report_total_excel={paths['total_excel']}")
        print(f"portfolio_report_json={paths['json']}")
    for line in portfolio_report.format_terminal_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
