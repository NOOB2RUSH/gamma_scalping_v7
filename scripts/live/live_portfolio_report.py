from __future__ import annotations

import argparse
from types import SimpleNamespace

import _bootstrap  # noqa: F401
import core
from core.live import account_report, portfolio_report
import capture_intraday_quotes


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
    products = tuple(core.config.available_live_products())
    intraday_captures = {
        product: _capture_intraday_for_report(product, "default")
        for product in products
    }
    payload = portfolio_report.build_portfolio_report(
        products=products,
        source=args.source,
        date=args.date,
        persist_history=not args.no_history,
    )
    if not args.no_write:
        paths = portfolio_report.write_portfolio_report(payload)
        print(f"portfolio_report_total_excel={paths['total_excel']}")
        print(f"portfolio_report_json={paths['json']}")
    for product in payload.get("products", []):
        subaccount = (payload.get("subaccounts") or {}).get(product)
        if not subaccount:
            continue
        for line in account_report.format_intraday_data_usage(
            subaccount,
            capture_result=intraday_captures.get(product),
        ):
            print(line)
    for line in portfolio_report.format_terminal_summary(payload):
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
