from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import core
from core.live import hedge_importer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import broker ETF hedge holding/trade exports into the live account."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument(
        "--holding-file",
        default=None,
        help="Security holding CSV path. Defaults to newest live_hold/证券持仓查询*.csv.",
    )
    parser.add_argument(
        "--trade-file",
        default=None,
        help="Security trade CSV path. Defaults to newest live_hold/证券委托查询_实时成交*.csv.",
    )
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--date",
        default=None,
        help="Trade date for generated hedge fill. Defaults to date parsed from files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inferred hedge fill without writing account state.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = hedge_importer.import_hedge_files(
        args.product,
        holding_file=args.holding_file,
        trade_file=args.trade_file,
        account_id=args.account_id,
        date=args.date,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    print(f"holding_file={result['holding_file']}")
    print(f"trade_file={result['trade_file']}")
    print(f"trade_date={result['trade_date']}")
    print(
        f"holding_rows={result['holding_rows']} "
        f"trade_rows={result['trade_rows']} "
        f"matched_trade_rows={result['matched_trade_rows']}"
    )
    print(f"dry_run={result['dry_run']}")
    for item in result["applied"]:
        fill = item["fill"]
        prefix = "DRY_RUN" if item["dry_run"] else "CONFIRMED"
        print(
            f"{prefix} {fill['action']} "
            f"target_qty={fill['target_hedge_qty']:.0f} "
            f"trade_qty={fill['trade_etf_qty']:.0f} "
            f"entry_price={fill['entry_price']:.6f} "
            f"trade_price={fill['price']:.6f} "
            f"cash_delta={fill['cash_delta']:.2f} "
            f"margin={fill['margin']:.2f} "
            f"underlying={fill['underlying_order_book_id']}"
        )
    for item in result["skipped"]:
        fill = item["fill"]
        print(
            f"SKIPPED reason={item['reason']} "
            f"target_qty={fill['target_hedge_qty']:.0f} "
            f"entry_price={fill['entry_price']:.6f}"
        )
    for warning in result["warnings"]:
        print(f"WARNING {warning}")


if __name__ == "__main__":
    main()
