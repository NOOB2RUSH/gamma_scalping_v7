from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import core
from core.live import holding_importer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import broker holding snapshot and auto-confirm supported live fills."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument(
        "--file",
        default=None,
        help="Holding CSV path. Defaults to the newest CSV under live_hold/.",
    )
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--date",
        default=None,
        help="Trade date for generated fills. Defaults to date parsed from filename.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Import total holdings instead of only rows with today's open quantity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inferred fills without writing account state.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = holding_importer.import_holding_file(
        args.product,
        file_path=args.file,
        account_id=args.account_id,
        date=args.date,
        include_existing=args.include_existing,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    print(f"file={result['file']}")
    print(f"trade_date={result['trade_date']}")
    print(f"input_rows={result['input_rows']} usable_rows={result['usable_rows']}")
    print(f"dry_run={result['dry_run']}")
    for item in result["applied"]:
        fill = item["fill"]
        prefix = "DRY_RUN" if item["dry_run"] else "CONFIRMED"
        if fill["action"] in {"option_mark_update", "option_hedge_mark_update"}:
            print(
                f"{prefix} {fill['action']} side={fill['side']} "
                f"qty={fill.get('call_qty')}/{fill.get('put_qty')} "
                f"call={fill.get('call_code')} put={fill.get('put_code')} "
                f"last_call_px={fill.get('last_call_price')} "
                f"last_put_px={fill.get('last_put_price')} "
                f"last_option_value={fill.get('last_option_value')} "
                f"margin={fill.get('option_margin')}"
            )
        elif fill["action"] in {"open_option_hedge", "close_option_hedge"}:
            print(
                f"{prefix} {fill['action']} side={fill['side']} "
                f"qty={fill.get('call_qty')}/{fill.get('put_qty')} "
                f"call={fill.get('call_code')} put={fill.get('put_code')} "
                f"price={fill.get('price', fill.get('entry_price'))} "
                f"cash_delta={fill['cash_delta']:.2f}"
            )
        else:
            print(
                f"{prefix} {fill['action']} side={fill['side']} "
                f"qty={fill['call_qty']}/{fill['put_qty']} "
                f"strike={fill['strike']} expiry={fill['expiry']} "
                f"call={fill['call_code']} put={fill['put_code']} "
                f"call_px={fill['entry_call_price']} put_px={fill['entry_put_price']} "
                f"cash_delta={fill['cash_delta']:.2f} margin={fill['option_margin']:.2f}"
            )
    for item in result["skipped"]:
        fill = item["fill"]
        print(
            f"SKIPPED side={item['side']} reason={item['reason']} "
            f"call={fill['call_code']} put={fill['put_code']}"
        )
    for warning in result["warnings"]:
        print(f"WARNING {warning}")


if __name__ == "__main__":
    main()
