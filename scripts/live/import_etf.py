from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401
import core
from core.live import etf_importer


def main():
    parser = argparse.ArgumentParser(description="Import standard ETF holding and trade-detail exports.")
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--holding-file", default=None)
    parser.add_argument("--trade-file", default=None)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--date", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = etf_importer.import_etf_files(
        args.product,
        holding_file=args.holding_file,
        trade_file=args.trade_file,
        account_id=args.account_id,
        date=args.date,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
