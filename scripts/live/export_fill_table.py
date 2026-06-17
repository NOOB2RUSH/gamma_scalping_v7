from __future__ import annotations

import argparse

import pandas as pd

import _bootstrap  # noqa: F401
import core
from core.live import account, storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export confirmed fills as one flat trade table."
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Hide voided fills.",
    )
    parser.add_argument("--out", help="CSV output path. Defaults to output/live/<product>.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = account.list_fill_table(
        args.product,
        account_id=args.account_id,
        include_voided=not args.active_only,
        limit=args.limit,
    )
    df = pd.DataFrame(rows)
    if args.out:
        path = args.out
    else:
        path = (
            storage.output_dir(args.product)
            / f"{storage.local_now_stamp()}_fill_table.csv"
        )
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"fill_table_csv={path}")
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
