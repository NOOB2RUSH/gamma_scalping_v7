from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import core
from core.live import intraday_pnl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate read-only intraday close-to-current PnL and Greeks PnL "
            "from the latest local quote snapshot."
        )
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--date",
        default=None,
        help="Quote snapshot date to use. Default: latest local snapshot.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = intraday_pnl.calculate_intraday_pnl(
        args.product,
        account_id=args.account_id,
        date=args.date,
    )
    if args.json:
        print(intraday_pnl.intraday_pnl_json(payload))
        return
    for line in intraday_pnl.format_intraday_pnl(payload):
        print(line)


if __name__ == "__main__":
    main()
