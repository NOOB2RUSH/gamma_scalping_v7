from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import account
from core.live.runtime import load_product_config


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize a live shadow account.")
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--cash", type=float, default=None)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_product_config(args.product)
    initial_cash = args.cash if args.cash is not None else config.backtest.initial_cash
    state = account.initialize_account(
        args.product,
        initial_cash,
        account_id=args.account_id,
        reset=args.reset,
    )
    print(f"initialized {args.product}/{args.account_id}")
    print(f"cash={state.cash:.2f}")


if __name__ == "__main__":
    main()
