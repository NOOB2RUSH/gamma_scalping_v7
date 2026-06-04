from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import account


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the live shadow account from non-voided fills."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=None,
        help="Initial cash used when rebuilding. Defaults to product config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state = account.rebuild_account(
        args.product,
        account_id=args.account_id,
        initial_cash=args.initial_cash,
    )
    print(f"rebuilt_account={args.product}/{args.account_id}")
    print(f"cash={state.cash:.2f}")
    for side, position in state.positions.items():
        if position is None:
            print(f"position.{side}=None")
        else:
            print(
                f"position.{side}={position['call_code']}/{position['put_code']} "
                f"qty={position['call_qty']}/{position['put_qty']} "
                f"strike={position['strike']} expiry={position['expiry']}"
            )


if __name__ == "__main__":
    main()
