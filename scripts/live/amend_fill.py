from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import account, storage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Void a confirmed fill, optionally insert a replacement, and rebuild account state."
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--fill-id", type=int, required=True)
    parser.add_argument("--replacement-fill", help="Path to corrected fill JSON.")
    parser.add_argument("--reason", default="amended_by_user")
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
    replacement = None
    if args.replacement_fill:
        replacement = storage.read_json(args.replacement_fill)

    result = account.amend_fill(
        args.product,
        args.fill_id,
        replacement_fill=replacement,
        reason=args.reason,
        account_id=args.account_id,
        initial_cash=args.initial_cash,
    )
    state = result["account"]
    print(f"voided_fill_id={result['voided_fill_id']}")
    if result["replacement_fill_id"] is not None:
        print(f"replacement_fill_id={result['replacement_fill_id']}")
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
