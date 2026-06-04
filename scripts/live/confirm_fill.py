from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import account, storage


def parse_args():
    parser = argparse.ArgumentParser(description="Apply a manually confirmed fill.")
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--fill", required=True, help="Path to fill JSON.")
    parser.add_argument("--account-id", default="default")
    return parser.parse_args()


def main():
    args = parse_args()
    fill = storage.read_json(args.fill)
    state = account.record_fill(args.product, fill, account_id=args.account_id)
    print(f"fill_applied={fill['action']}")
    print(f"cash={state.cash:.2f}")


if __name__ == "__main__":
    main()
