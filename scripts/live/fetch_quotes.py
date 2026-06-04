from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import market_data


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch/save one live quote snapshot.")
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--source", default="local", choices=["local", "akshare"])
    parser.add_argument("--date", default="latest")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = market_data.fetch_quote_snapshot(args.product, args.source, args.date)
    print("snapshot saved")
    for key, value in metadata.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
