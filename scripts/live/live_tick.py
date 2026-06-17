from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import market_data, report, signal_engine, storage


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch quotes and generate one signal.")
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--source", default="local", choices=["local", "akshare"])
    parser.add_argument("--date", default="latest")
    return parser.parse_args()


def main():
    args = parse_args()
    snapshot = market_data.fetch_quote_snapshot(args.product, args.source, args.date)
    payload = signal_engine.generate_signal(
        args.product,
        args.account_id,
        snapshot["quote_date"],
        quote_snapshot=snapshot,
    )
    payload["quote_snapshot"] = snapshot
    report_path = report.write_signal_report(args.product, payload)
    storage.write_json(report_path.with_suffix(".json"), payload)
    print(f"quote_date={snapshot['quote_date']}")
    print(f"signal_report={report_path}")
    print("read_only=True")
    for line in report.format_signal_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
