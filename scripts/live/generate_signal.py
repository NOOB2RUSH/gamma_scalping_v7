from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import core
from core.live import report, signal_engine, storage


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one simulated live signal.")
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--date", default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = signal_engine.generate_signal(args.product, args.account_id, args.date)
    report_path = report.write_signal_report(args.product, payload)
    json_path = report_path.with_suffix(".json")
    storage.write_json(json_path, payload)
    print(f"signal_report={report_path}")
    print(f"signal_json={json_path}")
    for line in report.format_signal_summary(payload):
        print(line)


if __name__ == "__main__":
    main()
