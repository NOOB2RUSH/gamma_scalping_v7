from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest_strategies import create_strategy  # noqa: E402
from core.live.replay import discover_signal_events, run_snapshot_replay  # noqa: E402
from core.live.replay_report import (  # noqa: E402
    build_live_comparison,
    write_live_comparison_report,
)
from core.live.runtime import load_product_config  # noqa: E402


PRODUCTS = ("50etf", "300etf", "500etf", "kc50etf")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay current live_straddle policy over immutable live snapshots."
    )
    parser.add_argument("--product", required=True, choices=PRODUCTS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--initial-account",
        default=None,
        help=(
            "Optional JSON containing an AccountState payload or a saved signal "
            "whose account field should seed the isolated replay."
        ),
    )
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-actual-comparison",
        action="store_true",
        help="Skip read-only comparison against broker-derived account history.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_product_config(args.product, start=args.start, end=args.end)
    plugin = create_strategy("live_straddle", config)
    initial_account = _load_initial_account(args.initial_account)
    daily, trades, plans = run_snapshot_replay(
        args.product,
        config=config,
        start=args.start,
        end=args.end,
        initial_account=initial_account,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("output/replay/live_straddle")
        / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        / args.product
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = output_dir / "snapshot_replay_daily.csv"
    trades_path = output_dir / "snapshot_replay_trades.csv"
    plans_path = output_dir / "snapshot_replay_plans.json"
    metadata_path = output_dir / "strategy_metadata.json"
    snapshot_manifest_path = output_dir / "snapshot_input_manifest.json"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    plans_path.write_text(
        json.dumps(plans, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    snapshot_manifest = _snapshot_manifest(
        args.product,
        start=args.start,
        end=args.end,
    )
    snapshot_manifest_path.write_text(
        json.dumps(snapshot_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        **plugin.metadata(),
        "replay_mode": "current_policy_immutable_signal_snapshots",
        "product": args.product,
        "start": str(daily["date"].min().date()),
        "end": str(daily["date"].max().date()),
        "account_state_source": (
            str(args.initial_account)
            if args.initial_account
            else "first_signal_embedded_account"
        ),
        "storage_isolation": {
            "reads_live_account_sqlite_for_replay": False,
            "writes_live_account_sqlite": False,
            "writes_live_feature_history": False,
            "actual_comparison_account_sqlite_access": (
                "disabled" if args.no_actual_comparison else "sqlite_mode_ro"
            ),
        },
        "policy_source_sha256": _source_fingerprint(),
        "snapshot_input_manifest_sha256": snapshot_manifest["manifest_sha256"],
        "snapshot_input_count": len(snapshot_manifest["inputs"]),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    result = {
        "daily": str(daily_path),
        "trades": str(trades_path),
        "plans": str(plans_path),
        "metadata": str(metadata_path),
        "snapshot_manifest": str(snapshot_manifest_path),
    }
    if not args.no_actual_comparison:
        comparison, matches, actual_fills = build_live_comparison(
            args.product,
            daily,
            trades,
            account_id=args.account_id,
        )
        result["comparison"] = write_live_comparison_report(
            args.product,
            comparison,
            matches,
            trades,
            actual_fills,
            output_dir=output_dir,
            metadata=metadata,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _load_initial_account(path):
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("account", payload)


def _source_fingerprint():
    digest = hashlib.sha256()
    for relative in (
        "core/strategy.py",
        "core/live/signal_engine.py",
        "core/live/replay.py",
        "core/live/replay_report.py",
        "core/live/etf_netting.py",
        "core/backtest_strategies/live_straddle.py",
    ):
        path = PROJECT_ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _snapshot_manifest(product, *, start=None, end=None):
    events = discover_signal_events(product, end=end)
    start_date = pd.Timestamp(start).normalize() if start is not None else None
    replay_events = [
        event
        for event in events
        if start_date is None or event.quote_date >= start_date
    ]
    if not replay_events:
        raise ValueError(f"No replay events available for manifest: {product}")
    previous = [event for event in events if event.quote_date < replay_events[0].quote_date]
    previous_close_event = previous[-1] if previous else None
    rows = []
    for event in events:
        if event.quote_date >= replay_events[0].quote_date:
            role = "replay_event"
        elif previous_close_event is not None and event == previous_close_event:
            role = "feature_and_previous_close_seed"
        else:
            role = "feature_seed"
        row = {
            "role": role,
            "quote_date": str(event.quote_date.date()),
            "signal_timestamp": event.signal_timestamp.isoformat(),
            "snapshot_stamp": event.snapshot_stamp,
            "signal_path": str(event.signal_path),
            "signal_sha256": _file_sha256(event.signal_path),
            "etf_snapshot": str(event.etf_snapshot),
            "option_snapshot": str(event.option_snapshot),
            "etf_snapshot_sha256": None,
            "option_snapshot_sha256": None,
        }
        if role != "feature_seed":
            row["etf_snapshot_sha256"] = _file_sha256(event.etf_snapshot)
            row["option_snapshot_sha256"] = _file_sha256(event.option_snapshot)
        rows.append(row)
    serialized = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema_version": 1,
        "product": product,
        "inputs": rows,
        "manifest_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
