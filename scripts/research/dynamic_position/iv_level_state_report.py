"""Study the relationship between absolute ATM-IV level and daily IV state."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.dynamic_position.iv_daily_state_report import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    STATE_ORDER,
    classify_iv_daily_state,
    load_atm_iv,
)


LEVEL_ORDER = ("low_iv", "mid_iv", "high_iv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-tab absolute ATM IV regimes against same-day ATM IV up/down/flat states."
        )
    )
    parser.add_argument("--product", default="kc50etf", help="Option product configuration.")
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument(
        "--low-iv-threshold",
        type=float,
        default=0.22,
        help="ATM IV at or below this value is low IV; defaults to 0.22 (22%%).",
    )
    parser.add_argument(
        "--high-iv-threshold",
        type=float,
        default=0.30,
        help="ATM IV at or above this value is high IV; defaults to 0.30 (30%%).",
    )
    parser.add_argument(
        "--up-threshold",
        type=float,
        default=0.04,
        help="Daily relative ATM IV rise threshold; defaults to 0.04 (+4%%).",
    )
    parser.add_argument(
        "--down-threshold",
        type=float,
        default=0.04,
        help="Daily relative ATM IV fall threshold as a positive number; defaults to 0.04 (-4%%).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def add_iv_level(classified: pd.DataFrame, low_threshold: float, high_threshold: float) -> pd.DataFrame:
    """Add level labels using yesterday's ATM IV to predict today's IV state."""
    if low_threshold <= 0 or high_threshold <= 0:
        raise ValueError("Absolute IV thresholds must be positive.")
    if low_threshold >= high_threshold:
        raise ValueError("--low-iv-threshold must be lower than --high-iv-threshold.")

    result = classified.copy()
    iv = pd.to_numeric(result["predictor_atm_iv"], errors="coerce")
    level = pd.Series("unclassified", index=result.index, dtype="object")
    level.loc[iv.notna()] = "mid_iv"
    level.loc[iv.le(low_threshold)] = "low_iv"
    level.loc[iv.ge(high_threshold)] = "high_iv"
    result["atm_iv_level"] = level
    return result


def build_relationship_summary(classified: pd.DataFrame) -> pd.DataFrame:
    """Return P(daily state | absolute IV level), together with total-sample deviation."""
    valid = classified[
        classified["atm_iv_level"].isin(LEVEL_ORDER)
        & classified["iv_state"].isin(STATE_ORDER)
    ].copy()
    total_state_rates = valid["iv_state"].value_counts(normalize=True)
    rows = []
    for level in LEVEL_ORDER:
        group = valid[valid["atm_iv_level"] == level]
        level_days = len(group)
        for state in STATE_ORDER:
            days = int((group["iv_state"] == state).sum())
            probability = days / level_days if level_days else float("nan")
            rows.append(
                {
                    "atm_iv_level": level,
                    "iv_state": state,
                    "days": days,
                    "level_days": level_days,
                    "state_probability_given_level": probability,
                    "all_level_share": level_days / len(valid) if len(valid) else float("nan"),
                    "all_sample_state_probability": total_state_rates.get(state, float("nan")),
                    "probability_minus_all_sample": probability - total_state_rates.get(state, float("nan")),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    atm_iv = load_atm_iv(args.product, args.start, args.end)
    classified = classify_iv_daily_state(atm_iv, args.up_threshold, args.down_threshold)
    classified = add_iv_level(classified, args.low_iv_threshold, args.high_iv_threshold)
    summary = build_relationship_summary(classified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{args.product}_previous_iv_vs_daily_state"
    daily_path = args.output_dir / f"{stem}.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    classified.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(
        f"product={args.product} daily_state_threshold=+/-{args.up_threshold:.2%} "
        f"iv_level_thresholds=low<={args.low_iv_threshold:.2%}, high>={args.high_iv_threshold:.2%}"
    )
    print(summary.to_string(index=False))
    print(f"daily_report={daily_path}")
    print(f"summary_report={summary_path}")


if __name__ == "__main__":
    main()
