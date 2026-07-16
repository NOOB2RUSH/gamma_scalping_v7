"""按前一交易日 ATM IV 变动划分日度状态，并输出频率统计报告。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "research" / "dynamic_position"
STATE_ORDER = ("iv_down", "flat", "iv_up")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify each day using only its ATM IV and the prior trading day's ATM IV."
    )
    parser.add_argument(
        "--product",
        choices=core.config.available_products(),
        default=core.config.CONFIG.data.product,
        help="Option product configuration.",
    )
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument(
        "--up-threshold",
        type=float,
        default=0.04,
        help="Relative ATM IV rise threshold; defaults to 0.04 (+4%%).",
    )
    parser.add_argument(
        "--down-threshold",
        type=float,
        default=0.04,
        help="Relative ATM IV fall threshold as a positive number; defaults to 0.04 (-4%%).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV reports.",
    )
    return parser.parse_args()


def _set_product_config(product: str):
    config = core.config.load_config(product)
    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    return config


def load_atm_iv(product: str, start: str | None, end: str | None) -> pd.Series:
    """Load the daily ATM IV series through the project's regular feature pipeline."""
    config = _set_product_config(product)
    start_date = pd.Timestamp(start or config.backtest.start)
    end_date = pd.Timestamp(end or config.backtest.end)
    etf_by_date = core.data_loader.load_etf_series(start_date, end_date)
    hedge_by_date = core.data_loader.load_hedge_series(start_date, end_date)
    opt_by_date = core.data_loader.load_opt_series(start_date, end_date)
    opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
    calendar = core.data_loader.load_etf_trading_calendar()
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date, opt_by_date, calendar, start_date, end_date
    )
    features = core.cache.get_vol_features(
        etf_by_date, opt_by_date, calendar, enriched, start_date, end_date
    )
    result = pd.to_numeric(features["atm_iv"], errors="coerce").copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def classify_iv_daily_state(
    atm_iv: pd.Series, up_threshold: float, down_threshold: float
) -> pd.DataFrame:
    """Classify each row strictly against the immediately preceding trading-day row."""
    if up_threshold < 0 or down_threshold < 0:
        raise ValueError("Thresholds must be non-negative.")

    current_iv = pd.to_numeric(atm_iv, errors="coerce").sort_index()
    previous_iv = current_iv.shift(1)
    relative_change = current_iv.div(previous_iv).sub(1.0)
    valid_comparison = current_iv.notna() & previous_iv.notna() & previous_iv.gt(0)

    state = pd.Series("unclassified", index=current_iv.index, dtype="object")
    state.loc[valid_comparison] = "flat"
    state.loc[valid_comparison & relative_change.ge(up_threshold)] = "iv_up"
    state.loc[valid_comparison & relative_change.le(-down_threshold)] = "iv_down"

    return pd.DataFrame(
        {
            "date": current_iv.index.strftime("%Y-%m-%d"),
            "atm_iv": current_iv.to_numpy(),
            "previous_trading_day_atm_iv": previous_iv.to_numpy(),
            "predictor_atm_iv": previous_iv.to_numpy(),
            "atm_iv_relative_change": relative_change.to_numpy(),
            "iv_state": state.to_numpy(),
        }
    )


def build_state_summary(classified: pd.DataFrame) -> pd.DataFrame:
    classified_days = classified[classified["iv_state"].isin(STATE_ORDER)]
    total = len(classified_days)
    rows = []
    for state in STATE_ORDER:
        days = int((classified_days["iv_state"] == state).sum())
        rows.append(
            {
                "iv_state": state,
                "days": days,
                "share_of_classified_days": days / total if total else float("nan"),
            }
        )
    rows.append(
        {
            "iv_state": "unclassified",
            "days": int((classified["iv_state"] == "unclassified").sum()),
            "share_of_classified_days": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.up_threshold < 0 or args.down_threshold < 0:
        raise ValueError("--up-threshold and --down-threshold must be non-negative.")

    atm_iv = load_atm_iv(args.product, args.start, args.end)
    classified = classify_iv_daily_state(
        atm_iv, up_threshold=args.up_threshold, down_threshold=args.down_threshold
    )
    summary = build_state_summary(classified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{args.product}_iv_daily_state"
    daily_path = args.output_dir / f"{stem}.csv"
    summary_path = args.output_dir / f"{stem}_summary.csv"
    classified.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(
        f"product={args.product} up_threshold={args.up_threshold:.2%} "
        f"down_threshold={args.down_threshold:.2%}"
    )
    print(summary.to_string(index=False))
    print(f"daily_report={daily_path}")
    print(f"summary_report={summary_path}")


if __name__ == "__main__":
    main()
