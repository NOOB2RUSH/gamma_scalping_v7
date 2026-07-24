"""统计相邻交易日 ATM IV 绝对值差的经验分布。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.dynamic_position.iv_daily_state_report import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    load_atm_iv,
)


DEFAULT_QUANTILES = (0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 0.995)
DEFAULT_FIXED_THRESHOLDS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the distribution of |ATM IV(t) - ATM IV(t-1)| using "
            "adjacent trading-day rows."
        )
    )
    parser.add_argument("--product", default="300etf")
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def build_daily_absolute_changes(atm_iv: pd.Series) -> pd.DataFrame:
    """Compare each ATM IV strictly with the preceding trading-day row."""
    current = pd.to_numeric(atm_iv, errors="coerce").sort_index()
    previous = current.shift(1)
    signed_change = current - previous
    result = pd.DataFrame(
        {
            "date": current.index,
            "atm_iv": current,
            "previous_trading_day_atm_iv": previous,
            "atm_iv_absolute_change_signed": signed_change,
            "atm_iv_absolute_change": signed_change.abs(),
        }
    )
    result["atm_iv_absolute_change_percentage_points"] = (
        result["atm_iv_absolute_change"] * 100.0
    )
    result["next_trading_day_atm_iv_change_signed"] = signed_change.shift(-1)
    result.index.name = None
    return result.reset_index(drop=True).assign(
        date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")
    )


def _valid_changes(samples: pd.DataFrame) -> pd.DataFrame:
    return samples.dropna(
        subset=["atm_iv_absolute_change_signed", "atm_iv_absolute_change"]
    ).copy()


def build_descriptive_summary(samples: pd.DataFrame) -> pd.DataFrame:
    values = _valid_changes(samples)["atm_iv_absolute_change"]
    return pd.DataFrame(
        [
            {
                "sample_days": int(values.count()),
                "mean_absolute_change": values.mean(),
                "mean_absolute_change_percentage_points": values.mean() * 100.0,
                "std_absolute_change": values.std(),
                "std_absolute_change_percentage_points": values.std() * 100.0,
                "maximum_absolute_change": values.max(),
                "maximum_absolute_change_percentage_points": values.max() * 100.0,
            }
        ]
    )


def _threshold_statistics(valid: pd.DataFrame, threshold: float) -> dict:
    event = valid[valid["atm_iv_absolute_change"] >= threshold]
    upward = event[event["atm_iv_absolute_change_signed"] > 0]
    downward = event[event["atm_iv_absolute_change_signed"] < 0]
    next_day_available = upward[
        upward["next_trading_day_atm_iv_change_signed"].notna()
    ]
    next_day_pullback = next_day_available[
        next_day_available["next_trading_day_atm_iv_change_signed"] < 0
    ]
    return {
        "ivspike_threshold": threshold,
        "ivspike_threshold_percentage_points": threshold * 100.0,
        "event_days": len(event),
        "event_share": len(event) / len(valid) if len(valid) else np.nan,
        "upward_event_days": len(upward),
        "downward_event_days": len(downward),
        "upward_events_with_next_day": len(next_day_available),
        "upward_events_followed_by_next_day_pullback": len(next_day_pullback),
        "upward_next_day_pullback_share": (
            len(next_day_pullback) / len(next_day_available)
            if len(next_day_available)
            else np.nan
        ),
    }


def build_quantile_summary(
    samples: pd.DataFrame,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    valid = _valid_changes(samples)
    values = valid["atm_iv_absolute_change"]
    rows = []
    for quantile in quantiles:
        threshold = float(values.quantile(quantile))
        rows.append(
            {
                "quantile": quantile,
                **_threshold_statistics(valid, threshold),
            }
        )
    return pd.DataFrame(rows)


def build_fixed_threshold_summary(
    samples: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_FIXED_THRESHOLDS,
) -> pd.DataFrame:
    valid = _valid_changes(samples)
    return pd.DataFrame(
        [_threshold_statistics(valid, float(threshold)) for threshold in thresholds]
    )


def plot_distribution(
    samples: pd.DataFrame,
    quantiles: pd.DataFrame,
    output_path: Path,
    product: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    values = _valid_changes(samples)["atm_iv_absolute_change"].to_numpy(dtype=float)
    fig, (ax_hist, ax_tail) = plt.subplots(1, 2, figsize=(14, 6))

    ax_hist.hist(values * 100.0, bins=60, color="tab:blue", alpha=0.72)
    for quantile, color in [(0.95, "tab:orange"), (0.975, "tab:red"), (0.99, "tab:purple")]:
        row = quantiles[np.isclose(quantiles["quantile"], quantile)]
        if row.empty:
            continue
        threshold_pp = float(row.iloc[0]["ivspike_threshold_percentage_points"])
        ax_hist.axvline(
            threshold_pp,
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"q{quantile:.1%} = {threshold_pp:.2f} pp",
        )
    ax_hist.set_title("Absolute ATM IV Change Distribution")
    ax_hist.set_xlabel("|ATM IV(t) - ATM IV(t-1)| (percentage points)")
    ax_hist.set_ylabel("Trading days")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.2)

    sorted_values = np.sort(values)
    exceedance = (len(sorted_values) - np.arange(len(sorted_values))) / len(sorted_values)
    ax_tail.plot(sorted_values * 100.0, exceedance, color="tab:red", linewidth=2.0)
    ax_tail.set_yscale("log")
    ax_tail.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_tail.set_title("Empirical Exceedance Probability")
    ax_tail.set_xlabel("IVspike threshold (percentage points)")
    ax_tail.set_ylabel("P(|daily ATM IV change| >= threshold)")
    ax_tail.grid(True, alpha=0.25, which="both")

    fig.suptitle(f"{product.upper()}: Adjacent-Day Absolute ATM IV Change")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    atm_iv = load_atm_iv(args.product, args.start, args.end)
    samples = build_daily_absolute_changes(atm_iv)
    descriptive = build_descriptive_summary(samples)
    quantiles = build_quantile_summary(samples)
    fixed_thresholds = build_fixed_threshold_summary(samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{args.product}_atm_iv_absolute_change_distribution"
    samples_path = args.output_dir / f"{stem}_samples.csv"
    descriptive_path = args.output_dir / f"{stem}_descriptive.csv"
    quantiles_path = args.output_dir / f"{stem}_quantiles.csv"
    thresholds_path = args.output_dir / f"{stem}_fixed_thresholds.csv"
    chart_path = args.output_dir / f"{stem}.png"
    samples.to_csv(samples_path, index=False, encoding="utf-8-sig")
    descriptive.to_csv(descriptive_path, index=False, encoding="utf-8-sig")
    quantiles.to_csv(quantiles_path, index=False, encoding="utf-8-sig")
    fixed_thresholds.to_csv(thresholds_path, index=False, encoding="utf-8-sig")
    plot_distribution(samples, quantiles, chart_path, args.product)

    print(descriptive.to_string(index=False))
    print(quantiles.to_string(index=False))
    print(f"samples={samples_path}")
    print(f"descriptive={descriptive_path}")
    print(f"quantiles={quantiles_path}")
    print(f"fixed_thresholds={thresholds_path}")
    print(f"chart={chart_path}")


if __name__ == "__main__":
    main()
