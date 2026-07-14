"""Plot smoothed conditional daily-IV-state probabilities against absolute ATM IV."""

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
    STATE_ORDER,
    classify_iv_daily_state,
    load_atm_iv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot kernel-smoothed P(IV daily state | absolute ATM IV)."
    )
    parser.add_argument("--product", default="kc50etf", help="Option product configuration.")
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument("--up-threshold", type=float, default=0.04)
    parser.add_argument("--down-threshold", type=float, default=0.04)
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=0.03,
        help="Gaussian-kernel bandwidth in absolute IV units; defaults to 0.03 (3%%).",
    )
    parser.add_argument("--grid-points", type=int, default=100)
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=30,
        help="Number of absolute-IV bins used by the sample-count bars; defaults to 30.",
    )
    parser.add_argument(
        "--range-lower-percentile",
        type=float,
        default=0.01,
        help="Lower IV percentile used for the plotted range; defaults to 0.01.",
    )
    parser.add_argument(
        "--range-upper-percentile",
        type=float,
        default=0.99,
        help="Upper IV percentile used for the plotted range; defaults to 0.99.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def kernel_smoothed_state_probabilities(
    classified: pd.DataFrame,
    bandwidth: float,
    grid_points: int,
    lower_percentile: float,
    upper_percentile: float,
) -> pd.DataFrame:
    """Estimate P(state | ATM IV=x) on a grid with Gaussian-kernel weights."""
    if bandwidth <= 0:
        raise ValueError("--bandwidth must be positive.")
    if grid_points < 2:
        raise ValueError("--grid-points must be at least 2.")
    if not 0 <= lower_percentile < upper_percentile <= 1:
        raise ValueError("Plot-range percentiles must satisfy 0 <= lower < upper <= 1.")

    valid = classified[classified["iv_state"].isin(STATE_ORDER)].copy()
    valid["predictor_atm_iv"] = pd.to_numeric(
        valid["predictor_atm_iv"], errors="coerce"
    )
    valid = valid.dropna(subset=["predictor_atm_iv"])
    if valid.empty:
        raise ValueError("No classified days with valid ATM IV are available.")

    iv = valid["predictor_atm_iv"].to_numpy(dtype=float)
    grid = np.linspace(
        np.quantile(iv, lower_percentile),
        np.quantile(iv, upper_percentile),
        grid_points,
    )
    distances = (iv[:, None] - grid[None, :]) / bandwidth
    weights = np.exp(-0.5 * distances**2)
    total_weight = weights.sum(axis=0)

    result = pd.DataFrame(
        {
            "previous_trading_day_atm_iv": grid,
            "effective_sample_size": total_weight**2 / (weights**2).sum(axis=0),
        }
    )
    for state in STATE_ORDER:
        indicator = (valid["iv_state"].to_numpy() == state).astype(float)
        result[f"probability_{state}"] = indicator @ weights / total_weight
    return result


def plot_probability_curves(
    curves: pd.DataFrame,
    classified: pd.DataFrame,
    output_path: Path,
    product: str,
    bandwidth: float,
    histogram_bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    colors = {"iv_down": "tab:blue", "flat": "tab:gray", "iv_up": "tab:red"}
    labels = {"iv_down": "IV down", "flat": "Flat", "iv_up": "IV up"}
    fig, ax = plt.subplots(figsize=(12, 7))
    valid_iv = pd.to_numeric(
        classified.loc[
            classified["iv_state"].isin(STATE_ORDER), "predictor_atm_iv"
        ],
        errors="coerce",
    ).dropna()
    x_min = curves["previous_trading_day_atm_iv"].min()
    x_max = curves["previous_trading_day_atm_iv"].max()
    histogram_iv = valid_iv[valid_iv.between(x_min, x_max)]
    ax_count = ax.twinx()
    counts, bin_edges = np.histogram(histogram_iv, bins=histogram_bins, range=(x_min, x_max))
    ax_count.bar(
        bin_edges[:-1],
        counts,
        width=np.diff(bin_edges),
        align="edge",
        color="tab:green",
        alpha=0.18,
        edgecolor="none",
        label="Sample days",
        zorder=0,
    )
    ax_count.set_ylabel("Sample days per IV bin", color="tab:green")
    ax_count.tick_params(axis="y", labelcolor="tab:green")
    ax_count.set_ylim(bottom=0)
    for state in STATE_ORDER:
        ax.plot(
            curves["previous_trading_day_atm_iv"],
            curves[f"probability_{state}"],
            label=labels[state],
            color=colors[state],
            linewidth=2.0,
            zorder=3,
        )
    ax.set_title(
        f"{product.upper()}: Next-Day IV State Probability by Previous Close ATM IV\n"
        f"Gaussian-kernel bandwidth = {bandwidth:.1%}"
    )
    ax.set_xlabel("Previous trading-day close ATM IV")
    ax.set_ylabel("Conditional probability")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    lines, line_labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax_count.get_legend_handles_labels()
    ax.legend(lines + bars, line_labels + bar_labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    atm_iv = load_atm_iv(args.product, args.start, args.end)
    classified = classify_iv_daily_state(atm_iv, args.up_threshold, args.down_threshold)
    curves = kernel_smoothed_state_probabilities(
        classified,
        bandwidth=args.bandwidth,
        grid_points=args.grid_points,
        lower_percentile=args.range_lower_percentile,
        upper_percentile=args.range_upper_percentile,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{args.product}_previous_iv_vs_daily_state_curve"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    curves.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive.")
    plot_probability_curves(
        curves,
        classified,
        png_path,
        args.product,
        args.bandwidth,
        args.histogram_bins,
    )
    print(f"curve_data={csv_path}")
    print(f"chart={png_path}")


if __name__ == "__main__":
    main()
