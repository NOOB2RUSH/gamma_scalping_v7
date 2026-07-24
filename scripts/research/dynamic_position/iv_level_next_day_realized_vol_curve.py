"""Study next-trading-day realized volatility conditional on today's ATM IV."""

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

import core  # noqa: E402
from scripts.research.dynamic_position.iv_daily_state_report import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    load_atm_iv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot next-trading-day realized volatility conditional on today's close ATM IV."
        )
    )
    parser.add_argument(
        "--product",
        choices=core.config.available_products(),
        default=core.config.CONFIG.data.product,
        help="Option product configuration.",
    )
    parser.add_argument("--start", default=None, help="Inclusive predictor-date start.")
    parser.add_argument("--end", default=None, help="Inclusive predictor-date end.")
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
        help="Number of ATM-IV bins used by the sample-count bars; defaults to 30.",
    )
    parser.add_argument(
        "--range-lower-percentile",
        type=float,
        default=0.01,
        help="Lower ATM-IV percentile used for the plotted range; defaults to 0.01.",
    )
    parser.add_argument(
        "--range-upper-percentile",
        type=float,
        default=0.99,
        help="Upper ATM-IV percentile used for the plotted range; defaults to 0.99.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_etf_close_through_next_trading_day(atm_iv: pd.Series) -> pd.Series:
    """Load close prices through the first available trading day after the IV sample."""
    valid_iv = pd.to_numeric(atm_iv, errors="coerce").dropna().sort_index()
    if valid_iv.empty:
        raise ValueError("ATM IV series has no valid values.")

    calendar = core.data_loader.load_etf_trading_calendar()
    last_predictor_date = valid_iv.index.max()
    later_dates = calendar[calendar > last_predictor_date]
    price_end = later_dates[0] if len(later_dates) else last_predictor_date
    etf_by_date = core.data_loader.load_etf_series(valid_iv.index.min(), price_end)
    return core.vol_engine.build_daily_ohlc_df(etf_by_date)["close"]


def build_next_day_realized_volatility(
    atm_iv: pd.Series,
    close: pd.Series,
    annual_days: int = 252,
) -> pd.DataFrame:
    """Pair annualized IV_t with annualized |log(close_{t+1} / close_t)|."""
    if annual_days <= 0:
        raise ValueError("annual_days must be positive.")
    predictor_iv = pd.to_numeric(atm_iv, errors="coerce").sort_index()
    close = pd.to_numeric(close, errors="coerce").sort_index()
    next_close = close.shift(-1)
    next_trade_date = pd.Series(close.index, index=close.index).shift(-1)

    result = pd.DataFrame({"today_atm_iv": predictor_iv})
    result["today_close"] = close.reindex(result.index)
    result["next_trading_day"] = next_trade_date.reindex(result.index)
    result["next_close"] = next_close.reindex(result.index)
    valid_close = result["today_close"].gt(0) & result["next_close"].gt(0)
    result["next_day_log_return"] = np.nan
    result.loc[valid_close, "next_day_log_return"] = np.log(
        result.loc[valid_close, "next_close"]
        / result.loc[valid_close, "today_close"]
    )
    result["next_day_realized_volatility"] = (
        result["next_day_log_return"].abs() * np.sqrt(annual_days)
    )
    result["today_atm_iv_minus_next_day_realized_volatility"] = (
        result["today_atm_iv"] - result["next_day_realized_volatility"]
    )
    result["next_day_realized_to_atm_iv_ratio"] = np.nan
    valid_ratio = valid_close & result["today_atm_iv"].gt(0)
    result.loc[valid_ratio, "next_day_realized_to_atm_iv_ratio"] = (
        result.loc[valid_ratio, "next_day_realized_volatility"]
        / result.loc[valid_ratio, "today_atm_iv"]
    )
    result.index.name = "date"
    return result.dropna(
        subset=[
            "today_atm_iv",
            "next_trading_day",
            "next_day_realized_volatility",
            "today_atm_iv_minus_next_day_realized_volatility",
            "next_day_realized_to_atm_iv_ratio",
        ]
    )


def kernel_smoothed_realized_volatility(
    samples: pd.DataFrame,
    bandwidth: float,
    grid_points: int,
    lower_percentile: float,
    upper_percentile: float,
) -> pd.DataFrame:
    """Estimate E[annualized next-day realized volatility | ATM IV=x]."""
    if bandwidth <= 0:
        raise ValueError("--bandwidth must be positive.")
    if grid_points < 2:
        raise ValueError("--grid-points must be at least 2.")
    if not 0 <= lower_percentile < upper_percentile <= 1:
        raise ValueError("Plot-range percentiles must satisfy 0 <= lower < upper <= 1.")

    valid = samples.dropna(
        subset=[
            "today_atm_iv",
            "next_day_realized_volatility",
            "today_atm_iv_minus_next_day_realized_volatility",
            "next_day_realized_to_atm_iv_ratio",
        ]
    ).copy()
    if valid.empty:
        raise ValueError("No valid IV/next-day-realized-volatility pairs are available.")

    iv = valid["today_atm_iv"].to_numpy(dtype=float)
    realized_vol = valid["next_day_realized_volatility"].to_numpy(dtype=float)
    iv_minus_realized_vol = valid[
        "today_atm_iv_minus_next_day_realized_volatility"
    ].to_numpy(dtype=float)
    realized_to_iv_ratio = valid["next_day_realized_to_atm_iv_ratio"].to_numpy(
        dtype=float
    )
    grid = np.linspace(
        np.quantile(iv, lower_percentile),
        np.quantile(iv, upper_percentile),
        grid_points,
    )
    distances = (iv[:, None] - grid[None, :]) / bandwidth
    weights = np.exp(-0.5 * distances**2)
    total_weight = weights.sum(axis=0)
    return pd.DataFrame(
        {
            "today_atm_iv": grid,
            "expected_next_day_realized_volatility": realized_vol @ weights
            / total_weight,
            "expected_today_atm_iv_minus_next_day_realized_volatility": (
                iv_minus_realized_vol @ weights / total_weight
            ),
            "expected_next_day_realized_to_atm_iv_ratio": realized_to_iv_ratio
            @ weights
            / total_weight,
            "effective_sample_size": total_weight**2 / (weights**2).sum(axis=0),
        }
    )


def plot_realized_volatility_curve(
    curves: pd.DataFrame,
    samples: pd.DataFrame,
    output_path: Path,
    product: str,
    bandwidth: float,
    histogram_bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    x_min = curves["today_atm_iv"].min()
    x_max = curves["today_atm_iv"].max()
    histogram_iv = samples.loc[
        samples["today_atm_iv"].between(x_min, x_max), "today_atm_iv"
    ]
    counts, bin_edges = np.histogram(
        histogram_iv,
        bins=histogram_bins,
        range=(x_min, x_max),
    )
    unconditional_mean = samples["next_day_realized_volatility"].mean()
    unconditional_ratio = samples["next_day_realized_to_atm_iv_ratio"].mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax_count = ax.twinx()
    ax_ratio = ax.twinx()
    ax_ratio.spines["right"].set_position(("axes", 1.12))
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

    ax_ratio.plot(
        curves["today_atm_iv"],
        curves["expected_next_day_realized_to_atm_iv_ratio"],
        color="tab:orange",
        linestyle="--",
        linewidth=2.0,
        label="E[annualized realized / ATM IV | ATM IV]",
        zorder=3,
    )
    ax_ratio.axhline(
        unconditional_ratio,
        color="tab:orange",
        linestyle=":",
        linewidth=1.2,
        label="Unconditional annualized realized / ATM IV",
        zorder=2,
    )
    ax_ratio.set_ylabel(
        "Annualized next-day realized / today's ATM IV", color="tab:orange"
    )
    ax_ratio.tick_params(axis="y", labelcolor="tab:orange")
    ax_ratio.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.plot(
        curves["today_atm_iv"],
        curves["expected_next_day_realized_volatility"],
        color="tab:purple",
        linewidth=2.2,
        label="E[annualized next-day realized vol | ATM IV]",
        zorder=3,
    )
    ax.axhline(
        unconditional_mean,
        color="tab:gray",
        linestyle="--",
        linewidth=1.3,
        label="Unconditional mean",
        zorder=2,
    )
    ax.set_title(
        f"{product.upper()}: Next-Day Realized Volatility by Today's Close ATM IV\n"
        f"Realized volatility = sqrt({core.config.CONFIG.vol.annual_days}) "
        f"* |log(Close(t+1) / Close(t))|; "
        f"Gaussian-kernel bandwidth = {bandwidth:.1%}"
    )
    ax.set_xlabel("Today's close ATM IV")
    ax.set_ylabel("Annualized next-trading-day realized volatility")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)
    lines, line_labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax_count.get_legend_handles_labels()
    ratio_lines, ratio_labels = ax_ratio.get_legend_handles_labels()
    ax.legend(
        lines + ratio_lines + bars,
        line_labels + ratio_labels + bar_labels,
        loc="best",
    )
    fig.subplots_adjust(right=0.80)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_iv_minus_realized_volatility_curve(
    curves: pd.DataFrame,
    samples: pd.DataFrame,
    output_path: Path,
    product: str,
    bandwidth: float,
    histogram_bins: int,
) -> None:
    """Plot annualized RV and the absolute IV-minus-RV spread on separate axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    x_min = curves["today_atm_iv"].min()
    x_max = curves["today_atm_iv"].max()
    histogram_iv = samples.loc[
        samples["today_atm_iv"].between(x_min, x_max), "today_atm_iv"
    ]
    counts, bin_edges = np.histogram(
        histogram_iv,
        bins=histogram_bins,
        range=(x_min, x_max),
    )
    unconditional_rv = samples["next_day_realized_volatility"].mean()
    unconditional_gap = samples[
        "today_atm_iv_minus_next_day_realized_volatility"
    ].mean()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax_count = ax.twinx()
    ax_gap = ax.twinx()
    ax_gap.spines["right"].set_position(("axes", 1.12))
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

    ax.plot(
        curves["today_atm_iv"],
        curves["expected_next_day_realized_volatility"],
        color="tab:purple",
        linewidth=2.2,
        label="E[annualized next-day realized vol | ATM IV]",
        zorder=3,
    )
    ax.axhline(
        unconditional_rv,
        color="tab:gray",
        linestyle="--",
        linewidth=1.3,
        label="Unconditional annualized realized vol",
        zorder=2,
    )
    ax_gap.plot(
        curves["today_atm_iv"],
        curves["expected_today_atm_iv_minus_next_day_realized_volatility"],
        color="tab:orange",
        linestyle="--",
        linewidth=2.0,
        label="E[ATM IV − annualized realized vol | ATM IV]",
        zorder=3,
    )
    ax_gap.axhline(
        unconditional_gap,
        color="tab:orange",
        linestyle=":",
        linewidth=1.2,
        label="Unconditional ATM IV − annualized realized vol",
        zorder=2,
    )
    ax_gap.axhline(0, color="tab:orange", alpha=0.45, linewidth=0.8, zorder=1)
    ax_gap.set_ylabel("Today's ATM IV − annualized next-day realized vol", color="tab:orange")
    ax_gap.tick_params(axis="y", labelcolor="tab:orange")
    ax_gap.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.set_title(
        f"{product.upper()}: ATM IV Minus Next-Day Realized Volatility\n"
        f"Realized volatility = sqrt({core.config.CONFIG.vol.annual_days}) "
        f"* |log(Close(t+1) / Close(t))|; "
        f"Gaussian-kernel bandwidth = {bandwidth:.1%}"
    )
    ax.set_xlabel("Today's close ATM IV")
    ax.set_ylabel("Annualized next-trading-day realized volatility")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)
    lines, line_labels = ax.get_legend_handles_labels()
    bars, bar_labels = ax_count.get_legend_handles_labels()
    gap_lines, gap_labels = ax_gap.get_legend_handles_labels()
    ax.legend(
        lines + gap_lines + bars,
        line_labels + gap_labels + bar_labels,
        loc="best",
    )
    fig.subplots_adjust(right=0.80)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.histogram_bins <= 0:
        raise ValueError("--histogram-bins must be positive.")

    atm_iv = load_atm_iv(args.product, args.start, args.end)
    close = load_etf_close_through_next_trading_day(atm_iv)
    samples = build_next_day_realized_volatility(
        atm_iv,
        close,
        annual_days=core.config.CONFIG.vol.annual_days,
    )
    curves = kernel_smoothed_realized_volatility(
        samples,
        bandwidth=args.bandwidth,
        grid_points=args.grid_points,
        lower_percentile=args.range_lower_percentile,
        upper_percentile=args.range_upper_percentile,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{args.product}_today_iv_vs_next_day_realized_vol_curve"
    samples_path = args.output_dir / f"{stem}_samples.csv"
    curves_path = args.output_dir / f"{stem}.csv"
    chart_path = args.output_dir / f"{stem}.png"
    gap_chart_path = args.output_dir / f"{stem}_iv_minus_realized_vol.png"
    samples.reset_index().to_csv(samples_path, index=False, encoding="utf-8-sig")
    curves.to_csv(curves_path, index=False, encoding="utf-8-sig")
    plot_realized_volatility_curve(
        curves,
        samples,
        chart_path,
        args.product,
        args.bandwidth,
        args.histogram_bins,
    )
    plot_iv_minus_realized_volatility_curve(
        curves,
        samples,
        gap_chart_path,
        args.product,
        args.bandwidth,
        args.histogram_bins,
    )
    print(f"samples={samples_path}")
    print(f"curve_data={curves_path}")
    print(f"chart={chart_path}")
    print(f"iv_minus_realized_vol_chart={gap_chart_path}")


if __name__ == "__main__":
    main()
