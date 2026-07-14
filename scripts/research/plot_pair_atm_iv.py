"""Plot paired ATM-IV levels and log changes from a research panel CSV."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "output" / "research" / "20260706_104713_atm_iv_log_changes.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "research"
DEFAULT_CORR_INPUT = PROJECT_ROOT / "output" / "research" / "20260706_104713_300etf_vs_50etf_atm_iv_log_change_corr_60d.csv"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--product-a", default="300etf")
    parser.add_argument("--product-b", default="50etf")
    parser.add_argument("--corr-input", type=Path, default=DEFAULT_CORR_INPUT)
    parser.add_argument("--shift-date", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _paired_data(frame: pd.DataFrame, product_a: str, product_b: str) -> pd.DataFrame:
    columns = [
        "date",
        f"atm_iv_{product_a}", f"atm_iv_{product_b}",
        f"atm_iv_log_change_{product_a}", f"atm_iv_log_change_{product_b}",
    ]
    result = frame.loc[:, columns].copy()
    result["date"] = pd.to_datetime(result["date"])
    return result.dropna(subset=[f"atm_iv_{product_a}", f"atm_iv_{product_b}"]).sort_values("date")


def _add_shift_markers(axis, dates):
    for date in dates:
        axis.axvline(date, color="tab:purple", linestyle="--", linewidth=1.0, alpha=0.75)


def _break_on_data_gaps(data: pd.DataFrame, column: str, max_gap_days: int = 7) -> pd.Series:
    """Return a plotting series that does not interpolate missing-date gaps."""
    values = pd.to_numeric(data[column], errors="coerce").copy()
    values.loc[data["date"].diff().dt.days.gt(max_gap_days)] = float("nan")
    return values


def main():
    args = parse_args()
    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    data = _paired_data(frame, args.product_a, args.product_b)
    shifts = [pd.Timestamp(value) for value in args.shift_date]
    label_a, label_b = args.product_a.upper(), args.product_b.upper()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{args.product_a}_vs_{args.product_b}"

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(data["date"], _break_on_data_gaps(data, f"atm_iv_{args.product_a}"), label=f"{label_a} ATM IV", linewidth=1.25)
    axis.plot(data["date"], _break_on_data_gaps(data, f"atm_iv_{args.product_b}"), label=f"{label_b} ATM IV", linewidth=1.25)
    _add_shift_markers(axis, shifts)
    axis.set(title=f"{label_a} vs {label_b}: ATM IV levels", xlabel="Date", ylabel="ATM IV")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    iv_path = args.output_dir / f"{stem}_atm_iv_levels.png"
    fig.savefig(iv_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(data["date"], _break_on_data_gaps(data, f"atm_iv_log_change_{args.product_a}"), label=f"{label_a} ln(IV_t / IV_t-1)", linewidth=0.8, alpha=0.85)
    axis.plot(data["date"], _break_on_data_gaps(data, f"atm_iv_log_change_{args.product_b}"), label=f"{label_b} ln(IV_t / IV_t-1)", linewidth=0.8, alpha=0.85)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    _add_shift_markers(axis, shifts)
    axis.set(title=f"{label_a} vs {label_b}: ATM IV log changes", xlabel="Date", ylabel="Log change")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    change_path = args.output_dir / f"{stem}_atm_iv_log_changes.png"
    fig.savefig(change_path, dpi=180)
    plt.close(fig)

    ratio = (
        pd.to_numeric(data[f"atm_iv_{args.product_a}"], errors="coerce")
        / pd.to_numeric(data[f"atm_iv_{args.product_b}"], errors="coerce")
    )
    ratio.loc[~pd.to_numeric(data[f"atm_iv_{args.product_b}"], errors="coerce").gt(0)] = float("nan")
    gap_starts = data["date"].diff().dt.days.gt(7)
    ratio.loc[gap_starts] = float("nan")
    ratio_ma = ratio.rolling(20, min_periods=10).mean()
    # ``rolling`` skips NaNs, so explicitly clear the first post-gap value too.
    ratio_ma.loc[gap_starts] = float("nan")
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(data["date"], ratio, label=f"{label_a} ATM IV / {label_b} ATM IV", linewidth=1.0, alpha=0.75)
    axis.plot(data["date"], ratio_ma, label="20-observation rolling mean", linewidth=1.7)
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    _add_shift_markers(axis, shifts)
    axis.set(title=f"{label_a} vs {label_b}: ATM IV ratio", xlabel="Date", ylabel="ATM IV ratio")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    ratio_path = args.output_dir / f"{stem}_atm_iv_ratio.png"
    fig.savefig(ratio_path, dpi=180)
    plt.close(fig)

    if args.corr_input.exists():
        corr_frame = pd.read_csv(args.corr_input, encoding="utf-8-sig")
        corr_col = next(column for column in corr_frame if column.startswith("atm_iv_log_change_corr_"))
        corr_dates = pd.to_datetime(corr_frame["date"])
        corr = pd.to_numeric(corr_frame[corr_col], errors="coerce")
        fig, axis = plt.subplots(figsize=(14, 6))
        axis.plot(corr_dates, corr, label="60D rolling Pearson correlation", color="tab:purple", linewidth=1.35)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        _add_shift_markers(axis, shifts)
        axis.set(title=f"{label_a} vs {label_b}: ATM IV log-change correlation", xlabel="Date", ylabel="Correlation", ylim=(-1.05, 1.05))
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        fig.tight_layout()
        corr_path = args.output_dir / f"{stem}_atm_iv_log_change_corr_60d.png"
        fig.savefig(corr_path, dpi=180)
        plt.close(fig)
    else:
        corr_path = None

    print(f"rows={len(data)} first={data['date'].min():%Y-%m-%d} last={data['date'].max():%Y-%m-%d}")
    print(f"atm_iv_chart={iv_path}")
    print(f"log_change_chart={change_path}")
    print(f"ratio_chart={ratio_path}")
    if corr_path is not None:
        print(f"correlation_chart={corr_path}")


if __name__ == "__main__":
    main()
