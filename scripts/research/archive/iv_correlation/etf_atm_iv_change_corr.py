from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research.archive.iv_correlation.etf_atm_iv_ratios import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PAIRS,
    PRODUCT_LABELS,
    _features_for_pairs,
    _parse_pairs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute ATM IV log changes and rolling correlations for ETF option pairs."
        )
    )
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help=(
            "Pair as product_a/product_b, e.g. 300etf/50etf. "
            "Can be passed multiple times. Defaults to the research pair set."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV and chart outputs.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Rolling correlation window for ATM IV log changes.",
    )
    return parser.parse_args()


def add_atm_iv_log_change(features, product):
    result = features.copy()
    iv = pd.to_numeric(result["atm_iv"], errors="coerce")
    valid_iv = iv.where(iv.gt(0))
    result[f"atm_iv_log_change_{product}"] = np.log(valid_iv / valid_iv.shift(1))
    return result


def _features_with_log_changes(features_by_product):
    return {
        product: add_atm_iv_log_change(features, product)
        for product, features in features_by_product.items()
    }


def build_log_change_panel(features_by_product):
    product_order = [
        product for product in PRODUCT_LABELS.keys() if product in features_by_product
    ]
    product_order.extend(
        product
        for product in sorted(features_by_product.keys())
        if product not in product_order
    )
    panel = None
    for product in product_order:
        columns = ["atm_iv", f"atm_iv_log_change_{product}"]
        product_frame = features_by_product[product][columns].rename(
            columns={"atm_iv": f"atm_iv_{product}"}
        )
        panel = product_frame if panel is None else panel.join(product_frame, how="outer")
    panel = panel.sort_index()
    panel.insert(0, "date", panel.index.strftime("%Y-%m-%d"))
    return panel.reset_index(drop=True)


def build_corr_frame(product_a, product_b, features_by_product, rolling_window=60):
    feature_a = features_by_product[product_a]
    feature_b = features_by_product[product_b]
    joined = feature_b.join(
        feature_a,
        how="inner",
        lsuffix=f"_{product_b}",
        rsuffix=f"_{product_a}",
    )

    change_a_col = f"atm_iv_log_change_{product_a}"
    change_b_col = f"atm_iv_log_change_{product_b}"
    corr_col = f"atm_iv_log_change_corr_{product_a}_vs_{product_b}_{rolling_window}d"
    change_a = pd.to_numeric(joined[change_a_col], errors="coerce")
    change_b = pd.to_numeric(joined[change_b_col], errors="coerce")
    joined[corr_col] = change_a.rolling(
        rolling_window,
        min_periods=rolling_window,
    ).corr(change_b)
    joined.insert(0, "date", joined.index.strftime("%Y-%m-%d"))
    return joined.reset_index(drop=True)


def write_pair_outputs(frame, product_a, product_b, output_dir, rolling_window, timestamp):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_{product_a}_vs_{product_b}_atm_iv_log_change_corr_{rolling_window}d"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    label_a = PRODUCT_LABELS.get(product_a, product_a.upper())
    label_b = PRODUCT_LABELS.get(product_b, product_b.upper())
    dates = pd.to_datetime(frame["date"])
    change_a_col = f"atm_iv_log_change_{product_a}"
    change_b_col = f"atm_iv_log_change_{product_b}"
    corr_col = f"atm_iv_log_change_corr_{product_a}_vs_{product_b}_{rolling_window}d"
    change_a = pd.to_numeric(frame[change_a_col], errors="coerce")
    change_b = pd.to_numeric(frame[change_b_col], errors="coerce")
    corr = pd.to_numeric(frame[corr_col], errors="coerce")

    fig, (ax_change, ax_corr) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    fig.suptitle(
        f"{label_a} vs {label_b} ATM IV Log Change Correlation",
        fontsize=14,
    )
    ax_change.plot(dates, change_a, label=f"{label_a} ln(IV_t / IV_t-1)", linewidth=1.0)
    ax_change.plot(dates, change_b, label=f"{label_b} ln(IV_t / IV_t-1)", linewidth=1.0)
    ax_change.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_change.set_title("ATM IV log changes")
    ax_change.set_ylabel("Log change")
    ax_change.grid(True, alpha=0.25)
    ax_change.legend(loc="best")

    ax_corr.plot(
        dates,
        corr,
        label=f"{rolling_window}D rolling Pearson correlation",
        linewidth=1.4,
        color="tab:purple",
    )
    ax_corr.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_corr.axhline(1.0, color="tab:green", linewidth=0.8, linestyle=":", alpha=0.45)
    ax_corr.axhline(-1.0, color="tab:red", linewidth=0.8, linestyle=":", alpha=0.45)
    ax_corr.set_title("Rolling correlation")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_xlabel("Date")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.grid(True, alpha=0.25)
    ax_corr.legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path


def _pair_summary(frame, product_a, product_b, rolling_window):
    corr_col = f"atm_iv_log_change_corr_{product_a}_vs_{product_b}_{rolling_window}d"
    corr = pd.to_numeric(frame[corr_col], errors="coerce").dropna()
    if corr.empty:
        return {
            "pair": f"{product_a}/{product_b}",
            "rows": len(frame),
            "valid": 0,
        }
    valid_dates = frame.loc[corr.index, "date"]
    return {
        "pair": f"{product_a}/{product_b}",
        "rows": len(frame),
        "valid": len(corr),
        "first_valid": valid_dates.iloc[0],
        "last_valid": valid_dates.iloc[-1],
        "mean": corr.mean(),
        "median": corr.median(),
        "min": corr.min(),
        "max": corr.max(),
        "latest": corr.iloc[-1],
    }


def write_log_change_panel(panel, output_dir, rolling_window, timestamp):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{timestamp}_atm_iv_log_changes.csv"
    panel.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for column in panel.columns:
        if not column.startswith("atm_iv_log_change_"):
            continue
        product = column.removeprefix("atm_iv_log_change_")
        values = pd.to_numeric(panel[column], errors="coerce").dropna()
        if values.empty:
            summary_rows.append({"product": product, "valid": 0})
            continue
        valid_dates = panel.loc[values.index, "date"]
        summary_rows.append(
            {
                "product": product,
                "valid": len(values),
                "first_valid": valid_dates.iloc[0],
                "last_valid": valid_dates.iloc[-1],
                "mean": values.mean(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "latest": values.iloc[-1],
                "rolling_window_for_pair_corr": rolling_window,
            }
        )
    summary_path = output_dir / f"{timestamp}_atm_iv_log_changes_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return csv_path, summary_path


def main():
    args = parse_args()
    if args.rolling_window <= 1:
        raise ValueError("--rolling-window must be greater than 1.")

    pairs = _parse_pairs(args.pair) if args.pair else DEFAULT_PAIRS
    features_by_product = _features_for_pairs(pairs, start=args.start, end=args.end)
    features_by_product = _features_with_log_changes(features_by_product)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_change_panel = build_log_change_panel(features_by_product)
    log_change_csv, log_change_summary_csv = write_log_change_panel(
        log_change_panel,
        args.output_dir,
        args.rolling_window,
        timestamp,
    )
    print(f"log_change_csv={log_change_csv}")
    print(f"log_change_summary_csv={log_change_summary_csv}")

    summaries = []
    for product_a, product_b in pairs:
        frame = build_corr_frame(
            product_a,
            product_b,
            features_by_product,
            rolling_window=args.rolling_window,
        )
        csv_path, png_path = write_pair_outputs(
            frame,
            product_a,
            product_b,
            args.output_dir,
            args.rolling_window,
            timestamp,
        )
        summary = _pair_summary(frame, product_a, product_b, args.rolling_window)
        summary["csv"] = str(csv_path)
        summary["chart"] = str(png_path)
        summaries.append(summary)
        print(
            f"{summary['pair']}: rows={summary.get('rows')} valid={summary.get('valid')} "
            f"first_valid={summary.get('first_valid')} last_valid={summary.get('last_valid')} "
            f"mean={summary.get('mean', float('nan')):.4f} "
            f"latest={summary.get('latest', float('nan')):.4f} "
            f"csv={csv_path} chart={png_path}"
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = args.output_dir / f"{timestamp}_atm_iv_log_change_corr_{args.rolling_window}d_summary.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
