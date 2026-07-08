from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "research"
PRODUCT_LABELS = {
    "50etf": "50ETF",
    "300etf": "300ETF",
    "500etf": "500ETF",
    "kc50etf": "KC50ETF",
}
DEFAULT_PAIRS = (
    ("300etf", "50etf"),
    ("500etf", "50etf"),
    ("300etf", "500etf"),
    ("300etf", "kc50etf"),
    ("500etf", "kc50etf"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute historical ATM IV ratios for ETF option products."
    )
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help=(
            "Ratio pair as numerator/denominator, e.g. 300etf/50etf. "
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
        default=20,
        help="Rolling mean window for each ratio chart.",
    )
    return parser.parse_args()


def _parse_pairs(values):
    if not values:
        return DEFAULT_PAIRS
    pairs = []
    for value in values:
        if "/" not in value:
            raise ValueError(f"pair must use numerator/denominator format: {value}")
        numerator, denominator = value.split("/", 1)
        pairs.append((numerator.strip().lower(), denominator.strip().lower()))
    return tuple(pairs)


def _set_product_config(product):
    cfg = core.config.load_config(product)
    core.config.CONFIG = cfg
    core.vol_engine.CONFIG = cfg
    return cfg


def _product_features(product, start=None, end=None):
    cfg = _set_product_config(product)
    start = pd.Timestamp(start or cfg.backtest.start)
    end = pd.Timestamp(end or cfg.backtest.end)
    etf_by_date = core.data_loader.load_etf_series(start, end)
    hedge_by_date = core.data_loader.load_hedge_series(start, end)
    opt_by_date = core.data_loader.load_opt_series(start, end)
    opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        start,
        end,
    )
    features = core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched,
        start,
        end,
    )
    result = features[["atm_iv", "atm_strike", "atm_expiry", "atm_dte"]].copy()
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def _features_for_pairs(pairs, start=None, end=None):
    products = sorted({product for pair in pairs for product in pair})
    return {
        product: _product_features(product, start=start, end=end)
        for product in products
    }


def build_ratio_frame(numerator, denominator, features_by_product, rolling_window=20):
    numerator_features = features_by_product[numerator]
    denominator_features = features_by_product[denominator]
    joined = denominator_features.join(
        numerator_features,
        how="inner",
        lsuffix=f"_{denominator}",
        rsuffix=f"_{numerator}",
    )

    denominator_iv = pd.to_numeric(joined[f"atm_iv_{denominator}"], errors="coerce")
    numerator_iv = pd.to_numeric(joined[f"atm_iv_{numerator}"], errors="coerce")
    ratio_col = f"iv_ratio_{numerator}_over_{denominator}"
    diff_col = f"iv_diff_{numerator}_minus_{denominator}"
    joined[ratio_col] = numerator_iv / denominator_iv
    joined.loc[~denominator_iv.gt(0), ratio_col] = pd.NA
    joined[diff_col] = numerator_iv - denominator_iv
    joined[f"{ratio_col}_ma"] = joined[ratio_col].rolling(
        rolling_window,
        min_periods=max(2, rolling_window // 4),
    ).mean()
    joined.insert(0, "date", joined.index.strftime("%Y-%m-%d"))
    return joined.reset_index(drop=True)


def write_ratio_outputs(frame, numerator, denominator, output_dir, rolling_window):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{numerator}_over_{denominator}_atm_iv_ratio"
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    ratio_col = f"iv_ratio_{numerator}_over_{denominator}"
    ma_col = f"{ratio_col}_ma"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    dates = pd.to_datetime(frame["date"])
    ratio = pd.to_numeric(frame[ratio_col], errors="coerce")
    ma = pd.to_numeric(frame[ma_col], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        dates,
        ratio,
        label=f"{numerator.upper()} ATM IV / {denominator.upper()} ATM IV",
        linewidth=1.2,
    )
    ax.plot(dates, ma, label=f"{rolling_window}D rolling mean", linewidth=1.6)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(f"{numerator.upper()} vs {denominator.upper()} ATM IV Ratio")
    ax.set_ylabel("IV ratio")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path


def write_pair_iv_curve_output(frame, numerator, denominator, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{numerator}_vs_{denominator}_atm_iv_curves"
    png_path = output_dir / f"{stem}.png"

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    numerator_label = PRODUCT_LABELS.get(numerator, numerator.upper())
    denominator_label = PRODUCT_LABELS.get(denominator, denominator.upper())
    dates = pd.to_datetime(frame["date"])
    numerator_iv = pd.to_numeric(frame[f"atm_iv_{numerator}"], errors="coerce")
    denominator_iv = pd.to_numeric(frame[f"atm_iv_{denominator}"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, numerator_iv, label=f"{numerator_label} ATM IV", linewidth=1.3)
    ax.plot(dates, denominator_iv, label=f"{denominator_label} ATM IV", linewidth=1.3)
    ax.set_title(f"{numerator_label} vs {denominator_label} ATM IV")
    ax.set_ylabel("ATM IV")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def write_pair_iv_diff_overlay_output(frame, numerator, denominator, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{numerator}_vs_{denominator}_atm_iv_diff_overlay"
    png_path = output_dir / f"{stem}.png"

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    numerator_label = PRODUCT_LABELS.get(numerator, numerator.upper())
    denominator_label = PRODUCT_LABELS.get(denominator, denominator.upper())
    dates = pd.to_datetime(frame["date"])
    numerator_iv = pd.to_numeric(frame[f"atm_iv_{numerator}"], errors="coerce")
    denominator_iv = pd.to_numeric(frame[f"atm_iv_{denominator}"], errors="coerce")
    diff_col = f"iv_diff_{numerator}_minus_{denominator}"
    iv_diff = pd.to_numeric(frame[diff_col], errors="coerce")

    fig, ax_iv = plt.subplots(figsize=(12, 5))
    numerator_line = ax_iv.plot(
        dates,
        numerator_iv,
        label=f"{numerator_label} ATM IV",
        linewidth=1.3,
    )[0]
    denominator_line = ax_iv.plot(
        dates,
        denominator_iv,
        label=f"{denominator_label} ATM IV",
        linewidth=1.3,
    )[0]
    ax_iv.set_title(
        f"{numerator_label} vs {denominator_label} ATM IV and Difference"
    )
    ax_iv.set_ylabel("ATM IV")
    ax_iv.set_xlabel("Date")
    ax_iv.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_iv.grid(True, alpha=0.25)

    ax_diff = ax_iv.twinx()
    diff_line = ax_diff.plot(
        dates,
        iv_diff,
        label=f"{numerator_label} - {denominator_label}",
        color="tab:red",
        linewidth=1.2,
        linestyle="--",
        alpha=0.85,
    )[0]
    ax_diff.axhline(0.0, color="tab:red", linewidth=0.8, alpha=0.35)
    ax_diff.set_ylabel("ATM IV difference")
    ax_diff.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    lines = [numerator_line, denominator_line, diff_line]
    ax_iv.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return png_path


def _pair_summary(frame, numerator, denominator):
    ratio_col = f"iv_ratio_{numerator}_over_{denominator}"
    ratio = pd.to_numeric(frame[ratio_col], errors="coerce").dropna()
    if ratio.empty:
        return {
            "pair": f"{numerator}/{denominator}",
            "rows": len(frame),
            "valid": 0,
        }
    valid_dates = frame.loc[ratio.index, "date"]
    return {
        "pair": f"{numerator}/{denominator}",
        "rows": len(frame),
        "valid": len(ratio),
        "first_valid": valid_dates.iloc[0],
        "last_valid": valid_dates.iloc[-1],
        "mean": ratio.mean(),
        "median": ratio.median(),
        "min": ratio.min(),
        "max": ratio.max(),
        "latest": ratio.iloc[-1],
    }


def main():
    args = parse_args()
    pairs = _parse_pairs(args.pair)
    features_by_product = _features_for_pairs(pairs, start=args.start, end=args.end)

    summaries = []
    for numerator, denominator in pairs:
        frame = build_ratio_frame(
            numerator,
            denominator,
            features_by_product,
            rolling_window=args.rolling_window,
        )
        csv_path, png_path = write_ratio_outputs(
            frame,
            numerator,
            denominator,
            args.output_dir,
            args.rolling_window,
        )
        iv_curve_path = write_pair_iv_curve_output(
            frame,
            numerator,
            denominator,
            args.output_dir,
        )
        iv_diff_overlay_path = write_pair_iv_diff_overlay_output(
            frame,
            numerator,
            denominator,
            args.output_dir,
        )
        summary = _pair_summary(frame, numerator, denominator)
        summary["csv"] = str(csv_path)
        summary["chart"] = str(png_path)
        summary["ratio_chart"] = str(png_path)
        summary["iv_curve_chart"] = str(iv_curve_path)
        summary["iv_diff_overlay_chart"] = str(iv_diff_overlay_path)
        summaries.append(summary)
        print(
            f"{summary['pair']}: rows={summary.get('rows')} valid={summary.get('valid')} "
            f"first_valid={summary.get('first_valid')} last_valid={summary.get('last_valid')} "
            f"mean={summary.get('mean', float('nan')):.4f} "
            f"latest={summary.get('latest', float('nan')):.4f} "
            f"iv_curve={iv_curve_path} "
            f"iv_diff_overlay={iv_diff_overlay_path}"
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = args.output_dir / "atm_iv_ratio_summary.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
