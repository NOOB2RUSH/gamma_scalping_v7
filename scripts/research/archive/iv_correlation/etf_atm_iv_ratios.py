from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
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
    parser.add_argument(
        "--boll-window",
        type=int,
        default=20,
        help="Bollinger band rolling window for the ratio subplot.",
    )
    parser.add_argument(
        "--boll-std",
        type=float,
        default=2.0,
        help="Bollinger band standard-deviation multiplier for the ratio subplot.",
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


def write_combined_chart_output(
    frame,
    numerator,
    denominator,
    output_dir,
    rolling_window,
    boll_window,
    boll_std,
    timestamp,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamp}_{numerator}_vs_{denominator}_atm_iv_combined"
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
    ratio_col = f"iv_ratio_{numerator}_over_{denominator}"
    ma_col = f"{ratio_col}_ma"
    ratio = pd.to_numeric(frame[ratio_col], errors="coerce")
    ma = pd.to_numeric(frame[ma_col], errors="coerce")
    band_min_periods = max(2, boll_window // 4)
    boll_mid = ratio.rolling(boll_window, min_periods=band_min_periods).mean()
    boll_sd = ratio.rolling(boll_window, min_periods=band_min_periods).std()
    boll_upper = boll_mid + boll_std * boll_sd
    boll_lower = boll_mid - boll_std * boll_sd

    fig, (ax_iv, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    fig.suptitle(
        f"{numerator_label} vs {denominator_label} ATM IV Research",
        fontsize=14,
    )

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
    ax_iv.set_title("ATM IV and Difference")
    ax_iv.set_ylabel("ATM IV")
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

    iv_lines = [numerator_line, denominator_line, diff_line]
    ax_iv.legend(iv_lines, [line.get_label() for line in iv_lines], loc="best")

    ax_ratio.plot(
        dates,
        ratio,
        label=f"{numerator_label} ATM IV / {denominator_label} ATM IV",
        linewidth=1.2,
    )
    ax_ratio.plot(dates, ma, label=f"{rolling_window}D rolling mean", linewidth=1.6)
    ax_ratio.plot(
        dates,
        boll_upper,
        label=f"Bollinger upper ({boll_window}D, {boll_std:g}σ)",
        color="tab:green",
        linewidth=0.9,
        alpha=0.85,
    )
    ax_ratio.plot(
        dates,
        boll_lower,
        label=f"Bollinger lower ({boll_window}D, {boll_std:g}σ)",
        color="tab:green",
        linewidth=0.9,
        alpha=0.85,
    )
    ax_ratio.fill_between(
        dates,
        boll_lower,
        boll_upper,
        color="tab:green",
        alpha=0.10,
        linewidth=0,
    )
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_ratio.set_title("ATM IV Ratio")
    ax_ratio.set_ylabel("IV ratio")
    ax_ratio.set_xlabel("Date")
    ax_ratio.grid(True, alpha=0.25)
    ax_ratio.legend(loc="best")

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
    if args.rolling_window <= 0:
        raise ValueError("--rolling-window must be positive.")
    if args.boll_window <= 0:
        raise ValueError("--boll-window must be positive.")
    if args.boll_std <= 0:
        raise ValueError("--boll-std must be positive.")

    pairs = _parse_pairs(args.pair)
    features_by_product = _features_for_pairs(pairs, start=args.start, end=args.end)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for numerator, denominator in pairs:
        frame = build_ratio_frame(
            numerator,
            denominator,
            features_by_product,
            rolling_window=args.rolling_window,
        )
        combined_chart_path = write_combined_chart_output(
            frame,
            numerator,
            denominator,
            args.output_dir,
            args.rolling_window,
            args.boll_window,
            args.boll_std,
            timestamp,
        )
        summary = _pair_summary(frame, numerator, denominator)
        summary["chart"] = str(combined_chart_path)
        summary["combined_chart"] = str(combined_chart_path)
        print(
            f"{summary['pair']}: rows={summary.get('rows')} valid={summary.get('valid')} "
            f"first_valid={summary.get('first_valid')} last_valid={summary.get('last_valid')} "
            f"mean={summary.get('mean', float('nan')):.4f} "
            f"latest={summary.get('latest', float('nan')):.4f} "
            f"combined_chart={combined_chart_path}"
        )


if __name__ == "__main__":
    main()
