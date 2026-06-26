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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the historical ATM IV ratio: kc50etf ATM IV / 50etf ATM IV."
        )
    )
    parser.add_argument("--start", default=None, help="Inclusive start date, e.g. 2023-06-05.")
    parser.add_argument("--end", default=None, help="Inclusive end date, e.g. 2026-05-27.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the CSV and chart outputs.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=20,
        help="Rolling mean window for the ratio chart.",
    )
    return parser.parse_args()


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


def build_ratio_frame(start=None, end=None, rolling_window=20):
    iv_50 = _product_features("50etf", start=start, end=end)
    iv_kc50 = _product_features("kc50etf", start=start, end=end)
    joined = iv_50.join(
        iv_kc50,
        how="inner",
        lsuffix="_50etf",
        rsuffix="_kc50etf",
    )
    joined = joined.rename(
        columns={
            "atm_iv_50etf": "atm_iv_50etf",
            "atm_iv_kc50etf": "atm_iv_kc50etf",
        }
    )
    joined["iv_ratio_kc50_over_50"] = (
        pd.to_numeric(joined["atm_iv_kc50etf"], errors="coerce")
        / pd.to_numeric(joined["atm_iv_50etf"], errors="coerce")
    )
    joined.loc[
        ~pd.to_numeric(joined["atm_iv_50etf"], errors="coerce").gt(0),
        "iv_ratio_kc50_over_50",
    ] = pd.NA
    joined["iv_ratio_ma"] = joined["iv_ratio_kc50_over_50"].rolling(
        rolling_window,
        min_periods=max(2, rolling_window // 4),
    ).mean()
    joined.insert(0, "date", joined.index.strftime("%Y-%m-%d"))
    return joined.reset_index(drop=True)


def write_outputs(frame, output_dir, rolling_window):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "kc50etf_over_50etf_atm_iv_ratio.csv"
    png_path = output_dir / "kc50etf_over_50etf_atm_iv_ratio.png"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    dates = pd.to_datetime(frame["date"])
    ratio = pd.to_numeric(frame["iv_ratio_kc50_over_50"], errors="coerce")
    ma = pd.to_numeric(frame["iv_ratio_ma"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, ratio, label="KC50ETF ATM IV / 50ETF ATM IV", linewidth=1.2)
    ax.plot(dates, ma, label=f"{rolling_window}D rolling mean", linewidth=1.6)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("KC50ETF vs 50ETF ATM IV Ratio")
    ax.set_ylabel("IV ratio")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    return csv_path, png_path


def _summary(frame):
    ratio = pd.to_numeric(frame["iv_ratio_kc50_over_50"], errors="coerce").dropna()
    if ratio.empty:
        return "No valid ratio observations."
    return (
        f"rows={len(frame)} valid={len(ratio)} "
        f"start={frame['date'].iloc[0]} end={frame['date'].iloc[-1]} "
        f"mean={ratio.mean():.4f} median={ratio.median():.4f} "
        f"min={ratio.min():.4f} max={ratio.max():.4f} "
        f"latest={ratio.iloc[-1]:.4f}"
    )


def main():
    args = parse_args()
    frame = build_ratio_frame(
        start=args.start,
        end=args.end,
        rolling_window=args.rolling_window,
    )
    csv_path, png_path = write_outputs(frame, args.output_dir, args.rolling_window)
    print(_summary(frame))
    print(f"csv={csv_path}")
    print(f"chart={png_path}")


if __name__ == "__main__":
    main()
