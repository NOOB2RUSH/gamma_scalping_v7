"""Compare one-pair ATM straddle Gamma exposure across ETF option products.

This is a research-only script.  It reuses the historical data loader, cached
IV/Greeks chains, and ATM selector, but does not modify any strategy behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402


PRODUCTS = ("50etf", "300etf", "500etf", "kc50etf")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "research" / "gamma_exposure"
RETURN_SCENARIOS = (
    ("median_abs", "abs_log_return_median"),
    ("p75_abs", "abs_log_return_p75"),
    ("p90_abs", "abs_log_return_p90"),
    ("p95_abs", "abs_log_return_p95"),
    ("p99_abs", "abs_log_return_p99"),
    ("rms", "log_return_rms"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare historical ATM call+put Gamma, Cash Gamma, and typical "
            "daily moves across ETF option products."
        )
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCTS,
        default=list(PRODUCTS),
        help="Products to include; defaults to all four ETF option products.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Optional inclusive start used for every product.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional inclusive end used for every product.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Write CSV/JSON outputs without the comparison chart.",
    )
    return parser.parse_args()


def _set_product_config(product: str):
    config = core.config.load_config(product)
    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    return config


def _number(value) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def calculate_cash_gamma(
    call_gamma,
    put_gamma,
    call_multiplier,
    put_multiplier,
    spot,
) -> dict[str, float]:
    """Calculate one-long-pair Gamma exposure using each leg's multiplier."""
    call_gamma = _number(call_gamma)
    put_gamma = _number(put_gamma)
    call_multiplier = _number(call_multiplier)
    put_multiplier = _number(put_multiplier)
    spot = _number(spot)
    values = (call_gamma, put_gamma, call_multiplier, put_multiplier, spot)
    if any(not math.isfinite(value) for value in values):
        return {
            "call_cash_gamma": math.nan,
            "put_cash_gamma": math.nan,
            "pair_cash_gamma": math.nan,
        }
    if call_multiplier <= 0 or put_multiplier <= 0 or spot <= 0:
        return {
            "call_cash_gamma": math.nan,
            "put_cash_gamma": math.nan,
            "pair_cash_gamma": math.nan,
        }
    spot_squared = spot**2
    call_cash_gamma = call_gamma * call_multiplier * spot_squared
    put_cash_gamma = put_gamma * put_multiplier * spot_squared
    return {
        "call_cash_gamma": call_cash_gamma,
        "put_cash_gamma": put_cash_gamma,
        "pair_cash_gamma": call_cash_gamma + put_cash_gamma,
    }


def approximate_gamma_pnl(cash_gamma, return_value):
    """Return 1/2 * CashGamma * r^2 for one long option position or pair."""
    cash_gamma = _number(cash_gamma)
    return_value = _number(return_value)
    if not math.isfinite(cash_gamma) or not math.isfinite(return_value):
        return math.nan
    return 0.5 * cash_gamma * return_value**2


def build_daily_gamma_exposure(
    product: str,
    daily_ohlc: pd.DataFrame,
    enriched_opt_by_date: dict,
) -> pd.DataFrame:
    """Build close Gamma_t and next-trading-day return exposure observations."""
    close = pd.to_numeric(daily_ohlc["close"], errors="coerce").sort_index()
    log_return = np.log(close / close.shift(1))
    next_close = close.shift(-1)
    next_log_return = np.log(next_close / close)
    next_trading_day = pd.Series(close.index, index=close.index).shift(-1)

    rows = []
    for date, spot in close.items():
        row = {
            "date": pd.Timestamp(date),
            "product": product,
            "spot": _number(spot),
            "log_return": _number(log_return.get(date)),
            "abs_log_return": abs(_number(log_return.get(date))),
            "next_trading_day": next_trading_day.get(date),
            "next_close": _number(next_close.get(date)),
            "next_day_log_return": _number(next_log_return.get(date)),
            "atm_available": False,
            "status": "no_option_chain",
            "call_code": None,
            "put_code": None,
            "strike": math.nan,
            "expiry": pd.NaT,
            "dte": math.nan,
            "call_iv": math.nan,
            "put_iv": math.nan,
            "call_gamma": math.nan,
            "put_gamma": math.nan,
            "pair_gamma": math.nan,
            "call_contract_multiplier": math.nan,
            "put_contract_multiplier": math.nan,
            "call_cash_gamma": math.nan,
            "put_cash_gamma": math.nan,
            "pair_cash_gamma": math.nan,
            "long_pair_gamma_pnl_next_day": math.nan,
            "short_pair_gamma_pnl_next_day": math.nan,
        }
        chain_df = enriched_opt_by_date.get(date)
        if chain_df is None or chain_df.empty or not math.isfinite(row["spot"]):
            rows.append(row)
            continue

        atm = core.vol_engine.select_atm_from_chain(chain_df, row["spot"])
        if atm is None:
            row["status"] = "no_atm_pair"
            rows.append(row)
            continue

        call = atm["call"]
        put = atm["put"]
        call_gamma = _number(call.get("gamma"))
        put_gamma = _number(put.get("gamma"))
        call_multiplier = _number(call.get("contract_multiplier"))
        put_multiplier = _number(put.get("contract_multiplier"))
        exposure = calculate_cash_gamma(
            call_gamma,
            put_gamma,
            call_multiplier,
            put_multiplier,
            row["spot"],
        )
        pair_cash_gamma = exposure["pair_cash_gamma"]
        pair_gamma = (
            call_gamma + put_gamma
            if math.isfinite(call_gamma) and math.isfinite(put_gamma)
            else math.nan
        )
        gamma_pnl = approximate_gamma_pnl(
            pair_cash_gamma,
            row["next_day_log_return"],
        )
        row.update(
            {
                "atm_available": math.isfinite(pair_cash_gamma),
                "status": "ok" if math.isfinite(pair_cash_gamma) else "invalid_gamma",
                "call_code": call.get("order_book_id"),
                "put_code": put.get("order_book_id"),
                "strike": _number(atm.get("strike")),
                "expiry": atm.get("expiry"),
                "dte": _number(atm.get("dte")),
                "call_iv": _number(atm.get("call_iv")),
                "put_iv": _number(atm.get("put_iv")),
                "call_gamma": call_gamma,
                "put_gamma": put_gamma,
                "pair_gamma": pair_gamma,
                "call_contract_multiplier": call_multiplier,
                "put_contract_multiplier": put_multiplier,
                **exposure,
                "long_pair_gamma_pnl_next_day": gamma_pnl,
                "short_pair_gamma_pnl_next_day": -gamma_pnl,
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("date").reset_index(drop=True)
    return result


def _quantile(series: pd.Series, probability: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(probability)) if not values.empty else math.nan


def summarize_product(daily: pd.DataFrame) -> dict:
    """Summarize raw Gamma, Cash Gamma, returns, and mapped next-day PnL."""
    if daily.empty:
        raise ValueError("daily Gamma exposure data is empty")
    returns = pd.to_numeric(daily["log_return"], errors="coerce").dropna()
    abs_returns = returns.abs()
    valid_gamma = daily.loc[daily["atm_available"]].copy()
    matched = valid_gamma.dropna(
        subset=["pair_cash_gamma", "next_day_log_return"]
    )
    pair_gamma = pd.to_numeric(valid_gamma["pair_gamma"], errors="coerce")
    cash_gamma = pd.to_numeric(
        valid_gamma["pair_cash_gamma"], errors="coerce"
    )
    gamma_pnl = pd.to_numeric(
        matched["long_pair_gamma_pnl_next_day"], errors="coerce"
    )
    return {
        "product": str(daily["product"].iloc[0]),
        "sample_start": daily["date"].min(),
        "sample_end": daily["date"].max(),
        "price_days": int(len(daily)),
        "return_days": int(len(returns)),
        "atm_gamma_days": int(len(valid_gamma)),
        "atm_gamma_coverage": float(len(valid_gamma) / len(daily)),
        "matched_next_day_days": int(len(matched)),
        "spot_median": _quantile(daily["spot"], 0.50),
        "dte_median": _quantile(valid_gamma["dte"], 0.50),
        "call_gamma_median": _quantile(valid_gamma["call_gamma"], 0.50),
        "put_gamma_median": _quantile(valid_gamma["put_gamma"], 0.50),
        "pair_gamma_mean": float(pair_gamma.mean()),
        "pair_gamma_median": _quantile(pair_gamma, 0.50),
        "pair_gamma_p25": _quantile(pair_gamma, 0.25),
        "pair_gamma_p75": _quantile(pair_gamma, 0.75),
        "pair_gamma_p90": _quantile(pair_gamma, 0.90),
        "pair_cash_gamma_mean": float(cash_gamma.mean()),
        "pair_cash_gamma_median": _quantile(cash_gamma, 0.50),
        "pair_cash_gamma_p25": _quantile(cash_gamma, 0.25),
        "pair_cash_gamma_p75": _quantile(cash_gamma, 0.75),
        "pair_cash_gamma_p90": _quantile(cash_gamma, 0.90),
        "log_return_rms": float(np.sqrt(np.mean(np.square(returns)))),
        "abs_log_return_mean": float(abs_returns.mean()),
        "abs_log_return_median": _quantile(abs_returns, 0.50),
        "abs_log_return_p75": _quantile(abs_returns, 0.75),
        "abs_log_return_p90": _quantile(abs_returns, 0.90),
        "abs_log_return_p95": _quantile(abs_returns, 0.95),
        "abs_log_return_p99": _quantile(abs_returns, 0.99),
        "realized_long_gamma_pnl_mean": float(gamma_pnl.mean()),
        "realized_long_gamma_pnl_median": _quantile(gamma_pnl, 0.50),
        "realized_long_gamma_pnl_p75": _quantile(gamma_pnl, 0.75),
        "realized_long_gamma_pnl_p90": _quantile(gamma_pnl, 0.90),
        "realized_long_gamma_pnl_p95": _quantile(gamma_pnl, 0.95),
        "realized_long_gamma_pnl_p99": _quantile(gamma_pnl, 0.99),
    }


def build_scenario_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Map typical return magnitudes to PnL at each product's median Cash Gamma."""
    rows = []
    for _, product_summary in summary.iterrows():
        cash_gamma = product_summary["pair_cash_gamma_median"]
        for scenario, return_column in RETURN_SCENARIOS:
            move = product_summary[return_column]
            long_pnl = approximate_gamma_pnl(cash_gamma, move)
            rows.append(
                {
                    "product": product_summary["product"],
                    "scenario": scenario,
                    "return_metric": return_column,
                    "absolute_log_return": move,
                    "reference_pair_cash_gamma": cash_gamma,
                    "long_pair_gamma_pnl": long_pnl,
                    "short_pair_gamma_pnl": -long_pnl,
                }
            )
    return pd.DataFrame(rows)


def load_product_daily(product: str, start=None, end=None) -> pd.DataFrame:
    config = _set_product_config(product)
    start_date = pd.Timestamp(start or config.backtest.start)
    end_date = pd.Timestamp(end or config.backtest.end)
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    etf_by_date = core.data_loader.load_etf_series(start_date, end_date)
    hedge_by_date = core.data_loader.load_hedge_series(start_date, end_date)
    opt_by_date = core.data_loader.load_opt_series(start_date, end_date)
    opt_by_date = core.data_loader.attach_underlying_prices(
        opt_by_date,
        hedge_by_date,
    )
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        start_date,
        end_date,
    )
    daily_ohlc = core.vol_engine.build_daily_ohlc_df(etf_by_date)
    return build_daily_gamma_exposure(product, daily_ohlc, enriched)


def plot_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ordered = summary.set_index("product").reindex(PRODUCTS).dropna(how="all")
    positions = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(
        positions,
        ordered["pair_cash_gamma_median"],
        color="tab:blue",
        alpha=0.8,
    )
    axes[0].set_title("Median Cash Gamma: One Long ATM Call + Put")
    axes[0].set_ylabel("Cash Gamma")
    axes[0].set_xticks(positions, ordered.index.str.upper())
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.24
    axes[1].bar(
        positions - width,
        ordered["abs_log_return_median"],
        width=width,
        label="Median |r|",
    )
    axes[1].bar(
        positions,
        ordered["log_return_rms"],
        width=width,
        label="RMS r",
    )
    axes[1].bar(
        positions + width,
        ordered["abs_log_return_p95"],
        width=width,
        label="95th percentile |r|",
    )
    axes[1].set_title("Historical Daily ETF Moves")
    axes[1].set_ylabel("Absolute log-return magnitude")
    axes[1].set_xticks(positions, ordered.index.str.upper())
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("ETF Option Gamma Exposure Research")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    scenarios: pd.DataFrame,
    output_dir: Path,
    write_chart: bool = True,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_four_product_gamma_exposure"
    paths = {
        "daily": output_dir / f"{stem}_daily.csv",
        "summary": output_dir / f"{stem}_summary.csv",
        "scenarios": output_dir / f"{stem}_scenarios.csv",
        "metadata": output_dir / f"{stem}_metadata.json",
    }
    daily.to_csv(paths["daily"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    scenarios.to_csv(paths["scenarios"], index=False, encoding="utf-8-sig")
    metadata = {
        "unit": "one long ATM call plus one long ATM put",
        "cash_gamma_formula": (
            "call_gamma * call_multiplier * spot^2 + "
            "put_gamma * put_multiplier * spot^2"
        ),
        "gamma_pnl_formula": "0.5 * pair_cash_gamma * log_return^2",
        "exposure_timing": "close Gamma on t mapped to close(t+1)/close(t)",
        "short_position_sign": "negative of the reported long-pair Gamma PnL",
        "atm_selection": "same configured ATM selector used by the backtester",
        "products": summary["product"].tolist(),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if write_chart:
        paths["chart"] = output_dir / f"{stem}_comparison.png"
        plot_comparison(summary, paths["chart"])
    return paths


def main() -> None:
    args = parse_args()
    daily_frames = []
    summary_rows = []
    for product in dict.fromkeys(args.products):
        print(f"[gamma exposure] loading {product}...")
        product_daily = load_product_daily(product, args.start, args.end)
        daily_frames.append(product_daily)
        summary_rows.append(summarize_product(product_daily))

    daily = pd.concat(daily_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    scenarios = build_scenario_table(summary)
    paths = write_outputs(
        daily,
        summary,
        scenarios,
        args.output_dir,
        write_chart=not args.no_chart,
    )
    print(summary.to_string(index=False))
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
