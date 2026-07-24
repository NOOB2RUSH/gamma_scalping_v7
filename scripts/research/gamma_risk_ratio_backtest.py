"""Backtest Gamma-risk-ratio quantities in four fully independent accounts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402
import run as backtest_runner  # noqa: E402
from core.backtest_strategies import (  # noqa: E402
    create_strategy,
    resolve_strategy_config,
)


STRATEGY_ID = "dynamic_atm_iv_straddle"
DEFAULT_START = "20230605"
DEFAULT_END = "20251231"
DEFAULT_INITIAL_CASH_PER_PRODUCT = 1_000_000.0
DEFAULT_QUANTITIES = {
    "50etf": 35,
    "300etf": 20,
    "500etf": 10,
    "kc50etf": 40,
}
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "research"
    / "gamma_exposure"
    / "independent_ratio_backtest"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest four independent 1M accounts at the Gamma-risk ratio "
            "7:4:2:8, scaled to 10 pairs for 500ETF."
        )
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--initial-cash-per-product",
        type=float,
        default=DEFAULT_INITIAL_CASH_PER_PRODUCT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-chart", action="store_true")
    return parser.parse_args()


def _run_product(
    product: str,
    quantity: int,
    start: str,
    end: str,
    initial_cash: float,
):
    config = resolve_strategy_config(
        core.config.load_config(product),
        STRATEGY_ID,
    )
    config = replace(
        config,
        backtest=replace(
            config.backtest,
            start=start,
            end=end,
            initial_cash=initial_cash,
            long_qty=quantity,
            short_qty=quantity,
            proportional_position_sizing_enabled=False,
            dynamic_position_control_enabled=False,
        ),
    )
    backtest_runner.sync_config(config)
    etf_by_date, opt_by_date, hedge_by_date = backtest_runner.load_data()
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
    plugin = create_strategy(STRATEGY_ID, config)
    signals = plugin.build_signals(features)
    daily, trades = core.backtester.run_backtest(
        etf_by_date,
        opt_by_date,
        signals,
        initial_cash=initial_cash,
        long_qty=quantity,
        short_qty=quantity,
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched,
        hedge_by_date=hedge_by_date,
        strategy_plugin=plugin,
        compute_full_revaluation=False,
    )
    daily = fill_inactive_account_days(
        daily,
        pd.DatetimeIndex(etf_by_date).sort_values(),
        initial_cash,
    )
    return config, plugin, daily, trades


def fill_inactive_account_days(
    daily: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    initial_cash: float,
) -> pd.DataFrame:
    """Restore skipped flat-account dates as zero-PnL observations."""
    result = daily.reindex(pd.DatetimeIndex(trading_dates).sort_values()).copy()
    result.index.name = daily.index.name
    for column in ("nav", "cash"):
        result[column] = pd.to_numeric(result[column], errors="coerce").ffill()
        result[column] = result[column].fillna(initial_cash)
    for column in (
        "option_margin",
        "hedge_margin",
        "option_fee",
        "etf_fee",
        "daily_fee",
        "daily_nav_pnl",
        "daily_nav_pnl_before_fee",
    ):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    return result


def _daily_returns(nav: pd.Series, initial_cash: float) -> pd.Series:
    previous_nav = nav.shift(1)
    if not nav.empty:
        previous_nav.iloc[0] = initial_cash
    return (nav - previous_nav) / previous_nav


def performance_stats(
    nav: pd.Series,
    initial_cash: float,
    annual_days: int = 252,
) -> dict[str, float]:
    nav = pd.to_numeric(nav, errors="coerce").dropna()
    if nav.empty:
        raise ValueError("NAV series is empty")
    total_pnl = float(nav.iloc[-1] - initial_cash)
    total_return = float(nav.iloc[-1] / initial_cash - 1.0)
    trading_days = len(nav)
    annual_return = (
        float((1.0 + total_return) ** (annual_days / trading_days) - 1.0)
        if 1.0 + total_return > 0
        else math.nan
    )
    returns = _daily_returns(nav, initial_cash).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    return_std = returns.std(ddof=1)
    sharpe = (
        float(returns.mean() / return_std * math.sqrt(annual_days))
        if len(returns) > 1 and return_std > 0
        else math.nan
    )
    running_peak = pd.concat(
        [pd.Series([initial_cash]), nav.reset_index(drop=True)],
        ignore_index=True,
    ).cummax().iloc[1:].set_axis(nav.index)
    drawdown = nav / running_peak - 1.0
    max_drawdown_date = drawdown.idxmin()
    return {
        "initial_cash": float(initial_cash),
        "final_nav": float(nav.iloc[-1]),
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annual_return": annual_return,
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "max_drawdown_date": max_drawdown_date,
        "daily_return_volatility": float(returns.std(ddof=1)),
    }


def summarize_account(
    product: str,
    quantity: int,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict:
    stats = performance_stats(daily["nav"], initial_cash)
    option_margin = pd.to_numeric(daily["option_margin"], errors="coerce").fillna(0)
    hedge_margin = pd.to_numeric(daily["hedge_margin"], errors="coerce").fillna(0)
    total_margin = option_margin + hedge_margin
    nav = pd.to_numeric(daily["nav"], errors="coerce")
    option_fee = pd.to_numeric(daily["option_fee"], errors="coerce").fillna(0)
    etf_fee = pd.to_numeric(daily["etf_fee"], errors="coerce").fillna(0)
    cash = pd.to_numeric(daily["cash"], errors="coerce")
    trade_types = (
        trades["type"].astype(str)
        if not trades.empty and "type" in trades
        else pd.Series(dtype="object")
    )
    return {
        "product": product,
        "long_target_qty": quantity,
        "short_target_qty": quantity,
        "start": daily.index.min(),
        "end": daily.index.max(),
        "recorded_days": len(daily),
        "trade_records": len(trades),
        "long_entries": int(trade_types.eq("open_straddle").sum()),
        "short_entries": int(trade_types.eq("open_short_straddle").sum()),
        **stats,
        "total_fee": float((option_fee + etf_fee).sum()),
        "pnl_before_fee": float(stats["total_pnl"] + (option_fee + etf_fee).sum()),
        "min_cash": float(cash.min()),
        "negative_cash_days": int(cash.lt(0).sum()),
        "max_option_margin": float(option_margin.max()),
        "max_hedge_margin": float(hedge_margin.max()),
        "max_total_margin": float(total_margin.max()),
        "max_margin_to_nav": float((total_margin / nav).max()),
    }


def plot_independent_accounts(
    daily_by_product: dict[str, pd.DataFrame],
    initial_cash: float,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (product, daily) in zip(axes.flat, daily_by_product.items()):
        cumulative_return = daily["nav"] / initial_cash - 1.0
        running_peak = pd.concat(
            [pd.Series([initial_cash]), daily["nav"].reset_index(drop=True)],
            ignore_index=True,
        ).cummax().iloc[1:].set_axis(daily.index)
        drawdown = daily["nav"] / running_peak - 1.0
        ax.plot(
            cumulative_return.index,
            cumulative_return,
            color="tab:blue",
            linewidth=1.5,
            label="Cumulative return",
        )
        ax.fill_between(
            drawdown.index,
            drawdown,
            0,
            color="tab:red",
            alpha=0.25,
            label="Drawdown",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_title(
            f"{product.upper()} — independent 1M account, "
            f"target {DEFAULT_QUANTITIES[product]} pairs"
        )
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Independent Gamma-Risk-Ratio Backtests (No Portfolio Aggregation)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.initial_cash_per_product <= 0:
        raise ValueError("--initial-cash-per-product must be positive")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start > end:
        raise ValueError("start must be on or before end")

    daily_by_product = {}
    trades_by_product = {}
    configs = {}
    strategy_metadata = {}
    summary_rows = []
    for product, quantity in DEFAULT_QUANTITIES.items():
        print(
            f"[independent backtest] {product}: cash={args.initial_cash_per_product:.0f}, "
            f"long={quantity}, short={quantity}",
            flush=True,
        )
        config, plugin, daily, trades = _run_product(
            product,
            quantity,
            start.strftime("%Y%m%d"),
            end.strftime("%Y%m%d"),
            args.initial_cash_per_product,
        )
        configs[product] = asdict(config)
        strategy_metadata[product] = plugin.metadata()
        daily_by_product[product] = daily
        trades_by_product[product] = trades
        summary_rows.append(
            summarize_account(
                product,
                quantity,
                daily,
                trades,
                args.initial_cash_per_product,
            )
        )

    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_independent_gamma_ratio_35_20_10_40"
    paths = {
        "summary": args.output_dir / f"{stem}_summary.csv",
        "metadata": args.output_dir / f"{stem}_metadata.json",
    }
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    for product in DEFAULT_QUANTITIES:
        daily_by_product[product].to_csv(
            args.output_dir / f"{stem}_{product}_daily.csv",
            encoding="utf-8-sig",
        )
        trades_by_product[product].to_csv(
            args.output_dir / f"{stem}_{product}_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
    paths["metadata"].write_text(
        json.dumps(
            {
                "strategy_id": STRATEGY_ID,
                "ratio": "50etf:300etf:500etf:kc50etf = 7:4:2:8",
                "quantities": DEFAULT_QUANTITIES,
                "account_model": "four fully independent accounts",
                "initial_cash_per_product": args.initial_cash_per_product,
                "portfolio_aggregation": False,
                "shared_cash_or_margin": False,
                "configs": configs,
                "strategy_metadata": strategy_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.no_chart:
        paths["chart"] = args.output_dir / f"{stem}.png"
        plot_independent_accounts(
            daily_by_product,
            args.initial_cash_per_product,
            paths["chart"],
        )

    print(summary.to_string(index=False), flush=True)
    for name, path in paths.items():
        print(f"{name}={path}", flush=True)


if __name__ == "__main__":
    main()
