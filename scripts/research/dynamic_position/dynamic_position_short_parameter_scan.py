"""单次加载300ETF数据，扫描动态跨式short侧仓位参数。"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402
import run as backtest_runner  # noqa: E402
from core.backtest_strategies import create_strategy  # noqa: E402
from scripts.research.dynamic_position.iv_daily_state_report import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan dynamic-position short-side parameters with one data load."
    )
    parser.add_argument("--product", default="300etf")
    parser.add_argument("--pmin-values", type=int, nargs="+", default=[8])
    parser.add_argument("--pmax-values", type=int, nargs="+", default=[18, 20, 22, 24])
    parser.add_argument("--iv-max-values", type=float, nargs="+", default=[0.35, 0.40, 0.45])
    parser.add_argument("--iv-spike-values", type=float, nargs="+", default=[0.03])
    parser.add_argument("--short-steps-values", type=int, nargs="+", default=[10])
    parser.add_argument("--shock-start", default="2024-09-23")
    parser.add_argument("--shock-end", default="2024-10-09")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parameter_grid(
    pmin_values: list[int],
    pmax_values: list[int],
    iv_max_values: list[float],
    iv_spike_values: list[float],
    short_steps_values: list[int],
) -> list[dict]:
    return [
        {
            "min_qty": int(min_qty),
            "max_qty": int(max_qty),
            "iv_max": float(iv_max),
            "iv_spike": float(iv_spike),
            "short_steps": int(short_steps),
        }
        for min_qty, max_qty, iv_max, iv_spike, short_steps in itertools.product(
            pmin_values,
            pmax_values,
            iv_max_values,
            iv_spike_values,
            short_steps_values,
        )
    ]


def window_statistics(
    daily_pnl: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    window = daily_pnl.loc[pd.Timestamp(start) : pd.Timestamp(end), "nav"].dropna()
    if window.empty:
        return {
            "shock_return": float("nan"),
            "shock_max_drawdown": float("nan"),
        }
    drawdown = window.div(window.cummax()).sub(1.0)
    return {
        "shock_return": float(window.iloc[-1] / window.iloc[0] - 1.0),
        "shock_max_drawdown": float(drawdown.min()),
    }


def summarize_result(
    daily_pnl: pd.DataFrame,
    trades: pd.DataFrame,
    parameters: dict,
    shock_start: str,
    shock_end: str,
) -> dict:
    stats = backtest_runner.calc_return_stats(daily_pnl)
    breakdown = backtest_runner.calc_summary_breakdown(daily_pnl, trades)
    short_days = daily_pnl[daily_pnl["short_has_position"]]
    pair_qty = (
        short_days["short_position_call_qty"]
        + short_days["short_position_put_qty"]
    ) / 2.0
    annual_pnl = daily_pnl["daily_nav_pnl"].groupby(daily_pnl.index.year).sum()
    total_margin = daily_pnl["option_margin"] + daily_pnl["hedge_margin"]
    return {
        **parameters,
        "total_pnl": float(stats["total_pnl"]),
        "total_return": float(stats["total_return"]),
        "annual_return": float(stats["annual_return"]),
        "sharpe_ratio": float(stats["sharpe_ratio"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "max_drawdown_start": stats["max_drawdown_start"].strftime("%Y-%m-%d"),
        "max_drawdown_end": stats["max_drawdown_end"].strftime("%Y-%m-%d"),
        **window_statistics(daily_pnl, shock_start, shock_end),
        "worst_calendar_year_pnl": float(annual_pnl.min()),
        "max_option_margin": float(daily_pnl["option_margin"].max()),
        "max_total_margin": float(total_margin.max()),
        "min_cash": float(stats["min_cash"]),
        "option_fee": float(breakdown["option_fee"]),
        "etf_fee": float(breakdown["etf_fee"]),
        "short_position_days": int(len(short_days)),
        "average_pair_qty": float(pair_qty.mean()),
        "maximum_pair_qty": float(pair_qty.max()),
        "trade_records": int(len(trades)),
    }


def add_relative_columns(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    baseline = result[
        result["min_qty"].eq(10)
        & result["max_qty"].eq(20)
        & result["iv_max"].eq(0.35)
        & result["iv_spike"].eq(0.03)
        & result["short_steps"].eq(10)
    ]
    if baseline.empty:
        return result
    baseline = baseline.iloc[0]
    for column in [
        "total_pnl",
        "sharpe_ratio",
        "max_drawdown",
        "shock_return",
        "shock_max_drawdown",
        "max_option_margin",
        "max_total_margin",
    ]:
        result[f"{column}_vs_baseline"] = result[column] - baseline[column]
    return result


def load_shared_backtest_inputs(product: str):
    config = core.config.load_config(product)
    backtest_runner.sync_config(config)
    etf_by_date, opt_by_date, hedge_by_date = backtest_runner.load_data()
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        config.backtest.start,
        config.backtest.end,
    )
    features = core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched,
        config.backtest.start,
        config.backtest.end,
    )
    return (
        config,
        etf_by_date,
        opt_by_date,
        hedge_by_date,
        trading_calendar,
        enriched,
        features,
    )


def main() -> None:
    args = parse_args()
    grid = parameter_grid(
        args.pmin_values,
        args.pmax_values,
        args.iv_max_values,
        args.iv_spike_values,
        args.short_steps_values,
    )
    if not grid:
        raise ValueError("Parameter grid is empty.")

    (
        config,
        etf_by_date,
        opt_by_date,
        hedge_by_date,
        trading_calendar,
        enriched,
        features,
    ) = load_shared_backtest_inputs(args.product)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / (
        f"{timestamp}_{args.product}_dynamic_position_short_parameter_scan.csv"
    )

    rows = []
    for number, parameters in enumerate(grid, start=1):
        plugin = create_strategy("dynamic_position_straddle", config)
        plugin.dynamic_config = replace(plugin.dynamic_config, **parameters)
        plugin.dynamic_config.validate()
        signals = plugin.build_signals(features)
        daily_pnl, trades = core.backtester.run_backtest(
            copy.deepcopy(etf_by_date),
            copy.deepcopy(opt_by_date),
            signals,
            trading_calendar=trading_calendar,
            enriched_opt_by_date=copy.deepcopy(enriched),
            hedge_by_date=copy.deepcopy(hedge_by_date),
            strategy_plugin=plugin,
        )
        row = summarize_result(
            daily_pnl,
            trades,
            parameters,
            args.shock_start,
            args.shock_end,
        )
        rows.append(row)
        add_relative_columns(pd.DataFrame(rows)).to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"[{number}/{len(grid)}] {parameters} "
            f"pnl={row['total_pnl']:.2f} sharpe={row['sharpe_ratio']:.3f} "
            f"max_dd={row['max_drawdown']:.3%} "
            f"shock_dd={row['shock_max_drawdown']:.3%}",
            flush=True,
        )
        del daily_pnl, trades, signals, plugin
        gc.collect()

    results = add_relative_columns(pd.DataFrame(rows))
    results = results.sort_values(
        ["sharpe_ratio", "max_drawdown", "shock_max_drawdown"],
        ascending=[False, False, False],
    )
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\nTop results by Sharpe:")
    print(
        results[
            [
                "max_qty",
                "min_qty",
                "iv_max",
                "iv_spike",
                "short_steps",
                "total_pnl",
                "sharpe_ratio",
                "max_drawdown",
                "shock_max_drawdown",
                "max_option_margin",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    print(f"scan_report={output_path}")


if __name__ == "__main__":
    main()
