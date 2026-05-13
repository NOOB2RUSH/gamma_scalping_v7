from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

import core


# 调参网格放在这里。默认只扫最核心、最容易影响交易频率和收益的参数。
# 如果要扩大搜索范围，直接往列表里加值即可。
PARAM_GRID = {
    "strategy.open_iv_threshold": [0.14, 0.15, 0.16],
    "strategy.close_iv_threshold": [0.24, 0.27, 0.30],
    "strategy.roll_iv_threshold": [0.135, 0.15],
    "strategy.roll_dte_threshold": [3, 4, 5],
    "vol.atm_target_dte": [10, 15, 20],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="扫描核心回测参数，并按期末 NAV 排序输出结果。"
    )
    parser.add_argument(
        "--start",
        default=core.config.CONFIG.backtest.start,
        help="回测开始日期，默认读取 core/config.py。",
    )
    parser.add_argument(
        "--end",
        default=core.config.CONFIG.backtest.end,
        help="回测结束日期，默认读取 core/config.py。",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=None,
        help="只运行前 N 组参数，便于先做小样本试跑。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="结果 CSV 路径；默认写入 output/optimize_时间戳.csv。",
    )
    return parser.parse_args()


def iter_param_sets(param_grid):
    """把参数网格展开为逐组参数字典。"""
    keys = list(param_grid)
    value_lists = [param_grid[key] for key in keys]

    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def apply_param_set(base_config, param_set):
    """基于默认 CONFIG 生成新的配置对象，并同步到各模块的全局引用。"""
    strategy_updates = {}
    vol_updates = {}
    backtest_updates = {}

    for key, value in param_set.items():
        section, field = key.split(".", 1)
        if section == "strategy":
            strategy_updates[field] = value
        elif section == "vol":
            vol_updates[field] = value
        elif section == "backtest":
            backtest_updates[field] = value
        else:
            raise ValueError(f"未知参数分组: {key}")

    config = replace(
        base_config,
        strategy=replace(base_config.strategy, **strategy_updates),
        vol=replace(base_config.vol, **vol_updates),
        backtest=replace(base_config.backtest, **backtest_updates),
    )

    # 这些模块在 import 时缓存了 CONFIG 引用；调参时需要同步替换。
    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    core.strategy.CONFIG = config
    core.backtester.CONFIG = config
    core.position.CONFIG = config
    core.hedge.CONFIG = config
    return config


def make_output_path(output_arg):
    if output_arg:
        output_path = Path(output_arg)
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(core.config.CONFIG.report.output_root) / (
            f"optimize_{timestamp}.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def calc_max_drawdown(nav):
    """按净值曲线计算最大回撤金额和比例。"""
    running_max = nav.cummax()
    drawdown = nav - running_max
    drawdown_pct = nav / running_max - 1
    return drawdown.min(), drawdown_pct.min()


def summarize_result(param_set, daily_pnl, trades, initial_cash):
    final_nav = daily_pnl["nav"].iloc[-1]
    total_return = final_nav / initial_cash - 1
    max_drawdown, max_drawdown_pct = calc_max_drawdown(daily_pnl["nav"])

    if trades.empty or "type" not in trades.columns:
        open_count = 0
        roll_count = 0
        close_count = 0
    else:
        trade_type = trades["type"].astype(str)
        open_count = int(trade_type.eq("open_straddle").sum())
        roll_count = int(trade_type.eq("roll_open_straddle").sum())
        close_count = int(trade_type.str.contains("close_straddle", na=False).sum())

    row = {
        **param_set,
        "final_nav": final_nav,
        "total_pnl": final_nav - initial_cash,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "open_count": open_count,
        "roll_count": roll_count,
        "close_count": close_count,
        "held_days": int(daily_pnl["eod_has_position"].sum()),
        "trading_days": len(daily_pnl),
    }
    return row


def make_feature_cache_key(param_set):
    """只有 vol.* 参数会影响 features，其余策略参数可以复用同一份特征。"""
    return tuple(
        sorted((key, value) for key, value in param_set.items() if key.startswith("vol."))
    )


def make_enriched_chain_cache_key(param_set):
    """只有这些参数会改变单条期权链的 IV/Greeks 计算结果。"""
    chain_param_names = {
        "vol.annual_days",
        "vol.risk_free_rate",
        "vol.dividend_yield",
    }
    return tuple(
        sorted((key, value) for key, value in param_set.items() if key in chain_param_names)
    )


def run_one_param_set(
    base_config,
    param_set,
    etf_by_date,
    opt_by_date,
    trading_calendar,
    feature_cache,
    enriched_chain_cache,
):
    config = apply_param_set(base_config, param_set)
    enriched_cache_key = make_enriched_chain_cache_key(param_set)
    if enriched_cache_key not in enriched_chain_cache:
        enriched_chain_cache[enriched_cache_key] = (
            core.vol_engine.build_enriched_option_chains(
                etf_by_date,
                opt_by_date,
                trading_calendar=trading_calendar,
            )
        )
    enriched_opt_by_date = enriched_chain_cache[enriched_cache_key]

    feature_cache_key = make_feature_cache_key(param_set)
    if feature_cache_key not in feature_cache:
        feature_cache[feature_cache_key] = core.vol_engine.build_vol_features(
            etf_by_date,
            opt_by_date,
            trading_calendar=trading_calendar,
            enriched_opt_by_date=enriched_opt_by_date,
        )
    features = feature_cache[feature_cache_key]

    signals = core.strategy.build_signals(features)
    daily_pnl, trades = core.backtester.run_backtest(
        etf_by_date,
        opt_by_date,
        signals,
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
    )
    return summarize_result(param_set, daily_pnl, trades, config.backtest.initial_cash)


def main():
    args = parse_args()
    base_config = core.config.CONFIG
    output_path = make_output_path(args.output)

    print("加载数据...")
    etf_by_date = core.data_loader.load_etf_series(args.start, args.end)
    opt_by_date = core.data_loader.load_opt_series(args.start, args.end)
    trading_calendar = core.data_loader.load_etf_trading_calendar()

    param_sets = list(iter_param_sets(PARAM_GRID))
    if args.max_combinations is not None:
        param_sets = param_sets[: args.max_combinations]

    print(f"参数组合数: {len(param_sets)}")
    print(f"结果输出: {output_path}")

    rows = []
    feature_cache = {}
    enriched_chain_cache = {}
    for idx, param_set in enumerate(param_sets, start=1):
        print(f"[{idx}/{len(param_sets)}] {param_set}")
        try:
            row = run_one_param_set(
                base_config,
                param_set,
                etf_by_date,
                opt_by_date,
                trading_calendar,
                feature_cache,
                enriched_chain_cache,
            )
            row["error"] = ""
        except Exception as exc:
            row = {**param_set, "error": repr(exc)}
            print(f"  失败: {exc!r}")

        rows.append(row)
        result_df = pd.DataFrame(rows)
        if "final_nav" in result_df.columns:
            result_df = result_df.sort_values("final_nav", ascending=False)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        if row.get("error") == "":
            print(
                "  NAV="
                f"{row['final_nav']:,.2f}, "
                f"收益率={row['total_return']:.2%}, "
                f"最大回撤={row['max_drawdown_pct']:.2%}, "
                f"开仓={row['open_count']}, roll={row['roll_count']}"
            )

    result_df = pd.DataFrame(rows)
    if "final_nav" in result_df.columns:
        result_df = result_df.sort_values("final_nav", ascending=False)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n=== NAV 前 10 参数 ===")
    display_cols = list(PARAM_GRID) + [
        "final_nav",
        "total_return",
        "max_drawdown_pct",
        "open_count",
        "roll_count",
        "held_days",
        "error",
    ]
    display_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
