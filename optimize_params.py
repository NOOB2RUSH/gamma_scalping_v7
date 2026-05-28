from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

import core


# 本脚本只扫描 ATM IV 单阈值信号的四个核心参数：
# - long_open_iv_threshold: ATM IV 低于该值时允许买入跨式
# - long_close_iv_threshold: ATM IV 高于该值时平掉买入跨式
# - short_open_iv_percentile_threshold: ATM IV 百分位高于该值时允许卖出跨式
# - short_close_iv_percentile_threshold: ATM IV 百分位低于该值时平掉卖出跨式
SCAN_PARAM_NAMES = [
    "strategy.long_open_iv_threshold",
    "strategy.long_close_iv_threshold",
    "strategy.short_open_iv_percentile_threshold",
    "strategy.short_close_iv_percentile_threshold",
]

DIRECTION_PARAM_NAMES = {
    "both": SCAN_PARAM_NAMES,
    "long": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
    ],
    "short": [
        "strategy.short_open_iv_percentile_threshold",
        "strategy.short_close_iv_percentile_threshold",
    ],
    "experiment": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "backtest.long_qty",
        "backtest.short_qty",
    ],
    "capital": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "backtest.long_qty",
        "backtest.short_qty",
    ],
    "zz1000_simple": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "backtest.long_qty",
        "backtest.short_qty",
    ],
    "zz1000_wide": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
    ],
    "zz1000_return_fine": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
    ],
    "500etf_wide": [
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "strategy.short_open_pullback_iv_threshold",
        "backtest.short_qty",
    ],
    "500etf_both_wide": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "backtest.long_qty",
        "backtest.short_qty",
    ],
    "500etf_both_delta": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "strategy.short_open_pullback_iv_threshold",
        "backtest.long_qty",
        "backtest.short_qty",
    ],
    "500etf_full_defense": [
        "strategy.long_open_iv_threshold",
        "strategy.long_close_iv_threshold",
        "strategy.short_open_iv_threshold",
        "strategy.short_close_iv_threshold",
        "strategy.short_open_pullback_iv_threshold",
        "strategy.roll_cooldown_days",
    ],
}


# 粗扫：覆盖当前配置附近的主要区间，先找大致有效区域。
COARSE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.long_open_iv_threshold": [0.115, 0.125, 0.135, 0.145, 0.155],
    "strategy.long_close_iv_threshold": [0.20, 0.23, 0.26, 0.29, 0.32],
    "strategy.short_open_iv_percentile_threshold": [0.60, 0.70, 0.80, 0.90],
    "strategy.short_close_iv_percentile_threshold": [0.10, 0.20, 0.30, 0.40],
}

SHORT_COARSE_PARAM_GRID = {
    "strategy.enable_long_straddle": [False],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["percentile"],
    "strategy.enable_delta_hedge": [False],
    "strategy.short_stop_loss_enabled": [False],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_open_iv_percentile_threshold": [
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ],
    "strategy.short_close_iv_percentile_threshold": [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
    ],
}

LONG_COARSE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [False],
    "strategy.long_open_iv_threshold": [0.115, 0.125, 0.135, 0.145, 0.155],
    "strategy.long_close_iv_threshold": [0.20, 0.23, 0.26, 0.29, 0.32],
}

EXPERIMENT_COARSE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [False],
    "strategy.short_stop_loss_enabled": [False],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.long_open_iv_threshold": [0.120, 0.130, 0.140],
    "strategy.long_close_iv_threshold": [0.220, 0.250, 0.280],
    "strategy.short_open_iv_threshold": [0.150, 0.160, 0.170, 0.180],
    "strategy.short_close_iv_threshold": [0.120, 0.135, 0.150],
    "backtest.long_qty": [15, 25],
    "backtest.short_qty": [25, 30, 35],
}

CAPITAL_COARSE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [False],
    "strategy.short_stop_loss_enabled": [False],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.long_open_iv_threshold": [0.140, 0.145, 0.150],
    "strategy.long_close_iv_threshold": [0.295, 0.310, 0.325],
    "strategy.short_open_iv_threshold": [0.155, 0.160, 0.165],
    "strategy.short_close_iv_threshold": [0.110, 0.115, 0.120],
    "backtest.long_qty": [25, 30, 35],
    "backtest.short_qty": [35, 45, 55, 65],
}

ZZ1000_SIMPLE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [False],
    "strategy.short_stop_loss_enabled": [False],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.long_open_iv_threshold": [0.14, 0.15],
    "strategy.long_close_iv_threshold": [0.26, 0.30],
    "strategy.short_open_iv_threshold": [0.24, 0.28, 0.32],
    "strategy.short_close_iv_threshold": [0.16, 0.20],
    "backtest.long_qty": [1, 2],
    "backtest.short_qty": [1, 2, 3, 4],
}

ZZ1000_WIDE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [False],
    "strategy.short_stop_loss_enabled": [True],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "backtest.long_qty": [1],
    "backtest.short_qty": [1],
    "strategy.long_open_iv_threshold": [0.13, 0.14, 0.15, 0.16, 0.18],
    "strategy.long_close_iv_threshold": [0.20, 0.24, 0.28, 0.32, 0.38],
    "strategy.short_open_iv_threshold": [0.26, 0.30, 0.34, 0.40, 0.50],
    "strategy.short_close_iv_threshold": [0.12, 0.16, 0.20, 0.24, 0.28],
}

ZZ1000_RETURN_FINE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [False],
    "strategy.short_stop_loss_enabled": [True],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "backtest.long_qty": [1],
    "backtest.short_qty": [1],
    "strategy.long_open_iv_threshold": [0.130, 0.135, 0.140, 0.145, 0.150],
    "strategy.long_close_iv_threshold": [0.180, 0.190, 0.200, 0.210, 0.220],
    "strategy.short_open_iv_threshold": [0.240, 0.250, 0.260, 0.270, 0.280, 0.300],
    "strategy.short_close_iv_threshold": [0.100, 0.120, 0.140, 0.160, 0.180, 0.200, 0.220],
}


# 细扫：围绕粗扫前几名展开邻域。
ETF500_WIDE_PARAM_GRID = {
    "strategy.enable_long_straddle": [False],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [True],
    "strategy.short_stop_loss_enabled": [True],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.short_open_iv_threshold": [0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.34],
    "strategy.short_close_iv_threshold": [0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
    "strategy.short_open_pullback_iv_threshold": [0.30, 0.34, 0.38],
    "backtest.short_qty": [5, 10, 15, 20, 25],
}

ETF500_BOTH_WIDE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [True],
    "strategy.short_stop_loss_enabled": [True],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.long_open_iv_threshold": [0.12, 0.14, 0.16, 0.18],
    "strategy.long_close_iv_threshold": [0.24, 0.30, 0.36],
    "strategy.short_open_iv_threshold": [0.22, 0.25, 0.28, 0.32],
    "strategy.short_close_iv_threshold": [0.14, 0.17, 0.20],
    "strategy.short_open_pullback_iv_threshold": [0.34],
    "backtest.long_qty": [5, 10, 15],
    "backtest.short_qty": [5, 10, 15, 20],
}

ETF500_BOTH_DELTA_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [True],
    "strategy.short_stop_loss_enabled": [True],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.long_open_iv_threshold": [0.13, 0.135, 0.14, 0.145],
    "strategy.long_close_iv_threshold": [0.22, 0.24, 0.28, 0.32],
    "strategy.short_open_iv_threshold": [0.28, 0.29, 0.30, 0.32],
    "strategy.short_close_iv_threshold": [0.20, 0.22, 0.225, 0.24],
    "strategy.short_open_pullback_iv_threshold": [0.36, 0.38, 0.42, 0.44],
    "backtest.long_qty": [5, 10, 15],
    "backtest.short_qty": [5, 10, 15],
}

ETF500_FULL_DEFENSE_PARAM_GRID = {
    "strategy.enable_long_straddle": [True],
    "strategy.enable_short_straddle": [True],
    "strategy.short_signal_mode": ["absolute"],
    "strategy.enable_delta_hedge": [True],
    "strategy.short_stop_loss_enabled": [True],
    "strategy.short_volume_spike_exit_enabled": [True],
    "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
    "strategy.long_open_iv_threshold": [0.115, 0.125, 0.135, 0.145, 0.155],
    "strategy.long_close_iv_threshold": [0.20, 0.24, 0.28, 0.32, 0.36],
    "strategy.short_open_iv_threshold": [0.20, 0.22, 0.24, 0.26, 0.28, 0.30],
    "strategy.short_close_iv_threshold": [0.12, 0.15, 0.18, 0.20, 0.225, 0.24],
    "strategy.short_open_pullback_iv_threshold": [0.26, 0.30, 0.34, 0.38, 0.42, 0.46],
    "strategy.roll_cooldown_days": [1, 2, 3, 4, 5, 7],
}

FINE_OFFSETS = {
    "strategy.long_open_iv_threshold": [-0.005, 0.0, 0.005],
    "strategy.long_close_iv_threshold": [-0.015, 0.0, 0.015],
    "strategy.short_open_iv_threshold": [-0.005, 0.0, 0.005],
    "strategy.short_close_iv_threshold": [-0.005, 0.0, 0.005],
    "strategy.short_open_pullback_iv_threshold": [-0.02, 0.0, 0.02],
    "strategy.roll_cooldown_days": [-1, 0, 1],
    "strategy.short_open_iv_percentile_threshold": [-0.025, 0.0, 0.025],
    "strategy.short_close_iv_percentile_threshold": [-0.025, 0.0, 0.025],
    "backtest.long_qty": [-5, 0, 5],
    "backtest.short_qty": [-5, 0, 5],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="分阶段扫描 long/short ATM IV 开平仓阈值，并按 NAV/MaxDD 排序输出结果。"
    )
    parser.add_argument(
        "--product",
        choices=core.config.available_products(),
        default=core.config.CONFIG.data.product,
        help="交易品种配置，默认使用 50ETF。",
    )
    parser.add_argument(
        "--stage",
        choices=["coarse", "fine", "all"],
        default="coarse",
        help="扫描阶段：coarse 粗扫；fine 读取粗扫结果后细扫；all 先粗扫再细扫。",
    )
    parser.add_argument(
        "--direction",
        choices=[
            "both",
            "long",
            "short",
            "experiment",
            "capital",
            "zz1000_simple",
            "zz1000_wide",
            "zz1000_return_fine",
            "500etf_wide",
            "500etf_both_wide",
            "500etf_both_delta",
            "500etf_full_defense",
        ],
        default="both",
        help="扫描方向：short 会关闭 long，只调卖出跨式；long 会关闭 short；both 同时调两套阈值。",
    )
    parser.add_argument(
        "--objective",
        choices=["nav_drawdown", "nav_sharpe", "final_nav", "sharpe"],
        default="nav_drawdown",
        help="排序目标：默认 NAV/MaxDD；nav_sharpe 用于同时关注最终净值和夏普。",
    )
    parser.add_argument(
        "--coarse-result",
        default=None,
        help="细扫使用的粗扫 CSV。stage=fine 时建议传入。未传入则围绕 config 当前值细扫。",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="细扫时读取粗扫 NAV/MaxDD 前 N 组参数展开邻域。",
    )
    parser.add_argument(
        "--min-held-days",
        type=int,
        default=0,
        help="细扫选取粗扫种子时要求的最低持仓天数；为 0 时不限制。",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="回测开始日期，默认读取 core/config.py。",
    )
    parser.add_argument(
        "--end",
        default=None,
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
        help="结果 CSV 路径；默认写入 output/optimize_阶段_时间戳.csv。",
    )
    return parser.parse_args()


def unique_sorted(values):
    return sorted(set(round(float(value), 6) for value in values))


def build_fine_values(key, center):
    values = unique_sorted(center + offset for offset in FINE_OFFSETS[key])
    if key in {
        "strategy.roll_cooldown_days",
        "strategy.short_cooldown_after_long_iv_high_exit_days",
    }:
        return sorted({int(value) for value in values if int(value) >= 0})
    if key == "backtest.short_qty":
        return sorted({int(value) for value in values if int(value) > 0})
    if key.startswith("backtest."):
        return sorted({int(value) for value in values if int(value) > 0})
    return values


def iter_param_sets(param_grid):
    """展开参数网格，并过滤明显不合逻辑的组合。"""
    keys = list(param_grid)
    value_lists = [param_grid[key] for key in keys]

    for values in itertools.product(*value_lists):
        param_set = dict(zip(keys, values))
        if is_valid_param_set(param_set):
            yield param_set


def is_valid_param_set(param_set):
    """过滤开平仓阈值倒挂的组合。"""
    long_open = param_set.get("strategy.long_open_iv_threshold")
    long_close = param_set.get("strategy.long_close_iv_threshold")
    short_open = param_set.get("strategy.short_open_iv_percentile_threshold")
    short_close = param_set.get("strategy.short_close_iv_percentile_threshold")
    short_open_abs = param_set.get("strategy.short_open_iv_threshold")
    short_close_abs = param_set.get("strategy.short_close_iv_threshold")
    short_pullback_abs = param_set.get("strategy.short_open_pullback_iv_threshold")

    if long_open is not None and long_close is not None and long_open >= long_close:
        return False
    if (
        short_open is not None
        and short_close is not None
        and short_open <= short_close
    ):
        return False
    if (
        short_open_abs is not None
        and short_close_abs is not None
        and short_open_abs <= short_close_abs
    ):
        return False
    if (
        short_pullback_abs is not None
        and short_open_abs is not None
        and short_pullback_abs <= short_open_abs
    ):
        return False
    for percentile_value in (short_open, short_close):
        if percentile_value is not None and not 0 <= percentile_value <= 1:
            return False
    cooldown_days = param_set.get("strategy.roll_cooldown_days")
    if cooldown_days is not None and int(cooldown_days) < 0:
        return False
    for qty_key in ("backtest.long_qty", "backtest.short_qty"):
        qty = param_set.get(qty_key)
        if qty is not None and int(qty) <= 0:
            return False
    return True


def get_scan_param_names(direction):
    return DIRECTION_PARAM_NAMES[direction]


def get_coarse_param_grid(direction):
    if direction == "500etf_full_defense":
        return ETF500_FULL_DEFENSE_PARAM_GRID
    if direction == "500etf_wide":
        return ETF500_WIDE_PARAM_GRID
    if direction == "500etf_both_wide":
        return ETF500_BOTH_WIDE_PARAM_GRID
    if direction == "500etf_both_delta":
        return ETF500_BOTH_DELTA_PARAM_GRID
    if direction == "zz1000_return_fine":
        return ZZ1000_RETURN_FINE_PARAM_GRID
    if direction == "zz1000_wide":
        return ZZ1000_WIDE_PARAM_GRID
    if direction == "zz1000_simple":
        return ZZ1000_SIMPLE_PARAM_GRID
    if direction == "capital":
        return CAPITAL_COARSE_PARAM_GRID
    if direction == "experiment":
        return EXPERIMENT_COARSE_PARAM_GRID
    if direction == "short":
        return SHORT_COARSE_PARAM_GRID
    if direction == "long":
        return LONG_COARSE_PARAM_GRID
    return COARSE_PARAM_GRID


def get_direction_switches(direction):
    if direction == "500etf_full_defense":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [True],
            "strategy.short_stop_loss_enabled": [True],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
        }
    if direction == "500etf_wide":
        return {
            "strategy.enable_long_straddle": [False],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [True],
            "strategy.short_stop_loss_enabled": [True],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
        }
    if direction == "500etf_both_wide":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [True],
            "strategy.short_stop_loss_enabled": [True],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
            "strategy.short_open_pullback_iv_threshold": [0.34],
        }
    if direction == "500etf_both_delta":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [True],
            "strategy.short_stop_loss_enabled": [True],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
        }
    if direction == "zz1000_return_fine":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [False],
            "strategy.short_stop_loss_enabled": [True],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
            "backtest.long_qty": [1],
            "backtest.short_qty": [1],
        }
    if direction == "zz1000_wide":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [False],
            "strategy.short_stop_loss_enabled": [True],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
            "backtest.long_qty": [1],
            "backtest.short_qty": [1],
        }
    if direction == "zz1000_simple":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [False],
            "strategy.short_stop_loss_enabled": [False],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
        }
    if direction == "capital":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [False],
            "strategy.short_stop_loss_enabled": [False],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
        }
    if direction == "experiment":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["absolute"],
            "strategy.enable_delta_hedge": [False],
            "strategy.short_stop_loss_enabled": [False],
            "strategy.short_volume_spike_exit_enabled": [True],
            "strategy.short_cooldown_after_long_iv_high_exit_days": [3],
        }
    if direction == "short":
        return {
            "strategy.enable_long_straddle": [False],
            "strategy.enable_short_straddle": [True],
            "strategy.short_signal_mode": ["percentile"],
            "strategy.enable_delta_hedge": [False],
            "strategy.short_stop_loss_enabled": [False],
            "strategy.short_volume_spike_exit_enabled": [True],
        }
    if direction == "long":
        return {
            "strategy.enable_long_straddle": [True],
            "strategy.enable_short_straddle": [False],
        }
    return {
        "strategy.enable_long_straddle": [True],
        "strategy.enable_short_straddle": [True],
    }


def build_fine_param_sets_from_rows(rows, direction):
    """围绕粗扫优秀参数生成细扫组合。"""
    param_sets = []
    seen = set()
    scan_param_names = get_scan_param_names(direction)

    for _, row in rows.iterrows():
        fine_grid = get_direction_switches(direction)
        for key in scan_param_names:
            center = float(row[key])
            fine_grid[key] = build_fine_values(key, center)

        for param_set in iter_param_sets(fine_grid):
            identity = tuple((key, param_set[key]) for key in sorted(param_set))
            if identity in seen:
                continue
            seen.add(identity)
            param_sets.append(param_set)

    return param_sets


def build_fine_param_sets_from_config(base_config, direction):
    row = {
        "strategy.long_open_iv_threshold": base_config.strategy.long_open_iv_threshold,
        "strategy.long_close_iv_threshold": base_config.strategy.long_close_iv_threshold,
        "strategy.short_open_iv_percentile_threshold": (
            base_config.strategy.short_open_iv_percentile_threshold
        ),
        "strategy.short_close_iv_percentile_threshold": (
            base_config.strategy.short_close_iv_percentile_threshold
        ),
        "strategy.short_open_iv_threshold": base_config.strategy.short_open_iv_threshold,
        "strategy.short_close_iv_threshold": base_config.strategy.short_close_iv_threshold,
        "strategy.short_open_pullback_iv_threshold": (
            base_config.strategy.short_open_pullback_iv_threshold
        ),
        "strategy.roll_cooldown_days": base_config.strategy.roll_cooldown_days,
        "vol.atm_iv_percentile_window": base_config.vol.atm_iv_percentile_window,
        "backtest.long_qty": base_config.backtest.long_qty,
        "backtest.short_qty": base_config.backtest.short_qty,
    }
    return build_fine_param_sets_from_rows(pd.DataFrame([row]), direction)


def load_fine_param_sets(
    base_config,
    coarse_result_path,
    top_n,
    direction,
    objective,
    min_held_days,
):
    if coarse_result_path is None:
        print("未传入粗扫结果，细扫将围绕 core/config.py 当前阈值展开。", flush=True)
        return build_fine_param_sets_from_config(base_config, direction)

    coarse_df = pd.read_csv(coarse_result_path)
    coarse_df = ensure_nav_drawdown_score(coarse_df)
    coarse_df = ensure_nav_sharpe_score(coarse_df)
    coarse_df = ensure_cash_usage_metrics(coarse_df, base_config.backtest.initial_cash)
    scan_param_names = get_scan_param_names(direction)
    missing_cols = [col for col in scan_param_names + ["final_nav"] if col not in coarse_df]
    if missing_cols:
        raise ValueError(f"粗扫结果缺少必要列: {missing_cols}")

    valid_df = coarse_df[coarse_df.get("error", "").fillna("").eq("")]
    valid_df = filter_by_min_held_days(valid_df, min_held_days)
    top_rows = sort_result_df(valid_df, objective).head(top_n)
    if top_rows.empty:
        raise ValueError("粗扫结果中没有可用于细扫的成功回测。")
    return build_fine_param_sets_from_rows(top_rows, direction)


def apply_param_set(base_config, param_set):
    """生成新配置，并同步到各模块的 CONFIG 引用。"""
    strategy_updates = {}
    vol_updates = {}
    backtest_updates = {}

    for key, value in param_set.items():
        section, field = key.split(".", 1)
        if section == "strategy":
            if field in {
                "roll_cooldown_days",
                "short_cooldown_after_long_iv_high_exit_days",
            }:
                value = int(value)
            strategy_updates[field] = value
        elif section == "vol":
            if field == "atm_iv_percentile_window":
                value = int(value)
            vol_updates[field] = value
        elif section == "backtest":
            if field in {"long_qty", "short_qty"}:
                value = int(value)
            backtest_updates[field] = value
        else:
            raise ValueError(f"未知参数分组: {key}")

    config = replace(
        base_config,
        strategy=replace(base_config.strategy, **strategy_updates),
        vol=replace(base_config.vol, **vol_updates),
        backtest=replace(base_config.backtest, **backtest_updates),
    )

    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    core.strategy.CONFIG = config
    core.backtester.CONFIG = config
    core.position.CONFIG = config
    core.hedge.CONFIG = config
    if hasattr(core, "analytics"):
        core.analytics.CONFIG = config
    return config


def make_output_path(output_arg, stage):
    if output_arg:
        output_path = Path(output_arg)
    else:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(core.config.CONFIG.report.output_root) / (
            f"optimize_{stage}_{timestamp}.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def calc_max_drawdown(nav):
    running_max = nav.cummax()
    drawdown = nav - running_max
    drawdown_pct = nav / running_max - 1
    return drawdown.min(), drawdown_pct.min()


def calc_nav_drawdown_score(final_nav, max_drawdown_pct):
    maxdd_abs = abs(max_drawdown_pct)
    if pd.isna(maxdd_abs) or maxdd_abs == 0:
        return pd.NA
    return final_nav / maxdd_abs


def calc_sharpe_ratio(nav):
    daily_return = nav.pct_change().dropna()
    if daily_return.empty:
        return pd.NA
    std = daily_return.std()
    if pd.isna(std) or std == 0:
        return pd.NA
    annual_days = getattr(core.config.CONFIG.vol, "annual_days", 252)
    return daily_return.mean() / std * (annual_days ** 0.5)


def calc_sortino_ratio(nav):
    daily_return = nav.pct_change().dropna()
    if daily_return.empty:
        return pd.NA
    downside = daily_return[daily_return < 0]
    downside_std = downside.std()
    if pd.isna(downside_std) or downside_std == 0:
        return pd.NA
    annual_days = getattr(core.config.CONFIG.vol, "annual_days", 252)
    return daily_return.mean() / downside_std * (annual_days ** 0.5)


def calc_annual_return(final_nav, initial_cash, trading_days):
    if trading_days <= 0 or initial_cash <= 0:
        return pd.NA
    annual_days = getattr(core.config.CONFIG.vol, "annual_days", 252)
    return (final_nav / initial_cash) ** (annual_days / trading_days) - 1


def calc_ulcer_index(nav):
    running_max = nav.cummax()
    drawdown_pct = nav / running_max - 1
    return ((drawdown_pct.pow(2).mean()) ** 0.5)


def calc_nav_trend_r2(nav):
    if len(nav) < 2:
        return pd.NA
    y = nav.reset_index(drop=True).astype(float)
    x = pd.Series(range(len(y)), dtype="float64")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    ss_x = (x_centered ** 2).sum()
    ss_y = (y_centered ** 2).sum()
    if ss_x == 0 or ss_y == 0:
        return pd.NA
    slope = (x_centered * y_centered).sum() / ss_x
    fitted = y.mean() + slope * x_centered
    ss_res = ((y - fitted) ** 2).sum()
    return 1 - ss_res / ss_y


def calc_pnl_concentration(daily_pnl, total_pnl):
    pnl = pd.to_numeric(daily_pnl, errors="coerce").fillna(0.0)
    positive = pnl[pnl > 0].sort_values(ascending=False)
    positive_total = positive.sum()
    if positive_total <= 0:
        top5_share = pd.NA
        top10_share = pd.NA
        hhi = pd.NA
    else:
        top5_share = positive.head(5).sum() / positive_total
        top10_share = positive.head(10).sum() / positive_total
        weights = positive / positive_total
        hhi = (weights ** 2).sum()

    max_gain = positive.iloc[0] if not positive.empty else 0.0
    max_gain_share_total_pnl = (
        max_gain / total_pnl if total_pnl > 0 else pd.NA
    )
    positive_day_ratio = (pnl > 0).mean() if len(pnl) > 0 else pd.NA
    return {
        "top5_profit_share": top5_share,
        "top10_profit_share": top10_share,
        "profit_hhi": hhi,
        "max_gain_share_total_pnl": max_gain_share_total_pnl,
        "positive_day_ratio": positive_day_ratio,
    }


def calc_nav_sharpe_score(final_nav, sharpe_ratio):
    if pd.isna(sharpe_ratio):
        return pd.NA
    return final_nav * sharpe_ratio


def sort_result_df(result_df, objective="nav_drawdown"):
    if result_df.empty or "nav_drawdown_score" not in result_df.columns:
        return result_df
    if "cash_negative_days" not in result_df.columns:
        result_df = result_df.copy()
        result_df["cash_negative_days"] = 0
    if objective == "nav_sharpe":
        return result_df.sort_values(
            ["cash_negative_days", "nav_sharpe_score", "final_nav", "sharpe_ratio"],
            ascending=[True, False, False, False],
            na_position="last",
        )
    if objective == "final_nav":
        return result_df.sort_values(
            ["cash_negative_days", "final_nav", "sharpe_ratio"],
            ascending=[True, False, False],
            na_position="last",
        )
    if objective == "sharpe":
        return result_df.sort_values(
            ["cash_negative_days", "sharpe_ratio", "final_nav"],
            ascending=[True, False, False],
            na_position="last",
        )
    return result_df.sort_values(
        ["cash_negative_days", "nav_drawdown_score", "final_nav"],
        ascending=[True, False, False],
        na_position="last",
    )


def ensure_nav_drawdown_score(df):
    if (
        "nav_drawdown_score" not in df.columns
        and {"final_nav", "max_drawdown_pct"}.issubset(df.columns)
    ):
        df = df.copy()
        maxdd_abs = df["max_drawdown_pct"].abs()
        df["nav_drawdown_score"] = df["final_nav"] / maxdd_abs
        df.loc[maxdd_abs.eq(0) | maxdd_abs.isna(), "nav_drawdown_score"] = pd.NA
    return df


def ensure_nav_sharpe_score(df):
    if (
        "nav_sharpe_score" not in df.columns
        and {"final_nav", "sharpe_ratio"}.issubset(df.columns)
    ):
        df = df.copy()
        df["nav_sharpe_score"] = df["final_nav"] * df["sharpe_ratio"]
        df.loc[df["sharpe_ratio"].isna(), "nav_sharpe_score"] = pd.NA
    return df


def ensure_cash_usage_metrics(df, initial_cash=None):
    if df.empty or "min_cash" not in df.columns:
        return df
    df = df.copy()
    if initial_cash is None:
        initial_cash = core.config.CONFIG.backtest.initial_cash
    df["min_cash_ratio"] = df["min_cash"] / initial_cash
    df["cash_usage_ratio"] = 1 - df["min_cash_ratio"]
    return df


def filter_by_min_held_days(df, min_held_days):
    if min_held_days <= 0 or "held_days" not in df.columns:
        return df
    filtered = df[df["held_days"].fillna(0) >= min_held_days]
    if filtered.empty:
        print(
            f"没有满足 held_days >= {min_held_days} 的组合，退回不加持仓天数过滤。",
            flush=True,
        )
        return df
    return filtered


def summarize_result(param_set, daily_pnl, trades, initial_cash):
    final_nav = daily_pnl["nav"].iloc[-1]
    total_pnl = final_nav - initial_cash
    total_return = final_nav / initial_cash - 1
    max_drawdown, max_drawdown_pct = calc_max_drawdown(daily_pnl["nav"])
    nav_drawdown_score = calc_nav_drawdown_score(final_nav, max_drawdown_pct)
    sharpe_ratio = calc_sharpe_ratio(daily_pnl["nav"])
    sortino_ratio = calc_sortino_ratio(daily_pnl["nav"])
    nav_sharpe_score = calc_nav_sharpe_score(final_nav, sharpe_ratio)
    annual_return = calc_annual_return(final_nav, initial_cash, len(daily_pnl))
    calmar_ratio = (
        annual_return / abs(max_drawdown_pct)
        if not pd.isna(annual_return) and max_drawdown_pct != 0
        else pd.NA
    )
    ulcer_index = calc_ulcer_index(daily_pnl["nav"])
    nav_trend_r2 = calc_nav_trend_r2(daily_pnl["nav"])
    pnl_col = (
        daily_pnl["daily_nav_pnl"]
        if "daily_nav_pnl" in daily_pnl.columns
        else daily_pnl["nav"].diff()
    )
    concentration = calc_pnl_concentration(pnl_col, total_pnl)

    if trades.empty or "type" not in trades.columns:
        open_count = 0
        roll_count = 0
        close_count = 0
    else:
        trade_type = trades["type"].astype(str)
        open_count = int(trade_type.isin(["open_straddle", "open_short_straddle"]).sum())
        roll_count = int(trade_type.eq("roll_open_straddle").sum())
        close_count = int(trade_type.str.contains("close_straddle", na=False).sum())

    held_mask = daily_pnl["eod_has_position"]
    avg_call_qty = (
        daily_pnl.loc[held_mask, "eod_position_call_qty"].mean()
        if held_mask.any()
        else 0.0
    )
    long_held_mask = daily_pnl["long_has_position"]
    short_held_mask = daily_pnl["short_has_position"]
    avg_long_qty = (
        daily_pnl.loc[long_held_mask, "long_position_call_qty"].mean()
        if long_held_mask.any()
        else 0.0
    )
    avg_short_qty = (
        daily_pnl.loc[short_held_mask, "short_position_call_qty"].mean()
        if short_held_mask.any()
        else 0.0
    )
    cash_negative_days = (
        int(daily_pnl["cash_negative_warning"].sum())
        if "cash_negative_warning" in daily_pnl.columns
        else int((daily_pnl["cash"] < 0).sum())
    )

    return {
        **param_set,
        "final_nav": final_nav,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "nav_drawdown_score": nav_drawdown_score,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "nav_sharpe_score": nav_sharpe_score,
        "ulcer_index": ulcer_index,
        "nav_trend_r2": nav_trend_r2,
        **concentration,
        "min_cash": daily_pnl["cash"].min(),
        "min_cash_ratio": daily_pnl["cash"].min() / initial_cash,
        "cash_usage_ratio": 1 - daily_pnl["cash"].min() / initial_cash,
        "cash_negative_days": cash_negative_days,
        "open_count": open_count,
        "roll_count": roll_count,
        "close_count": close_count,
        "held_days": int(held_mask.sum()),
        "avg_call_qty_when_held": avg_call_qty,
        "long_held_days": int(long_held_mask.sum()),
        "short_held_days": int(short_held_mask.sum()),
        "avg_long_qty_when_held": avg_long_qty,
        "avg_short_qty_when_held": avg_short_qty,
        "trading_days": len(daily_pnl),
    }


def make_feature_cache_key(param_set):
    """只有 vol.* 参数会改变 features；本轮阈值扫描会复用同一份特征。"""
    return tuple(
        sorted((key, value) for key, value in param_set.items() if key.startswith("vol."))
    )


def make_enriched_chain_cache_key(param_set):
    """只有这些参数会改变单日期权链的 IV/Greeks 计算结果。"""
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
    start,
    end,
):
    config = apply_param_set(base_config, param_set)
    enriched_cache_key = make_enriched_chain_cache_key(param_set)
    if enriched_cache_key not in enriched_chain_cache:
        enriched_chain_cache[enriched_cache_key] = core.cache.get_enriched_option_chains(
            etf_by_date,
            opt_by_date,
            trading_calendar,
            start,
            end,
        )
    enriched_opt_by_date = enriched_chain_cache[enriched_cache_key]

    feature_cache_key = make_feature_cache_key(param_set)
    if feature_cache_key not in feature_cache:
        feature_cache[feature_cache_key] = core.cache.get_vol_features(
            etf_by_date,
            opt_by_date,
            trading_calendar,
            enriched_opt_by_date,
            start,
            end,
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


def run_scan(
    stage_name,
    param_sets,
    output_path,
    base_config,
    etf_by_date,
    opt_by_date,
    trading_calendar,
    start,
    end,
    direction,
    objective,
):
    print(f"\n===== {stage_name} =====", flush=True)
    print(f"参数组合数: {len(param_sets)}", flush=True)
    print(f"排序目标: {objective}", flush=True)
    print(f"结果输出: {output_path}", flush=True)

    rows = []
    feature_cache = {}
    enriched_chain_cache = {}
    scan_start_time = time.perf_counter()

    for idx, param_set in enumerate(param_sets, start=1):
        case_start_time = time.perf_counter()
        print(f"[{idx}/{len(param_sets)}] {param_set}", flush=True)
        try:
            row = run_one_param_set(
                base_config,
                param_set,
                etf_by_date,
                opt_by_date,
                trading_calendar,
                feature_cache,
                enriched_chain_cache,
                start,
                end,
            )
            row["error"] = ""
        except Exception as exc:
            row = {**param_set, "error": repr(exc)}
            print(f"  失败: {exc!r}", flush=True)

        rows.append(row)
        result_df = pd.DataFrame(rows)
        result_df = ensure_nav_drawdown_score(result_df)
        result_df = ensure_nav_sharpe_score(result_df)
        result_df = ensure_cash_usage_metrics(result_df, base_config.backtest.initial_cash)
        result_df = sort_result_df(result_df, objective)
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        if row.get("error") == "":
            elapsed = time.perf_counter() - scan_start_time
            case_seconds = time.perf_counter() - case_start_time
            avg_seconds = elapsed / idx
            remaining_seconds = avg_seconds * (len(param_sets) - idx)
            print(
                "  NAV="
                f"{row['final_nav']:,.2f}, "
                f"收益率={row['total_return']:.2%}, "
                f"夏普={row['sharpe_ratio']:.2f}, "
                f"最大回撤={row['max_drawdown_pct']:.2%}, "
                f"开仓={row['open_count']}, "
                f"roll={row['roll_count']}, "
                f"持仓天数={row['held_days']}, "
                f"平均张数={row['avg_call_qty_when_held']:.1f}, "
                f"本组耗时={case_seconds:.1f}s, "
                f"预计剩余={remaining_seconds / 60:.1f}min",
                flush=True,
            )

    result_df = pd.DataFrame(rows)
    result_df = ensure_nav_drawdown_score(result_df)
    result_df = ensure_nav_sharpe_score(result_df)
    result_df = ensure_cash_usage_metrics(result_df, base_config.backtest.initial_cash)
    result_df = sort_result_df(result_df, objective)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n===== {stage_name} {objective} 前 10 参数 =====", flush=True)
    display_cols = get_scan_param_names(direction) + [
        "final_nav",
        "total_return",
        "annual_return",
        "max_drawdown_pct",
        "nav_drawdown_score",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "nav_sharpe_score",
        "ulcer_index",
        "nav_trend_r2",
        "top5_profit_share",
        "top10_profit_share",
        "profit_hhi",
        "max_gain_share_total_pnl",
        "positive_day_ratio",
        "min_cash",
        "min_cash_ratio",
        "cash_usage_ratio",
        "cash_negative_days",
        "open_count",
        "roll_count",
        "held_days",
        "avg_call_qty_when_held",
        "long_held_days",
        "short_held_days",
        "avg_long_qty_when_held",
        "avg_short_qty_when_held",
        "error",
    ]
    display_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[display_cols].head(10).to_string(index=False), flush=True)
    return result_df


def limit_param_sets(param_sets, max_combinations):
    if max_combinations is None:
        return param_sets
    return param_sets[:max_combinations]


def main():
    args = parse_args()
    product_config = core.config.load_config(args.product)
    backtest_updates = {
        key: value
        for key, value in {"start": args.start, "end": args.end}.items()
        if value is not None
    }
    if backtest_updates:
        product_config = replace(
            product_config,
            backtest=replace(product_config.backtest, **backtest_updates),
        )
    base_config = apply_param_set(product_config, {})
    start = args.start or base_config.backtest.start
    end = args.end or base_config.backtest.end

    print("加载数据...", flush=True)
    etf_by_date = core.data_loader.load_etf_series(start, end)
    opt_by_date = core.data_loader.load_opt_series(start, end)
    print(
        "调参品种: "
        f"{base_config.data.product}, "
        f"数据区间: {start} - {end}",
        flush=True,
    )
    trading_calendar = core.data_loader.load_etf_trading_calendar()

    if args.stage in ("coarse", "all"):
        coarse_param_sets = list(iter_param_sets(get_coarse_param_grid(args.direction)))
        coarse_param_sets = limit_param_sets(coarse_param_sets, args.max_combinations)
        coarse_output = make_output_path(args.output, "coarse")
        coarse_df = run_scan(
            "粗扫",
            coarse_param_sets,
            coarse_output,
            base_config,
            etf_by_date,
            opt_by_date,
            trading_calendar,
            start,
            end,
            args.direction,
            args.objective,
        )
    else:
        coarse_df = None
        coarse_output = args.coarse_result

    if args.stage in ("fine", "all"):
        if args.stage == "all":
            valid_coarse = coarse_df[coarse_df["error"].fillna("").eq("")]
            valid_coarse = ensure_nav_drawdown_score(valid_coarse)
            valid_coarse = ensure_nav_sharpe_score(valid_coarse)
            valid_coarse = ensure_cash_usage_metrics(
                valid_coarse,
                base_config.backtest.initial_cash,
            )
            valid_coarse = filter_by_min_held_days(
                valid_coarse,
                args.min_held_days,
            )
            top_rows = sort_result_df(valid_coarse, args.objective).head(args.top_n)
            fine_param_sets = build_fine_param_sets_from_rows(top_rows, args.direction)
            fine_output_arg = None
        else:
            fine_param_sets = load_fine_param_sets(
                base_config,
                args.coarse_result,
                args.top_n,
                args.direction,
                args.objective,
                args.min_held_days,
            )
            fine_output_arg = args.output

        fine_param_sets = limit_param_sets(fine_param_sets, args.max_combinations)
        fine_output = make_output_path(fine_output_arg, "fine")
        run_scan(
            "细扫",
            fine_param_sets,
            fine_output,
            base_config,
            etf_by_date,
            opt_by_date,
            trading_calendar,
            start,
            end,
            args.direction,
            args.objective,
        )

        if args.stage == "all":
            print(f"\n粗扫结果: {coarse_output}", flush=True)
            print(f"细扫结果: {fine_output}", flush=True)


if __name__ == "__main__":
    main()
