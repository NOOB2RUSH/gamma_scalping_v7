import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import core

CONFIG = core.config.CONFIG


def parse_args():
    parser = argparse.ArgumentParser(description="运行期权策略回测。")
    parser.add_argument(
        "--product",
        choices=core.config.available_products(),
        default=CONFIG.data.product,
        help="交易品种配置，默认使用 50ETF。",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="回测开始日期，默认使用所选品种配置文件中的 start。",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="回测结束日期，默认使用所选品种配置文件中的 end。",
    )
    parser.add_argument(
        "--test-date",
        default=None,
        help="烟雾测试日期，默认使用所选品种配置文件中的 test_date。",
    )
    parser.add_argument("--initial-cash", type=float, default=None)
    parser.add_argument(
        "--dynamic-position-control",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--proportional-position-sizing",
        action="store_true",
        default=None,
    )
    parser.add_argument("--max-margin-to-nav-ratio", type=float, default=None)
    return parser.parse_args()


def sync_config(config):
    """把运行时配置同步到各模块。"""
    global CONFIG
    CONFIG = config
    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    core.strategy.CONFIG = config
    core.backtester.CONFIG = config
    core.position.CONFIG = config
    core.hedge.CONFIG = config
    core.analytics.CONFIG = config


def select_runtime_config(args):
    """读取品种配置，并应用命令行日期覆盖。"""
    selected_config = core.config.load_config(args.product)
    backtest_updates = {
        key: value
        for key, value in {
            "start": args.start,
            "end": args.end,
            "test_date": args.test_date,
            "initial_cash": args.initial_cash,
            "dynamic_position_control_enabled": args.dynamic_position_control,
            "proportional_position_sizing_enabled": args.proportional_position_sizing,
            "max_margin_to_nav_ratio": args.max_margin_to_nav_ratio,
        }.items()
        if value is not None
    }
    if backtest_updates:
        selected_config = replace(
            selected_config,
            backtest=replace(selected_config.backtest, **backtest_updates),
        )
    return selected_config


def load_data():
    etf_by_date = core.data_loader.load_etf_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    opt_by_date = core.data_loader.load_opt_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    hedge_by_date = core.data_loader.load_hedge_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
    return etf_by_date, opt_by_date, hedge_by_date


def build_daily_report(features, daily_pnl):
    """生成对外日报：只展示收盘后持仓口径，隐藏内部 PnL 归因用的旧仓明细。"""
    feature_cols = [
        col
        for col in CONFIG.report.daily_feature_cols
        if col in features.columns and col not in daily_pnl.columns
    ]
    report = daily_pnl.join(features[feature_cols], how="left")

    # pnl_* 是内部用来解释“昨日持仓到今日收盘”损益的旧仓 Greeks，
    # 对外输出时只保留收盘后仓位，避免和当前持仓口径混淆。
    internal_pnl_cols = [col for col in report.columns if col.startswith("pnl_")]
    report = report.drop(columns=internal_pnl_cols)

    eod_rename = {
        col: col.removeprefix("eod_")
        for col in report.columns
        if col.startswith("eod_")
    }
    return report.rename(columns=eod_rename)


def format_money(value):
    return f"{value:,.2f}"


def format_pct(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def format_number(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2f}"


def calc_return_stats(daily_pnl):
    initial_cash = CONFIG.backtest.initial_cash
    final_nav = daily_pnl["nav"].iloc[-1]
    total_return = final_nav / initial_cash - 1
    trading_days = len(daily_pnl)
    annual_return = (1 + total_return) ** (CONFIG.vol.annual_days / trading_days) - 1
    daily_return = daily_pnl["nav"].pct_change().dropna()
    daily_return_std = daily_return.std()
    sharpe_ratio = (
        daily_return.mean() / daily_return_std * (CONFIG.vol.annual_days ** 0.5)
        if len(daily_return) > 1 and daily_return_std != 0
        else pd.NA
    )
    drawdown = daily_pnl["nav"] / daily_pnl["nav"].cummax() - 1
    max_drawdown = drawdown.min()
    max_drawdown_end = drawdown.idxmin()
    max_drawdown_start = daily_pnl.loc[:max_drawdown_end, "nav"].idxmax()
    cash_negative_days = (
        int(daily_pnl["cash_negative_warning"].sum())
        if "cash_negative_warning" in daily_pnl.columns
        else int((daily_pnl["cash"] < 0).sum())
    )

    return {
        "initial_cash": initial_cash,
        "final_nav": final_nav,
        "total_pnl": final_nav - initial_cash,
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_start": max_drawdown_start,
        "max_drawdown_end": max_drawdown_end,
        "trading_days": trading_days,
        "start_date": daily_pnl.index[0],
        "end_date": daily_pnl.index[-1],
        "min_cash": daily_pnl["cash"].min(),
        "cash_negative_days": cash_negative_days,
    }


def calc_summary_breakdown(daily_pnl, trades):
    """汇总 Greeks 和手续费解释项；未解释项只作为差额展示，不当作校验依据。"""
    total_pnl = daily_pnl["nav"].iloc[-1] - CONFIG.backtest.initial_cash
    total_pnl_before_fee = daily_pnl["daily_nav_pnl_before_fee"].sum(skipna=True)
    option_fee = 0.0
    option_fee_by_side = {"long": 0.0, "short": 0.0}
    if not trades.empty and "fee" in trades.columns:
        option_trade_mask = (
            trades["type"].str.contains("straddle", na=False)
            | trades["type"].isin(
                [
                    "option_delta_hedge_combination",
                    "gamma_neutral_option_delta_hedge",
                    "close_option_delta_hedge",
                ]
            )
        )
        option_fee = trades.loc[option_trade_mask, "fee"].sum()
        if "side" in trades.columns:
            for side in option_fee_by_side:
                side_mask = option_trade_mask & trades["side"].eq(side)
                option_fee_by_side[side] = trades.loc[side_mask, "fee"].sum()
    liquidity_warnings = {
        "warning_trades": 0,
        "warning_legs": 0,
        "missing_volume_trades": 0,
    }
    if not trades.empty and "liquidity_warning" in trades.columns:
        option_trade_mask = trades["type"].str.contains("straddle", na=False)
        warning_mask = trades["liquidity_warning"].eq(True)
        liquidity_warnings["warning_trades"] = int(
            (option_trade_mask & warning_mask).sum()
        )
        for leg_col in ["call_liquidity_warning", "put_liquidity_warning"]:
            if leg_col in trades.columns:
                liquidity_warnings["warning_legs"] += int(
                    trades.loc[option_trade_mask, leg_col].eq(True).sum()
                )
        if "liquidity_volume_missing_legs" in trades.columns:
            missing_mask = trades["liquidity_volume_missing_legs"].fillna("").ne("")
            liquidity_warnings["missing_volume_trades"] = int(
                (option_trade_mask & missing_mask).sum()
            )

    etf_fee = daily_pnl["etf_fee"].sum()
    components = {
        "total_before_fee": total_pnl_before_fee,
        "delta": daily_pnl["delta_pnl"].sum(skipna=True),
        "gamma": daily_pnl["gamma_pnl"].sum(skipna=True),
        "vega": daily_pnl["vega_pnl"].sum(skipna=True),
        "theta": daily_pnl["theta_pnl"].sum(skipna=True),
        "greeks_trading": daily_pnl["greeks_pnl"].sum(skipna=True),
        "unexplained_trading_before_fee": (
            daily_pnl["greeks_unexplained_pnl_before_fee"].sum(skipna=True)
        ),
        "option_fee": -option_fee,
        "etf_fee": -etf_fee,
    }
    components["by_side"] = {}
    for side in ["long", "short"]:
        prefix = f"{side}_"
        delta_col = (
            f"{prefix}total_delta_pnl"
            if f"{prefix}total_delta_pnl" in daily_pnl.columns
            else f"{prefix}delta_pnl"
        )
        hedge_delta_col = (
            f"{prefix}hedge_delta_pnl"
            if f"{prefix}hedge_delta_pnl" in daily_pnl.columns
            else None
        )
        total_delta = daily_pnl[delta_col].sum(skipna=True)
        components["by_side"][side] = {
            "delta": total_delta,
            "option_delta": daily_pnl[f"{prefix}delta_pnl"].sum(skipna=True),
            "hedge_delta": (
                daily_pnl[hedge_delta_col].sum(skipna=True)
                if hedge_delta_col is not None
                else 0.0
            ),
            "gamma": daily_pnl[f"{prefix}gamma_pnl"].sum(skipna=True),
            "vega": daily_pnl[f"{prefix}vega_pnl"].sum(skipna=True),
            "theta": daily_pnl[f"{prefix}theta_pnl"].sum(skipna=True),
            "greeks_trading": (
                total_delta
                + daily_pnl[f"{prefix}gamma_pnl"].sum(skipna=True)
                + daily_pnl[f"{prefix}vega_pnl"].sum(skipna=True)
                + daily_pnl[f"{prefix}theta_pnl"].sum(skipna=True)
            ),
            "option_fee": -option_fee_by_side[side],
            "days": int(daily_pnl[f"{prefix}greeks_explainable_day"].sum()),
        }
    components["liquidity_warnings"] = liquidity_warnings
    components["explained_subtotal"] = sum(
        value for value in components.values() if isinstance(value, (int, float))
    )
    components["unexplained"] = total_pnl - components["explained_subtotal"]
    return components


def calc_explain_ratio_stats(daily_pnl, ratio_col):
    """汇总解释力；算术均值容易被实际 PnL 接近 0 的单日放大，改用绝对 PnL 加权口径。"""
    explainable = daily_pnl["greeks_explainable_day"] == True
    ratios = daily_pnl.loc[explainable, ratio_col].dropna()
    if ratios.empty:
        return {"days": 0, "weighted_mean": pd.NA, "median": pd.NA}

    actual_abs = daily_pnl.loc[ratios.index, "daily_nav_pnl_before_fee"].abs()
    residual_abs = daily_pnl.loc[
        ratios.index,
        "greeks_unexplained_pnl_before_fee",
    ].abs()
    weighted_mean = 1 - residual_abs.sum() / actual_abs.sum()

    return {
        "days": len(ratios),
        "weighted_mean": weighted_mean,
        "median": ratios.median(),
    }


def print_breakdown_item(name, value, total_pnl):
    share = value / total_pnl if total_pnl != 0 else pd.NA
    share_text = "NA" if pd.isna(share) else format_pct(share)
    print(f"{name}: {format_money(value)} ({share_text})")


def print_side_breakdown(side_name, side_breakdown):
    total_greeks = side_breakdown["greeks_trading"]
    print(f"\n--- {side_name} Greeks 分解 ---")
    print(f"可解释交易日: {side_breakdown['days']}")
    print_breakdown_item("Delta PnL（期权 + ETF hedge）", side_breakdown["delta"], total_greeks)
    print(f"  期权腿 delta: {format_money(side_breakdown['option_delta'])}")
    print(f"  ETF hedge delta: {format_money(side_breakdown['hedge_delta'])}")
    print_breakdown_item("Gamma PnL", side_breakdown["gamma"], total_greeks)
    print_breakdown_item("Vega PnL", side_breakdown["vega"], total_greeks)
    print_breakdown_item("Theta PnL", side_breakdown["theta"], total_greeks)
    print_breakdown_item("Greeks PnL", total_greeks, total_greeks)
    print_breakdown_item("期权手续费", side_breakdown["option_fee"], total_greeks)


def print_summary(daily_pnl, trades):
    stats = calc_return_stats(daily_pnl)
    breakdown = calc_summary_breakdown(daily_pnl, trades)
    explain_stats = calc_explain_ratio_stats(
        daily_pnl,
        "greeks_explain_ratio_before_fee",
    )

    print("\n=== 回测概要 ===")
    print(f"区间: {stats['start_date'].date()} -> {stats['end_date'].date()}")
    print(f"交易日数: {stats['trading_days']}")
    print(f"初始资金: {format_money(stats['initial_cash'])}")
    print(f"期末净值: {format_money(stats['final_nav'])}")
    print(f"总损益: {format_money(stats['total_pnl'])}")
    print(f"总收益率: {format_pct(stats['total_return'])}")
    print(f"年化收益率: {format_pct(stats['annual_return'])}")
    print(f"夏普比率: {format_number(stats['sharpe_ratio'])}")
    print(
        "最大回撤: "
        f"{format_pct(stats['max_drawdown'])} "
        f"({stats['max_drawdown_start'].date()} -> {stats['max_drawdown_end'].date()})"
    )
    print(f"最大期权保证金: {format_money(daily_pnl['option_margin'].max())}")
    print(f"最大 ETF 保证金: {format_money(daily_pnl['hedge_margin'].max())}")
    print(f"最低现金: {format_money(stats['min_cash'])}")
    print(f"爆仓预警天数（现金<0）: {stats['cash_negative_days']}")

    print("\n=== 损益分解 ===")
    total_pnl = stats["total_pnl"]
    total_before_fee = breakdown["total_before_fee"]
    total_margin = daily_pnl["option_margin"] + daily_pnl["hedge_margin"]
    max_margin_to_nav = (
        daily_pnl["margin_to_nav_ratio"].max()
        if "margin_to_nav_ratio" in daily_pnl.columns
        else (total_margin / daily_pnl["nav"]).max()
    )
    margin_limit_breach_days = (
        int(daily_pnl["margin_limit_breach"].sum())
        if "margin_limit_breach" in daily_pnl.columns
        else 0
    )
    print_breakdown_item("Delta PnL", breakdown["delta"], total_before_fee)
    print_breakdown_item("Gamma PnL", breakdown["gamma"], total_before_fee)
    print_breakdown_item("Vega PnL", breakdown["vega"], total_before_fee)
    print_breakdown_item("Theta PnL", breakdown["theta"], total_before_fee)
    print_breakdown_item("Greeks PnL", breakdown["greeks_trading"], total_before_fee)
    print_breakdown_item(
        "未解释 PnL（手续费前）",
        breakdown["unexplained_trading_before_fee"],
        total_before_fee,
    )
    print_side_breakdown("Long", breakdown["by_side"]["long"])
    print_side_breakdown("Short", breakdown["by_side"]["short"])

    print("\n=== 手续费 ===")
    print_breakdown_item("期权手续费", breakdown["option_fee"], total_pnl)
    print_breakdown_item("ETF 手续费", breakdown["etf_fee"], total_pnl)
    print(f"手续费前实际 PnL: {format_money(total_before_fee)}")
    print(f"手续费后实际 PnL: {format_money(total_pnl)}")

    print("\n=== Greeks 解释力（手续费前） ===")
    print(
        f"days={explain_stats['days']}, "
        f"abs加权mean={format_pct(explain_stats['weighted_mean'])}, "
        f"median={format_pct(explain_stats['median'])}"
    )


def calc_liquidity_warning_stats(trades):
    stats = {
        "warning_trades": 0,
        "warning_legs": 0,
        "missing_volume_trades": 0,
        "worst_warning": None,
    }
    if trades.empty or "liquidity_warning" not in trades.columns:
        return stats

    option_trade_mask = trades["type"].str.contains("straddle", na=False)
    warning_mask = trades["liquidity_warning"].eq(True)
    stats["warning_trades"] = int((option_trade_mask & warning_mask).sum())
    warning_legs = []
    option_trades = trades.loc[option_trade_mask]
    for leg in ["call", "put"]:
        leg_col = f"{leg}_liquidity_warning"
        if leg_col in trades.columns:
            stats["warning_legs"] += int(option_trades[leg_col].eq(True).sum())
            for _, trade in option_trades.iterrows():
                leg_warning = trade.get(leg_col, False)
                if not (leg_warning is True or str(leg_warning).lower() == "true"):
                    continue
                volume = trade.get(f"{leg}_volume")
                qty = abs(float(trade.get(f"trade_{leg}_qty", 0) or 0))
                if pd.isna(volume) or float(volume) <= 0:
                    continue
                volume = float(volume)
                warning_legs.append(
                    {
                        "date": trade.get("date"),
                        "type": trade.get("type"),
                        "side": trade.get("side"),
                        "leg": leg,
                        "code": trade.get(f"{leg}_code"),
                        "qty": qty,
                        "volume": volume,
                        "ratio": qty / volume,
                    }
                )
    if warning_legs:
        stats["worst_warning"] = max(warning_legs, key=lambda item: item["ratio"])
    if "liquidity_volume_missing_legs" in trades.columns:
        missing_mask = trades["liquidity_volume_missing_legs"].fillna("").ne("")
        stats["missing_volume_trades"] = int((option_trade_mask & missing_mask).sum())
    return stats


def print_liquidity_warning_summary(trades):
    stats = calc_liquidity_warning_stats(trades)
    print("\n=== 成交量预警 ===")
    print(f"触发预警交易次数: {stats['warning_trades']}（腿次数: {stats['warning_legs']}）")
    print(f"成交量缺失交易次数: {stats['missing_volume_trades']}")
    worst_warning = stats["worst_warning"]
    if worst_warning is not None:
        print(
            "最严重预警: "
            f"{worst_warning['date']} {worst_warning['type']} "
            f"{worst_warning['side']} {worst_warning['leg']} "
            f"{worst_warning['code']}, "
            f"交易张数={worst_warning['qty']:.0f}, "
            f"成交量={worst_warning['volume']:.0f}, "
            f"占比={worst_warning['ratio']:.2%}"
        )


def _summary_value_for_csv(value):
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    return value


def save_summary_files(output_dir, daily_pnl, trades):
    """把回测概要同时保存为人读 txt 和机器读 csv。"""
    stats = calc_return_stats(daily_pnl)
    breakdown = calc_summary_breakdown(daily_pnl, trades)
    explain_stats = calc_explain_ratio_stats(
        daily_pnl,
        "greeks_explain_ratio_before_fee",
    )
    liquidity_stats = calc_liquidity_warning_stats(trades)
    total_pnl = stats["total_pnl"]
    total_before_fee = breakdown["total_before_fee"]
    total_margin = daily_pnl["option_margin"] + daily_pnl["hedge_margin"]
    max_margin_to_nav = (
        daily_pnl["margin_to_nav_ratio"].max()
        if "margin_to_nav_ratio" in daily_pnl.columns
        else (total_margin / daily_pnl["nav"]).max()
    )
    margin_limit_breach_days = (
        int(daily_pnl["margin_limit_breach"].sum())
        if "margin_limit_breach" in daily_pnl.columns
        else 0
    )

    lines = [
        "=== 回测概要 ===",
        f"区间: {stats['start_date'].date()} -> {stats['end_date'].date()}",
        f"交易日数: {stats['trading_days']}",
        f"初始资金: {format_money(stats['initial_cash'])}",
        f"期末净值: {format_money(stats['final_nav'])}",
        f"总损益: {format_money(stats['total_pnl'])}",
        f"总收益率: {format_pct(stats['total_return'])}",
        f"年化收益率: {format_pct(stats['annual_return'])}",
        f"夏普比率: {format_number(stats['sharpe_ratio'])}",
        (
            "最大回撤: "
            f"{format_pct(stats['max_drawdown'])} "
            f"({stats['max_drawdown_start'].date()} -> "
            f"{stats['max_drawdown_end'].date()})"
        ),
        f"最大期权保证金: {format_money(daily_pnl['option_margin'].max())}",
        f"最大 ETF 保证金: {format_money(daily_pnl['hedge_margin'].max())}",
        f"最大合计保证金: {format_money(total_margin.max())}",
        f"最大保证金/净值: {format_pct(max_margin_to_nav)}",
        f"保证金占用超限天数: {margin_limit_breach_days}",
        f"最低现金: {format_money(stats['min_cash'])}",
        f"爆仓预警天数（现金<0）: {stats['cash_negative_days']}",
        "",
        "=== 损益分解 ===",
        f"Delta PnL: {format_money(breakdown['delta'])}",
        f"Gamma PnL: {format_money(breakdown['gamma'])}",
        f"Vega PnL: {format_money(breakdown['vega'])}",
        f"Theta PnL: {format_money(breakdown['theta'])}",
        f"Greeks PnL: {format_money(breakdown['greeks_trading'])}",
        (
            "未解释 PnL（手续费前）: "
            f"{format_money(breakdown['unexplained_trading_before_fee'])}"
        ),
        "",
        "--- Long Greeks 分解 ---",
        f"可解释交易日: {breakdown['by_side']['long']['days']}",
        (
            "Delta PnL（期权 + ETF hedge）: "
            f"{format_money(breakdown['by_side']['long']['delta'])}"
        ),
        (
            "期权腿 delta: "
            f"{format_money(breakdown['by_side']['long']['option_delta'])}"
        ),
        (
            "ETF hedge delta: "
            f"{format_money(breakdown['by_side']['long']['hedge_delta'])}"
        ),
        f"Gamma PnL: {format_money(breakdown['by_side']['long']['gamma'])}",
        f"Vega PnL: {format_money(breakdown['by_side']['long']['vega'])}",
        f"Theta PnL: {format_money(breakdown['by_side']['long']['theta'])}",
        f"Greeks PnL: {format_money(breakdown['by_side']['long']['greeks_trading'])}",
        "",
        "--- Short Greeks 分解 ---",
        f"可解释交易日: {breakdown['by_side']['short']['days']}",
        (
            "Delta PnL（期权 + ETF hedge）: "
            f"{format_money(breakdown['by_side']['short']['delta'])}"
        ),
        (
            "期权腿 delta: "
            f"{format_money(breakdown['by_side']['short']['option_delta'])}"
        ),
        (
            "ETF hedge delta: "
            f"{format_money(breakdown['by_side']['short']['hedge_delta'])}"
        ),
        f"Gamma PnL: {format_money(breakdown['by_side']['short']['gamma'])}",
        f"Vega PnL: {format_money(breakdown['by_side']['short']['vega'])}",
        f"Theta PnL: {format_money(breakdown['by_side']['short']['theta'])}",
        f"Greeks PnL: {format_money(breakdown['by_side']['short']['greeks_trading'])}",
        "",
        "=== 手续费 ===",
        f"期权手续费: {format_money(breakdown['option_fee'])}",
        f"ETF 手续费: {format_money(breakdown['etf_fee'])}",
        f"手续费前实际 PnL: {format_money(total_before_fee)}",
        f"手续费后实际 PnL: {format_money(total_pnl)}",
        "",
        "=== Greeks 解释力（手续费前） ===",
        (
            f"days={explain_stats['days']}, "
            f"abs加权mean={format_pct(explain_stats['weighted_mean'])}, "
            f"median={format_pct(explain_stats['median'])}"
        ),
        "",
        "=== 成交量预警 ===",
        (
            f"触发预警交易次数: {liquidity_stats['warning_trades']} "
            f"（腿次数: {liquidity_stats['warning_legs']}）"
        ),
        f"成交量缺失交易次数: {liquidity_stats['missing_volume_trades']}",
    ]

    worst_warning = liquidity_stats["worst_warning"]
    if worst_warning is not None:
        lines.append(
            "最严重预警: "
            f"{worst_warning['date']} {worst_warning['type']} "
            f"{worst_warning['side']} {worst_warning['leg']} "
            f"{worst_warning['code']}, "
            f"交易张数={worst_warning['qty']:.0f}, "
            f"成交量={worst_warning['volume']:.0f}, "
            f"占比={worst_warning['ratio']:.2%}"
        )

    (output_dir / "backtest_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8-sig",
    )

    rows = [
        ("initial_cash", stats["initial_cash"]),
        ("final_nav", stats["final_nav"]),
        ("total_pnl", stats["total_pnl"]),
        ("total_return", stats["total_return"]),
        ("annual_return", stats["annual_return"]),
        ("sharpe_ratio", stats["sharpe_ratio"]),
        ("max_drawdown", stats["max_drawdown"]),
        ("max_drawdown_start", stats["max_drawdown_start"]),
        ("max_drawdown_end", stats["max_drawdown_end"]),
        ("trading_days", stats["trading_days"]),
        ("start_date", stats["start_date"]),
        ("end_date", stats["end_date"]),
        ("max_option_margin", daily_pnl["option_margin"].max()),
        ("max_etf_margin", daily_pnl["hedge_margin"].max()),
        ("max_total_margin", total_margin.max()),
        ("max_margin_to_nav_ratio", max_margin_to_nav),
        ("margin_limit_breach_days", margin_limit_breach_days),
        ("min_cash", stats["min_cash"]),
        ("cash_negative_days", stats["cash_negative_days"]),
        ("delta_pnl", breakdown["delta"]),
        ("gamma_pnl", breakdown["gamma"]),
        ("vega_pnl", breakdown["vega"]),
        ("theta_pnl", breakdown["theta"]),
        ("greeks_pnl", breakdown["greeks_trading"]),
        (
            "unexplained_pnl_before_fee",
            breakdown["unexplained_trading_before_fee"],
        ),
        ("option_fee", breakdown["option_fee"]),
        ("etf_fee", breakdown["etf_fee"]),
        ("actual_pnl_before_fee", total_before_fee),
        ("actual_pnl_after_fee", total_pnl),
        ("greeks_explainable_days", explain_stats["days"]),
        ("greeks_explain_ratio_weighted_mean", explain_stats["weighted_mean"]),
        ("greeks_explain_ratio_median", explain_stats["median"]),
        ("liquidity_warning_trades", liquidity_stats["warning_trades"]),
        ("liquidity_warning_legs", liquidity_stats["warning_legs"]),
        ("liquidity_missing_volume_trades", liquidity_stats["missing_volume_trades"]),
    ]
    pd.DataFrame(
        [{"metric": key, "value": _summary_value_for_csv(value)} for key, value in rows]
    ).to_csv(output_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig")


def make_output_dir():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(CONFIG.report.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_runtime_config(output_dir):
    """保存本次运行实际使用的配置，方便回看结果时确认参数是否生效。"""
    (output_dir / "runtime_config.json").write_text(
        json.dumps(asdict(CONFIG), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _runtime_surface_config():
    filters = core.vol_surface.SurfaceFilters(
        min_dte=CONFIG.vol.surface_min_dte,
        min_volume=CONFIG.vol.surface_min_volume,
        max_spread_pct=CONFIG.vol.surface_max_spread_pct,
        min_abs_delta=CONFIG.vol.surface_min_abs_delta,
        max_abs_delta=CONFIG.vol.surface_max_abs_delta,
    )
    return core.vol_surface.SurfaceConfig(
        standard_dtes=CONFIG.vol.surface_standard_dtes,
        annual_days=CONFIG.vol.annual_days,
        risk_free_rate=CONFIG.vol.risk_free_rate,
        filters=filters,
    )


def plot_full_sample_vol_surface(output_dir, enriched_opt_by_date):
    """用整个回测期样本汇总平均曲面；该图有前视偏差，只用于诊断展示。"""
    observation_mode = getattr(CONFIG.vol, "iv_observation_mode", "legacy")
    surface_plot_enabled = (
        observation_mode == "surface_percentile"
        or (
            observation_mode == "legacy"
            and CONFIG.vol.surface_atm_iv_enabled
        )
    )
    if (
        not surface_plot_enabled
        or not CONFIG.report.enable_surface_full_sample_plot
    ):
        return None

    surface_config = _runtime_surface_config()
    smile_rows = []
    raw_points = []
    for date in sorted(enriched_opt_by_date):
        surface = core.vol_surface.build_daily_surface(
            enriched_opt_by_date[date],
            config=surface_config,
            standard_dtes=CONFIG.vol.surface_standard_dtes,
            k_grid_mode=CONFIG.vol.surface_k_grid_mode,
            allow_term_extrapolate=CONFIG.vol.surface_allow_term_extrapolate,
            term_extrapolate_mode=CONFIG.vol.surface_term_extrapolate_mode,
        )
        points = surface.get("points")
        if points is not None and not points.empty:
            raw_points.append(points.assign(sample_date=date))

        for target_dte, smile in surface.get("smiles", {}).items():
            if smile is None or smile.empty:
                continue
            df = smile[["target_dte", "log_moneyness", "surface_iv"]].copy()
            df["sample_date"] = date
            df["target_dte"] = float(target_dte)
            smile_rows.append(df)

    if not smile_rows:
        return None

    all_smiles = pd.concat(smile_rows, ignore_index=True).dropna(
        subset=["log_moneyness", "surface_iv"]
    )
    k_low = all_smiles["log_moneyness"].quantile(0.02)
    k_high = all_smiles["log_moneyness"].quantile(0.98)
    if pd.isna(k_low) or pd.isna(k_high) or k_low >= k_high:
        return None
    k_grid = np.linspace(float(k_low), float(k_high), 41)

    normalized_rows = []
    for (sample_date, target_dte), group in all_smiles.groupby(
        ["sample_date", "target_dte"]
    ):
        group = group.sort_values("log_moneyness")
        k = group["log_moneyness"].to_numpy(dtype=float)
        iv = group["surface_iv"].to_numpy(dtype=float)
        valid_grid = k_grid[(k_grid >= k.min()) & (k_grid <= k.max())]
        if len(valid_grid) == 0:
            continue
        interp_iv = np.interp(valid_grid, k, iv)
        for grid_k, surface_iv in zip(valid_grid, interp_iv):
            normalized_rows.append(
                {
                    "target_dte": float(target_dte),
                    "log_moneyness": float(grid_k),
                    "surface_iv": float(surface_iv),
                }
            )

    normalized = pd.DataFrame(normalized_rows)
    if normalized.empty:
        return None

    average_smile = (
        normalized.groupby(["target_dte", "log_moneyness"], as_index=False)[
            "surface_iv"
        ]
        .mean()
        .sort_values(["target_dte", "log_moneyness"])
    )
    average_smile["total_variance"] = (
        average_smile["surface_iv"] ** 2
        * (average_smile["target_dte"] / float(CONFIG.vol.annual_days))
    )

    smiles = {
        dte: group.copy()
        for dte, group in average_smile.groupby("target_dte", sort=True)
    }
    points = (
        pd.concat(raw_points, ignore_index=True)
        if raw_points
        else pd.DataFrame(columns=["log_moneyness", "dte", "surface_iv"])
    )
    max_dte = CONFIG.vol.surface_raw_point_max_dte
    if max_dte is not None and not points.empty:
        points = points[points["dte"] <= max_dte]
    if len(points) > 20000:
        points = points.sample(20000, random_state=42)

    output_path = output_dir / "full_sample_vol_surface.png"
    core.vol_surface.plot_vol_surface(
        {"points": points, "smiles": smiles},
        output_path=output_path,
        title=(
            f"{CONFIG.data.product} full-sample average fixed-tenor IV surface "
            "(diagnostic, look-ahead biased)"
        ),
        include_raw_points=not points.empty,
        raw_point_max_dte=max_dte,
        invert_log_moneyness_axis=True,
        invert_dte_axis=True,
        show=False,
    )
    (output_dir / "surface_limitations.txt").write_text(
        "full_sample_vol_surface.png 使用整个回测区间的期权样本汇总平均曲面，"
        "因此包含前视偏差；它只用于诊断换月跳变和曲面覆盖，不参与逐日交易信号。\n"
        "逐日信号使用每个交易日当日可见期权链提取的 30 天固定期限 ATM IV "
        "及其历史分位数。\n",
        encoding="utf-8",
    )
    return output_path


def main():
    args = parse_args()
    sync_config(select_runtime_config(args))
    print(
        "运行品种: "
        f"{CONFIG.data.product}, "
        f"数据区间: {CONFIG.backtest.start} - {CONFIG.backtest.end}",
        flush=True,
    )

    etf_by_date, opt_by_date, hedge_by_date = load_data()
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    enriched_opt_by_date = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    features = core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched_opt_by_date,
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    signals = core.strategy.build_signals(features)
    daily_pnl, trades = core.backtester.run_backtest(
        etf_by_date,
        opt_by_date,
        signals,
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
        hedge_by_date=hedge_by_date,
    )
    output_dir = make_output_dir()
    save_runtime_config(output_dir)

    daily_report = build_daily_report(features, daily_pnl)
    daily_report.to_csv(output_dir / "daily_feature_position.csv", encoding="utf-8-sig")
    trades.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    core.analytics.plot_vol_features(
        features,
        backtest_df=daily_pnl,
        output_path=output_dir / "vol_features.png",
        show=False,
    )
    core.analytics.plot_cumulative_greeks_pnl(
        daily_pnl,
        output_path=output_dir / "cumulative_greeks_pnl.png",
        show=False,
    )
    core.analytics.plot_cumulative_actual_vs_greeks_pnl(
        daily_pnl,
        output_path=output_dir / "cumulative_actual_vs_greeks_pnl.png",
        show=False,
    )
    surface_path = plot_full_sample_vol_surface(output_dir, enriched_opt_by_date)
    if surface_path is not None:
        print(f"full-sample vol surface: {surface_path}")

    save_summary_files(output_dir, daily_pnl, trades)
    print_summary(daily_pnl, trades)
    print_liquidity_warning_summary(trades)
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
