from pathlib import Path

import pandas as pd

import core

CONFIG = core.config.CONFIG


def load_data():
    etf_by_date = core.data_loader.load_etf_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    opt_by_date = core.data_loader.load_opt_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    return etf_by_date, opt_by_date


def build_daily_report(features, daily_pnl):
    feature_cols = [
        col for col in CONFIG.report.daily_feature_cols if col in features.columns
    ]
    return daily_pnl.join(features[feature_cols], how="left")


def format_money(value):
    return f"{value:,.2f}"


def format_pct(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def calc_return_stats(daily_pnl):
    initial_cash = CONFIG.backtest.initial_cash
    final_nav = daily_pnl["nav"].iloc[-1]
    total_return = final_nav / initial_cash - 1
    trading_days = len(daily_pnl)
    annual_return = (1 + total_return) ** (CONFIG.vol.annual_days / trading_days) - 1

    return {
        "initial_cash": initial_cash,
        "final_nav": final_nav,
        "total_pnl": final_nav - initial_cash,
        "total_return": total_return,
        "annual_return": annual_return,
        "trading_days": trading_days,
        "start_date": daily_pnl.index[0],
        "end_date": daily_pnl.index[-1],
    }


def calc_summary_breakdown(daily_pnl, trades):
    """汇总 Greeks 和手续费解释项；未解释项只作为差额展示，不当作校验依据。"""
    total_pnl = daily_pnl["nav"].iloc[-1] - CONFIG.backtest.initial_cash
    total_pnl_before_fee = daily_pnl["daily_nav_pnl_before_fee"].sum(skipna=True)
    option_fee = 0.0
    if not trades.empty and "fee" in trades.columns:
        option_trade_mask = trades["type"].str.contains("straddle", na=False)
        option_fee = trades.loc[option_trade_mask, "fee"].sum()

    etf_fee = daily_pnl["etf_fee"].sum()
    components = {
        "total_before_fee": total_pnl_before_fee,
        "delta": daily_pnl["delta_pnl"].sum(skipna=True),
        "gamma": daily_pnl["gamma_pnl"].sum(skipna=True),
        "vega": daily_pnl["vega_pnl"].sum(skipna=True),
        "theta": daily_pnl["theta_pnl"].sum(skipna=True),
        "theta_calendar": daily_pnl["theta_calendar_pnl"].sum(skipna=True),
        "greeks_trading": daily_pnl["greeks_pnl"].sum(skipna=True),
        "greeks_calendar": daily_pnl["greeks_calendar_pnl"].sum(skipna=True),
        "unexplained_trading_before_fee": (
            daily_pnl["greeks_unexplained_pnl_before_fee"].sum(skipna=True)
        ),
        "unexplained_calendar_before_fee": (
            daily_pnl["greeks_calendar_unexplained_pnl_before_fee"].sum(skipna=True)
        ),
        "option_fee": -option_fee,
        "etf_fee": -etf_fee,
    }
    components["explained_subtotal"] = sum(components.values())
    components["unexplained"] = total_pnl - components["explained_subtotal"]
    return components


def calc_explain_ratio_stats(daily_pnl, ratio_col):
    explainable = daily_pnl["greeks_explainable_day"] == True
    ratios = daily_pnl.loc[explainable, ratio_col].dropna()
    if ratios.empty:
        return {"days": 0, "mean": pd.NA, "median": pd.NA}

    return {
        "days": len(ratios),
        "mean": ratios.mean(),
        "median": ratios.median(),
    }


def print_breakdown_item(name, value, total_pnl):
    share = value / total_pnl if total_pnl != 0 else pd.NA
    share_text = "NA" if pd.isna(share) else format_pct(share)
    print(f"{name}: {format_money(value)} ({share_text})")


def print_summary(daily_pnl, trades):
    stats = calc_return_stats(daily_pnl)
    breakdown = calc_summary_breakdown(daily_pnl, trades)
    explain_stats = calc_explain_ratio_stats(
        daily_pnl,
        "greeks_explain_ratio_before_fee",
    )
    calendar_explain_stats = calc_explain_ratio_stats(
        daily_pnl,
        "greeks_calendar_explain_ratio_before_fee",
    )

    print("\n=== 回测概要 ===")
    print(f"区间: {stats['start_date'].date()} -> {stats['end_date'].date()}")
    print(f"交易日数: {stats['trading_days']}")
    print(f"初始资金: {format_money(stats['initial_cash'])}")
    print(f"期末净值: {format_money(stats['final_nav'])}")
    print(f"总损益: {format_money(stats['total_pnl'])}")
    print(f"总收益率: {format_pct(stats['total_return'])}")
    print(f"年化收益率: {format_pct(stats['annual_return'])}")
    print(f"最大 ETF 保证金: {format_money(daily_pnl['hedge_margin'].max())}")
    print(f"最低现金: {format_money(daily_pnl['cash'].min())}")

    print("\n=== 损益分解（交易日 Theta 口径） ===")
    total_pnl = stats["total_pnl"]
    total_before_fee = breakdown["total_before_fee"]
    print_breakdown_item("Delta PnL", breakdown["delta"], total_before_fee)
    print_breakdown_item("Gamma PnL", breakdown["gamma"], total_before_fee)
    print_breakdown_item("Vega PnL", breakdown["vega"], total_before_fee)
    print_breakdown_item("Theta PnL（交易日）", breakdown["theta"], total_before_fee)
    print_breakdown_item("Greeks PnL（交易日）", breakdown["greeks_trading"], total_before_fee)
    print_breakdown_item(
        "未解释 PnL（手续费前）",
        breakdown["unexplained_trading_before_fee"],
        total_before_fee,
    )

    print("\n=== 损益分解（日历日 Theta 口径） ===")
    print_breakdown_item("Delta PnL", breakdown["delta"], total_before_fee)
    print_breakdown_item("Gamma PnL", breakdown["gamma"], total_before_fee)
    print_breakdown_item("Vega PnL", breakdown["vega"], total_before_fee)
    print_breakdown_item(
        "Theta PnL（日历日）",
        breakdown["theta_calendar"],
        total_before_fee,
    )
    print_breakdown_item(
        "Greeks PnL（日历日）",
        breakdown["greeks_calendar"],
        total_before_fee,
    )
    print_breakdown_item(
        "未解释 PnL（手续费前）",
        breakdown["unexplained_calendar_before_fee"],
        total_before_fee,
    )

    print("\n=== 手续费 ===")
    print_breakdown_item("期权手续费", breakdown["option_fee"], total_pnl)
    print_breakdown_item("ETF 手续费", breakdown["etf_fee"], total_pnl)
    print(f"手续费前实际 PnL: {format_money(total_before_fee)}")
    print(f"手续费后实际 PnL: {format_money(total_pnl)}")

    print("\n=== Greeks 解释力（手续费前） ===")
    print(
        "交易日 Theta: "
        f"days={explain_stats['days']}, "
        f"mean={format_pct(explain_stats['mean'])}, "
        f"median={format_pct(explain_stats['median'])}"
    )
    print(
        "日历日 Theta: "
        f"days={calendar_explain_stats['days']}, "
        f"mean={format_pct(calendar_explain_stats['mean'])}, "
        f"median={format_pct(calendar_explain_stats['median'])}"
    )


def make_output_dir():
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(CONFIG.report.output_root) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def smoke_test_data_load():
    test_date = pd.Timestamp(CONFIG.backtest.test_date)
    etf_by_date = core.data_loader.load_etf_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )
    opt_by_date = core.data_loader.load_opt_series(
        CONFIG.backtest.start,
        CONFIG.backtest.end,
    )

    print("=== data load ===")
    print(f"etf days: {len(etf_by_date)}")
    print(f"option days: {len(opt_by_date)}")
    print(f"first etf date: {min(etf_by_date)}")
    print(f"last etf date: {max(etf_by_date)}")
    print(f"test date exists in etf: {test_date in etf_by_date}")
    print(f"test date exists in option: {test_date in opt_by_date}")

    return etf_by_date, opt_by_date


def smoke_test_hv(etf_by_date):
    daily_ohlc = core.vol_engine.build_daily_ohlc_df(etf_by_date)
    hv_df = core.vol_engine.calculate_yz_hv(
        daily_ohlc,
        rolling_windows=(5, 20),
        annual_days=CONFIG.vol.annual_days,
    )

    print("\n=== realized vol ===")
    print(hv_df[["close", "yz_hv5", "yz_hv20"]].tail(10))

    return daily_ohlc, hv_df


def smoke_test_atm_iv(daily_ohlc, opt_by_date):
    test_date = pd.Timestamp(CONFIG.backtest.test_date)
    spot = daily_ohlc.loc[test_date, "close"]
    chain = opt_by_date[test_date]

    atm = core.vol_engine.calc_atm_iv_for_day(
        daily_opt_chain=chain,
        spot=spot,
        target_dte=CONFIG.vol.atm_target_dte,
        target_dte_min=CONFIG.vol.atm_target_dte_min,
        target_dte_max=CONFIG.vol.atm_target_dte_max,
        atm_moneyness_tol=CONFIG.vol.atm_moneyness_tol,
    )

    print("\n=== atm iv ===")
    print(f"date: {test_date.date()}")
    print(f"spot: {spot}")

    if atm is None:
        print("no valid atm call+put pair found")
        return None

    print(f"strike: {atm['strike']}")
    print(f"expiry: {atm['expiry'].date()}")
    print(f"dte: {atm['dte']}")
    print(f"call code: {atm['call']['order_book_id']}")
    print(f"put code: {atm['put']['order_book_id']}")
    print(f"call mid: {atm['call']['mid']}")
    print(f"put mid: {atm['put']['mid']}")
    print(f"call iv: {atm['call_iv']}")
    print(f"put iv: {atm['put_iv']}")
    print(f"atm iv: {atm['atm_iv']}")

    return atm


def smoke_test_greeks():
    date = pd.Timestamp(CONFIG.backtest.test_date)

    smoke_date = date.strftime("%Y%m%d")
    etf_by_date = core.data_loader.load_etf_series(smoke_date, smoke_date)
    opt_by_date = core.data_loader.load_opt_series(smoke_date, smoke_date)

    spot = etf_by_date[date].iloc[0]["close"]
    atm = core.vol_engine.calc_atm_iv_for_day(opt_by_date[date], spot)

    print("\n=== greeks smoke test ===")
    print(f"date: {date.date()}")
    print(f"spot: {spot}")
    print(f"strike: {atm['strike']}")
    print(f"expiry: {atm['expiry'].date()}")
    print(f"dte: {atm['dte']}")

    for side in ["call", "put"]:
        row = atm[side]
        print(f"\n{side.upper()}")
        print(f"code: {row['order_book_id']}")
        print(f"mid: {row['mid']}")
        print(f"iv: {row['iv']}")
        print(f"delta: {row['delta']}")
        print(f"gamma: {row['gamma']}")
        print(f"vega: {row['vega']}")
        print(f"theta: {row['theta']}")

    position_greeks = core.strategy.calc_position_greeks(
        atm["call"],
        atm["put"],
    )

    print("\nPOSITION GREEKS")
    print(position_greeks)


def main():
    etf_by_date, opt_by_date = load_data()
    features = core.vol_engine.build_vol_features(etf_by_date, opt_by_date)
    signals = core.strategy.build_signals(features)
    daily_pnl, trades = core.backtester.run_backtest(etf_by_date, opt_by_date, signals)

    output_dir = make_output_dir()

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

    print_summary(daily_pnl, trades)
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
