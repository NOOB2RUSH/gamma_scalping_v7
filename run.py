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


def print_summary(daily_pnl):
    print("\n=== backtest summary ===")
    print(f"final nav: {daily_pnl['nav'].iloc[-1]}")
    print(f"max hedge margin: {daily_pnl['hedge_margin'].max()}")
    print(f"min cash: {daily_pnl['cash'].min()}")


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
    signals = core.strategy.build_signal_df(features)
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

    print_summary(daily_pnl)
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
