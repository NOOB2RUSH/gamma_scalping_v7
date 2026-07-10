"""中证1000股指期权独立配置。"""

from .config_schema import (
    AppConfig,
    BacktestConfig,
    DataConfig,
    ReportConfig,
    StrategyConfig,
    VolConfig,
)

CONFIG = AppConfig(
    data=DataConfig(
        product="zz1000",
        etf_dir="data/zz1000/index",
        opt_dir="data/zz1000/option",
        hedge_etf_dir="data/zz1000/hedge_etf",
    ),
    backtest=BacktestConfig(
        # 当前 AKShare 本地连续样本为 2022-08-01 到 2025-12-31。
        start="20220801",
        end="20260101",
        test_date="20220801",
        initial_cash=1_000_000,
        min_cash_reserve=50_000,
        # 中证1000股指期权
        long_qty=1,
        short_qty=1,
        etf_fee_rate=0.00005,
        option_fee_per_contract=2.0,
        liquidity_warning_volume_ratio=0.005,
    ),
    strategy=StrategyConfig(
        enable_long_straddle=True,
        enable_short_straddle=True,
        long_open_iv_threshold=0.14,
        long_close_iv_threshold=0.20,
        min_exit_dte=3,
        short_signal_mode="absolute",
        short_open_iv_threshold=0.25,
        short_close_iv_threshold=0.22,
        short_open_pullback_iv_threshold=0.30,
        short_open_iv_percentile_threshold=0.75,
        short_close_iv_percentile_threshold=0.60,
        short_low_iv_open_threshold=0.21,
        short_low_iv_close_threshold=0.21,
        short_low_iv_hv_spread_threshold=0.01,
        short_low_iv_close_spread_threshold=0.0,
        short_low_iv_hv_col="yz_hv60",
        short_low_iv_overlay_enabled=True,
        short_stop_loss_enabled=True,
        short_daily_loss_aum_threshold=-0.015,
        enable_delta_hedge=True,
        short_volume_spike_exit_enabled=True,
        short_volume_spike_multiplier=1.5,
        short_cooldown_after_long_iv_high_exit_days=3,
        roll_dte_threshold=7,
    ),
    vol=VolConfig(
        annual_days=252,
        hv_windows=(60,),
        atm_iv_percentile_window=252,
        atm_target_dte=20,
        atm_target_dte_min=7,
        atm_target_dte_max=30,
        # 中证1000行权价档位是指数点位，ATM 偏离容忍度不能沿用 50ETF 的 0.10 元。
        atm_moneyness_tol=100,
        # 若目标 DTE 合约成交量过低，优先回到同 strike 的更近月活跃合约。
        atm_min_total_volume=5000,
        atm_low_volume_search_near_month=True,
        # 中证1000股指期权合约乘数为每点 100 元。
        contract_multiplier=100,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    ),
    report=ReportConfig(
        output_root="output/backtest",
        daily_feature_cols=("yz_hv60", "atm_iv_percentile"),
    ),
)
