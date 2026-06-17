"""南方中证500ETF期权独立配置。"""

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
        product="500etf",
        etf_dir="data/500etf/etf",
        opt_dir="data/500etf/option",
        hedge_etf_dir="data/500etf/etf",
    ),
    backtest=BacktestConfig(
        start="20220919",
        end="20260521",
        test_date="20220919",
        initial_cash=10_000_000,
        min_cash_reserve=50_000,
        long_qty=10,
        short_qty=10,
        etf_fee_rate=0.00005,
        option_fee_per_contract=2.0,
        # 500ETF 期权单合约历史成交量不是交易所逐合约原始数据，
        # 而是用 Sina 合约活跃度按上交所认购/认沽总量校准后的估算值。
        # 因此成交量预警只适合作为粗略流动性提示，不宜视为精确成交约束。
        liquidity_warning_volume_ratio=0.005,
    ),
    strategy=StrategyConfig(
        enable_long_straddle=True,
        enable_short_straddle=True,
        long_open_iv_threshold=0.120,
        long_close_iv_threshold=0.185,
        min_exit_dte=3,
        short_signal_mode="absolute",
        short_open_iv_threshold=0.180,
        short_close_iv_threshold=0.115,
        short_open_pullback_iv_threshold=0.25,
        short_open_iv_percentile_threshold=0.75,
        short_close_iv_percentile_threshold=0.60,
        short_stop_loss_enabled=True,
        short_stop_loss_rate=0.2,
        enable_delta_hedge=True,
        # 该退出信号依赖估算后的单合约成交量，适合捕捉明显放量，
        # 但不代表真实逐合约成交量的严格放大。
        short_volume_spike_exit_enabled=True,
        short_volume_spike_multiplier=1.5,
        short_cooldown_after_long_iv_high_exit_days=3,
        roll_dte_threshold=7,
        roll_strike_mismatch_days=2,
        roll_cooldown_days=6,
    ),
    vol=VolConfig(
        annual_days=252,
        hv_windows=(60,),
        atm_iv_percentile_window=252,
        atm_target_dte=20,
        atm_target_dte_min=7,
        atm_target_dte_max=30,
        atm_moneyness_tol=0.25,
        # ATM 低成交量过滤使用同一套估算成交量口径；
        # 可用于避开显著冷门合约，但不要把阈值理解为真实精确张数。
        atm_min_total_volume=5000,
        atm_low_volume_search_near_month=True,
        contract_multiplier=10000,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    ),
    report=ReportConfig(
        output_root="output/backtest",
        daily_feature_cols=("yz_hv60", "atm_iv_percentile"),
    ),
)
