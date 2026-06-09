"""华夏科创50ETF期权独立配置。"""

from .config_schema import (
    AppConfig,
    BacktestConfig,
    DataConfig,
    ReferenceCurveConfig,
    ReportConfig,
    StrategyConfig,
    VolConfig,
)

CONFIG = AppConfig(
    data=DataConfig(
        product="kc50etf",
        etf_dir="data/kc50etf/etf",
        opt_dir="data/kc50etf/option",
        hedge_etf_dir="data/kc50etf/etf",
    ),
    backtest=BacktestConfig(
        start="20230605",
        end="20260527",
        test_date="20230605",
        initial_cash=10_000_000,
        min_cash_reserve=50_000,
        long_qty=10,
        short_qty=80,
        etf_fee_rate=0.00005,
        option_fee_per_contract=2.0,
        proportional_position_sizing_enabled=True,
        position_sizing_base_nav=1_000_000.0,
        dynamic_position_control_enabled=True,
        max_margin_to_nav_ratio=0.80,
        # 科创50ETF 期权单合约历史成交量不是交易所逐合约原始数据，
        # 而是用 Sina 合约活跃度按上交所认购/认沽总量校准后的估算值。
        # 因此成交量预警只适合作为粗略流动性提示，不宜视为精确成交约束。
        liquidity_warning_volume_ratio=0.005,
    ),
    strategy=StrategyConfig(
        enable_long_straddle=True,
        enable_short_straddle=True,
        long_open_iv_threshold=0.150,
        long_close_iv_threshold=0.285,
        min_exit_dte=3,
        short_signal_mode="absolute",
        short_open_iv_threshold=0.250,
        short_close_iv_threshold=0.195,
        short_open_pullback_iv_threshold=0.40,
        short_open_iv_percentile_threshold=0.75,
        short_close_iv_percentile_threshold=0.60,
        short_stop_loss_enabled=False,
        short_stop_loss_rate=0.40,
        enable_delta_hedge=True,
        delta_hedge_tolerance_ratio=0.10,
        allow_etf_short_hedge=False,
        enable_option_delta_hedge=True,
        option_delta_hedge_combination_enabled=True,
        option_delta_hedge_max_itm_ratio=0.10,
        # 该退出信号依赖估算后的单合约成交量，适合捕捉明显放量，
        # 但不代表真实逐合约成交量的严格放大。
        short_volume_spike_exit_enabled=True,
        short_volume_spike_multiplier=1.5,
        short_cooldown_after_long_iv_high_exit_days=0,
        roll_dte_threshold=7,
        roll_strike_mismatch_days=2,
        roll_cooldown_days=10,
    ),
    vol=VolConfig(
        annual_days=252,
        hv_windows=(60,),
        atm_iv_percentile_window=252,
        atm_target_dte=20,
        atm_target_dte_min=7,
        atm_target_dte_max=30,
        atm_moneyness_tol=0.10,
        # ATM 低成交量过滤使用同一套估算成交量口径；
        # 可用于避开显著冷门合约，但不要把阈值理解为真实精确张数。
        atm_min_total_volume=5000,
        atm_low_volume_search_near_month=True,
        contract_multiplier=10000,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    ),
    report=ReportConfig(
        output_root="output",
        daily_feature_cols=("yz_hv60", "atm_iv_percentile"),
    ),
    reference=ReferenceCurveConfig(
        enable_always_atm=False,
        always_atm_side="short",
        always_atm_qty=5,
        always_atm_enable_delta_hedge=False,
        enable_experiment=False,
        experiment_short_signal_mode="absolute",
        experiment_enable_delta_hedge=False,
        experiment_enable_long_straddle=True,
        experiment_enable_short_straddle=True,
        experiment_short_stop_loss_enabled=True,
        experiment_short_volume_spike_exit_enabled=True,
        experiment_short_cooldown_after_long_iv_high_exit_days=6,
        experiment_long_qty=None,
        experiment_short_qty=None,
    ),
)
