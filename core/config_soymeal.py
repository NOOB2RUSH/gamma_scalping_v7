"""豆粕期权独立配置。

第一版使用豆粕主连 M0 作为现有单标的回测引擎的近似标的。
严格的期货期权版本需要按每个期权合约对应的标的期货合约分别取价。
"""

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
        product="soymeal",
        etf_dir="data/soymeal/underlying",
        opt_dir="data/soymeal/option",
        hedge_etf_dir=None,
    ),
    backtest=BacktestConfig(
        start="20190918",
        end="20260526",
        test_date="20260520",
        initial_cash=1_000_000,
        min_cash_reserve=50_000,
        long_qty=10,
        short_qty=10,
        etf_fee_rate=0.0,
        option_fee_per_contract=2.0,
        liquidity_warning_volume_ratio=0.005,
    ),
    strategy=StrategyConfig(
        enable_long_straddle=True,
        enable_short_straddle=False,
        long_open_iv_threshold=0.18,
        long_close_iv_threshold=0.30,
        min_exit_dte=3,
        short_signal_mode="absolute",
        short_open_iv_threshold=0.28,
        short_close_iv_threshold=0.18,
        short_open_pullback_iv_threshold=0.35,
        short_stop_loss_enabled=True,
        short_stop_loss_rate=0.2,
        # 现有 delta hedge 模块按 ETF/单一标的现金买卖处理；豆粕期货对冲需单独适配保证金。
        enable_delta_hedge=False,
        short_volume_spike_exit_enabled=True,
        short_volume_spike_multiplier=1.5,
        short_cooldown_after_long_iv_high_exit_days=3,
        roll_dte_threshold=7,
        roll_strike_mismatch_days=2,
        roll_cooldown_days=3,
    ),
    vol=VolConfig(
        annual_days=252,
        hv_windows=(60,),
        atm_iv_percentile_window=252,
        atm_target_dte=20,
        atm_target_dte_min=5,
        atm_target_dte_max=45,
        # 豆粕价格在数千元/吨，行权价间距通常为 50；这里使用绝对价差。
        atm_moneyness_tol=100.0,
        # Sina 商品期权历史日线只返回有成交记录的合约日线，成交量可用但期权链可能不完整。
        atm_min_total_volume=0,
        atm_low_volume_search_near_month=False,
        contract_multiplier=10,
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
        always_atm_qty=10,
        always_atm_enable_delta_hedge=False,
        enable_experiment=False,
    ),
)
