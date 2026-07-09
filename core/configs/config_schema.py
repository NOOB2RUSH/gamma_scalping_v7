from dataclasses import dataclass, field


@dataclass(frozen=True)
class BacktestConfig:
    start: str
    end: str
    test_date: str
    initial_cash: float = 1_000_000
    min_cash_reserve: float = 50_000
    long_qty: int = 40
    short_qty: int = 30
    etf_fee_rate: float = 0.00005
    option_fee_per_contract: float = 2.0
    liquidity_warning_volume_ratio: float = 0.005
    proportional_position_sizing_enabled: bool = False
    position_sizing_base_nav: float = 1_000_000.0
    dynamic_position_control_enabled: bool = False
    max_margin_to_nav_ratio: float = 0.80


@dataclass(frozen=True)
class DataConfig:
    product: str
    etf_dir: str
    opt_dir: str
    hedge_etf_dir: str | None = None


@dataclass(frozen=True)
class StrategyConfig:
    # 信号开关。
    enable_long_straddle: bool = True
    enable_short_straddle: bool = True

    # 买入跨式：可在 absolute / percentile 两种 IV 信号口径之间切换。
    long_signal_mode: str = "absolute"
    long_open_iv_threshold: float = 0.15
    long_close_iv_threshold: float = 0.28
    long_open_iv_percentile_threshold: float = 0.20
    long_close_iv_percentile_threshold: float = 0.60
    min_exit_dte: int = 3

    # 卖出跨式：可在 absolute / percentile 两种 ATM IV 信号口径之间切换。
    short_signal_mode: str = "absolute"
    short_open_iv_threshold: float = 0.16
    short_close_iv_threshold: float = 0.10
    short_open_pullback_iv_threshold: float | None = None
    short_open_iv_percentile_threshold: float = 0.75
    short_close_iv_percentile_threshold: float = 0.60
    short_low_iv_open_threshold: float = 0.17
    short_low_iv_close_threshold: float = 0.22
    short_low_iv_hv_spread_threshold: float = 0.03
    short_low_iv_close_spread_threshold: float = 0.0
    short_low_iv_hv_col: str = "yz_hv60"
    short_low_iv_overlay_enabled: bool = False
    short_stop_loss_enabled: bool = False
    short_daily_loss_aum_threshold: float = -0.015
    enable_delta_hedge: bool = False
    delta_hedge_tolerance_ratio: float = 0.05
    allow_etf_short_hedge: bool = True
    enable_option_delta_hedge: bool = False
    atm_rebalance_target_pair_qty: int | None = None

    # 卖方持仓期间，若持仓合约 call+put 成交量较开仓时明显放大，主动退出。
    short_volume_spike_exit_enabled: bool = True
    short_volume_spike_multiplier: float = 1.5
    short_cooldown_after_long_iv_high_exit_days: int = 3

    # 展期信号：按到期日和 ATM 档位偏离触发。
    roll_dte_threshold: int = 7
    roll_cooldown_days: int = 1


@dataclass(frozen=True)
class VolConfig:
    annual_days: int = 252
    hv_windows: tuple[int, ...] = (60,)
    atm_iv_percentile_window: int = 252
    # IV 观察/信号模式：
    # legacy: 兼容旧配置；simple_atm_absolute: 简单 ATM + 绝对 IV 阈值；
    # surface_percentile: 固定期限曲面 + 历史分位数阈值。
    iv_observation_mode: str = "legacy"
    atm_target_dte: int = 20
    atm_target_dte_min: int = 7
    atm_target_dte_max: int = 30
    atm_selection_mode: str = "target_dte"
    atm_moneyness_tol: float = 0.10
    atm_min_total_volume: float = 0.0
    atm_low_volume_search_near_month: bool = False
    contract_multiplier: int = 10000
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0

    # 固定期限波动率曲面。默认关闭，以免改变既有 ETF 配置行为。
    surface_atm_iv_enabled: bool = False
    surface_atm_target_dte: int = 30
    surface_standard_dtes: tuple[int, ...] = (15, 30, 45, 60, 90)
    surface_min_dte: int = 7
    surface_min_volume: float = 1.0
    surface_max_spread_pct: float | None = 0.50
    surface_min_abs_delta: float | None = 0.05
    surface_max_abs_delta: float | None = 0.95
    surface_allow_term_extrapolate: bool = False
    surface_term_extrapolate_mode: str = "linear"
    surface_k_grid_mode: str = "union"
    surface_raw_point_max_dte: int | None = 120


@dataclass(frozen=True)
class ReportConfig:
    output_root: str = "output/backtest"
    daily_feature_cols: tuple[str, ...] = ("yz_hv60", "atm_iv_percentile")
    enable_surface_full_sample_plot: bool = False


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    backtest: BacktestConfig
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
