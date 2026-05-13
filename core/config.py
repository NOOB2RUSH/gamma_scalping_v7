from dataclasses import dataclass, field


@dataclass(frozen=True)
class BacktestConfig:
    start: str = "20180102"
    end: str = "20260101"
    test_date: str = "20180202"
    initial_cash: float = 1_000_000
    call_qty: int = 25
    put_qty: int = 25
    etf_fee_rate: float = 0.00005
    option_fee_per_contract: float = 2.0


@dataclass(frozen=True)
class StrategyConfig:
    open_iv_threshold: float = 0.15
    close_iv_threshold: float = 0.3
    min_exit_dte: int = 3
    roll_iv_threshold: float = 0.155
    roll_dte_threshold: int = 7
    roll_strike_mismatch_days: int = 2
    roll_cooldown_days: int = 1


@dataclass(frozen=True)
class VolConfig:
    annual_days: int = 252
    hv_windows: tuple[int, ...] = (60,)
    atm_target_dte: int = 20
    atm_target_dte_min: int = 7
    atm_target_dte_max: int = 30
    atm_moneyness_tol: float = 0.10
    contract_multiplier: int = 10000
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0


@dataclass(frozen=True)
class ReportConfig:
    output_root: str = "output"
    daily_feature_cols: tuple[str, ...] = ("yz_hv60",)


@dataclass(frozen=True)
class AppConfig:
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    vol: VolConfig = field(default_factory=VolConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


CONFIG = AppConfig()
