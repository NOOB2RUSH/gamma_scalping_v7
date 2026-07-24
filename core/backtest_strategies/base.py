from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BacktestStrategy(ABC):
    """Strategy behavior contract used exclusively by the historical backtester."""

    strategy_id: str

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def build_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Return the complete signal frame used by the backtest engine."""

    @abstractmethod
    def entry_target_qty(self, feature_row: pd.Series, max_qty: int, side: str) -> int:
        """Return the desired per-leg quantity for a new or rolled position."""

    def existing_position_target_qty(
        self,
        feature_row: pd.Series,
        side: str,
    ) -> int | None:
        """Return a per-leg target for an existing position, or ``None`` to hold."""
        return None

    def roll_target_qty(
        self,
        feature_row: pd.Series,
        max_qty: int,
        side: str,
        current_qty: int,
    ) -> int:
        """Return the replacement quantity when rolling an existing position."""
        return self.entry_target_qty(feature_row, max_qty, side)

    @property
    def preserve_position_qty_during_roll(self) -> bool:
        """Whether a contract switch must first retain the current pair quantity."""
        return False

    def roll_candidate_target_qty(
        self,
        call_row: pd.Series,
        put_row: pd.Series,
        max_qty: int,
        side: str,
    ) -> int | None:
        """Size a selected roll candidate before execution, or use legacy sizing."""
        return None

    @property
    def enable_roll(self) -> bool:
        """Whether the shared DTE/strike roll executor is enabled."""
        return True

    @property
    def enable_strike_roll(self) -> bool:
        """Whether moving one strike step away may trigger a roll."""
        return True

    @property
    def attempt_roll_without_current_entry_signal(self) -> bool:
        """Whether a roll attempt should inspect the replacement contract first."""
        return False

    @property
    def evaluate_roll_entry_on_candidate(self) -> bool:
        """Whether replacement sizing should use the candidate ATM IV."""
        return False

    @property
    def close_if_roll_candidate_unavailable(self) -> bool:
        """Whether to close when no replacement satisfies the entry rule."""
        return False

    def metadata(self) -> dict:
        """Return strategy-specific metadata persisted with backtest outputs."""
        return {"strategy_id": self.strategy_id}

    @abstractmethod
    def get_close_reason(self, feature_row: pd.Series, position_dte: int) -> str | None:
        """Return a long-position close reason, or ``None`` to keep holding."""

    @abstractmethod
    def get_short_close_reason(
        self,
        feature_row: pd.Series,
        position_dte: int,
        position: dict | None = None,
    ) -> str | None:
        """Return a short-position close reason, or ``None`` to keep holding."""

    @abstractmethod
    def short_entry_regime(self, feature_row: pd.Series) -> str | None:
        """Return the short-entry regime persisted on a new short position."""

    @property
    @abstractmethod
    def default_short_entry_regime(self) -> str:
        """Return the fallback regime used when rolling an existing short position."""

    @abstractmethod
    def is_short_daily_loss_stop(self, daily_pnl: float, aum: float) -> bool:
        """Return whether a short position should exit because of its daily loss."""

    @abstractmethod
    def has_short_volume_spike(self, position: dict, call_row, put_row) -> bool:
        """Return whether a short position should exit because of volume expansion."""

    @property
    @abstractmethod
    def roll_dte_threshold(self) -> int:
        """DTE threshold used by the generic roll executor."""

    @property
    @abstractmethod
    def short_cooldown_after_long_iv_high_exit_days(self) -> int:
        """Trading-day cooldown following a long IV-high exit."""

    @property
    @abstractmethod
    def enable_delta_hedge(self) -> bool:
        """Whether the strategy requests the shared delta-hedge executor."""

    @property
    @abstractmethod
    def delta_hedge_tolerance_ratio(self) -> float:
        """Normalized delta tolerance for the shared hedge executor."""

    @property
    def delta_residual_abs_tolerance(self) -> float:
        """Absolute delta tolerance; zero preserves legacy plugin behavior."""
        return 0.0

    @property
    @abstractmethod
    def allow_etf_short_hedge(self) -> bool:
        """Whether the shared hedge executor may hold a short ETF hedge."""

    @property
    @abstractmethod
    def enable_atm_straddle_rebalance(self) -> bool:
        """Whether shared ATM-leg delta rebalancing is enabled."""

    @property
    def enable_atm_straddle_shape_rebalance(self) -> bool:
        """Whether ATM-leg shape adjustment may run before an ETF hedge."""
        return self.enable_atm_straddle_rebalance

    @property
    def force_liquidate_adjusted_options(self) -> bool:
        """Whether exchange-adjusted contracts force same-day full liquidation."""
        return False

    @property
    def use_live_delta_execution_plan(self) -> bool:
        """Whether delta control delegates to the live target-state planner."""
        return False

    @property
    def persist_model_entry_volume_baseline(self) -> bool:
        """Whether simulated opens retain a model-derived volume baseline."""
        return True
