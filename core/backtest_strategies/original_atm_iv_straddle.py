from __future__ import annotations

from dataclasses import replace

import pandas as pd

from .base import BacktestStrategy


class OriginalAtmIvStraddleStrategy(BacktestStrategy):
    """Minimal absolute-ATM-IV straddle strategy with daily delta hedging."""

    strategy_id = "original_atm_iv_straddle"
    strategy_name = "原始 ATM IV 跨式策略"

    def __init__(self, config):
        strategy_config = replace(
            config.strategy,
            short_stop_loss_enabled=False,
            enable_delta_hedge=True,
            delta_hedge_tolerance_ratio=0.0,
            delta_residual_abs_tolerance=0.0,
            allow_etf_short_hedge=False,
            enable_atm_straddle_rebalance=True,
            short_volume_spike_exit_enabled=False,
            short_cooldown_after_long_iv_high_exit_days=0,
        )
        super().__init__(replace(config, strategy=strategy_config))

    def build_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        signals = features_df.copy()
        if "atm_iv" not in signals:
            signals["atm_iv"] = pd.NA
        atm_iv = pd.to_numeric(signals["atm_iv"], errors="coerce")
        signals["signal_iv"] = atm_iv
        signals["signal_iv_percentile"] = pd.NA
        signals["prev_atm_iv"] = atm_iv.shift(1)
        signals["prev_signal_iv"] = signals["prev_atm_iv"]
        signals["long_open_signal"] = (
            bool(self.config.strategy.enable_long_straddle)
            & atm_iv.le(self.config.strategy.long_open_iv_threshold)
        )
        signals["short_open_signal"] = (
            bool(self.config.strategy.enable_short_straddle)
            & atm_iv.ge(self.config.strategy.short_open_iv_threshold)
        )
        signals["short_open_regime"] = pd.Series(
            pd.NA,
            index=signals.index,
            dtype="object",
        )
        signals.loc[signals["short_open_signal"], "short_open_regime"] = "absolute"
        return signals

    def entry_target_qty(
        self,
        feature_row: pd.Series,
        max_qty: int,
        side: str,
    ) -> int:
        iv = pd.to_numeric(feature_row.get("atm_iv"), errors="coerce")
        if pd.isna(iv):
            return 0
        if side == "long":
            return (
                int(max_qty)
                if self.config.strategy.enable_long_straddle
                and float(iv) <= self.config.strategy.long_open_iv_threshold
                else 0
            )
        if side == "short":
            return (
                int(max_qty)
                if self.config.strategy.enable_short_straddle
                and float(iv) >= self.config.strategy.short_open_iv_threshold
                else 0
            )
        raise ValueError(f"Unsupported position side: {side}")

    def get_close_reason(
        self,
        feature_row: pd.Series,
        position_dte: int,
    ) -> str | None:
        iv = pd.to_numeric(feature_row.get("atm_iv"), errors="coerce")
        if pd.notna(iv) and float(iv) >= self.config.strategy.long_close_iv_threshold:
            return "iv_high"
        return None

    def get_short_close_reason(
        self,
        feature_row: pd.Series,
        position_dte: int,
        position: dict | None = None,
    ) -> str | None:
        iv = pd.to_numeric(feature_row.get("atm_iv"), errors="coerce")
        if pd.notna(iv) and float(iv) <= self.config.strategy.short_close_iv_threshold:
            return "short_iv_low"
        return None

    def short_entry_regime(self, feature_row: pd.Series) -> str | None:
        return (
            "absolute"
            if self.entry_target_qty(feature_row, 1, "short") > 0
            else None
        )

    @property
    def default_short_entry_regime(self) -> str:
        return "absolute"

    def is_short_daily_loss_stop(self, daily_pnl: float, aum: float) -> bool:
        return False

    def has_short_volume_spike(self, position: dict, call_row, put_row) -> bool:
        return False

    @property
    def enable_roll(self) -> bool:
        return True

    @property
    def enable_strike_roll(self) -> bool:
        return True

    @property
    def attempt_roll_without_current_entry_signal(self) -> bool:
        return True

    @property
    def evaluate_roll_entry_on_candidate(self) -> bool:
        return True

    @property
    def close_if_roll_candidate_unavailable(self) -> bool:
        return True

    @property
    def roll_dte_threshold(self) -> int:
        return int(self.config.strategy.roll_dte_threshold)

    @property
    def short_cooldown_after_long_iv_high_exit_days(self) -> int:
        return 0

    @property
    def enable_delta_hedge(self) -> bool:
        return True

    @property
    def delta_hedge_tolerance_ratio(self) -> float:
        return 0.0

    @property
    def allow_etf_short_hedge(self) -> bool:
        return False

    @property
    def enable_atm_straddle_rebalance(self) -> bool:
        return True

    @property
    def enable_atm_straddle_shape_rebalance(self) -> bool:
        return False

    def metadata(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "signal_iv": "atm_iv",
            "long_open_iv_threshold": self.config.strategy.long_open_iv_threshold,
            "long_close_iv_threshold": self.config.strategy.long_close_iv_threshold,
            "short_open_iv_threshold": self.config.strategy.short_open_iv_threshold,
            "short_close_iv_threshold": self.config.strategy.short_close_iv_threshold,
            "delta_hedge": {
                "frequency": "daily",
                "negative_delta": "buy_etf",
                "positive_delta": "rebalance_option_legs",
                "tolerance_ratio": 0.0,
                "residual_abs_tolerance": 0.0,
            },
            "disabled_behaviors": [
                "short_daily_loss_stop",
                "short_volume_spike_exit",
                "short_entry_cooldown",
                "dynamic_position_resize",
                "spike_pullback_wait",
            ],
            "roll": {
                "trigger": "dte_or_strike_tracking",
                "dte_threshold": self.roll_dte_threshold,
                "candidate": "next_month_atm",
                "entry_rule": "candidate_atm_iv",
                "close_if_unavailable_or_entry_rule_fails": True,
            },
            "strike_tracking": {
                "trigger": "held_strike_at_least_one_step_from_atm",
                "preserve_expiry": True,
                "reset_etf_hedge_before_switch": True,
                "entry_rule": "unconditional_position_maintenance",
            },
        }
