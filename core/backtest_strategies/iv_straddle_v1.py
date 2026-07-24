from __future__ import annotations

import pandas as pd

from .base import BacktestStrategy


LONG_SIGNAL_MODES = {"absolute", "percentile"}
SHORT_SIGNAL_MODES = {"absolute", "percentile", "low_iv_hv_spread"}
IV_OBSERVATION_MODES = {"legacy", "simple_atm_absolute", "surface_percentile"}


class IvStraddleV1Strategy(BacktestStrategy):
    """Backtest-only copy of the current live IV straddle decision behavior."""

    strategy_id = "iv_straddle_v1"

    def _iv_observation_mode(self) -> str:
        mode = getattr(self.config.vol, "iv_observation_mode", "legacy")
        if mode not in IV_OBSERVATION_MODES:
            raise ValueError(f"Unsupported IV observation mode: {mode}")
        return mode

    def _long_signal_mode(self) -> str:
        observation_mode = self._iv_observation_mode()
        if observation_mode == "simple_atm_absolute":
            return "absolute"
        if observation_mode == "surface_percentile":
            return "percentile"
        mode = getattr(self.config.strategy, "long_signal_mode", "absolute")
        if mode not in LONG_SIGNAL_MODES:
            raise ValueError(f"Unsupported long signal mode: {mode}")
        return mode

    def _short_signal_mode(self) -> str:
        observation_mode = self._iv_observation_mode()
        if observation_mode == "simple_atm_absolute":
            return "absolute"
        if observation_mode == "surface_percentile":
            return "percentile"
        mode = self.config.strategy.short_signal_mode
        if mode not in SHORT_SIGNAL_MODES:
            raise ValueError(f"Unsupported short signal mode: {mode}")
        return mode

    def _signal_iv(self, feature_row: pd.Series):
        if self._iv_observation_mode() == "simple_atm_absolute":
            return feature_row.get("atm_iv", pd.NA)
        value = feature_row.get("signal_iv", pd.NA)
        return feature_row.get("atm_iv", pd.NA) if pd.isna(value) else value

    def _signal_iv_percentile(self, feature_row: pd.Series):
        if self._iv_observation_mode() == "simple_atm_absolute":
            return pd.NA
        value = feature_row.get("signal_iv_percentile", pd.NA)
        return feature_row.get("atm_iv_percentile", pd.NA) if pd.isna(value) else value

    def _long_open_signal(self, feature_row: pd.Series) -> bool:
        if self._long_signal_mode() == "percentile":
            percentile = self._signal_iv_percentile(feature_row)
            return (
                pd.notna(percentile)
                and percentile <= self.config.strategy.long_open_iv_percentile_threshold
            )
        iv = self._signal_iv(feature_row)
        return pd.notna(iv) and iv <= self.config.strategy.long_open_iv_threshold

    def _long_close_reason(self, feature_row: pd.Series) -> str | None:
        if self._long_signal_mode() == "percentile":
            percentile = self._signal_iv_percentile(feature_row)
            if (
                pd.notna(percentile)
                and percentile >= self.config.strategy.long_close_iv_percentile_threshold
            ):
                return "iv_percentile_high"
            return None
        iv = self._signal_iv(feature_row)
        if pd.notna(iv) and iv >= self.config.strategy.long_close_iv_threshold:
            return "iv_high"
        return None

    def _low_iv_hv_spread_open_signal(self, feature_row: pd.Series) -> bool:
        iv = self._signal_iv(feature_row)
        hv = feature_row.get(self.config.strategy.short_low_iv_hv_col, pd.NA)
        return bool(
            pd.notna(iv)
            and pd.notna(hv)
            and iv <= self.config.strategy.short_low_iv_open_threshold
            and iv - hv >= self.config.strategy.short_low_iv_hv_spread_threshold
        )

    def _absolute_short_open_signal(self, feature_row: pd.Series) -> bool:
        iv = self._signal_iv(feature_row)
        if pd.isna(iv) or iv < self.config.strategy.short_open_iv_threshold:
            return False
        pullback_threshold = self.config.strategy.short_open_pullback_iv_threshold
        if pullback_threshold is not None and iv >= pullback_threshold:
            previous_iv = feature_row.get("prev_signal_iv", pd.NA)
            if pd.isna(previous_iv):
                previous_iv = feature_row.get("prev_atm_iv", pd.NA)
            return pd.notna(previous_iv) and iv < previous_iv
        return True

    def _short_open_regime(self, feature_row: pd.Series) -> str | None:
        mode = self._short_signal_mode()
        if mode == "percentile":
            percentile = self._signal_iv_percentile(feature_row)
            return (
                "percentile"
                if pd.notna(percentile)
                and percentile >= self.config.strategy.short_open_iv_percentile_threshold
                else None
            )
        if mode == "low_iv_hv_spread":
            return "low_iv_hv_spread" if self._low_iv_hv_spread_open_signal(feature_row) else None
        if self._absolute_short_open_signal(feature_row):
            return "absolute"
        if (
            self.config.strategy.short_low_iv_overlay_enabled
            and self._low_iv_hv_spread_open_signal(feature_row)
        ):
            return "low_iv_hv_spread"
        return None

    def _short_close_reason(
        self, feature_row: pd.Series, mode: str | None = None
    ) -> str | None:
        mode = mode or self._short_signal_mode()
        if mode == "percentile":
            percentile = self._signal_iv_percentile(feature_row)
            if (
                pd.notna(percentile)
                and percentile <= self.config.strategy.short_close_iv_percentile_threshold
            ):
                return "short_iv_percentile_low"
            return None
        if mode == "low_iv_hv_spread":
            iv = self._signal_iv(feature_row)
            hv = feature_row.get(self.config.strategy.short_low_iv_hv_col, pd.NA)
            if pd.isna(iv) or pd.isna(hv):
                return None
            if iv >= self.config.strategy.short_low_iv_close_threshold:
                return "short_low_iv_high"
            if iv - hv <= self.config.strategy.short_low_iv_close_spread_threshold:
                return "short_low_iv_spread_gone"
            return None
        iv = self._signal_iv(feature_row)
        if pd.notna(iv) and iv <= self.config.strategy.short_close_iv_threshold:
            return "short_iv_low"
        return None

    def build_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        signals = features_df.copy()
        if "atm_iv" not in signals:
            signals["atm_iv"] = pd.NA
        if "atm_iv_percentile" not in signals:
            signals["atm_iv_percentile"] = pd.NA
        signals["signal_iv"] = signals.get("signal_iv", signals["atm_iv"]).fillna(
            signals["atm_iv"]
        )
        if self._iv_observation_mode() == "simple_atm_absolute":
            signals["signal_iv"] = signals["atm_iv"]
            signals["signal_iv_percentile"] = pd.NA
        else:
            signals["signal_iv_percentile"] = signals.get(
                "signal_iv_percentile", signals["atm_iv_percentile"]
            ).fillna(signals["atm_iv_percentile"])
        signals["prev_atm_iv"] = signals["atm_iv"].shift(1)
        signals["prev_signal_iv"] = signals["signal_iv"].shift(1)
        long_open = signals.apply(self._long_open_signal, axis=1)
        short_regime = signals.apply(self._short_open_regime, axis=1)
        signals["long_open_signal"] = self.config.strategy.enable_long_straddle & long_open
        signals["short_open_regime"] = short_regime
        signals["short_open_signal"] = (
            self.config.strategy.enable_short_straddle & short_regime.notna()
        )
        return signals

    def entry_target_qty(self, feature_row: pd.Series, max_qty: int, side: str) -> int:
        if side == "short":
            return max_qty if self._short_open_regime(feature_row) is not None else 0
        return max_qty if self._long_open_signal(feature_row) else 0

    def get_close_reason(self, feature_row: pd.Series, position_dte: int) -> str | None:
        return self._long_close_reason(feature_row) or (
            "near_expiry" if position_dte <= self.config.strategy.min_exit_dte else None
        )

    def get_short_close_reason(
        self,
        feature_row: pd.Series,
        position_dte: int,
        position: dict | None = None,
    ) -> str | None:
        mode = position.get("short_entry_regime") if position else None
        return self._short_close_reason(feature_row, mode) or (
            "near_expiry" if position_dte <= self.config.strategy.min_exit_dte else None
        )

    def short_entry_regime(self, feature_row: pd.Series) -> str | None:
        return self._short_open_regime(feature_row)

    @property
    def default_short_entry_regime(self) -> str:
        return self._short_signal_mode()

    def is_short_daily_loss_stop(self, daily_pnl: float, aum: float) -> bool:
        return bool(
            self.config.strategy.short_stop_loss_enabled
            and aum > 0
            and daily_pnl / aum < self.config.strategy.short_daily_loss_aum_threshold
        )

    def has_short_volume_spike(self, position: dict, call_row, put_row) -> bool:
        if not self.config.strategy.short_volume_spike_exit_enabled:
            return False
        entry_volume = position.get("entry_total_volume")
        call_volume = call_row.get("volume")
        put_volume = put_row.get("volume")
        try:
            entry_volume = float(entry_volume)
            current_volume = float(call_volume) + float(put_volume)
        except (TypeError, ValueError):
            return False
        return entry_volume > 0 and (
            current_volume
            >= entry_volume * self.config.strategy.short_volume_spike_multiplier
        )

    @property
    def roll_dte_threshold(self) -> int:
        return int(self.config.strategy.roll_dte_threshold)

    @property
    def short_cooldown_after_long_iv_high_exit_days(self) -> int:
        return int(self.config.strategy.short_cooldown_after_long_iv_high_exit_days)

    @property
    def enable_delta_hedge(self) -> bool:
        return bool(self.config.strategy.enable_delta_hedge)

    @property
    def delta_hedge_tolerance_ratio(self) -> float:
        return float(self.config.strategy.delta_hedge_tolerance_ratio)

    @property
    def allow_etf_short_hedge(self) -> bool:
        return bool(self.config.strategy.allow_etf_short_hedge)

    @property
    def enable_atm_straddle_rebalance(self) -> bool:
        return bool(self.config.strategy.enable_atm_straddle_rebalance)
