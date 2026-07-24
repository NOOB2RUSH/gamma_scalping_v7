from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np
import pandas as pd

from .dynamic_position_config import load_dynamic_position_straddle_config
from .iv_straddle_v1 import IvStraddleV1Strategy


class DynamicPositionStraddleStrategy(IvStraddleV1Strategy):
    """Deprecated prototype of stepwise IV-based position sizing."""

    strategy_id = "dynamic_position_straddle"
    strategy_name = "[已过期] 旧版动态仓位跨式原型"
    deprecated = True
    replacement_strategy_id = "dynamic_atm_iv_straddle"
    deprecation_message = (
        "Strategy 'dynamic_position_straddle' is a deprecated legacy prototype; "
        "use 'dynamic_atm_iv_straddle' for new strategy development."
    )
    SPIKE_CHANGE_COL = "atm_iv_absolute_change_signed"
    SPIKE_EVENT_COL = "short_iv_spike_event"
    UNDERLYING_LOG_RETURN_COL = "underlying_close_log_return"
    UNDERLYING_SPIKE_EVENT_COL = "short_underlying_return_spike_event"
    RISK_SPIKE_EVENT_COL = "short_risk_spike_event"
    SPIKE_WAITING_COL = "short_iv_spike_waiting"
    SPIKE_PULLBACK_COL = "short_iv_spike_pullback_confirmed"

    def __init__(self, config):
        super().__init__(config)
        self.dynamic_config = load_dynamic_position_straddle_config(
            config.data.product
        )

    def _signal_iv(self, feature_row: pd.Series):
        """This strategy always sizes from the observed close ATM IV."""
        return feature_row.get("atm_iv", pd.NA)

    def _long_signal_mode(self) -> str:
        return "absolute"

    def _short_signal_mode(self) -> str:
        return "absolute"

    def _long_open_signal(self, feature_row: pd.Series) -> bool:
        iv = self._signal_iv(feature_row)
        return pd.notna(iv) and iv <= self.dynamic_config.iv_long

    def _absolute_short_open_signal(self, feature_row: pd.Series) -> bool:
        iv = self._signal_iv(feature_row)
        return pd.notna(iv) and iv >= self.dynamic_config.iv_short

    def build_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add a no-lookahead spike/pullback state to the regular signal frame."""
        enriched = features_df.copy()
        atm_iv = pd.to_numeric(enriched.get("atm_iv"), errors="coerce")
        signed_change = atm_iv.diff()
        iv_spike_event = signed_change.ge(
            self.dynamic_config.iv_spike - 1e-12
        )
        if "close" in enriched:
            close = pd.to_numeric(enriched["close"], errors="coerce")
        else:
            close = pd.Series(np.nan, index=enriched.index, dtype=float)
        previous_close = close.shift(1)
        valid_close = close.gt(0) & previous_close.gt(0)
        close_log_return = pd.Series(np.nan, index=enriched.index, dtype=float)
        close_log_return.loc[valid_close] = np.log(
            close.loc[valid_close] / previous_close.loc[valid_close]
        )
        underlying_spike_event = close_log_return.abs().ge(
            self.dynamic_config.underlying_log_return_spike - 1e-12
        )
        risk_spike_event = iv_spike_event | underlying_spike_event

        waiting = False
        waiting_values = []
        pullback_values = []
        for change, is_spike in zip(signed_change, risk_spike_event):
            pullback_confirmed = False
            if bool(is_spike):
                waiting = True
            elif waiting and pd.notna(change) and float(change) < 0:
                waiting = False
                pullback_confirmed = True
            waiting_values.append(waiting)
            pullback_values.append(pullback_confirmed)

        enriched[self.SPIKE_CHANGE_COL] = signed_change
        enriched["atm_iv_absolute_change"] = signed_change.abs()
        enriched[self.SPIKE_EVENT_COL] = iv_spike_event
        enriched[self.UNDERLYING_LOG_RETURN_COL] = close_log_return
        enriched[self.UNDERLYING_SPIKE_EVENT_COL] = underlying_spike_event
        enriched[self.RISK_SPIKE_EVENT_COL] = risk_spike_event
        enriched[self.SPIKE_WAITING_COL] = waiting_values
        enriched[self.SPIKE_PULLBACK_COL] = pullback_values
        signals = super().build_signals(enriched)
        signals.loc[signals[self.SPIKE_WAITING_COL], "short_open_regime"] = None
        signals["short_open_signal"] = (
            self.config.strategy.enable_short_straddle
            & signals["short_open_regime"].notna()
        )
        return signals

    def _is_waiting_for_short_pullback(self, feature_row: pd.Series) -> bool:
        value = feature_row.get(self.SPIKE_WAITING_COL, False)
        return bool(value) if pd.notna(value) else False

    @staticmethod
    def _ladder_qty(
        iv: float,
        start_iv: float,
        end_iv: float,
        min_qty: int,
        max_qty: int,
        steps: int,
    ) -> int:
        """Map an IV level to its downward-rounded position ladder quantity."""
        if end_iv <= start_iv:
            raise ValueError("ladder end_iv must be greater than start_iv")
        progress = (float(iv) - start_iv) / (end_iv - start_iv)
        level = min(steps, max(0, math.floor(progress * steps + 1e-12)))
        return min_qty + math.floor(level * (max_qty - min_qty) / steps)

    def target_position_qty(self, feature_row: pd.Series, side: str) -> int | None:
        iv = self._signal_iv(feature_row)
        if pd.isna(iv):
            return None
        iv = float(iv)
        cfg = self.dynamic_config
        if side == "short":
            return self._ladder_qty(
                iv,
                cfg.iv_short,
                cfg.iv_max,
                cfg.min_qty,
                cfg.max_qty,
                cfg.short_steps,
            )
        if side == "long":
            return self._ladder_qty(
                -iv,
                -cfg.iv_long,
                -cfg.iv_min,
                cfg.min_qty,
                cfg.max_qty,
                cfg.long_steps,
            )
        raise ValueError(f"Unsupported position side: {side}")

    def entry_target_qty(
        self,
        feature_row: pd.Series,
        max_qty: int,
        side: str,
    ) -> int:
        if side == "short" and not self._absolute_short_open_signal(feature_row):
            return 0
        if side == "long" and not self._long_open_signal(feature_row):
            return 0
        target = self.target_position_qty(feature_row, side)
        return 0 if target is None else target

    def existing_position_target_qty(
        self,
        feature_row: pd.Series,
        side: str,
    ) -> int | None:
        if side == "short" and self._is_waiting_for_short_pullback(feature_row):
            return None
        return self.target_position_qty(feature_row, side)

    def roll_target_qty(
        self,
        feature_row: pd.Series,
        max_qty: int,
        side: str,
        current_qty: int,
    ) -> int:
        if side == "short" and self._is_waiting_for_short_pullback(feature_row):
            return int(current_qty)
        return self.entry_target_qty(feature_row, max_qty, side)

    def metadata(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "strategy_status": "deprecated",
            "deprecated": True,
            "replacement_strategy_id": self.replacement_strategy_id,
            "deprecation_message": self.deprecation_message,
            "dynamic_position_config": asdict(self.dynamic_config),
        }
