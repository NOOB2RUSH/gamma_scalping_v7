from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from core import position, strategy

from .base import BacktestStrategy


class LiveStraddleStrategy(BacktestStrategy):
    """Historical adapter for the current live straddle policy.

    Strategy predicates deliberately delegate to ``core.strategy`` instead of
    copying them.  The live signal engine uses those same functions, making the
    shared implementation the source of truth while the backtester remains the
    owner of simulated cash, fills and positions.
    """

    strategy_id = "live_straddle"
    strategy_name = "Live 跨式镜像策略"
    strategy_status = "live_mirror"

    def build_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return strategy.build_signals(features_df, config=self.config)

    def entry_target_qty(
        self,
        feature_row: pd.Series,
        max_qty: int,
        side: str,
    ) -> int:
        if side == "short":
            return strategy.calc_short_entry_target_qty(
                feature_row,
                max_qty,
                config=self.config,
            )
        if side == "long":
            return strategy.calc_entry_target_qty(
                feature_row,
                max_qty,
                config=self.config,
            )
        raise ValueError(f"Unsupported position side: {side}")

    def roll_target_qty(
        self,
        feature_row: pd.Series,
        max_qty: int,
        side: str,
        current_qty: int,
    ) -> int:
        # Live restores the configured per-leg target whenever a required roll
        # finds an eligible replacement; it does not re-run the entry threshold.
        return int(max_qty)

    def get_close_reason(
        self,
        feature_row: pd.Series,
        position_dte: int,
    ) -> str | None:
        return strategy.get_close_reason(
            feature_row,
            position_dte,
            config=self.config,
        )

    def get_short_close_reason(
        self,
        feature_row: pd.Series,
        position_dte: int,
        position: dict | None = None,
    ) -> str | None:
        return strategy.get_short_close_reason(
            feature_row,
            position_dte,
            position,
            config=self.config,
        )

    def short_entry_regime(self, feature_row: pd.Series) -> str | None:
        return strategy.get_short_open_regime(feature_row, config=self.config)

    @property
    def default_short_entry_regime(self) -> str:
        return strategy._short_signal_mode(self.config)

    def is_short_daily_loss_stop(self, daily_pnl: float, aum: float) -> bool:
        return strategy.is_short_daily_loss_aum_stop(
            daily_pnl,
            aum,
            config=self.config,
        )

    def has_short_volume_spike(self, position_state: dict, call_row, put_row) -> bool:
        return position.has_short_volume_spike(
            position_state,
            call_row,
            put_row,
            config=self.config,
        )

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
        return False

    @property
    def close_if_roll_candidate_unavailable(self) -> bool:
        return True

    @property
    def roll_dte_threshold(self) -> int:
        return int(self.config.strategy.roll_dte_threshold)

    @property
    def short_cooldown_after_long_iv_high_exit_days(self) -> int:
        return int(
            self.config.strategy.short_cooldown_after_long_iv_high_exit_days
        )

    @property
    def enable_delta_hedge(self) -> bool:
        return bool(self.config.strategy.enable_delta_hedge)

    @property
    def delta_hedge_tolerance_ratio(self) -> float:
        return float(self.config.strategy.delta_hedge_tolerance_ratio)

    @property
    def delta_residual_abs_tolerance(self) -> float:
        return max(
            0.0,
            float(self.config.strategy.delta_residual_abs_tolerance),
        )

    @property
    def allow_etf_short_hedge(self) -> bool:
        return bool(self.config.strategy.allow_etf_short_hedge)

    @property
    def enable_atm_straddle_rebalance(self) -> bool:
        return bool(self.config.strategy.enable_atm_straddle_rebalance)

    @property
    def enable_atm_straddle_shape_rebalance(self) -> bool:
        return bool(self.config.strategy.enable_atm_straddle_rebalance)

    @property
    def force_liquidate_adjusted_options(self) -> bool:
        return True

    @property
    def use_live_delta_execution_plan(self) -> bool:
        return True

    @property
    def persist_model_entry_volume_baseline(self) -> bool:
        # Broker-confirmed live positions do not currently carry signal-time
        # option volume fields, so retaining them only in historical state would
        # create a backtest-only volume-spike exit.
        return False

    def metadata(self) -> dict:
        effective_config = asdict(self.config)
        serialized = json.dumps(
            effective_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        source_hashes = _policy_source_hashes()
        source_manifest = json.dumps(
            source_hashes,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "metadata_schema_version": 1,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "strategy_status": self.strategy_status,
            "policy_source": "core.strategy_and_shared_execution_contracts",
            "live_product": self.config.data.product,
            "effective_config_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
            "effective_config": effective_config,
            "policy_source_files_sha256": source_hashes,
            "policy_source_manifest_sha256": hashlib.sha256(
                source_manifest.encode("utf-8")
            ).hexdigest(),
            "entry_qty_per_leg": {
                "long": int(self.config.backtest.long_qty),
                "short": int(self.config.backtest.short_qty),
            },
            "state_contract": {
                "persist_model_entry_volume_baseline": (
                    self.persist_model_entry_volume_baseline
                ),
                "reason": "broker_confirmed_live_positions_do_not_persist_entry_volume",
            },
            "roll": {
                "dte_threshold": self.roll_dte_threshold,
                "strike_trigger": "at_least_one_strike_step",
                "replacement_qty": "configured_per_leg_target",
                "close_if_unavailable": True,
            },
            "delta_control": {
                "execution_plan_source": "core.live.signal_engine._delta_hedge_plan",
                "etf_netting_source": "core.live.etf_netting.netted_etf_advice_items",
                "tolerance_ratio": self.delta_hedge_tolerance_ratio,
                "absolute_tolerance": self.delta_residual_abs_tolerance,
                "allow_short_etf": self.allow_etf_short_hedge,
                "option_legs_before_etf_fine_tuning": (
                    self.enable_atm_straddle_rebalance
                ),
            },
        }


def _policy_source_hashes() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        "core/strategy.py",
        "core/position.py",
        "core/backtester.py",
        "core/live/signal_engine.py",
        "core/live/etf_netting.py",
        "core/backtest_strategies/live_straddle.py",
    )
    return {
        relative: hashlib.sha256((project_root / relative).read_bytes()).hexdigest()
        for relative in paths
    }
