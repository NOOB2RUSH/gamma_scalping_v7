from __future__ import annotations

import warnings

from .base import BacktestStrategy
from .configs import (
    available_strategy_config_ids,
    load_strategy_config_overrides,
    resolve_strategy_config,
)
from .dynamic_atm_iv_straddle import DynamicAtmIvStraddleStrategy
from .dynamic_position_straddle import DynamicPositionStraddleStrategy
from .iv_straddle_v1 import IvStraddleV1Strategy
from .live_straddle import LiveStraddleStrategy
from .original_atm_iv_straddle import OriginalAtmIvStraddleStrategy


_STRATEGY_TYPES = {
    DynamicAtmIvStraddleStrategy.strategy_id: DynamicAtmIvStraddleStrategy,
    DynamicPositionStraddleStrategy.strategy_id: DynamicPositionStraddleStrategy,
    IvStraddleV1Strategy.strategy_id: IvStraddleV1Strategy,
    LiveStraddleStrategy.strategy_id: LiveStraddleStrategy,
    OriginalAtmIvStraddleStrategy.strategy_id: OriginalAtmIvStraddleStrategy,
}


def available_strategy_ids() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGY_TYPES))


def deprecated_strategy_ids() -> tuple[str, ...]:
    """Return registered strategies retained only for legacy comparisons."""
    return tuple(
        sorted(
            strategy_id
            for strategy_id, strategy_type in _STRATEGY_TYPES.items()
            if getattr(strategy_type, "deprecated", False)
        )
    )


def create_strategy(strategy_id: str, config) -> BacktestStrategy:
    try:
        strategy_type = _STRATEGY_TYPES[strategy_id]
    except KeyError as exc:
        choices = ", ".join(available_strategy_ids())
        raise ValueError(f"Unknown backtest strategy: {strategy_id}. Choices: {choices}") from exc
    if getattr(strategy_type, "deprecated", False):
        warnings.warn(
            strategy_type.deprecation_message,
            FutureWarning,
            stacklevel=2,
        )
    return strategy_type(config)


__all__ = [
    "BacktestStrategy",
    "available_strategy_config_ids",
    "available_strategy_ids",
    "create_strategy",
    "deprecated_strategy_ids",
    "load_strategy_config_overrides",
    "resolve_strategy_config",
]
