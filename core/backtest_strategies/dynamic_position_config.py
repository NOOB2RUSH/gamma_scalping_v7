from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicPositionStraddleConfig:
    """IV ladder and position ladder used by the dynamic straddle plugin."""

    min_qty: int
    max_qty: int
    short_steps: int
    long_steps: int
    iv_max: float
    iv_min: float
    iv_short: float
    iv_long: float
    iv_spike: float
    underlying_log_return_spike: float

    def validate(self) -> None:
        if self.min_qty <= 0:
            raise ValueError("dynamic min_qty must be positive")
        if self.max_qty < self.min_qty:
            raise ValueError("dynamic max_qty must be >= min_qty")
        if self.short_steps <= 0 or self.long_steps <= 0:
            raise ValueError("dynamic short_steps and long_steps must be positive")
        if not 0 < self.iv_min < self.iv_long:
            raise ValueError("dynamic IV levels must satisfy 0 < iv_min < iv_long")
        if not self.iv_short < self.iv_max:
            raise ValueError("dynamic IV levels must satisfy iv_short < iv_max")
        if self.iv_spike <= 0:
            raise ValueError("dynamic iv_spike must be positive")
        if self.underlying_log_return_spike <= 0:
            raise ValueError(
                "dynamic underlying_log_return_spike must be positive"
            )


DYNAMIC_POSITION_STRADDLE_CONFIGS = {
    "300etf": DynamicPositionStraddleConfig(
        min_qty=8,
        max_qty=24,
        short_steps=10,
        long_steps=10,
        iv_max=0.35,
        iv_min=0.05,
        iv_short=0.155,
        iv_long=0.08,
        iv_spike=0.03,
        underlying_log_return_spike=0.03,
    ),
}


def load_dynamic_position_straddle_config(
    product: str,
) -> DynamicPositionStraddleConfig:
    try:
        result = DYNAMIC_POSITION_STRADDLE_CONFIGS[str(product).lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(DYNAMIC_POSITION_STRADDLE_CONFIGS))
        raise ValueError(
            f"Dynamic position straddle is not configured for {product}. "
            f"Configured products: {choices}"
        ) from exc
    result.validate()
    return result
