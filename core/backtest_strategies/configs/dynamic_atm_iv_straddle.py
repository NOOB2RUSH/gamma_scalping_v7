"""Per-product defaults for the active dynamic ATM-IV straddle plugin."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AbsoluteIvPositionLadder:
    """Short-straddle quantity ladder driven only by the absolute ATM IV."""

    min_qty: int
    max_qty: int
    steps: int
    full_position_iv: float

    def validate(self, *, open_iv: float) -> None:
        if self.min_qty <= 0:
            raise ValueError("position ladder min_qty must be positive")
        if self.max_qty < self.min_qty:
            raise ValueError("position ladder max_qty must be >= min_qty")
        if self.steps <= 0:
            raise ValueError("position ladder steps must be positive")
        if self.full_position_iv <= open_iv:
            raise ValueError(
                "position ladder full_position_iv must exceed short open IV"
            )


def _profile(
    *,
    long_open: float,
    long_close: float,
    short_open: float,
    short_close: float,
    target_dte: int,
    roll_dte: int,
    quantity: int,
):
    return {
        "backtest": {
            "long_qty": quantity,
            "short_qty": quantity,
        },
        "strategy": {
            "enable_long_straddle": False,
            "enable_short_straddle": True,
            "long_signal_mode": "absolute",
            "long_open_iv_threshold": long_open,
            "long_close_iv_threshold": long_close,
            "short_signal_mode": "absolute",
            "short_open_iv_threshold": short_open,
            "short_close_iv_threshold": short_close,
            "short_stop_loss_enabled": False,
            "enable_delta_hedge": True,
            "delta_hedge_tolerance_ratio": 0.0,
            "delta_residual_abs_tolerance": 0.0,
            "allow_etf_short_hedge": False,
            "enable_atm_straddle_rebalance": True,
            "short_volume_spike_exit_enabled": False,
            "short_cooldown_after_long_iv_high_exit_days": 0,
            "roll_dte_threshold": roll_dte,
        },
        "vol": {
            "atm_target_dte": target_dte,
        },
    }


PRODUCT_OVERRIDES = {
    "50etf": _profile(
        long_open=0.14,
        long_close=0.16,
        short_open=0.20,
        short_close=0.175,
        target_dte=15,
        roll_dte=7,
        quantity=35,
    ),
    "300etf": _profile(
        long_open=0.14,
        long_close=0.16,
        short_open=0.20,
        short_close=0.16,
        target_dte=25,
        roll_dte=5,
        quantity=20,
    ),
    "500etf": _profile(
        long_open=0.12,
        long_close=0.22,
        short_open=0.24,
        short_close=0.18,
        target_dte=10,
        roll_dte=7,
        quantity=10,
    ),
    "kc50etf": _profile(
        long_open=0.20,
        long_close=0.23,
        short_open=0.35,
        short_close=0.31,
        target_dte=15,
        roll_dte=7,
        quantity=40,
    ),
}


# Conservative first version: the Gamma-risk-ratio quantity is the maximum,
# roughly half of it is held at the short-entry IV, and the interval is split
# into five downward-rounded steps.  Full-position IVs are rounded common-window
# historical p95 ATM-IV levels.
POSITION_LADDERS = {
    "50etf": AbsoluteIvPositionLadder(18, 35, 5, 0.23),
    "300etf": AbsoluteIvPositionLadder(10, 20, 5, 0.24),
    "500etf": AbsoluteIvPositionLadder(5, 10, 5, 0.30),
    "kc50etf": AbsoluteIvPositionLadder(20, 40, 5, 0.70),
}


def load_position_ladder(product: str) -> AbsoluteIvPositionLadder:
    try:
        return POSITION_LADDERS[str(product).lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(POSITION_LADDERS))
        raise ValueError(
            f"Unsupported dynamic ATM-IV ladder product: {product}. "
            f"Choices: {choices}"
        ) from exc
