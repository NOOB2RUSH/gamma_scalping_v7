"""Per-product defaults for the original absolute-ATM-IV straddle plugin."""


def _profile(
    *,
    long_open: float,
    long_close: float,
    short_open: float,
    short_close: float,
    target_dte: int,
    roll_dte: int,
):
    return {
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
    ),
    "300etf": _profile(
        long_open=0.14,
        long_close=0.16,
        short_open=0.20,
        short_close=0.16,
        target_dte=25,
        roll_dte=5,
    ),
    "500etf": _profile(
        long_open=0.12,
        long_close=0.22,
        short_open=0.24,
        short_close=0.18,
        target_dte=10,
        roll_dte=7,
    ),
    "kc50etf": _profile(
        long_open=0.20,
        long_close=0.23,
        short_open=0.35,
        short_close=0.31,
        target_dte=15,
        roll_dte=7,
    ),
}
