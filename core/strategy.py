import math

import pandas as pd

from .config import CONFIG


LONG_SIGNAL_MODES = {"absolute", "percentile"}
SHORT_SIGNAL_MODES = {"absolute", "percentile", "low_iv_hv_spread"}
IV_OBSERVATION_MODES = {"legacy", "simple_atm_absolute", "surface_percentile"}
ETF_HEDGE_LOT_SIZE = 100


def round_etf_hedge_target(qty, lot_size=ETF_HEDGE_LOT_SIZE):
    """Round an ETF hedge target to the nearest executable board lot."""
    qty = float(qty)
    lot_size = int(lot_size)
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if abs(qty) < 1e-12:
        return 0.0
    lots = math.floor(abs(qty) / lot_size + 0.5)
    return float(math.copysign(lots * lot_size, qty))


def option_delta_capacity(positions, option_hedges=None, default_multiplier=10000):
    """Return the underlying-unit capacity used to normalize account delta."""
    capacity = 0.0
    for position in (positions or {}).values():
        if position is None:
            continue
        qty = max(
            abs(float(position.get("call_qty", 0) or 0)),
            abs(float(position.get("put_qty", 0) or 0)),
        )
        multiplier = float(
            position.get("contract_multiplier", default_multiplier)
            or default_multiplier
        )
        capacity += qty * multiplier

    for position in option_hedges or []:
        qty = abs(
            float(
                position.get(
                    "qty",
                    position.get("call_qty", position.get("put_qty", 0)),
                )
                or 0
            )
        )
        multiplier = float(
            position.get("contract_multiplier", default_multiplier)
            or default_multiplier
        )
        capacity += qty * multiplier
    return capacity


def normalized_account_delta(
    account_delta,
    positions,
    option_hedges=None,
    default_multiplier=10000,
):
    capacity = option_delta_capacity(
        positions,
        option_hedges=option_hedges,
        default_multiplier=default_multiplier,
    )
    if capacity <= 0:
        return 0.0, 0.0
    return float(account_delta) / capacity, capacity


def _iv_observation_mode():
    mode = getattr(CONFIG.vol, "iv_observation_mode", "legacy")
    if mode not in IV_OBSERVATION_MODES:
        raise ValueError(
            "CONFIG.vol.iv_observation_mode 只能是 "
            "'legacy'、'simple_atm_absolute' 或 'surface_percentile'"
        )
    return mode


def _long_signal_mode():
    """读取买入跨式主信号口径。"""
    observation_mode = _iv_observation_mode()
    if observation_mode == "simple_atm_absolute":
        return "absolute"
    if observation_mode == "surface_percentile":
        return "percentile"

    mode = getattr(CONFIG.strategy, "long_signal_mode", "absolute")
    if mode not in LONG_SIGNAL_MODES:
        raise ValueError(
            "CONFIG.strategy.long_signal_mode 只能是 'absolute' 或 'percentile'"
        )
    return mode


def _short_signal_mode():
    """读取卖出跨式主信号口径。"""
    observation_mode = _iv_observation_mode()
    if observation_mode == "simple_atm_absolute":
        return "absolute"
    if observation_mode == "surface_percentile":
        return "percentile"

    mode = CONFIG.strategy.short_signal_mode
    if mode not in SHORT_SIGNAL_MODES:
        raise ValueError(
            "CONFIG.strategy.short_signal_mode 只能是 "
            "'absolute'、'percentile' 或 'low_iv_hv_spread'"
        )
    return mode


def _signal_iv(feature_row):
    """策略使用的 IV：按观察模式在简单 ATM 和曲面 ATM 之间切换。"""
    if _iv_observation_mode() == "simple_atm_absolute":
        return feature_row.get("atm_iv", pd.NA)

    value = feature_row.get("signal_iv", pd.NA)
    if pd.isna(value):
        value = feature_row.get("atm_iv", pd.NA)
    return value


def _signal_iv_percentile(feature_row):
    """策略使用 IV 的历史分位数。"""
    if _iv_observation_mode() == "simple_atm_absolute":
        return pd.NA

    value = feature_row.get("signal_iv_percentile", pd.NA)
    if pd.isna(value):
        value = feature_row.get("atm_iv_percentile", pd.NA)
    return value


def _calc_long_open_signal(feature_row):
    mode = _long_signal_mode()
    if mode == "percentile":
        iv_percentile = _signal_iv_percentile(feature_row)
        return (
            pd.notna(iv_percentile)
            and iv_percentile <= CONFIG.strategy.long_open_iv_percentile_threshold
        )

    iv = _signal_iv(feature_row)
    return pd.notna(iv) and iv <= CONFIG.strategy.long_open_iv_threshold


def _get_long_close_reason_by_signal(feature_row):
    mode = _long_signal_mode()
    if mode == "percentile":
        iv_percentile = _signal_iv_percentile(feature_row)
        if (
            pd.notna(iv_percentile)
            and iv_percentile >= CONFIG.strategy.long_close_iv_percentile_threshold
        ):
            return "iv_percentile_high"
        return None

    iv = _signal_iv(feature_row)
    if pd.notna(iv) and iv >= CONFIG.strategy.long_close_iv_threshold:
        return "iv_high"
    return None


def _calc_low_iv_hv_spread_open_signal(feature_row):
    """低 IV 收租：IV 处于低位，但仍高于 HV 一定安全垫时开仓。"""
    atm_iv = _signal_iv(feature_row)
    hv = feature_row.get(CONFIG.strategy.short_low_iv_hv_col, pd.NA)
    if pd.isna(atm_iv) or pd.isna(hv):
        return False

    iv_is_low = atm_iv <= CONFIG.strategy.short_low_iv_open_threshold
    spread_is_positive = (
        atm_iv - hv >= CONFIG.strategy.short_low_iv_hv_spread_threshold
    )
    return iv_is_low and spread_is_positive


def _calc_absolute_short_open_signal(feature_row):
    """高 IV 卖波动：达到普通阈值即可开仓，极端高位需先确认 IV 回落。"""
    atm_iv = _signal_iv(feature_row)
    if pd.isna(atm_iv) or atm_iv < CONFIG.strategy.short_open_iv_threshold:
        return False

    pullback_threshold = CONFIG.strategy.short_open_pullback_iv_threshold
    if pullback_threshold is not None and atm_iv >= pullback_threshold:
        prev_atm_iv = feature_row.get("prev_signal_iv", pd.NA)
        if pd.isna(prev_atm_iv):
            prev_atm_iv = feature_row.get("prev_atm_iv", pd.NA)
        return pd.notna(prev_atm_iv) and atm_iv < prev_atm_iv

    return True


def _short_open_regime(feature_row):
    """返回卖方开仓信号来源；没有信号时返回 None。"""
    mode = _short_signal_mode()

    if mode == "percentile":
        atm_iv_percentile = _signal_iv_percentile(feature_row)
        if (
            pd.notna(atm_iv_percentile)
            and atm_iv_percentile >= CONFIG.strategy.short_open_iv_percentile_threshold
        ):
            return "percentile"
        return None

    if mode == "low_iv_hv_spread":
        if _calc_low_iv_hv_spread_open_signal(feature_row):
            return "low_iv_hv_spread"
        return None

    if _calc_absolute_short_open_signal(feature_row):
        return "absolute"
    if (
        CONFIG.strategy.short_low_iv_overlay_enabled
        and _calc_low_iv_hv_spread_open_signal(feature_row)
    ):
        return "low_iv_hv_spread"
    return None


def _calc_short_open_signal(feature_row):
    """按配置选择卖出跨式开仓口径。"""
    return _short_open_regime(feature_row) is not None


def _get_short_close_reason_by_signal(feature_row, mode=None):
    """按信号口径选择卖出跨式平仓原因。"""
    mode = mode or _short_signal_mode()

    if mode == "percentile":
        atm_iv_percentile = _signal_iv_percentile(feature_row)
        if (
            pd.notna(atm_iv_percentile)
            and atm_iv_percentile <= CONFIG.strategy.short_close_iv_percentile_threshold
        ):
            return "short_iv_percentile_low"
        return None

    if mode == "low_iv_hv_spread":
        atm_iv = _signal_iv(feature_row)
        hv = feature_row.get(CONFIG.strategy.short_low_iv_hv_col, pd.NA)
        if pd.isna(atm_iv) or pd.isna(hv):
            return None

        if atm_iv >= CONFIG.strategy.short_low_iv_close_threshold:
            return "short_low_iv_high"
        if atm_iv - hv <= CONFIG.strategy.short_low_iv_close_spread_threshold:
            return "short_low_iv_spread_gone"
        return None

    atm_iv = _signal_iv(feature_row)
    if pd.notna(atm_iv) and atm_iv <= CONFIG.strategy.short_close_iv_threshold:
        return "short_iv_low"
    return None


def calc_entry_target_qty(feature_row, max_qty):
    """买入跨式：按配置选择绝对 IV 或历史分位数信号。"""
    if not _calc_long_open_signal(feature_row):
        return 0
    return max_qty


def calc_short_entry_target_qty(feature_row, max_qty):
    """卖出跨式：按配置选择 ATM IV 绝对值、百分位或低 IV 收租信号。"""
    if not _calc_short_open_signal(feature_row):
        return 0
    return max_qty


def get_short_open_regime(feature_row):
    """给回测引擎记录本次 short 仓位的信号来源。"""
    return _short_open_regime(feature_row)


def build_signals(features_df):
    """生成交易信号；short 信号口径由 short_signal_mode 决定。"""
    signals_df = features_df.copy()
    if "atm_iv" not in signals_df.columns:
        signals_df["atm_iv"] = pd.NA
    if "atm_iv_percentile" not in signals_df.columns:
        signals_df["atm_iv_percentile"] = pd.NA
    if "signal_iv" not in signals_df.columns:
        signals_df["signal_iv"] = signals_df["atm_iv"]
    else:
        signals_df["signal_iv"] = signals_df["signal_iv"].fillna(signals_df["atm_iv"])
    if _iv_observation_mode() == "simple_atm_absolute":
        signals_df["signal_iv"] = signals_df["atm_iv"]
        signals_df["signal_iv_percentile"] = pd.NA
    elif "signal_iv_percentile" not in signals_df.columns:
        signals_df["signal_iv_percentile"] = signals_df["atm_iv_percentile"]
    else:
        signals_df["signal_iv_percentile"] = signals_df[
            "signal_iv_percentile"
        ].fillna(signals_df["atm_iv_percentile"])
    signals_df["prev_atm_iv"] = signals_df["atm_iv"].shift(1)
    signals_df["prev_signal_iv"] = signals_df["signal_iv"].shift(1)

    long_open = signals_df.apply(_calc_long_open_signal, axis=1)
    short_regime = signals_df.apply(_short_open_regime, axis=1)

    signals_df["long_open_signal"] = (
        CONFIG.strategy.enable_long_straddle & long_open
    )
    signals_df["short_open_regime"] = short_regime
    signals_df["short_open_signal"] = (
        CONFIG.strategy.enable_short_straddle & short_regime.notna()
    )
    return signals_df


def get_close_reason(feature_row, position_dte):
    """买入跨式在 ATM IV 升高或临近到期时平仓。"""
    close_reason = _get_long_close_reason_by_signal(feature_row)
    if close_reason is not None:
        return close_reason

    if position_dte <= CONFIG.strategy.min_exit_dte:
        return "near_expiry"

    return None


def get_short_close_reason(feature_row, position_dte, position=None):
    """卖出跨式按开仓信号来源平仓，或在临近到期时平仓。"""
    mode = None
    if position is not None:
        mode = position.get("short_entry_regime")
    close_reason = _get_short_close_reason_by_signal(feature_row, mode=mode)
    if close_reason is not None:
        return close_reason

    if position_dte <= CONFIG.strategy.min_exit_dte:
        return "near_expiry"

    return None


def is_short_stop_loss(position, current_market_value):
    """卖方跨式亏损超过初始合约价值一定比例时止损。"""
    if not CONFIG.strategy.short_stop_loss_enabled:
        return False

    if position.get("side") != "short":
        return False
    entry_value = position.get("entry_option_value", 0.0)
    if entry_value <= 0:
        return False
    loss = current_market_value - entry_value
    return loss > entry_value * CONFIG.strategy.short_stop_loss_rate


def calc_position_greeks(call_row, put_row, call_qty=1, put_qty=1, side="long"):
    """按合约乘数、张数和方向汇总跨式仓位 Greeks。"""
    call_multiplier = call_row["contract_multiplier"]
    put_multiplier = put_row["contract_multiplier"]
    call_scale = call_qty * call_multiplier
    put_scale = put_qty * put_multiplier
    direction = -1 if side == "short" else 1

    call_delta = direction * call_row["delta"] * call_scale
    put_delta = direction * put_row["delta"] * put_scale
    call_gamma = direction * call_row["gamma"] * call_scale
    put_gamma = direction * put_row["gamma"] * put_scale
    call_vega = direction * call_row["vega"] * call_scale
    put_vega = direction * put_row["vega"] * put_scale
    call_theta = direction * call_row["theta"] * call_scale
    put_theta = direction * put_row["theta"] * put_scale

    return {
        "delta": call_delta + put_delta,
        "gamma": call_gamma + put_gamma,
        "vega": call_vega + put_vega,
        "theta": call_theta + put_theta,
        "call_iv": call_row["iv"],
        "put_iv": put_row["iv"],
        "position_iv": (call_row["iv"] + put_row["iv"]) / 2,
        "call_delta": call_delta,
        "put_delta": put_delta,
        "call_gamma": call_gamma,
        "put_gamma": put_gamma,
        "call_vega": call_vega,
        "put_vega": put_vega,
        "call_theta": call_theta,
        "put_theta": put_theta,
    }
