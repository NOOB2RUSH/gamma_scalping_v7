import pandas as pd

from .config import CONFIG


SHORT_SIGNAL_MODES = {"absolute", "percentile", "low_iv_hv_spread"}


def _short_signal_mode():
    """读取卖出跨式主信号口径。"""
    mode = CONFIG.strategy.short_signal_mode
    if mode not in SHORT_SIGNAL_MODES:
        raise ValueError(
            "CONFIG.strategy.short_signal_mode 只能是 "
            "'absolute'、'percentile' 或 'low_iv_hv_spread'"
        )
    return mode


def _calc_low_iv_hv_spread_open_signal(feature_row):
    """低 IV 收租：IV 处于低位，但仍高于 HV 一定安全垫时开仓。"""
    atm_iv = feature_row.get("atm_iv", pd.NA)
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
    atm_iv = feature_row.get("atm_iv", pd.NA)
    if pd.isna(atm_iv) or atm_iv < CONFIG.strategy.short_open_iv_threshold:
        return False

    pullback_threshold = CONFIG.strategy.short_open_pullback_iv_threshold
    if pullback_threshold is not None and atm_iv >= pullback_threshold:
        prev_atm_iv = feature_row.get("prev_atm_iv", pd.NA)
        return pd.notna(prev_atm_iv) and atm_iv < prev_atm_iv

    return True


def _short_open_regime(feature_row):
    """返回卖方开仓信号来源；没有信号时返回 None。"""
    mode = _short_signal_mode()

    if mode == "percentile":
        atm_iv_percentile = feature_row.get("atm_iv_percentile", pd.NA)
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
        atm_iv_percentile = feature_row.get("atm_iv_percentile", pd.NA)
        if (
            pd.notna(atm_iv_percentile)
            and atm_iv_percentile <= CONFIG.strategy.short_close_iv_percentile_threshold
        ):
            return "short_iv_percentile_low"
        return None

    if mode == "low_iv_hv_spread":
        atm_iv = feature_row.get("atm_iv", pd.NA)
        hv = feature_row.get(CONFIG.strategy.short_low_iv_hv_col, pd.NA)
        if pd.isna(atm_iv) or pd.isna(hv):
            return None

        if atm_iv >= CONFIG.strategy.short_low_iv_close_threshold:
            return "short_low_iv_high"
        if atm_iv - hv <= CONFIG.strategy.short_low_iv_close_spread_threshold:
            return "short_low_iv_spread_gone"
        return None

    atm_iv = feature_row.get("atm_iv", pd.NA)
    if pd.notna(atm_iv) and atm_iv <= CONFIG.strategy.short_close_iv_threshold:
        return "short_iv_low"
    return None


def calc_entry_target_qty(feature_row, max_qty):
    """买入跨式：ATM IV 低于开仓阈值时持有固定最大张数。"""
    atm_iv = feature_row.get("atm_iv", pd.NA)
    if pd.isna(atm_iv) or atm_iv > CONFIG.strategy.long_open_iv_threshold:
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
    signals_df["prev_atm_iv"] = signals_df["atm_iv"].shift(1)

    long_open = signals_df["atm_iv"] <= CONFIG.strategy.long_open_iv_threshold
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
    atm_iv = feature_row.get("atm_iv", pd.NA)
    if pd.notna(atm_iv) and atm_iv >= CONFIG.strategy.long_close_iv_threshold:
        return "iv_high"

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
