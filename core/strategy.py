import pandas as pd

from .config import CONFIG


def _get_atm_iv(feature_row):
    """读取当天 ATM IV 绝对值；缺失时返回 NA，避免误触发交易。"""
    return feature_row.get("atm_iv", pd.NA)


def _get_prev_atm_iv(feature_row):
    """读取上一交易日 ATM IV，用于卖方开仓的回落确认。"""
    return feature_row.get("prev_atm_iv", pd.NA)


def _get_atm_iv_percentile(feature_row):
    """读取当天 ATM IV 百分位；窗口不足或缺失时返回 NA，避免误触发交易。"""
    return feature_row.get("atm_iv_percentile", pd.NA)


def _get_short_low_iv_hv(feature_row):
    """读取低 IV 卖空实验使用的历史波动率列；默认使用 yz_hv60。"""
    return feature_row.get(CONFIG.strategy.short_low_iv_hv_col, pd.NA)


def _short_signal_mode():
    """读取卖出跨式信号口径。"""
    mode = CONFIG.strategy.short_signal_mode
    if mode not in {"absolute", "percentile", "low_iv_hv_spread"}:
        raise ValueError(
            "CONFIG.strategy.short_signal_mode 只能是 "
            "'absolute'、'percentile' 或 'low_iv_hv_spread'"
        )
    return mode


def _calc_low_iv_hv_spread_open_signal(feature_row):
    """低 IV 卖空实验：IV 处于低位，但仍高于 HV 一定安全垫时开仓。"""
    atm_iv = _get_atm_iv(feature_row)
    hv = _get_short_low_iv_hv(feature_row)
    if pd.isna(atm_iv) or pd.isna(hv):
        return False

    iv_is_low = atm_iv <= CONFIG.strategy.short_low_iv_open_threshold
    spread_is_positive = (
        atm_iv - hv >= CONFIG.strategy.short_low_iv_hv_spread_threshold
    )
    return iv_is_low and spread_is_positive


def _calc_short_open_signal(feature_row):
    """按配置选择卖出跨式开仓口径。"""
    mode = _short_signal_mode()
    if mode == "percentile":
        atm_iv_percentile = _get_atm_iv_percentile(feature_row)
        return (
            pd.notna(atm_iv_percentile)
            and atm_iv_percentile >= CONFIG.strategy.short_open_iv_percentile_threshold
        )
    if mode == "low_iv_hv_spread":
        return _calc_low_iv_hv_spread_open_signal(feature_row)

    atm_iv = _get_atm_iv(feature_row)
    if pd.isna(atm_iv) or atm_iv < CONFIG.strategy.short_open_iv_threshold:
        return False

    pullback_threshold = CONFIG.strategy.short_open_pullback_iv_threshold
    if pullback_threshold is not None and atm_iv >= pullback_threshold:
        prev_atm_iv = _get_prev_atm_iv(feature_row)
        return pd.notna(prev_atm_iv) and atm_iv < prev_atm_iv

    return True


def _get_short_close_reason_by_signal(feature_row):
    """按配置选择卖出跨式平仓口径。"""
    mode = _short_signal_mode()
    if mode == "percentile":
        atm_iv_percentile = _get_atm_iv_percentile(feature_row)
        if (
            pd.notna(atm_iv_percentile)
            and atm_iv_percentile <= CONFIG.strategy.short_close_iv_percentile_threshold
        ):
            return "short_iv_percentile_low"
        return None

    if mode == "low_iv_hv_spread":
        atm_iv = _get_atm_iv(feature_row)
        hv = _get_short_low_iv_hv(feature_row)
        if pd.isna(atm_iv) or pd.isna(hv):
            return None

        if atm_iv >= CONFIG.strategy.short_low_iv_close_threshold:
            return "short_low_iv_high"
        if atm_iv - hv <= CONFIG.strategy.short_low_iv_close_spread_threshold:
            return "short_low_iv_spread_gone"
        return None

    atm_iv = _get_atm_iv(feature_row)
    if pd.notna(atm_iv) and atm_iv <= CONFIG.strategy.short_close_iv_threshold:
        return "short_iv_low"
    return None


def calc_entry_target_qty(feature_row, max_qty):
    """买入跨式：ATM IV 低于开仓阈值时持有固定最大张数。"""
    atm_iv = _get_atm_iv(feature_row)
    if pd.isna(atm_iv) or atm_iv > CONFIG.strategy.long_open_iv_threshold:
        return 0
    return max_qty


def calc_short_entry_target_qty(feature_row, max_qty):
    """卖出跨式：按配置选择 ATM IV 绝对值或百分位开仓。"""
    if not _calc_short_open_signal(feature_row):
        return 0
    return max_qty


def build_signals(features_df):
    """生成交易信号；short 信号口径由 short_signal_mode 决定。"""
    signals_df = features_df.copy()
    if "atm_iv" not in signals_df.columns:
        signals_df["atm_iv"] = pd.NA
    if "atm_iv_percentile" not in signals_df.columns:
        signals_df["atm_iv_percentile"] = pd.NA
    signals_df["prev_atm_iv"] = signals_df["atm_iv"].shift(1)

    long_open = signals_df["atm_iv"] <= CONFIG.strategy.long_open_iv_threshold
    short_open = signals_df.apply(_calc_short_open_signal, axis=1)

    signals_df["long_open_signal"] = (
        CONFIG.strategy.enable_long_straddle & long_open
    )
    signals_df["short_open_signal"] = (
        CONFIG.strategy.enable_short_straddle & short_open
    )
    return signals_df


def get_close_reason(feature_row, position_dte):
    """买入跨式在 ATM IV 升高或临近到期时平仓。"""
    atm_iv = _get_atm_iv(feature_row)
    if pd.notna(atm_iv) and atm_iv >= CONFIG.strategy.long_close_iv_threshold:
        return "iv_high"

    if position_dte <= CONFIG.strategy.min_exit_dte:
        return "near_expiry"

    return None


def get_short_close_reason(feature_row, position_dte):
    """卖出跨式按配置口径平仓，或在临近到期时平仓。"""
    close_reason = _get_short_close_reason_by_signal(feature_row)
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
