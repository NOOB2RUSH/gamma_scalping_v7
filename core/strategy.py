from .config import CONFIG


def build_signals(features_df):
    """根据波动率特征生成交易信号。"""
    signals_df = features_df.copy()

    # 入场规则：ATM IV 低于阈值时开仓。
    signals_df["open_signal"] = signals_df["atm_iv"] < CONFIG.strategy.open_iv_threshold
    return signals_df


def get_close_reason(position_iv, position_dte):
    """返回平仓原因；没有触发平仓时返回 None。"""
    if position_iv > CONFIG.strategy.close_iv_threshold:
        return "iv_high"

    if position_dte <= CONFIG.strategy.min_exit_dte:
        return "near_expiry"

    return None


def should_roll_position(
    feature_row,
    position_dte,
    position_strike,
    strike_mismatch_days,
    in_roll_cooldown,
):
    """判断是否需要把当前跨式移到新的 ATM 合约。"""
    if in_roll_cooldown:
        return False

    iv_still_low = feature_row["atm_iv"] < CONFIG.strategy.roll_iv_threshold
    dte_too_low = position_dte <= CONFIG.strategy.roll_dte_threshold

    # 行权价按期权档位判断；连续偏离多日后才 roll，避免 ATM 档位来回跳导致过度换仓。
    strike_roll_ready = (
        position_strike != feature_row["atm_strike"]
        and strike_mismatch_days >= CONFIG.strategy.roll_strike_mismatch_days
    )
    return iv_still_low and (dte_too_low or strike_roll_ready)


def calc_position_greeks(call_row, put_row, call_qty=1, put_qty=1):
    """按合约乘数和张数汇总跨式仓位 Greeks。"""
    call_multiplier = call_row["contract_multiplier"]
    put_multiplier = put_row["contract_multiplier"]
    call_scale = call_qty * call_multiplier
    put_scale = put_qty * put_multiplier

    return {
        "delta": call_row["delta"] * call_scale + put_row["delta"] * put_scale,
        "gamma": call_row["gamma"] * call_scale + put_row["gamma"] * put_scale,
        "vega": call_row["vega"] * call_scale + put_row["vega"] * put_scale,
        "theta": call_row["theta"] * call_scale + put_row["theta"] * put_scale,
        "call_iv": call_row["iv"],
        "put_iv": put_row["iv"],
        # 组合 IV 使用 call 和 put 的简单平均，保持原有策略口径。
        "position_iv": (call_row["iv"] + put_row["iv"]) / 2,
        "call_delta": call_row["delta"] * call_scale,
        "put_delta": put_row["delta"] * put_scale,
        "call_gamma": call_row["gamma"] * call_scale,
        "put_gamma": put_row["gamma"] * put_scale,
        "call_vega": call_row["vega"] * call_scale,
        "put_vega": put_row["vega"] * put_scale,
        "call_theta": call_row["theta"] * call_scale,
        "put_theta": put_row["theta"] * put_scale,
    }
