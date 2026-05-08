from .config import CONFIG


def generate_open_signal(feature_row, open_iv_threshold=None):
    if open_iv_threshold is None:
        open_iv_threshold = CONFIG.strategy.open_iv_threshold
    return feature_row["atm_iv"] < open_iv_threshold


def build_signal_df(features_df, open_iv_threshold=None):
    if open_iv_threshold is None:
        open_iv_threshold = CONFIG.strategy.open_iv_threshold
    signals_df = features_df.copy()
    signals_df["open_signal"] = signals_df.apply(
        lambda row: generate_open_signal(row, open_iv_threshold),
        axis=1,
    )
    return signals_df


def get_close_reason(
    position_iv, position_dte, close_iv_threshold=None, min_exit_dte=None
):
    if close_iv_threshold is None:
        close_iv_threshold = CONFIG.strategy.close_iv_threshold
    if min_exit_dte is None:
        min_exit_dte = CONFIG.strategy.min_exit_dte

    if position_iv > close_iv_threshold:
        return "iv_high"

    if position_dte <= min_exit_dte:
        return "near_expiry"

    return None


def calc_position_moneyness(position_strike, spot):
    return abs(position_strike / spot - 1)


def calc_position_iv(call_row, put_row):
    return (call_row["iv"] + put_row["iv"]) / 2


def should_roll_position(
    feature_row,
    position_dte,
    position_strike,
    spot,
    roll_iv_threshold=None,
    roll_dte_threshold=None,
    roll_moneyness_threshold=None,
):
    if roll_iv_threshold is None:
        roll_iv_threshold = CONFIG.strategy.roll_iv_threshold
    if roll_dte_threshold is None:
        roll_dte_threshold = CONFIG.strategy.roll_dte_threshold
    if roll_moneyness_threshold is None:
        roll_moneyness_threshold = CONFIG.strategy.roll_moneyness_threshold

    iv_still_low = feature_row["atm_iv"] < roll_iv_threshold
    dte_too_low = position_dte <= roll_dte_threshold
    strike_too_far = (
        calc_position_moneyness(position_strike, spot) > roll_moneyness_threshold
    )
    return iv_still_low and (dte_too_low or strike_too_far)


def calc_position_greeks(call_row, put_row, call_qty=1, put_qty=1):
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
        "position_iv": calc_position_iv(call_row, put_row),
        "call_delta": call_row["delta"] * call_scale,
        "put_delta": put_row["delta"] * put_scale,
        "call_gamma": call_row["gamma"] * call_scale,
        "put_gamma": put_row["gamma"] * put_scale,
        "call_vega": call_row["vega"] * call_scale,
        "put_vega": put_row["vega"] * put_scale,
        "call_theta": call_row["theta"] * call_scale,
        "put_theta": put_row["theta"] * put_scale,
    }
