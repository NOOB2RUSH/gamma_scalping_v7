import math

from .config import CONFIG


def value(position, call_row, put_row):
    """按 mid 价格计算当前跨式仓位市值。"""
    multiplier = position["contract_multiplier"]
    return (
        call_row["mid"] * position["call_qty"] * multiplier
        + put_row["mid"] * position["put_qty"] * multiplier
    )


def signed_value(position, call_row, put_row):
    """按持仓方向计算 NAV 中的期权价值：买方为资产，卖方为负债。"""
    market_value = value(position, call_row, put_row)
    if position.get("side", "long") == "short":
        return -market_value
    return market_value


def margin_value(position):
    """返回当前期权仓位冻结保证金；买方跨式没有期权保证金。"""
    return position.get("option_margin", 0.0)


def calc_trade_value(call_row, put_row, call_qty, put_qty):
    """按指定张数计算本次跨式交易金额。"""
    multiplier = call_row["contract_multiplier"]
    return (
        call_row["mid"] * call_qty * multiplier
        + put_row["mid"] * put_qty * multiplier
    )


def margin_call(spot, strike, option_price, multiplier=10000):
    """按交易所公式估算认购义务仓单张保证金；价格 P 暂用 mid 近似。"""
    otm = max(strike - spot, 0)
    m_value = max(0.12 * spot - otm, 0.07 * spot)
    return (option_price + m_value) * multiplier


def margin_put(spot, strike, option_price, multiplier=10000):
    """按交易所公式估算认沽义务仓单张保证金；价格 P 暂用 mid 近似。"""
    otm = max(spot - strike, 0)
    m_value = max(0.12 * spot - otm, 0.07 * strike)
    return min(option_price + m_value, strike) * multiplier


def calc_short_margin(call_row, put_row, call_qty, put_qty, spot):
    """按 call/put 分腿公式计算卖出跨式总保证金。"""
    call_margin = margin_call(
        float(spot),
        float(call_row["strike_price"]),
        float(call_row["mid"]),
        float(call_row["contract_multiplier"]),
    )
    put_margin = margin_put(
        float(spot),
        float(put_row["strike_price"]),
        float(put_row["mid"]),
        float(put_row["contract_multiplier"]),
    )
    return call_margin * call_qty + put_margin * put_qty


def open_straddle(date, atm, call_qty=1, put_qty=1, side="long", spot=None):
    """根据 ATM 选择结果创建跨式仓位对象。"""
    call = atm["call"]
    put = atm["put"]
    position = {
        "entry_date": date,
        "call_code": call["order_book_id"],
        "put_code": put["order_book_id"],
        "strike": atm["strike"],
        "expiry": atm["expiry"],
        "call_qty": call_qty,
        "put_qty": put_qty,
        "entry_call_price": call["mid"],
        "entry_put_price": put["mid"],
        "entry_call_volume": call.get("volume"),
        "entry_put_volume": put.get("volume"),
        "entry_total_volume": (call.get("volume") or 0) + (put.get("volume") or 0),
        "contract_multiplier": call["contract_multiplier"],
        "side": side,
        "entry_option_value": 0.0,
        "option_margin": 0.0,
    }
    market_value = value(position, call, put)
    position["entry_option_value"] = market_value
    if side == "short":
        if spot is None:
            spot = atm["strike"]
        position["option_margin"] = calc_short_margin(
            call,
            put,
            call_qty,
            put_qty,
            spot,
        )
    position["last_option_value"] = signed_value(position, call, put)
    return position


def find_rows(position, chain_df):
    """从当日期权链中找回当前持仓对应的 call 和 put 行。"""
    call_rows = chain_df[chain_df["order_book_id"] == position["call_code"]]
    put_rows = chain_df[chain_df["order_book_id"] == position["put_code"]]
    return call_rows.iloc[0], put_rows.iloc[0]


def trade_fields(position):
    """交易流水中通用的期权合约字段。"""
    return {
        "call_code": position["call_code"],
        "put_code": position["put_code"],
        "strike": position["strike"],
        "expiry": position["expiry"],
        "side": position.get("side", "long"),
    }


def calc_option_fee(call_qty, put_qty, option_fee_per_contract=None):
    """按张数计算期权交易手续费，买卖双边在各自交易时收取。"""
    if option_fee_per_contract is None:
        option_fee_per_contract = CONFIG.backtest.option_fee_per_contract
    return (call_qty + put_qty) * option_fee_per_contract


def _get_row_value(row, field):
    """兼容 pandas Series 和 dict，读取合约行字段。"""
    if hasattr(row, "get"):
        return row.get(field)
    return row[field]


def _is_valid_number(value):
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)


def _build_liquidity_fields(call_row=None, put_row=None, call_qty=0, put_qty=0):
    """检查交易张数是否超过当日成交量的指定比例，只预警不阻止交易。"""
    ratio = CONFIG.backtest.liquidity_warning_volume_ratio
    fields = {
        "liquidity_warning_ratio": ratio,
        "liquidity_check_available": call_row is not None and put_row is not None,
        "liquidity_warning": False,
        "liquidity_warning_legs": "",
        "liquidity_volume_missing_legs": "",
        "call_volume": None,
        "put_volume": None,
        "call_liquidity_limit_qty": None,
        "put_liquidity_limit_qty": None,
        "call_liquidity_warning": False,
        "put_liquidity_warning": False,
    }
    if call_row is None or put_row is None:
        return fields

    warning_legs = []
    missing_legs = []
    for leg_name, row, qty in [
        ("call", call_row, abs(call_qty)),
        ("put", put_row, abs(put_qty)),
    ]:
        volume = _get_row_value(row, "volume")
        fields[f"{leg_name}_volume"] = volume
        if not _is_valid_number(volume):
            missing_legs.append(leg_name)
            continue

        limit_qty = float(volume) * ratio
        leg_warning = qty > limit_qty
        fields[f"{leg_name}_liquidity_limit_qty"] = limit_qty
        fields[f"{leg_name}_liquidity_warning"] = leg_warning
        if leg_warning:
            warning_legs.append(leg_name)

    fields["liquidity_warning"] = bool(warning_legs)
    fields["liquidity_warning_legs"] = ",".join(warning_legs)
    fields["liquidity_volume_missing_legs"] = ",".join(missing_legs)
    return fields


def has_liquidity_warning(call_row, put_row, call_qty, put_qty):
    """按交易预警口径判断一组 call/put 是否存在流动性风险。"""
    return _build_liquidity_fields(
        call_row,
        put_row,
        call_qty,
        put_qty,
    )["liquidity_warning"]


def has_short_volume_spike(position, call_row, put_row):
    """卖方持仓成交量放大止损：当前持仓合约成交量较开仓时显著放大。"""
    if not CONFIG.strategy.short_volume_spike_exit_enabled:
        return False

    entry_volume = position.get("entry_total_volume")
    if not _is_valid_number(entry_volume) or float(entry_volume) <= 0:
        return False

    call_volume = _get_row_value(call_row, "volume")
    put_volume = _get_row_value(put_row, "volume")
    if not _is_valid_number(call_volume) or not _is_valid_number(put_volume):
        return False

    current_volume = float(call_volume) + float(put_volume)
    return (
        current_volume
        >= float(entry_volume) * CONFIG.strategy.short_volume_spike_multiplier
    )


def open_trade(
    date,
    cash,
    atm,
    call_qty,
    put_qty,
    trades,
    trade_type,
    side="long",
    spot=None,
):
    """开跨式仓位并写入交易流水。"""
    position = open_straddle(date, atm, call_qty, put_qty, side=side, spot=spot)
    cost = value(position, atm["call"], atm["put"])
    fee = calc_option_fee(call_qty, put_qty)
    if side == "short":
        cash += cost - fee - position["option_margin"]
    else:
        cash -= cost + fee
    trades.append(
        {
            "date": date,
            "type": trade_type,
            "cash": cash,
            "fee": fee,
            "option_margin": position["option_margin"],
            "trade_call_qty": call_qty,
            "trade_put_qty": put_qty,
            "position_call_qty": position["call_qty"],
            "position_put_qty": position["put_qty"],
            **_build_liquidity_fields(atm["call"], atm["put"], call_qty, put_qty),
            **trade_fields(position),
        }
    )
    return cash, position, signed_value(position, atm["call"], atm["put"])


def close_trade(
    date,
    cash,
    position,
    call_row,
    put_row,
    trades,
    trade_type="close_straddle",
    exit_reason=None,
):
    """按当日价格平跨式仓位并写入交易流水。"""
    close_value = value(position, call_row, put_row)
    fee = calc_option_fee(position["call_qty"], position["put_qty"])
    side = position.get("side", "long")
    margin_change = -position.get("option_margin", 0.0)
    if side == "short":
        cash += -margin_change - close_value - fee
    else:
        cash += close_value - fee
    trades.append(
        {
            "date": date,
            "type": trade_type,
            "exit_reason": exit_reason,
            "cash": cash,
            "fee": fee,
            "option_margin": 0.0,
            "margin_change": margin_change,
            "trade_call_qty": -position["call_qty"],
            "trade_put_qty": -position["put_qty"],
            "position_call_qty": 0,
            "position_put_qty": 0,
            **_build_liquidity_fields(
                call_row,
                put_row,
                position["call_qty"],
                position["put_qty"],
            ),
            **trade_fields(position),
        }
    )
    return cash, close_value


def close_at_last_value(
    date,
    cash,
    position,
    trades,
    exit_reason="missing_option_data_last_price",
):
    close_value = abs(position["last_option_value"])
    fee = calc_option_fee(position["call_qty"], position["put_qty"])
    side = position.get("side", "long")
    margin_change = -position.get("option_margin", 0.0)
    if side == "short":
        cash += -margin_change - close_value - fee
    else:
        cash += close_value - fee
    trades.append(
        {
            "date": date,
            "type": "close_straddle",
            "exit_reason": exit_reason,
            "cash": cash,
            "fee": fee,
            "option_margin": 0.0,
            "margin_change": margin_change,
            "trade_call_qty": -position["call_qty"],
            "trade_put_qty": -position["put_qty"],
            "position_call_qty": 0,
            "position_put_qty": 0,
            **_build_liquidity_fields(),
            **trade_fields(position),
        }
    )
    return cash, close_value

