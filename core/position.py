from .config import CONFIG


def value(position, call_row, put_row):
    """按 mid 价格计算当前跨式仓位市值。"""
    multiplier = position["contract_multiplier"]
    return (
        call_row["mid"] * position["call_qty"] * multiplier
        + put_row["mid"] * position["put_qty"] * multiplier
    )


def open_straddle(date, atm, call_qty=1, put_qty=1):
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
        "contract_multiplier": call["contract_multiplier"],
    }
    position["last_option_value"] = value(position, call, put)
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
    }


def calc_option_fee(call_qty, put_qty, option_fee_per_contract=None):
    """按张数计算期权交易手续费，买卖双边在各自交易时收取。"""
    if option_fee_per_contract is None:
        option_fee_per_contract = CONFIG.backtest.option_fee_per_contract
    return (call_qty + put_qty) * option_fee_per_contract


def open_trade(date, cash, atm, call_qty, put_qty, trades, trade_type):
    """开跨式仓位并写入交易流水。"""
    position = open_straddle(date, atm, call_qty, put_qty)
    cost = value(position, atm["call"], atm["put"])
    fee = calc_option_fee(call_qty, put_qty)
    cash -= cost + fee
    trades.append(
        {
            "date": date,
            "type": trade_type,
            "cash_flow": -cost,
            "fee": fee,
            **trade_fields(position),
        }
    )
    return cash, position, cost


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
    cash += close_value - fee
    trades.append(
        {
            "date": date,
            "type": trade_type,
            "exit_reason": exit_reason,
            "cash_flow": close_value,
            "fee": fee,
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
    close_value = position["last_option_value"]
    fee = calc_option_fee(position["call_qty"], position["put_qty"])
    cash += close_value - fee
    trades.append(
        {
            "date": date,
            "type": "close_straddle",
            "exit_reason": exit_reason,
            "cash_flow": close_value,
            "fee": fee,
            **trade_fields(position),
        }
    )
    return cash, close_value

