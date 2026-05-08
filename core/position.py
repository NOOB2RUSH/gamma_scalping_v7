def value(position, call_row, put_row):
    multiplier = position["contract_multiplier"]
    return (
        call_row["mid"] * position["call_qty"] * multiplier
        + put_row["mid"] * position["put_qty"] * multiplier
    )


def open_straddle(date, atm, call_qty=1, put_qty=1):
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
    call_rows = chain_df[chain_df["order_book_id"] == position["call_code"]]
    put_rows = chain_df[chain_df["order_book_id"] == position["put_code"]]
    return call_rows.iloc[0], put_rows.iloc[0]


def trade_fields(position):
    return {
        "call_code": position["call_code"],
        "put_code": position["put_code"],
        "strike": position["strike"],
        "expiry": position["expiry"],
    }


def open_trade(date, cash, atm, call_qty, put_qty, trades, trade_type):
    position = open_straddle(date, atm, call_qty, put_qty)
    cost = value(position, atm["call"], atm["put"])
    cash -= cost
    trades.append({"date": date, "type": trade_type, "cash_flow": -cost, **trade_fields(position)})
    return cash, position, cost


def close_trade(date, cash, position, call_row, put_row, trades, trade_type="close_straddle", exit_reason=None):
    close_value = value(position, call_row, put_row)
    cash += close_value
    trades.append(
        {
            "date": date,
            "type": trade_type,
            "exit_reason": exit_reason,
            "cash_flow": close_value,
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
    cash += close_value
    trades.append(
        {
            "date": date,
            "type": "close_straddle",
            "exit_reason": exit_reason,
            "cash_flow": close_value,
            **trade_fields(position),
        }
    )
    return cash, close_value

