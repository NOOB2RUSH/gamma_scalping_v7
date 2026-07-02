from core.live import infinitrader


def test_compile_combination_option_hedge_signal_orders():
    payload = {
        "advice": [
            {
                "action": "DELTA_HEDGE",
                "priority": "action",
                "trade_etf_qty": -21800,
                "estimated_price": 4.931,
                "underlying_order_book_id": "510300.XSHG",
            },
            {
                "action": "OPTION_DELTA_HEDGE_COMBINATION",
                "priority": "action",
                "side": "short",
                "close_call_code": "10011704",
                "close_call_qty": 4,
                "estimated_close_call_price": 0.08535,
                "open_legs": [
                    {
                        "order_book_id": "10011699",
                        "qty": 3,
                        "estimated_price": 0.4352,
                    },
                    {
                        "order_book_id": "10011700",
                        "qty": 1,
                        "estimated_price": 0.3456,
                    },
                ],
                "trade_etf_qty": 5434,
                "estimated_price": 4.931,
                "underlying_order_book_id": "510300.XSHG",
            },
        ]
    }

    orders = infinitrader.compile_signal_orders(payload)

    assert [
        (
            order["instrument_id"],
            order["order_direction"],
            order["offset"],
            order["volume"],
        )
        for order in orders
    ] == [
        ("10011704", "buy", "1", 4),
        ("10011699", "sell", "0", 3),
        ("10011700", "sell", "0", 1),
        ("510300", "sell", None, 16400),
    ]
    assert all(order["exchange"] == "SSE" for order in orders)


def test_build_fills_from_combination_command():
    signal = {
        "product": "300etf",
        "account_id": "default",
        "date": "2026-06-23",
        "account": {
            "positions": {
                "short": {
                    "call_code": "10011704",
                    "put_code": "10011713",
                    "call_qty": 10,
                    "put_qty": 10,
                    "strike": 5.0,
                    "expiry": "2026-07-22",
                    "entry_call_price": 0.1625,
                    "entry_put_price": 0.0839,
                    "entry_option_value": 24640.0,
                    "option_margin": 141496.0,
                    "contract_multiplier": 10000,
                }
            }
        },
        "advice": [
            {
                "action": "DELTA_HEDGE",
                "priority": "action",
                "trade_etf_qty": -21800,
                "target_hedge_qty": 0,
                "estimated_price": 4.931,
                "underlying_order_book_id": "510300.XSHG",
            },
            {
                "action": "OPTION_DELTA_HEDGE_COMBINATION",
                "priority": "action",
                "side": "short",
                "close_source": "core_short_call",
                "close_call_code": "10011704",
                "close_call_qty": 4,
                "estimated_close_call_price": 0.08535,
                "estimated_close_margin_release": 24322.8,
                "open_call_qty": 4,
                "open_expiry": "2026-07-22",
                "open_legs": [
                    {
                        "order_book_id": "10011699",
                        "qty": 3,
                        "estimated_price": 0.4352,
                        "strike": 4.5,
                    },
                    {
                        "order_book_id": "10011700",
                        "qty": 1,
                        "estimated_price": 0.3456,
                        "strike": 4.6,
                    },
                ],
                "trade_etf_qty": 5434,
                "target_hedge_qty": 5434,
                "estimated_price": 4.931,
                "estimated_option_margin": 40180.8,
                "underlying_order_book_id": "510300.XSHG",
            },
        ],
    }
    command = {
        "product": "300etf",
        "account_id": "default",
        "date": "2026-06-23",
        "signal": signal,
        "orders": infinitrader.compile_signal_orders(signal),
    }

    fills = infinitrader.build_fills_from_command(command)

    assert [fill["action"] for fill in fills] == [
        "rebalance_straddle_legs",
        "open_option_hedge",
        "open_option_hedge",
        "delta_hedge",
    ]
    assert fills[0]["call_qty"] == 6
    assert fills[1]["order_book_id"] == "10011699"
    assert fills[3]["trade_etf_qty"] == -16400
    assert fills[3]["target_hedge_qty"] == 5400


def test_compile_roll_short_straddle_orders_close_then_open():
    payload = {
        "advice": [
            {
                "action": "ROLL_SHORT_STRADDLE",
                "priority": "action",
                "side": "short",
                "current_call_code": "CALL_OLD",
                "current_put_code": "PUT_OLD",
                "current_call_qty": 10,
                "current_put_qty": 10,
                "estimated_current_call_price": 0.12,
                "estimated_current_put_price": 0.08,
                "target_call_code": "CALL_NEW",
                "target_put_code": "PUT_NEW",
                "target_call_qty": 10,
                "target_put_qty": 10,
                "estimated_target_call_price": 0.10,
                "estimated_target_put_price": 0.09,
            }
        ]
    }

    orders = infinitrader.compile_signal_orders(payload)

    assert [
        (order["instrument_id"], order["order_direction"], order["offset"])
        for order in orders
    ] == [
        ("CALL_OLD", "buy", "1"),
        ("PUT_OLD", "buy", "1"),
        ("CALL_NEW", "sell", "0"),
        ("PUT_NEW", "sell", "0"),
    ]
