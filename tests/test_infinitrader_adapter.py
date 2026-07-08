from core.live import infinitrader


def test_compile_atm_straddle_delta_rebalance_orders_and_fill():
    payload = {
        "product": "300etf",
        "account_id": "default",
        "date": "2026-07-01",
        "account": {
            "positions": {
                "short": {
                    "call_code": "100CALL",
                    "put_code": "100PUT",
                    "call_qty": 10,
                    "put_qty": 10,
                    "strike": 5.0,
                    "expiry": "2026-07-22",
                    "entry_call_price": 0.10,
                    "entry_put_price": 0.10,
                    "entry_option_value": 20_000.0,
                    "option_margin": 100_000.0,
                    "contract_multiplier": 10000,
                }
            }
        },
        "advice": [
            {
                "action": "ATM_STRADDLE_DELTA_REBALANCE",
                "priority": "action",
                "side": "short",
                "close_put_code": "100PUT",
                "close_put_qty": 1,
                "estimated_close_put_price": 0.10,
                "open_call_code": "100CALL",
                "open_call_qty": 1,
                "estimated_open_call_price": 0.11,
                "target_call_qty": 11,
                "target_put_qty": 9,
                "estimated_option_margin": 105_000.0,
                "underlying_order_book_id": "510300.XSHG",
            }
        ],
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
        ("100PUT", "buy", "1", 1),
        ("100CALL", "sell", "0", 1),
    ]

    fills = infinitrader.build_fills_from_command(
        {"product": "300etf", "date": "2026-07-01", "signal": payload}
    )

    assert [fill["action"] for fill in fills] == ["rebalance_straddle_legs"]
    fill = fills[0]
    assert fill["call_qty"] == 11
    assert fill["put_qty"] == 9
    assert fill["option_margin"] == 105_000.0
    assert fill["cash_delta"] == 96.0


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
