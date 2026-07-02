from core.live import account, position_checker


def test_compare_option_positions_aggregates_core_and_option_hedges():
    local = account.AccountState(
        product="500etf",
        positions={
            "long": None,
            "short": {
                "side": "short",
                "call_code": "10011723",
                "put_code": "10011732",
                "call_qty": 1,
                "put_qty": 10,
            },
        },
        option_hedges=[
            {
                "side": "short",
                "option_type": "c",
                "order_book_id": "10011721",
                "qty": 5,
            }
        ],
    )
    broker_rows = [
        {"order_book_id": "10011721", "side": "short", "total_qty": 5},
        {"order_book_id": "10011723", "side": "short", "total_qty": 1},
        {"order_book_id": "10011732", "side": "short", "total_qty": 10},
    ]

    rows = position_checker._compare_option_positions(local, broker_rows)

    assert all(row["ok"] for row in rows)
    assert {
        (row["合约代码"], row["方向"], row["本地数量"], row["券商数量"])
        for row in rows
    } == {
        ("10011721", "short", 5.0, 5.0),
        ("10011723", "short", 1.0, 1.0),
        ("10011732", "short", 10.0, 10.0),
    }


def test_compare_option_positions_reports_quantity_mismatch():
    local = account.AccountState(
        product="500etf",
        positions={
            "long": None,
            "short": {
                "side": "short",
                "call_code": "10011723",
                "put_code": "10011732",
                "call_qty": 4,
                "put_qty": 10,
            },
        },
        option_hedges=[],
    )
    broker_rows = [
        {"order_book_id": "10011721", "side": "short", "total_qty": 5},
        {"order_book_id": "10011723", "side": "short", "total_qty": 1},
        {"order_book_id": "10011732", "side": "short", "total_qty": 10},
    ]

    rows = position_checker._compare_option_positions(local, broker_rows)
    by_code = {row["合约代码"]: row for row in rows}

    assert by_code["10011721"]["数量差异"] == -5.0
    assert by_code["10011721"]["ok"] is False
    assert by_code["10011723"]["数量差异"] == 3.0
    assert by_code["10011723"]["ok"] is False
    assert by_code["10011732"]["ok"] is True
