from core.live import account, holding_importer


def test_load_account_drops_option_hedge_that_overlaps_core_position(tmp_path):
    db_path = tmp_path / "account.db"
    short_fill = {
        "action": "open_short_straddle",
        "side": "short",
        "date": "2026-07-02",
        "call_code": "10011723",
        "put_code": "10011732",
        "call_qty": 12,
        "put_qty": 8,
        "strike": 9.0,
        "expiry": "2026-07-22",
        "entry_call_price": 0.14,
        "entry_put_price": 0.25,
        "contract_multiplier": 10000,
        "cash_delta": 0.0,
    }
    duplicate_hedge = {
        "action": "open_option_hedge",
        "side": "short",
        "option_type": "c",
        "order_book_id": "10011723",
        "qty": 11,
        "strike": 9.0,
        "expiry": "2026-07-22",
        "entry_price": 0.14,
        "contract_multiplier": 10000,
        "cash_delta": 0.0,
    }

    account.record_fill("500etf", short_fill, db_path=db_path)
    account.record_fill("500etf", duplicate_hedge, db_path=db_path)

    local = account.load_account("500etf", db_path=db_path)

    assert local.positions["short"]["call_qty"] == 12
    assert local.positions["short"]["put_qty"] == 8
    assert local.option_hedges == []


def test_holding_import_detects_option_hedge_that_overlaps_core_position():
    local = account.AccountState(
        product="500etf",
        positions={
            "long": None,
            "short": {
                "call_code": "10011723",
                "put_code": "10011732",
                "call_qty": 12,
                "put_qty": 8,
            },
        },
    )

    assert holding_importer._option_hedge_overlaps_core_position(
        local,
        {"order_book_id": "10011723", "side": "short", "qty": 11},
    )
    assert not holding_importer._option_hedge_overlaps_core_position(
        local,
        {"order_book_id": "10011721", "side": "short", "qty": 5},
    )
