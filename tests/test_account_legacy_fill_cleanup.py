import json

from core.live import account


def test_rebuild_skips_removed_option_hedge_fills(tmp_path):
    db_path = tmp_path / "account.sqlite"
    account.initialize_account("500etf", 1000.0, db_path=db_path)

    with account.connect(db_path) as conn:
        for action, payload in [
            (
                "open_option_hedge",
                {"action": "open_option_hedge", "cash_delta": 500.0},
            ),
            (
                "cash_adjustment",
                {"action": "cash_adjustment", "cash_delta": 10.0},
            ),
        ]:
            conn.execute(
                """
                insert into fills(account_id, action, payload, created_at)
                values (?, ?, ?, ?)
                """,
                ("default", action, json.dumps(payload), "2026-07-09T00:00:00"),
            )
        conn.commit()

    rebuilt = account.rebuild_account(
        "500etf",
        db_path=db_path,
        initial_cash=1000.0,
    )

    assert rebuilt.cash == 1010.0
    assert not hasattr(rebuilt, "option_hedges")
