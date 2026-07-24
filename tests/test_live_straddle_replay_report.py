from __future__ import annotations

import json
import hashlib
import sqlite3

import pandas as pd

from core.live.replay_report import (
    build_live_comparison,
    write_live_comparison_report,
)


def test_comparison_matches_execution_and_exposes_unvalued_differences(tmp_path):
    product_dir = tmp_path / "kc50etf"
    product_dir.mkdir()
    pd.DataFrame(
        [
            {
                "日期": "2026-07-22",
                "当日手续费": 100.0,
                "期权单日盈亏": 200.0,
                "ETF单日盈亏": -50.0,
                "总单日盈亏": 150.0,
                "净单日盈亏": 50.0,
                "账户Delta": 1000.0,
            }
        ]
    ).to_csv(
        product_dir / "default_account_summary_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "日期": "2026-07-22",
                "方向": "short",
                "合约代码": "CALL",
                "合约名称": "科创50购8月2000",
                "买卖": "卖",
                "总持仓": 10,
            },
            {
                "日期": "2026-07-22",
                "方向": "short",
                "合约代码": "PUT",
                "合约名称": "科创50沽8月2000",
                "买卖": "卖",
                "总持仓": 10,
            },
            {
                "日期": "2026-07-22",
                "方向": "hedge",
                "合约代码": "588000",
                "合约名称": "588000.XSHG",
                "买卖": "买",
                "总持仓": 1000,
            },
        ]
    ).to_csv(
        product_dir / "default_position_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    db_path = product_dir / "account.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table fills(
                id integer primary key,
                account_id text,
                action text,
                payload text,
                created_at text,
                voided_at text
            )
            """
        )
        payload = {
            "action": "open_short_straddle",
            "date": "2026-07-22",
            "side": "short",
            "call_code": "CALL",
            "put_code": "PUT",
            "call_qty": 10,
            "put_qty": 10,
            "entry_call_price": 0.11,
            "entry_put_price": 0.09,
            "contract_multiplier": 10000,
        }
        conn.execute(
            "insert into fills values(1, 'default', 'open_short_straddle', ?, '', null)",
            (json.dumps(payload),),
        )
    database_hash_before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    database_mtime_before = db_path.stat().st_mtime_ns

    theoretical_daily = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-22"),
                "product": "kc50etf",
                "theoretical_daily_pnl": 80.0,
                "ending_position_fingerprint": json.dumps(
                    {
                        "long": None,
                        "short": {
                            "call_code": "CALL",
                            "put_code": "PUT",
                            "call_qty": 10,
                            "put_qty": 10,
                        },
                    },
                    sort_keys=True,
                ),
                "ending_hedge_qty": 0.0,
            }
        ]
    )
    theoretical_trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-22"),
                "asset": "OPTION",
                "action": "OPEN_SHORT_STRADDLE",
                "side": "short",
                "call_code": "CALL",
                "put_code": "PUT",
                "trade_call_qty": 10,
                "trade_put_qty": 10,
                "call_price": 0.10,
                "put_price": 0.10,
                "contract_multiplier": 10000,
                "fee": 40.0,
            }
        ]
    )

    comparison, matches, actual_fills = build_live_comparison(
        "kc50etf",
        theoretical_daily,
        theoretical_trades,
        state_root=tmp_path,
    )

    row = comparison.iloc[0]
    assert row["matched_leg_count"] == 2
    assert abs(row["execution_slippage_pnl"]) < 1e-9
    assert row["fee_difference_pnl"] == -60.0
    assert bool(row["position_difference"]) is True
    assert row["position_difference_status"] == "present_not_monetarily_valued"
    assert row["unexplained_residual_pnl"] == 30.0
    assert row["comparison_scope_status"] == "interval_start_partial_theoretical_day"
    assert bool(row["pnl_comparable"]) is False
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == database_hash_before
    assert db_path.stat().st_mtime_ns == database_mtime_before

    paths = write_live_comparison_report(
        "kc50etf",
        comparison,
        matches,
        theoretical_trades,
        actual_fills,
        output_dir=tmp_path / "report",
        metadata={"start": "2026-07-22", "end": "2026-07-22"},
    )
    assert all((tmp_path / "report" / name).exists() for name in (
        "live_straddle_comparison.csv",
        "live_straddle_comparison.xlsx",
        "live_straddle_comparison.json",
        "live_straddle_comparison.md",
    ))
    assert set(paths) == {"csv", "xlsx", "json", "markdown"}


def test_execution_matching_splits_partial_fills_and_uses_option_multiplier():
    from core.live.replay_report import (
        _execution_leg,
        _match_execution_legs,
        _summarize_matches,
    )

    date = pd.Timestamp("2026-07-20")
    theoretical = pd.DataFrame(
        [_execution_leg(date, "OPTION", "CALL", 10, 0.10, 10_000, "theoretical", "short")]
    )
    actual = pd.DataFrame(
        [_execution_leg(date, "OPTION", "CALL", 6, 0.11, 10_000, "actual", "short")]
    )

    matches = _match_execution_legs(theoretical, actual)
    summary = _summarize_matches(matches).iloc[0]

    matched = matches[matches["match_status"] == "matched"].iloc[0]
    remaining = matches[matches["match_status"] == "theoretical_only"].iloc[0]
    assert matched["matched_qty"] == 6
    assert remaining["theoretical_qty"] == 4
    assert matched["execution_slippage_pnl"] == 600.0
    assert summary["unexecuted_leg_count"] == 1
    assert summary["unexecuted_notional"] == 4_000.0


def test_execution_matching_classifies_later_fill_as_delayed_without_fake_timing_pnl():
    from core.live.replay_report import (
        _execution_leg,
        _match_execution_legs,
        _summarize_matches,
    )

    theoretical = pd.DataFrame(
        [
            _execution_leg(
                pd.Timestamp("2026-07-20"),
                "ETF",
                "588000.XSHG",
                1_000,
                1.9,
                1,
                "theoretical",
                "etf",
            )
        ]
    )
    actual = pd.DataFrame(
        [
            _execution_leg(
                pd.Timestamp("2026-07-21"),
                "ETF",
                "588000.XSHG",
                1_000,
                2.0,
                1,
                "actual",
                "etf",
            )
        ]
    )

    matches = _match_execution_legs(theoretical, actual)
    summary = _summarize_matches(matches).iloc[0]

    assert list(matches["match_status"]) == ["delayed"]
    assert matches.iloc[0]["delay_days"] == 1
    assert matches.iloc[0]["execution_slippage_pnl"] == 0.0
    assert summary["delayed_leg_count"] == 1
    assert summary["delayed_notional"] == 1_900.0


def test_actual_single_leg_option_hedge_fills_are_comparable_execution_legs():
    from core.live.replay_report import _expand_actual_fills

    fills = pd.DataFrame(
        [
            {
                "fill_id": 1,
                "date": pd.Timestamp("2026-07-20"),
                "action": "open_option_hedge",
                "payload": {
                    "side": "short",
                    "option_type": "c",
                    "order_book_id": "CALL",
                    "qty": 4,
                    "entry_price": 0.20,
                    "contract_multiplier": 10_000,
                },
            },
            {
                "fill_id": 2,
                "date": pd.Timestamp("2026-07-21"),
                "action": "close_option_hedge",
                "payload": {
                    "side": "short",
                    "option_type": "c",
                    "order_book_id": "CALL",
                    "qty": 4,
                    "price": 0.25,
                    "contract_multiplier": 10_000,
                },
            },
        ]
    )

    legs = _expand_actual_fills(fills, product="kc50etf")

    assert list(legs["qty"]) == [4.0, -4.0]
    assert list(legs["cash_flow"]) == [8_000.0, -10_000.0]
    assert list(legs["code"]) == ["CALL", "CALL"]
