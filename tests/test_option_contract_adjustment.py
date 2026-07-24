import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import account, account_report, holding_importer


def _existing_500etf_position():
    return {
        "entry_date": "2026-07-13",
        "call_code": "10011720",
        "put_code": "10011729",
        "strike": 8.25,
        "expiry": "2026-07-22",
        "call_qty": 10,
        "put_qty": 10,
        "entry_call_price": 0.185,
        "entry_put_price": 0.1592,
        "contract_multiplier": 10000,
        "side": "short",
        "short_entry_regime": "absolute",
        "entry_option_value": 34420.0,
        "option_margin": 212202.0,
        "last_option_value": 33690.0,
        "last_call_price": 0.1887,
        "last_put_price": 0.1482,
    }


def _adjusted_metadata():
    return {
        "10011720": {
            "strike": 8.104,
            "expiry": "2026-07-22",
            "option_type": "C",
            "contract_multiplier": 10180,
            "contract_symbol": "XD500ETF\u8d2d7\u67088104A",
        },
        "10011729": {
            "strike": 8.104,
            "expiry": "2026-07-22",
            "option_type": "P",
            "contract_multiplier": 10180,
            "contract_symbol": "XD500ETF\u6cbd7\u67088104A",
        },
    }


def test_holding_import_creates_zero_cash_contract_adjustment():
    with TemporaryDirectory() as root:
        path = Path(root) / "holding_2026_07_15-13_15_51.csv"
        frame = pd.DataFrame(
            [
                {
                    "\u5408\u7ea6\u4ee3\u7801": "10011720",
                    "\u5408\u7ea6\u540d\u79f0": "XD500ETF\u8d2d7\u67088104A",
                    "\u4e70\u5356": "\u5356",
                    "\u603b\u6301\u4ed3": 10,
                    "\u4eca\u5f00\u4ed3": 0,
                    "\u5f00\u4ed3\u5747\u4ef7": 0.18143,
                    "\u6700\u65b0\u4ef7": 0.1582,
                    "\u5360\u7528\u4fdd\u8bc1\u91d1": 123896.0,
                    "\u6301\u4ed3\u7c7b\u578b": "\u4e49\u52a1\u4ed3",
                },
                {
                    "\u5408\u7ea6\u4ee3\u7801": "10011729",
                    "\u5408\u7ea6\u540d\u79f0": "XD500ETF\u6cbd7\u67088104A",
                    "\u4e70\u5356": "\u5356",
                    "\u603b\u6301\u4ed3": 10,
                    "\u4eca\u5f00\u4ed3": 0,
                    "\u5f00\u4ed3\u5747\u4ef7": 0.15609,
                    "\u6700\u65b0\u4ef7": 0.1234,
                    "\u5360\u7528\u4fdd\u8bc1\u91d1": 92956.0,
                    "\u6301\u4ed3\u7c7b\u578b": "\u4e49\u52a1\u4ed3",
                },
            ]
        )
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        state = account.AccountState(product="500etf")
        state.positions["short"] = _existing_500etf_position()
        initial_cash = state.cash

        with (
            mock.patch.object(
                holding_importer,
                "_resolve_holding_file",
                return_value=path,
            ),
            mock.patch.object(
                holding_importer,
                "_resolve_trade_detail_file",
                return_value=None,
            ),
            mock.patch.object(
                holding_importer,
                "_load_contract_metadata",
                return_value=_adjusted_metadata(),
            ),
            mock.patch.object(
                holding_importer.account_store,
                "load_account",
                return_value=state,
            ),
        ):
            result = holding_importer.import_holding_file(
                "500etf",
                date="2026-07-15",
                include_existing=False,
                dry_run=True,
            )

    fills = [item["fill"] for item in result["applied"]]
    assert [fill["action"] for fill in fills] == ["option_contract_adjustment"]
    fill = fills[0]
    assert fill["cash_delta"] == 0.0
    assert fill["strike"] == 8.104
    assert fill["contract_multiplier"] == 10180
    assert fill["entry_option_value"] == 34420.0
    assert math.isclose(fill["entry_call_price"], 0.185 * 10000 / 10180)
    assert math.isclose(fill["entry_put_price"], 0.1592 * 10000 / 10180)
    assert math.isclose(fill["last_option_value"], (0.1582 + 0.1234) * 10 * 10180)
    assert state.cash == initial_cash
    assert state.positions["short"]["contract_multiplier"] == 10180


def test_daily_pnl_rescales_previous_mark_without_dividend_jump():
    adjustment = {
        "adjustment_price_ratio": 10000 / 10180,
        "new_contract_multiplier": 10180,
    }
    current = pd.DataFrame(
        [{"\u65b9\u5411": "short", "\u603b\u6301\u4ed3": 10, "\u6301\u4ed3\u5747\u4ef7": 0.18172888}]
    )
    previous = pd.DataFrame(
        [{"\u65b9\u5411": "short", "\u603b\u6301\u4ed3": 10, "\u6700\u65b0\u4ef7": 0.2294, "\u6301\u4ed3\u5747\u4ef7": 0.185}]
    )
    row = account_report._position_report_row(
        {
            "product": "500etf",
            "account_id": "default",
            "date": "2026-07-15",
            "current_chain_metadata": {
                "10011720": {
                    "contract_symbol": "XD500ETF\u8d2d7\u67088104A",
                    "close": 0.1582,
                    "contract_multiplier": 10180,
                    "maturity_date": "2026-07-22",
                }
            },
        },
        "10011720",
        current,
        previous,
        [],
        {},
        adjustment,
    )

    expected = 0.2294 * 10 * 10000 - 0.1582 * 10 * 10180
    assert math.isclose(row["\u6301\u4ed3\u76c8\u4e8f"], expected)


def test_report_infers_adjustment_from_current_terms_without_adjustment_fill():
    previous = pd.DataFrame(
        [
            {
                "\u65b9\u5411": "short",
                "\u5408\u7ea6\u4ee3\u7801": "10011720",
                "\u603b\u6301\u4ed3": 10,
                "\u6700\u65b0\u4ef7": 0.2294,
                "\u884c\u6743\u4ef7": 8.25,
                "\u5408\u7ea6\u4e58\u6570": 10000,
            }
        ]
    )
    metadata = {
        "10011720": {
            "contract_symbol": "XD500ETF\u8d2d7\u67088104A",
            "strike_price": 8.104,
            "contract_multiplier": 10180,
        }
    }

    with mock.patch.object(account_report.account_store, "list_fills", return_value=[]):
        adjustments = account_report._option_contract_adjustments_by_code(
            "500etf",
            "default",
            "2026-07-15",
            previous_positions=previous,
            current_metadata=metadata,
        )

    adjustment = adjustments["10011720"]
    assert adjustment["new_contract_multiplier"] == 10180
    assert math.isclose(adjustment["adjustment_price_ratio"], 10000 / 10180)


def test_close_trade_broker_realized_pnl_is_exact_daily_pnl_anchor():
    result = account_report._daily_position_pnl_breakdown(
        current_qty=0,
        current_side="short",
        current_price=0.1601,
        previous_qty=10,
        previous_side="short",
        previous_price=0.2294 * 10000 / 10180,
        previous_cost=0.185 * 10000 / 10180,
        trade_rows=[
            {
                "\u4e70\u5356": "\u4e70",
                "\u6210\u4ea4\u4ef7\u683c": 0.1623,
                "\u6210\u4ea4\u6570\u91cf": 10,
                "\u5e73\u4ed3\u76c8\u4e8f": 6413.40,
            }
        ],
        multiplier=10180,
    )

    assert math.isclose(result["daily_pnl_decomposition"], 6413.40)


def test_dividend_is_written_as_existing_report_remark():
    payload = {
        "product": "500etf",
        "account_id": "default",
        "date": "2026-07-15",
        "position_history": pd.DataFrame(
            [
                {
                    "日期": "2026-07-14",
                    "账户ID": "default",
                    "方向": "hedge",
                    "总持仓": 34000,
                }
            ]
        ),
    }

    note = account_report._dividend_report_note(payload)

    assert "应收股息5,066.00元" in note
    assert "已计入单日盈亏" in note
    assert "2026-07-20" in note
