from pathlib import Path
import json

import pandas as pd
import pytest

from core.live import intraday_pnl


def test_intraday_pnl_uses_previous_close_positions_and_latest_snapshot(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.csv"
    position_path = tmp_path / "positions.csv"
    etf_path = tmp_path / "etf.parquet"
    option_path = tmp_path / "option.parquet"

    pd.DataFrame(
        [
            {
                "日期": "2026-06-25",
                "账户ID": "default",
                "标的价格": 100.0,
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "日期": "2026-06-25",
                "账户ID": "default",
                "方向": "short",
                "合约代码": "10000001",
                "合约名称": "test call",
                "总持仓": 1.0,
                "最新价": 1.0,
                "行权价": 100.0,
                "到期日": "2026-07-22",
                "剩余天数": 19.0,
                "IV": 0.20,
                "Delta": -10.0,
                "Gamma": 0.0,
                "Vega": -2.0,
                "Theta": 3.0,
            },
            {
                "日期": "2026-06-25",
                "账户ID": "default",
                "方向": "hedge",
                "合约代码": "510300",
                "总持仓": 5.0,
                "最新价": 100.0,
            },
            {
                "日期": "2026-06-26",
                "账户ID": "default",
                "方向": "short",
                "合约代码": "10000002",
                "合约名称": "same-day new call",
                "总持仓": 99.0,
                "最新价": 9.9,
            },
        ]
    ).to_csv(position_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"close": 101.0}]).to_parquet(etf_path)
    pd.DataFrame(
        [
            {
                "order_book_id": "10000001",
                "close": 1.2,
            },
            {
                "order_book_id": "10000002",
                "close": 8.8,
            },
        ]
    ).to_parquet(option_path)

    monkeypatch.setattr(
        intraday_pnl.storage,
        "account_report_summary_history_path",
        lambda product, account_id="default": Path(summary_path),
    )
    monkeypatch.setattr(
        intraday_pnl.storage,
        "account_report_position_history_path",
        lambda product, account_id="default": Path(position_path),
    )
    monkeypatch.setattr(
        intraday_pnl.market_data,
        "load_latest_quote_snapshot",
        lambda product, date="latest": {
            "quote_date": "2026-06-26",
            "snapshot_stamp": "20260626_145959",
            "etf_snapshot": str(etf_path),
            "option_snapshot": str(option_path),
            "metadata_path": str(tmp_path / "metadata.json"),
        },
    )
    monkeypatch.setattr(intraday_pnl, "_position_multiplier", lambda product, row: 100.0)
    monkeypatch.setattr(intraday_pnl, "_trading_day_step", lambda previous, current, product: 1.0)
    monkeypatch.setattr(
        intraday_pnl,
        "_current_greeks",
        lambda product, row, price, spot, signed_qty: {"iv": 0.22},
    )

    payload = intraday_pnl.calculate_intraday_pnl("300etf")

    assert payload["summary"]["previous_date"] == "2026-06-25"
    assert payload["summary"]["quote_date"] == "2026-06-26"
    assert payload["summary"]["option_count"] == 1
    assert payload["option_rows"][0]["code"] == "10000001"
    assert payload["summary"]["option_actual_pnl"] == pytest.approx(-20.0)
    assert payload["summary"]["hedge_actual_pnl"] == pytest.approx(5.0)
    assert payload["summary"]["actual_pnl"] == pytest.approx(-15.0)
    assert payload["summary"]["delta_pnl"] == pytest.approx(-5.0)
    assert payload["summary"]["vega_pnl"] == pytest.approx(-4.0)
    assert payload["summary"]["theta_pnl"] == pytest.approx(3.0)
    assert payload["summary"]["greeks_pnl"] == pytest.approx(-6.0)

    lines = intraday_pnl.format_intraday_pnl(payload)
    assert "盘中盈亏 300etf" in lines[0]
    assert lines == [
        "盘中盈亏 300etf 账户=default 2026-06-25昨收 -> 2026-06-26当前 快照=20260626_145959",
        "实际盈亏=-15.000000",
        "Greeks盈亏=-6.000000",
    ]
    assert "实际盈亏=-15.000000" in "\n".join(lines)
    assert "期权盈亏" not in "\n".join(lines)
    assert "ETF盈亏" not in "\n".join(lines)
    assert "合约代码" not in "\n".join(lines)

    json_payload = json.loads(intraday_pnl.intraday_pnl_json(payload))
    assert json_payload["汇总"]["品种"] == "300etf"
    assert json_payload["汇总"]["实际盈亏"] == pytest.approx(-15.0)
    assert json_payload["汇总"]["Greeks盈亏"] == pytest.approx(-6.0)
    assert "期权盈亏" not in json_payload["汇总"]
    assert "ETF盈亏" not in json_payload["汇总"]
    assert "期权明细" not in json_payload
    assert "ETF明细" not in json_payload
