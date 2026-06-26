from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core.live import reconciler


def test_reconcile_builds_account_closure_checks(tmp_path):
    summary_path = tmp_path / "summary.csv"
    rows = [
        {
            "日期": "2026-06-16",
            "账户ID": "default",
            "初始资金": 10_000_000.0,
            "估算权益": 10_000_118.0,
            "期权总盈亏": 100.0,
            "对冲总盈亏": 20.0,
            "手续费": 2.0,
            "当日手续费": 0.0,
            "期权单日盈亏": 0.0,
            "ETF单日盈亏": 0.0,
            "总单日盈亏": 0.0,
            "净单日盈亏": 0.0,
            "持仓盈亏": 0.0,
            "交易盈亏": 0.0,
            "当日盯市交易盈亏": 0.0,
            "当日盈亏分解合计": 0.0,
            "当日盈亏对账差额": 0.0,
            "单日GreeksPnL": 0.0,
            "期权单日GreeksPnL": 0.0,
            "对冲单日GreeksPnL": 0.0,
        },
        {
            "日期": "2026-06-17",
            "账户ID": "default",
            "初始资金": 10_000_000.0,
            "估算权益": 10_000_177.0,
            "期权总盈亏": 150.0,
            "对冲总盈亏": 30.0,
            "手续费": 3.0,
            "当日手续费": 1.0,
            "期权单日盈亏": 50.0,
            "ETF单日盈亏": 10.0,
            "总单日盈亏": 60.0,
            "净单日盈亏": 59.0,
            "持仓盈亏": 60.0,
            "交易盈亏": 0.0,
            "当日盯市交易盈亏": 0.0,
            "当日盈亏分解合计": 60.0,
            "当日盈亏对账差额": 0.0,
            "单日GreeksPnL": 60.0,
            "期权单日GreeksPnL": 50.0,
            "对冲单日GreeksPnL": 10.0,
        },
    ]
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    positions = pd.DataFrame(
        [
            {
                "日期": "2026-06-17",
                "合约代码": "1001",
                "到期日": "2026-07-22",
                "总持仓张数": 1,
                "持仓盈亏": 30.0,
                "交易盈亏": 0.0,
                "当日盯市交易盈亏": 0.0,
                "当日盈亏分解合计": 30.0,
            },
            {
                "日期": "2026-06-17",
                "合约代码": "1002",
                "到期日": "2026-07-22",
                "总持仓张数": 1,
                "持仓盈亏": 20.0,
                "交易盈亏": 0.0,
                "当日盯市交易盈亏": 0.0,
                "当日盈亏分解合计": 20.0,
            },
            {
                "日期": "2026-06-17",
                "合约代码": "510300",
                "到期日": None,
                "总持仓张数": 2,
                "持仓盈亏": 10.0,
                "交易盈亏": 0.0,
                "当日盯市交易盈亏": 0.0,
                "当日盈亏分解合计": 10.0,
            },
        ]
    )
    trades = pd.DataFrame([{"日期": "2026-06-17", "手续费": 1.0}])
    state = SimpleNamespace(
        positions={
            "short": {
                "call_code": "1001",
                "call_qty": 1,
                "put_code": "1002",
                "put_qty": 1,
            },
            "long": None,
        },
        option_hedges=[],
        hedge=SimpleNamespace(underlying_order_book_id="510300", qty=2),
    )

    with (
        mock.patch.object(
            reconciler.storage,
            "account_report_summary_history_path",
            return_value=summary_path,
        ),
        mock.patch.object(reconciler, "_load_position_report_frame", return_value=positions),
        mock.patch.object(reconciler, "_load_trade_report_frame", return_value=trades),
        mock.patch.object(
            reconciler,
            "_merge_latest_report_summary",
            side_effect=lambda _product, history: history,
        ),
        mock.patch.object(reconciler.account_store, "load_account", return_value=state),
        mock.patch.object(reconciler, "_record_reconciliation"),
    ):
        payload = reconciler.reconcile("300etf")

    assert payload["ok"] is True
    assert payload["start_date"] == "2026-06-17"
    assert payload["end_date"] == "2026-06-17"
    assert [row["date"] for row in payload["rows"]] == ["2026-06-17"]
    checks = {check["name"]: check for check in payload["checks"]}
    for name in [
        "total_vs_legs",
        "net_after_fee",
        "position_decomposition_sum",
    ]:
        assert checks[name]["ok"] is True
        assert checks[name]["residual"] == 0.0
        assert checks[name]["group"] == "report_check"
    for name in ["trade_fee_sum", "account_position_snapshot"]:
        assert checks[name]["ok"] is True
        assert checks[name]["residual"] == 0.0
        assert checks[name]["group"] == "source_check"
    assert checks["greeks_intraday_adjusted"]["group"] == "greeks_check"
    assert "greeks_explainability" not in checks
    assert "option_greeks_explainability" not in checks
    assert "hedge_greeks_explainability" not in checks

    lines = reconciler.format_terminal_summary(payload)
    assert "[Source Check]" in lines
    assert "[Report Check]" in lines
    assert "[Lifecycle Check]" in lines
    assert "[Greeks Check]" in lines
    assert any("残差=0.000000 比例=0.000000" in line for line in lines)
    assert all("actual" not in line.lower() for line in lines)
    report_index = lines.index("[Report Check]")
    lifecycle_index = lines.index("[Lifecycle Check]")
    greeks_index = lines.index("[Greeks Check]")
    assert lifecycle_index > report_index
    assert greeks_index > lifecycle_index
    greeks_lines = lines[greeks_index + 1 :]
    assert len(greeks_lines) == 1


def test_position_lifecycle_reconciles_open_mark_and_close_trade_pnl():
    positions = pd.DataFrame(
        [
            {
                "日期": "2026-06-15",
                "合约代码": "1001",
                "合约名称": "300ETF购7月5000",
                "交易方向": "空",
                "总持仓张数": 10,
                "最新价": 0.11,
                "持仓均价": 0.10,
                "持仓盈亏": 0.0,
                "交易盈亏": 0.0,
                "当日盯市交易盈亏": 0.0,
                "当日盈亏分解合计": 0.0,
                "到期日": "2026-07-22",
            },
            {
                "日期": "2026-06-16",
                "合约代码": "1001",
                "合约名称": "300ETF购7月5000",
                "交易方向": "空",
                "总持仓张数": 10,
                "最新价": 0.12,
                "持仓均价": 0.10,
                "持仓盈亏": -1000.0,
                "交易盈亏": 0.0,
                "当日盯市交易盈亏": 0.0,
                "当日盈亏分解合计": -1000.0,
                "到期日": "2026-07-22",
            },
            {
                "日期": "2026-06-17",
                "合约代码": "1001",
                "合约名称": "300ETF购7月5000",
                "交易方向": "空",
                "总持仓张数": 0,
                "最新价": 0.13,
                "持仓均价": 0.10,
                "持仓盈亏": -1000.0,
                "交易盈亏": -3000.0,
                "当日盯市交易盈亏": 0.0,
                "当日盈亏分解合计": -1000.0,
                "到期日": "2026-07-22",
            },
        ]
    )

    rows = reconciler._build_position_lifecycle_rows(
        "300etf",
        positions,
        start_date="2026-06-17",
        end_date="2026-06-17",
        abs_tolerance=1e-6,
        rel_tolerance=0.0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["closed"] is True
    assert row["start_date"] == "2026-06-15"
    assert row["end_date"] == "2026-06-17"
    assert abs(row["opening_mark_pnl"] - (-1000.0)) < 1e-6
    assert abs(row["decomposition_pnl"] - (-2000.0)) < 1e-6
    assert abs(row["adjusted_decomposition_pnl"] - (-3000.0)) < 1e-6
    assert abs(row["trade_pnl"] - (-3000.0)) < 1e-6
    assert abs(row["residual"]) < 1e-6
    assert row["ok"] is True

    check = reconciler._aggregate_lifecycle_check(rows)
    assert check["name"] == "position_lifecycle_pnl"
    assert check["group"] == "lifecycle_check"
    assert check["ok"] is True
    assert abs(check["residual"]) < 1e-6
