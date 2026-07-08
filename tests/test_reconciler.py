from types import SimpleNamespace
from unittest import mock

import pandas as pd

from core.live import reconciler


def test_fund_reconciliation_excludes_option_margin_change(tmp_path):
    fund_dir = tmp_path / "live_hold"
    fund_dir.mkdir()
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {
                "日期": "2026-06-30",
                "账户ID": "default",
                "净单日盈亏": 150.0,
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")
    (fund_dir / "实时资金_2026_06_29-15_00_00.csv").write_text(
        "投资者账号,币种,资金余额,总资产,保证金,期权市值\n"
        "期权-qq,CNY,0,900,200,0\n",
        encoding="utf-8-sig",
    )
    (fund_dir / "实时资金_2026_06_29-15_00_01.csv").write_text(
        "投资者账号,币种,资金余额,冻结资金,可用资金,可取资金,总资产,证券市值\n"
        "证券-zq,CNY,0,--,0,--,1950,0\n",
        encoding="utf-8-sig",
    )
    (fund_dir / "实时资金_2026_06_30-15_00_00.csv").write_text(
        "投资者账号,币种,资金余额,总资产,保证金,期权市值\n"
        "期权-qq,CNY,0,1100,150,0\n",
        encoding="utf-8-sig",
    )
    (fund_dir / "实时资金_2026_06_30-15_00_01.csv").write_text(
        "投资者账号,币种,资金余额,冻结资金,可用资金,可取资金,总资产,证券市值\n"
        "证券-zq,CNY,0,--,0,--,1950,0\n",
        encoding="utf-8-sig",
    )

    with (
        mock.patch.object(
            reconciler.storage,
            "account_report_summary_history_path",
            return_value=summary_path,
        ),
        mock.patch.object(reconciler.account_report, "_live_hold_dir", return_value=fund_dir),
        mock.patch.object(
            reconciler,
            "_merge_latest_report_summary",
            side_effect=lambda _product, history: history,
        ),
    ):
        rows = reconciler.fund_reconciliation_rows(
            account_id="default",
            products=("300etf",),
            report_date="2026-06-30",
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["券商总资产变化"] == 200.0
    assert row["期权保证金变化"] == -50.0
    assert row["剔除保证金后券商资产变化"] == 150.0
    assert row["资金对账差额"] == 0.0
    assert row["资金对账通过"] is True
    assert reconciler.format_fund_reconciliation_terminal(rows)[0].startswith(
        "资金对账 OK"
    )


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
            "单日DeltaPnL": 0.0,
            "单日GammaPnL": 0.0,
            "单日ThetaPnL": 0.0,
            "标的价格": 100.0,
            "对冲最新价": 10.0,
            "对冲持仓": 2.0,
            "Call IV": 0.20,
            "Put IV": 0.20,
            "Call Delta": 10.0,
            "Put Delta": 0.0,
            "Call Gamma": 0.0,
            "Put Gamma": 0.0,
            "Call Vega": 0.0,
            "Put Vega": 0.0,
            "Call Theta": 0.0,
            "Put Theta": 0.0,
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
            "单日DeltaPnL": 60.0,
            "单日GammaPnL": 0.0,
            "单日ThetaPnL": 0.0,
            "标的价格": 105.0,
            "对冲最新价": 15.0,
            "对冲持仓": 2.0,
            "Call IV": 0.20,
            "Put IV": 0.20,
            "Call Delta": 10.0,
            "Put Delta": 0.0,
            "Call Gamma": 0.0,
            "Put Gamma": 0.0,
            "Call Vega": 0.0,
            "Put Vega": 0.0,
            "Call Theta": 0.0,
            "Put Theta": 0.0,
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
    assert checks["trade_pnl_sum"]["ok"] is True
    assert checks["trade_pnl_sum"]["residual"] == 0.0
    assert checks["trade_pnl_sum"]["group"] == "source_check"
    assert checks["greeks_intraday_adjusted"]["group"] == "greeks_check"
    assert checks["greeks_report_recalc"]["group"] == "greeks_check"
    assert checks["greeks_report_recalc"]["ok"] is True
    assert checks["greeks_report_recalc"]["residual"] == 0.0
    for name in [
        "greeks_report_delta_recalc",
        "greeks_report_gamma_recalc",
        "greeks_report_theta_recalc",
    ]:
        assert checks[name]["group"] == "greeks_check"
        assert checks[name]["ok"] is True
        assert checks[name]["residual"] == 0.0
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
    assert len(greeks_lines) == 5


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
                "交易盈亏": -1000.0,
                "当日盯市交易盈亏": -1000.0,
                "当日盈亏分解合计": -1000.0,
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
                "交易盈亏": 0.0,
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
    assert abs(row["opening_mark_pnl"]) < 1e-6
    assert abs(row["decomposition_pnl"] - (-3000.0)) < 1e-6
    assert abs(row["adjusted_decomposition_pnl"] - (-3000.0)) < 1e-6
    assert abs(row["trade_pnl"] - (-1000.0)) < 1e-6
    assert abs(row["residual"]) < 1e-6
    assert row["ok"] is True

    check = reconciler._aggregate_lifecycle_check(rows)
    assert check["name"] == "position_lifecycle_pnl"
    assert check["group"] == "lifecycle_check"
    assert check["ok"] is True
    assert abs(check["residual"]) < 1e-6


def test_daily_checks_use_displayed_position_rows_with_strict_equality():
    history = pd.DataFrame(
        [
            {
                "日期": "2026-06-26",
                "期权单日盈亏": 0.0,
                "ETF单日盈亏": 0.0,
                "总单日盈亏(手续费前)": 0.0,
                "净单日盈亏": 0.0,
                "当日手续费": 0.0,
                "标的价格": 2.133,
            },
            {
                "日期": "2026-06-29",
                "期权单日盈亏": -235.0,
                "ETF单日盈亏": 686.8,
                "总单日盈亏(手续费前)": 451.8,
                "净单日盈亏": 451.8,
                "当日手续费": 0.0,
                "持仓盈亏": 79.5,
                "交易盈亏": 372.3,
                "当日盈亏分解合计": 451.8,
                "标的价格": 2.248,
            },
        ]
    )
    positions = pd.DataFrame(
        [
            {
                "日期": "2026-06-29",
                "合约代码": "10011740",
                "到期日": "2026-07-22",
                "持仓盈亏": -587.5,
                "交易盈亏": 352.5,
            },
            {
                "日期": "2026-06-29",
                "合约代码": "588000",
                "到期日": None,
                "持仓盈亏": 667.0,
                "交易盈亏": 0.0,
            },
        ]
    )

    rows = reconciler._build_daily_rows(
        "kc50etf",
        history,
        positions,
        pd.DataFrame(),
        account_id="default",
        start_date="2026-06-29",
        end_date="2026-06-29",
        abs_tolerance=100.0,
        rel_tolerance=0.25,
    )

    checks = {check["name"]: check for check in rows[0]["checks"]}
    assert checks["summary_decomposition"]["actual"] == 451.8
    assert checks["summary_decomposition"]["expected"] == 432.0
    assert checks["summary_decomposition"]["ok"] is False
    assert checks["position_decomposition_sum"]["ok"] is False
    assert checks["position_hedge_split"]["ok"] is False
    assert checks["position_trade_sum"]["ok"] is False
    assert rows[0]["ok"] is False


def test_trade_pnl_sum_recomputes_from_trade_rows():
    positions = pd.DataFrame(
        [
            {
                "日期": "2026-06-29",
                "合约代码": "588000",
                "最新价": 2.248,
                "到期日": None,
                "交易盈亏": 19.8,
            },
            {
                "日期": "2026-06-29",
                "合约代码": "10011741",
                "最新价": 0.25225,
                "到期日": "2026-07-22",
                "交易盈亏": -887.5,
            },
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "日期": "2026-06-29",
                "合约代码": "588000",
                "买卖": "买",
                "成交数量": 1100,
                "成交价格": 2.23,
            },
            {
                "日期": "2026-06-29",
                "合约代码": "10011741",
                "买卖": "卖",
                "成交数量": 5,
                "成交价格": 0.2345,
            },
        ]
    )

    check = reconciler._trade_pnl_sum_check(
        "kc50etf",
        "2026-06-29",
        positions,
        trades,
        abs_tolerance=0.0,
        rel_tolerance=0.0,
    )

    assert check["ok"] is True
    assert abs(check["actual"] - check["expected"]) < 1e-8
    assert check["residual"] == 0.0

    bad_positions = positions.copy()
    bad_positions.loc[bad_positions["合约代码"].eq("588000"), "交易盈亏"] = 0.0
    bad_check = reconciler._trade_pnl_sum_check(
        "kc50etf",
        "2026-06-29",
        bad_positions,
        trades,
        abs_tolerance=100.0,
        rel_tolerance=0.25,
    )

    assert bad_check["ok"] is False
    assert abs(bad_check["residual"] + 19.8) < 1e-8


def test_position_market_value_identity_reconciles_positions():
    previous_positions = pd.DataFrame(
        [
            {
                "合约代码": "588000",
                "交易方向": "多",
                "总持仓张数": 5800,
                "最新价": 2.133,
                "持仓盈亏": 0.0,
                "到期日": None,
            },
            {
                "合约代码": "10011740",
                "交易方向": "空",
                "总持仓张数": 4,
                "最新价": 0.205,
                "持仓盈亏": 0.0,
                "到期日": "2026-07-22",
            },
        ]
    )
    current_positions = pd.DataFrame(
        [
            {
                "合约代码": "588000",
                "交易方向": "多",
                "总持仓张数": 6900,
                "最新价": 2.248,
                "持仓盈亏": 667.0,
                "到期日": None,
            },
            {
                "合约代码": "10011740",
                "交易方向": "空",
                "总持仓张数": 0,
                "最新价": 0.287,
                "持仓盈亏": -3280.0,
                "到期日": "2026-07-22",
            },
            {
                "合约代码": "10011741",
                "交易方向": "空",
                "总持仓张数": 5,
                "最新价": 0.252,
                "持仓盈亏": 0.0,
                "到期日": "2026-07-22",
            },
        ]
    )

    check = reconciler._position_market_value_identity_check(
        "kc50etf",
        "2026-06-29",
        previous_positions,
        current_positions,
        abs_tolerance=0.0,
        rel_tolerance=0.0,
    )

    assert check["ok"] is True
    assert check["group"] == "source_check"
    assert abs(check["actual"] - (-1260.2)) < 1e-8
    assert abs(check["expected"] - (-1260.2)) < 1e-8
    assert check["residual"] == 0.0

    bad_positions = current_positions.copy()
    bad_positions.loc[bad_positions["合约代码"].eq("10011740"), "持仓盈亏"] = -2434.0
    bad_check = reconciler._position_market_value_identity_check(
        "kc50etf",
        "2026-06-29",
        previous_positions,
        bad_positions,
        abs_tolerance=100.0,
        rel_tolerance=0.25,
    )

    assert bad_check["ok"] is False
    assert abs(bad_check["residual"] + 846.0) < 1e-8


def test_terminal_summary_pins_daily_pnl_identity():
    report_checks = [
        {
            "name": "summary_decomposition",
            "group": "report_check",
            "skipped": False,
            "ok": True,
            "abs_residual": 0.0,
            "ratio": 0.0,
        },
        {
            "name": "position_decomposition_sum",
            "group": "report_check",
            "skipped": False,
            "ok": True,
            "abs_residual": 10.0,
            "ratio": 0.1,
        },
    ]

    displayed = reconciler._terminal_group_checks("report_check", report_checks)

    assert [check["name"] for check in displayed] == [
        "summary_decomposition",
        "position_decomposition_sum",
    ]

    source_checks = [
        {
            "name": "position_market_value_identity",
            "group": "source_check",
            "skipped": False,
            "ok": True,
            "abs_residual": 0.0,
            "ratio": 0.0,
        },
        {
            "name": "trade_pnl_sum",
            "group": "source_check",
            "skipped": False,
            "ok": True,
            "abs_residual": 10.0,
            "ratio": 0.1,
        },
    ]
    displayed = reconciler._terminal_group_checks("source_check", source_checks)

    assert [check["name"] for check in displayed] == [
        "position_market_value_identity",
        "trade_pnl_sum",
    ]


def test_report_close_to_close_greeks_recalc_ignores_vega():
    prev = pd.Series(
        {
            "日期": "2026-06-16",
            "标的价格": 100.0,
            "对冲最新价": 10.0,
            "对冲持仓": 2.0,
            "Call IV": 0.20,
            "Put IV": 0.20,
            "Call Delta": 10.0,
            "Put Delta": 0.0,
            "Call Gamma": 0.0,
            "Put Gamma": 0.0,
            "Call Vega": 100.0,
            "Put Vega": 100.0,
            "Call Theta": 0.0,
            "Put Theta": 0.0,
        }
    )
    current = pd.Series(
        {
            "日期": "2026-06-17",
            "标的价格": 105.0,
            "对冲最新价": 15.0,
            "Call IV": 0.90,
            "Put IV": 0.90,
        }
    )

    result = reconciler._report_close_to_close_greeks_pnl_without_vega(
        "300etf",
        prev,
        current,
    )

    assert result["total"] == 60.0
    assert result["delta"] == 60.0
    assert result["gamma"] == 0.0
    assert result["theta"] == 0.0
    assert "vega=ignored" in result["note"]
