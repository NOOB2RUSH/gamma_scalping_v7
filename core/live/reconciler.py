from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import account as account_store
from . import account_report
from . import storage


DEFAULT_ABS_TOLERANCE = 100.0
DEFAULT_REL_TOLERANCE = 0.25


CHECK_GROUP_LABELS = {
    "source_check": "Source Check",
    "report_check": "Report Check",
    "lifecycle_check": "Lifecycle Check",
    "greeks_check": "Greeks Check",
}


CHECK_DEFINITIONS = {
    "total_vs_legs": "总单日盈亏 = 期权单日盈亏 + ETF单日盈亏",
    "net_after_fee": "净单日盈亏 = 总单日盈亏 - 当日手续费",
    "summary_decomposition": "总单日盈亏 = 持仓盈亏 + 当日盯市交易盈亏",
    "summary_decomposition_residual": "当日盈亏对账差额 = 总单日盈亏 - 分解合计",
    "position_holding_sum": "汇总持仓盈亏 = 持仓记录持仓盈亏合计",
    "position_trade_sum": "汇总交易盈亏 = 持仓记录交易盈亏合计",
    "position_mark_trade_sum": "汇总盯市交易盈亏 = 持仓记录盯市交易盈亏合计",
    "position_decomposition_sum": "汇总分解合计 = 持仓记录分解合计",
    "position_option_split": "期权单日盈亏 = 期权持仓记录分解合计",
    "position_hedge_split": "ETF单日盈亏 = ETF持仓记录分解合计",
    "trade_fee_sum": "当日手续费 = 交易记录手续费合计",
    "account_position_snapshot": "账户当前持仓 = 最新持仓记录",
    "position_lifecycle_pnl": "持仓周期交易盈亏 = 周期盯市分解合计",
    "greeks_intraday_adjusted": "总单日盈亏 = 单日GreeksPnL",
}


CHECK_GROUPS = {
    "trade_fee_sum": "source_check",
    "account_position_snapshot": "source_check",
    "position_lifecycle_pnl": "lifecycle_check",
    "greeks_intraday_adjusted": "greeks_check",
}


def reconcile(
    product,
    broker_snapshot=None,
    account_id="default",
    start_date=None,
    end_date=None,
    abs_tolerance=DEFAULT_ABS_TOLERANCE,
    rel_tolerance=DEFAULT_REL_TOLERANCE,
):
    """Run account-level reconciliation checks for one live product.

    The broker_snapshot argument is kept for compatibility with older callers and
    is intentionally ignored by the current broker-import-driven workflow.
    """
    del broker_snapshot

    summary_path = storage.account_report_summary_history_path(product, account_id)
    if not Path(summary_path).exists():
        raise FileNotFoundError(f"Account summary history not found: {summary_path}")

    history = pd.read_csv(summary_path, encoding="utf-8-sig")
    if history.empty:
        raise ValueError(f"Account summary history is empty: {summary_path}")

    account_history = history[
        history["账户ID"].astype(str).eq(str(account_id))
    ].copy()
    if account_history.empty:
        raise ValueError(f"No summary history for account_id={account_id}")

    summary_frame = _merge_latest_report_summary(product, account_history)
    if start_date is None and end_date is None:
        latest_date = _latest_summary_date(summary_frame)
        if latest_date is not None:
            start_date = latest_date
            end_date = latest_date
    position_frame = _load_position_report_frame(product, account_id)
    trade_frame = _load_trade_report_frame(product, account_id)

    rows = _build_daily_rows(
        product,
        summary_frame,
        position_frame,
        trade_frame,
        account_id=account_id,
        start_date=start_date,
        end_date=end_date,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    checks = _aggregate_checks(rows)
    lifecycle_rows = _build_position_lifecycle_rows(
        product,
        position_frame,
        start_date=start_date,
        end_date=end_date,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    checks.append(_aggregate_lifecycle_check(lifecycle_rows))
    metrics = _aggregate_metrics(rows, checks)
    payload = {
        "product": product,
        "account_id": account_id,
        "ok": bool(rows) and all(check["ok"] for check in checks if not check["skipped"]),
        "mode": "account_reconciliation_by_layer",
        "start_date": start_date,
        "end_date": end_date,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "metrics": metrics,
        "checks": checks,
        "rows": rows,
        "lifecycle_rows": lifecycle_rows,
        "summary_history_path": str(summary_path),
    }
    _record_reconciliation(product, payload, account_id)
    return payload


def write_reconcile_report(product, payload):
    stamp = storage.local_now_stamp()
    path = storage.output_dir(product) / f"{stamp}_reconcile.md"
    metrics = payload.get("metrics", {})
    lines = [
        f"# Account Reconciliation: {product}",
        "",
        f"- account_id: {payload['account_id']}",
        f"- mode: {payload.get('mode')}",
        f"- ok: {payload['ok']}",
        f"- tolerance: abs<={_fmt(payload.get('abs_tolerance'))}, "
        f"rel<={_fmt(payload.get('rel_tolerance'))}",
        f"- rows: {metrics.get('row_count', 0)}",
        "",
        "## Check Summary",
    ]
    for group, group_label in CHECK_GROUP_LABELS.items():
        group_checks = [
            check for check in payload.get("checks", [])
            if check.get("group") == group
        ]
        if not group_checks:
            continue
        lines.extend(
            [
                "",
                f"### {group_label}",
                "",
                "| check | residual | ratio | rows | skipped | ok |",
                "|---|---:|---:|---:|---:|:---:|",
            ]
        )
        for check in group_checks:
            lines.append(
                "| {label} | {residual} | {ratio} | {rows} | {skipped} | {ok} |".format(
                    label=check.get("label", check.get("name")),
                    residual=_fmt(check.get("residual")),
                    ratio=_fmt(check.get("ratio")),
                    rows=check.get("row_count", 0),
                    skipped=check.get("skipped_count", 0),
                    ok="Y" if check.get("ok") else "N",
                )
            )

    lines.extend(
        [
            "",
            "## Daily Checks",
            "",
            "| date | layer | check | actual | expected | residual | ratio | ok | note |",
            "|---|---|---|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for row in payload.get("rows", []):
        for check in row.get("checks", []):
            lines.append(
                "| {date} | {layer} | {label} | {actual} | {expected} | {residual} | {ratio} | {ok} | {note} |".format(
                    date=row["date"],
                    layer=check.get("group_label", check.get("group")),
                    label=check.get("label", check.get("name")),
                    actual=_fmt(check.get("actual")),
                    expected=_fmt(check.get("expected")),
                    residual=_fmt(check.get("residual")),
                    ratio=_fmt(check.get("ratio")),
                    ok="Y" if check.get("ok") else "N",
                    note=str(check.get("note") or ""),
                )
            )
    lifecycle_rows = payload.get("lifecycle_rows") or []
    if lifecycle_rows:
        lines.extend(
            [
                "",
                "## Lifecycle Checks",
                "",
                "| code | name | start | end | rows | trade pnl | adjusted decomposition | residual | ratio | ok | note |",
                "|---|---|---|---|---:|---:|---:|---:|---:|:---:|---|",
            ]
        )
        for row in lifecycle_rows:
            lines.append(
                "| {code} | {name} | {start} | {end} | {rows} | {actual} | {expected} | {residual} | {ratio} | {ok} | {note} |".format(
                    code=row.get("contract_code"),
                    name=row.get("contract_name") or "",
                    start=row.get("start_date"),
                    end=row.get("end_date") or "",
                    rows=row.get("row_count", 0),
                    actual=_fmt(row.get("actual")),
                    expected=_fmt(row.get("expected")),
                    residual=_fmt(row.get("residual")),
                    ratio=_fmt(row.get("ratio")),
                    ok="Y" if row.get("ok") else "N",
                    note=str(row.get("note") or ""),
                )
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_terminal_summary(payload):
    metrics = payload.get("metrics", {})
    lines = [
        (
            f"账户对账 {payload['product']}/{payload['account_id']} "
            f"ok={payload['ok']} rows={metrics.get('row_count', 0)}"
        )
    ]
    checks = payload.get("checks", [])
    if not checks:
        lines.append("(none)")
        return lines
    for group, group_label in CHECK_GROUP_LABELS.items():
        group_checks = [check for check in checks if check.get("group") == group]
        if not group_checks:
            continue
        lines.append(f"[{group_label}]")
        display_checks = _terminal_group_checks(group, group_checks)
        for check in display_checks:
            status = "OK" if check.get("ok") else "FAIL"
            if check.get("skipped"):
                status = "SKIP"
            lines.append(
                f"{status} {check.get('label', check.get('name'))}: "
                f"残差={_fmt(check.get('residual'))} "
                f"比例={_fmt(check.get('ratio'))}"
            )
    return lines


def _terminal_group_checks(group, checks):
    if group == "greeks_check":
        return checks
    active = [check for check in checks if not check.get("skipped")]
    failed = [check for check in active if not check.get("ok")]
    if failed:
        return checks
    if not active:
        return checks[:1]
    return [
        max(
            active,
            key=lambda check: (
                _zero_if_none(check.get("abs_residual")),
                _zero_if_none(check.get("ratio")),
            ),
        )
    ]


def _build_daily_rows(
    product,
    history,
    position_frame,
    trade_frame,
    account_id,
    start_date=None,
    end_date=None,
    abs_tolerance=DEFAULT_ABS_TOLERANCE,
    rel_tolerance=DEFAULT_REL_TOLERANCE,
):
    required = ["日期"]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(f"Summary history missing columns: {missing}")

    frame = history.copy()
    frame["_date"] = pd.to_datetime(frame["日期"], errors="coerce")
    frame = frame.dropna(subset=["_date"]).sort_values("_date")
    rows = []
    start = _date_or_none(start_date)
    end = _date_or_none(end_date)

    for i in range(len(frame)):
        prev = frame.iloc[i - 1] if i > 0 else None
        current = frame.iloc[i]
        current_date = current["_date"].date()
        if start is not None and current_date < start:
            continue
        if end is not None and current_date > end:
            continue
        if prev is None:
            continue

        checks = []
        total_daily_pnl = _summary_total_daily_pnl(current)
        option_daily_pnl = _number(current.get("期权单日盈亏"))
        hedge_daily_pnl = _summary_hedge_daily_pnl(current)
        daily_fee = _number(current.get("当日手续费")) or 0.0
        net_daily_pnl = _summary_net_daily_pnl(current)
        date_positions = _rows_for_date(position_frame, current_date)

        checks.append(
            _check_value(
                "total_vs_legs",
                current_date,
                total_daily_pnl,
                _sum_optional(option_daily_pnl, hedge_daily_pnl),
                abs_tolerance,
                rel_tolerance,
            )
        )
        checks.append(
            _check_value(
                "net_after_fee",
                current_date,
                net_daily_pnl,
                None if total_daily_pnl is None else total_daily_pnl - daily_fee,
                abs_tolerance,
                rel_tolerance,
            )
        )
        decomposition = _number(current.get("当日盈亏分解合计"))
        holding_pnl = _number(current.get("持仓盈亏"))
        mark_trade_pnl = _number(current.get("当日盯市交易盈亏"))
        if _skip_new_position_pnl_decomposition(date_positions):
            checks.append(
                _skipped_check(
                    "summary_decomposition",
                    current_date,
                    "new_position_pnl_not_reconciled",
                )
            )
            checks.append(
                _skipped_check(
                    "summary_decomposition_residual",
                    current_date,
                    "new_position_pnl_not_reconciled",
                )
            )
        else:
            checks.append(
                _check_value(
                    "summary_decomposition",
                    current_date,
                    total_daily_pnl,
                    _sum_optional(holding_pnl, mark_trade_pnl, fallback=decomposition),
                    abs_tolerance,
                    rel_tolerance,
                )
            )
            checks.append(
                _check_value(
                    "summary_decomposition_residual",
                    current_date,
                    _number(current.get("当日盈亏对账差额")),
                    _difference(total_daily_pnl, decomposition),
                    abs_tolerance,
                    rel_tolerance,
                )
            )

        checks.extend(
            _position_checks(
                current_date,
                current,
                date_positions,
                abs_tolerance,
                rel_tolerance,
            )
        )

        date_trades = _rows_for_date(trade_frame, current_date)
        checks.append(
            _check_value(
                "trade_fee_sum",
                current_date,
                daily_fee,
                _sum_numeric_column(date_trades, "手续费"),
                abs_tolerance,
                rel_tolerance,
                note="no_trade_rows" if date_trades.empty else None,
            )
        )

        greeks_pnl = _number(current.get("单日GreeksPnL"))
        checks.append(
            _check_value(
                "greeks_intraday_adjusted",
                current_date,
                total_daily_pnl,
                greeks_pnl,
                abs_tolerance,
                rel_tolerance,
                note=(
                    f"greeks_pnl_scope={current.get('GreeksPnL口径') or 'unknown'};"
                    f"{current.get('GreeksPnL说明') or ''}"
                ),
            )
        )

        rows.append(
            {
                "date": str(current_date),
                "previous_date": str(prev["_date"].date()),
                "total_daily_pnl": total_daily_pnl,
                "net_daily_pnl": net_daily_pnl,
                "option_daily_pnl": option_daily_pnl,
                "hedge_daily_pnl": hedge_daily_pnl,
                "daily_fee": daily_fee,
                "greeks_pnl": greeks_pnl,
                "checks": checks,
                "ok": all(check["ok"] for check in checks if not check["skipped"]),
            }
        )

    snapshot_check = _account_position_snapshot_check(
        product,
        account_id,
        frame,
        position_frame,
        abs_tolerance,
        rel_tolerance,
    )
    if snapshot_check is not None:
        if rows and rows[-1]["date"] == snapshot_check["date"]:
            rows[-1]["checks"].append(snapshot_check)
            rows[-1]["ok"] = all(
                check["ok"] for check in rows[-1]["checks"] if not check["skipped"]
            )
        else:
            rows.append(
                {
                    "date": snapshot_check["date"],
                    "previous_date": None,
                    "checks": [snapshot_check],
                    "ok": snapshot_check["ok"],
                }
            )
    return rows


def _latest_summary_date(history):
    if history.empty or "日期" not in history.columns:
        return None
    dates = pd.to_datetime(history["日期"], errors="coerce").dropna()
    if dates.empty:
        return None
    return str(dates.max().date())


def _position_checks(date, summary_row, position_frame, abs_tolerance, rel_tolerance):
    if position_frame is None or position_frame.empty:
        return [
            _skipped_check(name, date, "no_position_rows")
            for name in [
                "position_holding_sum",
                "position_trade_sum",
                "position_mark_trade_sum",
                "position_decomposition_sum",
                "position_option_split",
                "position_hedge_split",
            ]
        ]

    position_decomposition = _position_decomposition_series(position_frame)
    option_mask = _position_option_mask(position_frame)
    hedge_mask = ~option_mask
    return [
        _check_value(
            "position_holding_sum",
            date,
            _number(summary_row.get("持仓盈亏")),
            _sum_numeric_column(position_frame, "持仓盈亏"),
            abs_tolerance,
            rel_tolerance,
        ),
        _check_value(
            "position_trade_sum",
            date,
            _number(summary_row.get("交易盈亏")),
            _sum_numeric_column(position_frame, "交易盈亏"),
            abs_tolerance,
            rel_tolerance,
        ),
        _check_value(
            "position_mark_trade_sum",
            date,
            _number(summary_row.get("当日盯市交易盈亏")),
            _sum_numeric_column(position_frame, "当日盯市交易盈亏"),
            abs_tolerance,
            rel_tolerance,
        ),
        _check_value(
            "position_decomposition_sum",
            date,
            _number(summary_row.get("当日盈亏分解合计")),
            float(position_decomposition.sum()) if position_decomposition is not None else None,
            abs_tolerance,
            rel_tolerance,
        ),
        _check_value(
            "position_option_split",
            date,
            _number(summary_row.get("期权单日盈亏")),
            (
                float(position_decomposition.loc[option_mask].sum())
                if position_decomposition is not None
                else None
            ),
            abs_tolerance,
            rel_tolerance,
        ),
        _check_value(
            "position_hedge_split",
            date,
            _summary_hedge_daily_pnl(summary_row),
            (
                float(position_decomposition.loc[hedge_mask].sum())
                if position_decomposition is not None
                else None
            ),
            abs_tolerance,
            rel_tolerance,
        ),
    ]


def _account_position_snapshot_check(
    product,
    account_id,
    summary_frame,
    position_frame,
    abs_tolerance,
    rel_tolerance,
):
    if summary_frame.empty or position_frame is None or position_frame.empty:
        return None
    latest_date = summary_frame["_date"].max().date()
    current_positions = _rows_for_date(position_frame, latest_date)
    if current_positions.empty:
        return _skipped_check("account_position_snapshot", latest_date, "no_position_rows")

    expected = _expected_account_quantities(product, account_id)
    actual = _actual_position_quantities(current_positions)
    codes = sorted(set(expected) | set(actual))
    if not codes:
        return _skipped_check("account_position_snapshot", latest_date, "no_active_positions")
    residual = sum(abs(actual.get(code, 0.0) - expected.get(code, 0.0)) for code in codes)
    denominator = sum(abs(value) for value in expected.values())
    return _check_value(
        "account_position_snapshot",
        latest_date,
        residual,
        0.0,
        abs_tolerance,
        rel_tolerance,
        denominator=denominator,
    )


def _build_position_lifecycle_rows(
    product,
    position_frame,
    start_date=None,
    end_date=None,
    abs_tolerance=DEFAULT_ABS_TOLERANCE,
    rel_tolerance=DEFAULT_REL_TOLERANCE,
):
    required = {"日期", "合约代码", "总持仓张数"}
    if position_frame is None or position_frame.empty or not required.issubset(
        position_frame.columns
    ):
        return []
    frame = position_frame.copy()
    frame["_lifecycle_date"] = pd.to_datetime(frame["日期"], errors="coerce")
    frame = frame.dropna(subset=["_lifecycle_date"])
    if frame.empty:
        return []
    frame["_lifecycle_code"] = frame["合约代码"].map(_position_code_key)
    frame = frame.loc[frame["_lifecycle_code"].astype(str).ne("")]
    if "到期日" in frame.columns:
        frame = frame.loc[_position_option_mask(frame)]
    if frame.empty:
        return []

    rows = []
    start = _date_or_none(start_date)
    end = _date_or_none(end_date)
    for _, group in frame.sort_values(
        ["_lifecycle_code", "_lifecycle_date"],
        kind="stable",
    ).groupby("_lifecycle_code", sort=False):
        active_rows = []
        was_active = False
        for _, row in group.iterrows():
            qty = abs(_number(row.get("总持仓张数")) or 0.0)
            has_pnl = any(
                (_number(row.get(column)) or 0.0) != 0.0
                for column in (
                    "持仓盈亏",
                    "交易盈亏",
                    "当日盯市交易盈亏",
                    "当日盈亏分解合计",
                )
            )
            if not was_active and qty <= 1e-9 and not has_pnl:
                continue
            if not active_rows:
                active_rows = [row]
            elif was_active or qty > 1e-9 or has_pnl:
                active_rows.append(row)
            was_active = qty > 1e-9
            if active_rows and qty <= 1e-9:
                lifecycle = _position_lifecycle_row(
                    product,
                    active_rows,
                    closed=True,
                    abs_tolerance=abs_tolerance,
                    rel_tolerance=rel_tolerance,
                )
                if _include_lifecycle_row(lifecycle, start, end):
                    rows.append(lifecycle)
                active_rows = []
                was_active = False
        if active_rows:
            lifecycle = _position_lifecycle_row(
                product,
                active_rows,
                closed=False,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            if _include_lifecycle_row(lifecycle, start, end):
                rows.append(lifecycle)
    return rows


def _position_lifecycle_row(
    product,
    rows,
    closed,
    abs_tolerance,
    rel_tolerance,
):
    first = rows[0]
    last = rows[-1]
    holding_pnl = _sum_row_values(rows, "持仓盈亏")
    mark_trade_pnl = _sum_row_values(rows, "当日盯市交易盈亏")
    decomposition_pnl = _sum_lifecycle_decomposition(rows)
    opening_mark_pnl = _opening_mark_pnl(product, first)
    adjusted_decomposition = decomposition_pnl + opening_mark_pnl
    trade_pnl = _sum_row_values(rows, "交易盈亏")
    residual = trade_pnl - adjusted_decomposition if closed else None
    denominator = abs(trade_pnl)
    ratio = _safe_ratio(abs(residual), denominator) if residual is not None else None
    tolerance = (
        max(abs_tolerance, rel_tolerance * denominator)
        if residual is not None
        else None
    )
    return {
        "contract_code": _position_code_key(first.get("合约代码")),
        "contract_name": first.get("合约名称"),
        "start_date": str(first["_lifecycle_date"].date()),
        "end_date": str(last["_lifecycle_date"].date()) if closed else None,
        "closed": closed,
        "row_count": len(rows),
        "holding_pnl": holding_pnl,
        "mark_trade_pnl": mark_trade_pnl,
        "opening_mark_pnl": opening_mark_pnl,
        "decomposition_pnl": decomposition_pnl,
        "adjusted_decomposition_pnl": adjusted_decomposition,
        "trade_pnl": trade_pnl,
        "actual": trade_pnl if closed else None,
        "expected": adjusted_decomposition if closed else None,
        "residual": residual,
        "abs_residual": abs(residual) if residual is not None else None,
        "denominator": denominator if closed else None,
        "ratio": ratio,
        "tolerance": tolerance,
        "skipped": not closed,
        "ok": True if not closed else abs(residual) <= tolerance,
        "note": None if closed else "open_lifecycle",
    }


def _sum_lifecycle_decomposition(rows):
    total = 0.0
    for row in rows:
        explicit = _number(row.get("当日盈亏分解合计"))
        if explicit is not None:
            total += explicit
        else:
            total += (_number(row.get("持仓盈亏")) or 0.0) + (
                _number(row.get("当日盯市交易盈亏")) or 0.0
            )
    return total


def _opening_mark_pnl(product, row):
    qty = abs(_number(row.get("总持仓张数")) or 0.0)
    latest = _number(row.get("最新价"))
    cost = _number(row.get("持仓均价"))
    if qty <= 1e-9 or latest is None or cost is None:
        return 0.0
    direction = -1.0 if str(row.get("交易方向") or "") == "空" else 1.0
    return direction * (latest - cost) * qty * _contract_multiplier(product)


def _include_lifecycle_row(row, start, end):
    end_text = row.get("end_date") or row.get("start_date")
    date = _date_or_none(end_text)
    if date is None:
        return False
    if start is not None and date < start:
        return False
    if end is not None and date > end:
        return False
    return True


def _aggregate_lifecycle_check(lifecycle_rows):
    active = [row for row in lifecycle_rows if not row.get("skipped")]
    skipped_count = len(lifecycle_rows) - len(active)
    if not active:
        return {
            "name": "position_lifecycle_pnl",
            "label": CHECK_DEFINITIONS["position_lifecycle_pnl"],
            "group": _check_group("position_lifecycle_pnl"),
            "group_label": _check_group_label("position_lifecycle_pnl"),
            "residual": None,
            "abs_residual": None,
            "ratio": None,
            "row_count": 0,
            "skipped_count": skipped_count,
            "skipped": True,
            "ok": True,
        }
    abs_residual = sum(row["abs_residual"] for row in active)
    denominator = sum(abs(row.get("denominator") or 0.0) for row in active)
    return {
        "name": "position_lifecycle_pnl",
        "label": CHECK_DEFINITIONS["position_lifecycle_pnl"],
        "group": _check_group("position_lifecycle_pnl"),
        "group_label": _check_group_label("position_lifecycle_pnl"),
        "residual": abs_residual,
        "signed_residual": sum(row["residual"] for row in active),
        "abs_residual": abs_residual,
        "ratio": _safe_ratio(abs_residual, denominator),
        "row_count": len(active),
        "skipped_count": skipped_count,
        "skipped": False,
        "ok": all(row["ok"] for row in active),
    }


def _aggregate_checks(rows):
    by_name = {
        name: []
        for name in CHECK_DEFINITIONS
        if _check_group(name) != "lifecycle_check"
    }
    for row in rows:
        for check in row.get("checks", []):
            by_name.setdefault(check["name"], []).append(check)

    aggregates = []
    for name, checks in by_name.items():
        non_skipped = [check for check in checks if not check.get("skipped")]
        skipped_count = len(checks) - len(non_skipped)
        if not non_skipped:
            aggregates.append(
                {
                    "name": name,
                    "label": CHECK_DEFINITIONS.get(name, name),
                    "group": _check_group(name),
                    "group_label": _check_group_label(name),
                    "residual": None,
                    "abs_residual": None,
                    "ratio": None,
                    "row_count": 0,
                    "skipped_count": skipped_count,
                    "skipped": True,
                    "ok": True,
                }
            )
            continue
        residual = sum(check["residual"] for check in non_skipped)
        abs_residual = sum(abs(check["residual"]) for check in non_skipped)
        denominator = sum(abs(check.get("denominator") or 0.0) for check in non_skipped)
        aggregates.append(
            {
                "name": name,
                "label": CHECK_DEFINITIONS.get(name, name),
                "group": _check_group(name),
                "group_label": _check_group_label(name),
                "residual": abs_residual,
                "signed_residual": residual,
                "abs_residual": abs_residual,
                "ratio": _safe_ratio(abs_residual, denominator),
                "row_count": len(non_skipped),
                "skipped_count": skipped_count,
                "skipped": False,
                "ok": all(check["ok"] for check in non_skipped),
            }
        )
    return aggregates


def _aggregate_metrics(rows, checks):
    total_daily_pnl = sum(
        _zero_if_none(row.get("total_daily_pnl")) for row in rows
    )
    total_greeks_pnl = sum(_zero_if_none(row.get("greeks_pnl")) for row in rows)
    greeks_check = next(
        (check for check in checks if check["name"] == "greeks_intraday_adjusted"),
        None,
    )
    residuals = [
        check["residual"]
        for row in rows
        for check in row.get("checks", [])
        if not check.get("skipped")
    ]
    return {
        "row_count": len(rows),
        "check_count": sum(check["row_count"] for check in checks),
        "failed_check_count": sum(
            1 for check in checks if not check["skipped"] and not check["ok"]
        ),
        "groups": _aggregate_group_metrics(checks),
        "total_daily_pnl": total_daily_pnl,
        "total_greeks_pnl": total_greeks_pnl,
        "total_residual": None if greeks_check is None else greeks_check.get("residual"),
        "explained_ratio": _safe_ratio(total_greeks_pnl, total_daily_pnl),
        "mean_abs_residual": (
            sum(abs(value) for value in residuals) / len(residuals)
            if residuals
            else None
        ),
        "rmse_residual": (
            math.sqrt(sum(value * value for value in residuals) / len(residuals))
            if residuals
            else None
        ),
    }


def _check_value(
    name,
    date,
    actual,
    expected,
    abs_tolerance,
    rel_tolerance,
    denominator=None,
    note=None,
):
    label = CHECK_DEFINITIONS.get(name, name)
    if actual is None or expected is None:
        return _skipped_check(name, date, "missing_value" if note is None else note)
    residual = float(actual) - float(expected)
    if abs(residual) < 1e-8:
        residual = 0.0
    denominator_value = (
        abs(float(denominator))
        if denominator is not None
        else max(abs(float(actual)), abs(float(expected)))
    )
    ratio = _safe_ratio(abs(residual), denominator_value)
    tolerance = max(abs_tolerance, denominator_value * rel_tolerance)
    return {
        "name": name,
        "label": label,
        "group": _check_group(name),
        "group_label": _check_group_label(name),
        "date": str(date),
        "actual": float(actual),
        "expected": float(expected),
        "residual": residual,
        "abs_residual": abs(residual),
        "denominator": denominator_value,
        "ratio": ratio,
        "tolerance": tolerance,
        "skipped": False,
        "ok": abs(residual) <= tolerance,
        "note": note,
    }


def _skipped_check(name, date, note):
    return {
        "name": name,
        "label": CHECK_DEFINITIONS.get(name, name),
        "group": _check_group(name),
        "group_label": _check_group_label(name),
        "date": str(date),
        "actual": None,
        "expected": None,
        "residual": None,
        "abs_residual": None,
        "denominator": None,
        "ratio": None,
        "tolerance": None,
        "skipped": True,
        "ok": True,
        "note": note,
    }


def _check_group(name):
    return CHECK_GROUPS.get(name, "report_check")


def _check_group_label(name):
    return CHECK_GROUP_LABELS[_check_group(name)]


def _aggregate_group_metrics(checks):
    metrics = {}
    for group, label in CHECK_GROUP_LABELS.items():
        group_checks = [check for check in checks if check.get("group") == group]
        active = [check for check in group_checks if not check.get("skipped")]
        metrics[group] = {
            "label": label,
            "check_count": sum(check.get("row_count", 0) for check in group_checks),
            "failed_check_count": sum(
                1 for check in active if not check.get("ok")
            ),
            "skipped_check_count": sum(
                check.get("skipped_count", 0) for check in group_checks
            ),
            "ok": all(check.get("ok") for check in active),
        }
    return metrics


def _merge_latest_report_summary(product, history):
    report = _latest_portfolio_report_frames(product)
    if report is None:
        return history
    summary = report.get("账户总体情况")
    if summary is None or summary.empty:
        return history
    product_summary = _filter_portfolio_product_rows(summary, product)
    if product_summary.empty:
        return history

    result = history.copy()
    result["_merge_date"] = pd.to_datetime(result["日期"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    product_summary = product_summary.copy()
    product_summary["_merge_date"] = pd.to_datetime(
        product_summary["日期"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    product_summary = product_summary.drop_duplicates("_merge_date", keep="last")
    for column in product_summary.columns:
        if column in {"_merge_date", "策略名称", "合约代码", "备注"}:
            continue
        if column not in result.columns:
            result[column] = None
        updates = product_summary.set_index("_merge_date")[column]
        mask = result["_merge_date"].isin(updates.index)
        result.loc[mask, column] = result.loc[mask, "_merge_date"].map(updates)
    return result.drop(columns=["_merge_date"])


def _load_position_report_frame(product, account_id):
    report = _latest_portfolio_report_frames(product)
    if report is not None:
        positions = report.get("持仓记录")
        if positions is not None and not positions.empty:
            filtered = _filter_portfolio_product_rows(positions, product)
            if not filtered.empty:
                return filtered

    position_path = storage.account_report_position_history_path(product, account_id)
    if not Path(position_path).exists():
        return pd.DataFrame()
    frame = pd.read_csv(position_path, encoding="utf-8-sig")
    if frame.empty:
        return frame
    return _position_history_to_report_frame(frame)


def _load_trade_report_frame(product, account_id):
    rows = []
    try:
        live_account = account_store.load_account(product, account_id=account_id)
        rows.extend(
            account_report._all_trade_rows_from_exports(
                product,
                not_before=live_account.reset_at,
            )
        )
        rows.extend(
            account_report._all_etf_trade_rows_from_exports(
                product,
                not_before=live_account.reset_at,
            )
        )
    except Exception:
        pass
    if not rows:
        report = _latest_portfolio_report_frames(product)
        if report is not None:
            trades = report.get("交易记录")
            if trades is not None and not trades.empty:
                return _filter_trade_rows_for_product(trades, product)
    return pd.DataFrame(rows, columns=account_report.TRADE_COLUMNS)


def _latest_portfolio_report_frames(product):
    try:
        from . import portfolio_report
    except Exception:
        return None
    out_dir = storage.portfolio_output_dir()
    path = portfolio_report._latest_account_report_path(out_dir)
    if path is None or not Path(path).exists():
        return None
    try:
        return account_report._read_report_workbook(path)
    except Exception:
        return None


def _filter_portfolio_product_rows(frame, product):
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "策略名称" not in frame.columns:
        return frame.copy()
    try:
        from . import portfolio_report

        mask = frame["策略名称"].map(
            lambda value: portfolio_report._canonical_strategy_name(value)
            == portfolio_report._strategy_display_name(product)
        )
    except Exception:
        mask = frame["策略名称"].astype(str).eq(str(product))
    return frame.loc[mask].copy()


def _filter_trade_rows_for_product(frame, product):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=account_report.TRADE_COLUMNS)
    result = frame.copy()
    marker = account_report.PRODUCT_CONTRACT_NAME_MARKERS.get(product)
    if marker is not None and "合约名称" in result.columns:
        result = result[
            result["合约名称"].astype(str).str.contains(marker, na=False)
            | result["类型"].astype(str).eq("ETF对冲")
        ]
    return result.reindex(columns=account_report.TRADE_COLUMNS)


def _position_history_to_report_frame(frame):
    result = pd.DataFrame()
    result["日期"] = frame.get("日期")
    result["合约代码"] = frame.get("合约代码")
    result["合约名称"] = frame.get("合约名称")
    result["交易方向"] = frame.get("方向").map(
        lambda value: "空" if str(value) == "short" else "多"
    )
    result["总持仓张数"] = frame.get("总持仓")
    result["AUM"] = frame.get("AUM")
    result["今日变化"] = None
    result["最新价"] = frame.get("最新价")
    result["持仓均价"] = frame.get("持仓均价")
    result["持仓盈亏"] = frame.get("持仓盈亏")
    result["交易盈亏"] = None
    result["到期日"] = frame.get("到期日")
    result["IV"] = frame.get("IV")
    result["当日盯市交易盈亏"] = None
    result["当日盈亏分解合计"] = None
    return result


def _rows_for_date(frame, date):
    if frame is None or frame.empty or "日期" not in frame.columns:
        return pd.DataFrame()
    dates = pd.to_datetime(frame["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame.loc[dates.eq(str(pd.Timestamp(date).date()))].copy()


def _position_decomposition_series(frame):
    if frame is None or frame.empty:
        return None
    if "当日盈亏分解合计" in frame.columns:
        values = pd.to_numeric(frame["当日盈亏分解合计"], errors="coerce")
        if values.notna().any():
            return values.fillna(0.0)
    if "持仓盈亏" not in frame.columns:
        return None
    if "当日盯市交易盈亏" not in frame.columns and "今日变化" in frame.columns:
        changes = pd.to_numeric(frame["今日变化"], errors="coerce").fillna(0.0)
        if changes.abs().gt(1e-9).any():
            return None
    holding = pd.to_numeric(frame["持仓盈亏"], errors="coerce")
    if not holding.notna().any():
        return None
    mark_trade = (
        pd.to_numeric(frame["当日盯市交易盈亏"], errors="coerce").fillna(0.0)
        if "当日盯市交易盈亏" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    return holding.fillna(0.0) + mark_trade


def _skip_new_position_pnl_decomposition(frame):
    if frame is None or frame.empty or "今日变化" not in frame.columns:
        return False
    changes = pd.to_numeric(frame["今日变化"], errors="coerce").fillna(0.0)
    if not changes.abs().gt(1e-9).any():
        return False
    if "当日盯市交易盈亏" not in frame.columns:
        return True
    values = pd.to_numeric(frame["当日盯市交易盈亏"], errors="coerce")
    return not values.notna().any()


def _position_option_mask(frame):
    if "到期日" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["到期日"].notna() & frame["到期日"].astype(str).str.strip().ne("")


def _expected_account_quantities(product, account_id):
    state = account_store.load_account(product, account_id=account_id)
    expected = {}
    for position in state.positions.values():
        if not position:
            continue
        for code_key, qty_key in [("call_code", "call_qty"), ("put_code", "put_qty")]:
            code = _position_code_key(position.get(code_key))
            qty = _number(position.get(qty_key)) or 0.0
            if code:
                expected[code] = expected.get(code, 0.0) + abs(qty)
    for hedge in state.option_hedges:
        code = _position_code_key(hedge.get("order_book_id"))
        qty = _number(hedge.get("qty")) or 0.0
        if code:
            expected[code] = expected.get(code, 0.0) + abs(qty)
    hedge_code = _position_code_key(state.hedge.underlying_order_book_id)
    if hedge_code and abs(state.hedge.qty) > 1e-9:
        expected[hedge_code] = expected.get(hedge_code, 0.0) + abs(state.hedge.qty)
    return expected


def _actual_position_quantities(frame):
    actual = {}
    if frame is None or frame.empty:
        return actual
    for _, row in frame.iterrows():
        code = _position_code_key(row.get("合约代码"))
        qty = _number(row.get("总持仓张数"))
        if not code or qty is None:
            continue
        actual[code] = actual.get(code, 0.0) + abs(qty)
    return actual


def _summary_total_daily_pnl(row):
    return _first_number(row, "总单日盈亏(手续费前)", "总单日盈亏")


def _summary_net_daily_pnl(row):
    explicit = _number(row.get("净单日盈亏"))
    if explicit is not None:
        return explicit
    total = _summary_total_daily_pnl(row)
    if total is None:
        return None
    return total - (_number(row.get("当日手续费")) or 0.0)


def _summary_hedge_daily_pnl(row):
    return _number(row.get("ETF单日盈亏"))


def _option_greeks_pnl(prev, current):
    explicit = _number(current.get("期权单日GreeksPnL"))
    if explicit is not None:
        return explicit

    spot_change = _value_change(prev, current, "标的价格")
    if spot_change is None:
        return None
    call_delta = _number(prev.get("Call Delta")) or 0.0
    put_delta = _number(prev.get("Put Delta")) or 0.0
    return (
        (call_delta + put_delta) * spot_change
        + (_number(current.get("期权单日GammaPnL")) or 0.0)
        + (_number(current.get("期权单日VegaPnL")) or 0.0)
        + (_number(current.get("期权单日ThetaPnL")) or 0.0)
    )


def _hedge_greeks_pnl(product, prev, current):
    explicit = _number(current.get("对冲单日GreeksPnL"))
    if explicit is not None:
        return explicit
    start_price = _first_number(prev, "对冲最新价", "标的价格")
    end_price = _first_number(current, "对冲最新价", "标的价格")
    if start_price is None or end_price is None:
        return None
    previous_qty = _number(prev.get("对冲持仓")) or 0.0
    try:
        trade_rows = account_report._security_trade_rows_by_date(product).get(
            str(current.get("日期")),
            [],
        )
        if trade_rows:
            return account_report._segmented_hedge_delta_pnl(
                previous_qty,
                start_price,
                end_price,
                trade_rows,
            )
    except Exception:
        pass
    return previous_qty * (end_price - start_price)


def _first_number(row, *columns):
    for column in columns:
        value = _number(row.get(column))
        if value is not None:
            return value
    return None


def _daily_fee_compensation(prev, current):
    daily_fee = _number(current.get("当日手续费"))
    if daily_fee is not None:
        return daily_fee
    return 0.0


def _sum_optional(*values, fallback=None):
    valid = [value for value in values if value is not None]
    if valid:
        return sum(valid)
    return fallback


def _sum_numeric_column(frame, column):
    if frame is None or frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.notna().any():
        return None
    return float(values.fillna(0.0).sum())


def _sum_row_values(rows, column):
    return sum((_number(row.get(column)) or 0.0) for row in rows)


def _contract_multiplier(product):
    try:
        return float(account_report._contract_multiplier(product))
    except Exception:
        return 1.0


def _value_change(prev, current, column):
    if prev is None:
        return None
    prev_value = _number(prev.get(column))
    current_value = _number(current.get(column))
    if prev_value is None or current_value is None:
        return None
    return current_value - prev_value


def _difference(left, right):
    if left is None or right is None:
        return None
    return left - right


def _zero_if_none(value):
    return 0.0 if value is None else float(value)


def _date_or_none(value):
    if value is None or value == "":
        return None
    return pd.Timestamp(value).date()


def _number(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "待设置", "全部", "nan", "NaN", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return None
    if abs(float(denominator)) < 1e-12:
        return 0.0 if abs(float(numerator)) < 1e-12 else None
    return float(numerator) / float(denominator)


def _position_code_key(value):
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _fmt(value):
    if value is None or pd.isna(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _record_reconciliation(product, payload, account_id):
    db_path = storage.account_db_path(product)
    with account_store.connect(db_path) as conn:
        conn.execute(
            """
            insert into reconciliations(account_id, payload, created_at)
            values (?, ?, ?)
            """,
            (
                account_id,
                json.dumps(payload, ensure_ascii=False, default=str),
                storage.utc_now_text(),
            ),
        )
        conn.commit()
