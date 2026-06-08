from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import account as account_store
from . import storage


DEFAULT_ABS_TOLERANCE = 100.0
DEFAULT_REL_TOLERANCE = 0.25


def reconcile(
    product,
    broker_snapshot=None,
    account_id="default",
    start_date=None,
    end_date=None,
    abs_tolerance=DEFAULT_ABS_TOLERANCE,
    rel_tolerance=DEFAULT_REL_TOLERANCE,
):
    """Validate how well daily Greeks PnL explains actual daily trading PnL.

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

    rows = _build_greeks_explainability_rows(
        product,
        account_history,
        start_date=start_date,
        end_date=end_date,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    metrics = _aggregate_metrics(rows)
    payload = {
        "product": product,
        "account_id": account_id,
        "ok": bool(rows) and all(row["ok"] for row in rows),
        "mode": "greeks_daily_pnl_explainability",
        "start_date": start_date,
        "end_date": end_date,
        "abs_tolerance": abs_tolerance,
        "rel_tolerance": rel_tolerance,
        "metrics": metrics,
        "rows": rows,
        "summary_history_path": str(summary_path),
    }
    _record_reconciliation(product, payload, account_id)
    return payload


def write_reconcile_report(product, payload):
    stamp = storage.local_now_stamp()
    path = storage.output_dir(product) / f"{stamp}_reconcile.md"
    metrics = payload.get("metrics", {})
    lines = [
        f"# Greeks PnL Reconciliation: {product}",
        "",
        f"- account_id: {payload['account_id']}",
        f"- mode: {payload.get('mode')}",
        f"- ok: {payload['ok']}",
        f"- tolerance: abs<={_fmt(payload.get('abs_tolerance'))}, "
        f"rel<={_fmt(payload.get('rel_tolerance'))}",
        f"- rows: {metrics.get('row_count', 0)}",
        f"- total_daily_pnl: {_fmt(metrics.get('total_daily_pnl'))}",
        f"- total_greeks_pnl: {_fmt(metrics.get('total_greeks_pnl'))}",
        f"- total_residual: {_fmt(metrics.get('total_residual'))}",
        f"- option_daily_pnl: {_fmt(metrics.get('total_option_daily_pnl'))}",
        f"- option_greeks_pnl: {_fmt(metrics.get('total_option_greeks_pnl'))}",
        f"- option_residual: {_fmt(metrics.get('total_option_residual'))}",
        f"- hedge_daily_pnl: {_fmt(metrics.get('total_hedge_daily_pnl'))}",
        f"- hedge_greeks_pnl: {_fmt(metrics.get('total_hedge_greeks_pnl'))}",
        f"- hedge_residual: {_fmt(metrics.get('total_hedge_residual'))}",
        f"- mean_abs_residual: {_fmt(metrics.get('mean_abs_residual'))}",
        f"- rmse_residual: {_fmt(metrics.get('rmse_residual'))}",
        f"- explained_ratio: {_fmt(metrics.get('explained_ratio'))}",
        "",
        "## Daily Checks",
        "",
        (
            "| date | option_daily_pnl | hedge_daily_pnl | total_daily_pnl "
            "| greeks_pnl | residual | residual_ratio | fee_compensation | ok |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {date} | {option_daily_pnl} | {hedge_daily_pnl} | "
            "{total_daily_pnl} | {greeks_pnl} | {residual} | "
            "{residual_ratio} | {fee_compensation} | {ok} |".format(
                date=row["date"],
                option_daily_pnl=_fmt(row["option_daily_pnl"]),
                hedge_daily_pnl=_fmt(row["hedge_daily_pnl"]),
                total_daily_pnl=_fmt(row["total_daily_pnl"]),
                fee_compensation=_fmt(row["fee_compensation"]),
                greeks_pnl=_fmt(row["greeks_pnl"]),
                residual=_fmt(row["residual"]),
                residual_ratio=_fmt(row["residual_ratio"]),
                ok="Y" if row["ok"] else "N",
            )
        )
    lines.extend(
        [
            "",
            "## Option Leg Checks",
            "",
            "| date | option_daily_pnl | option_greeks_pnl | option_residual | option_residual_ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("rows", []):
        lines.append(
            "| {date} | {actual} | {greeks} | {residual} | {ratio} |".format(
                date=row["date"],
                actual=_fmt(row.get("option_daily_pnl")),
                greeks=_fmt(row.get("option_greeks_pnl")),
                residual=_fmt(row.get("option_residual")),
                ratio=_fmt(row.get("option_residual_ratio")),
            )
        )
    lines.extend(
        [
            "",
            "## Hedge Leg Checks",
            "",
            "| date | hedge_daily_pnl | hedge_greeks_pnl | hedge_residual | hedge_residual_ratio | previous_hedge_qty | spot_change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload.get("rows", []):
        lines.append(
            "| {date} | {actual} | {greeks} | {residual} | {ratio} | {qty} | {spot_change} |".format(
                date=row["date"],
                actual=_fmt(row.get("hedge_daily_pnl")),
                greeks=_fmt(row.get("hedge_greeks_pnl")),
                residual=_fmt(row.get("hedge_residual")),
                ratio=_fmt(row.get("hedge_residual_ratio")),
                qty=_fmt(row.get("previous_hedge_qty")),
                spot_change=_fmt(row.get("spot_change")),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_terminal_summary(payload):
    metrics = payload.get("metrics", {})
    lines = [
        (
            f"Greeks解释能力对账={payload['product']}/{payload['account_id']} "
            f"ok={payload['ok']} rows={metrics.get('row_count', 0)}"
        ),
        (
            "合计: "
            f"总单日盈亏={_fmt(metrics.get('total_daily_pnl'))} "
            f"GreeksPnL={_fmt(metrics.get('total_greeks_pnl'))} "
            f"残差={_fmt(metrics.get('total_residual'))} "
            f"解释比例={_fmt(metrics.get('explained_ratio'))}"
        ),
        (
            f"平均绝对残差={_fmt(metrics.get('mean_abs_residual'))} "
            f"RMSE={_fmt(metrics.get('rmse_residual'))}"
        ),
        (
            "Option腿合计: "
            f"单日盈亏={_fmt(metrics.get('total_option_daily_pnl'))} "
            f"Greeks={_fmt(metrics.get('total_option_greeks_pnl'))} "
            f"残差={_fmt(metrics.get('total_option_residual'))} "
            f"解释比例={_fmt(metrics.get('total_option_explained_ratio'))}"
        ),
        (
            "Hedge腿合计: "
            f"单日盈亏={_fmt(metrics.get('total_hedge_daily_pnl'))} "
            f"Greeks={_fmt(metrics.get('total_hedge_greeks_pnl'))} "
            f"残差={_fmt(metrics.get('total_hedge_residual'))} "
            f"解释比例={_fmt(metrics.get('total_hedge_explained_ratio'))}"
        ),
        "",
        "逐日检查",
    ]
    if not payload.get("rows"):
        lines.append("(none)")
        return lines
    for row in payload["rows"]:
        marker = "OK" if row["ok"] else "FAIL"
        lines.append(
            f"{row['date']} {marker} "
            f"期权单日盈亏={_fmt(row['option_daily_pnl'])} "
            f"对冲单日盈亏={_fmt(row['hedge_daily_pnl'])} "
            f"总单日盈亏={_fmt(row['total_daily_pnl'])} "
            f"GreeksPnL={_fmt(row['greeks_pnl'])} "
            f"残差={_fmt(row['residual'])} "
            f"残差比例={_fmt(row['residual_ratio'])} "
            f"手续费补偿={_fmt(row['fee_compensation'])}"
        )
        lines.append(
            f"  Option腿: 单日盈亏={_fmt(row.get('option_daily_pnl'))} "
            f"Greeks={_fmt(row.get('option_greeks_pnl'))} "
            f"残差={_fmt(row.get('option_residual'))} "
            f"残差比例={_fmt(row.get('option_residual_ratio'))}"
        )
        lines.append(
            f"  Hedge腿: 单日盈亏={_fmt(row.get('hedge_daily_pnl'))} "
            f"Greeks={_fmt(row.get('hedge_greeks_pnl'))} "
            f"残差={_fmt(row.get('hedge_residual'))} "
            f"残差比例={_fmt(row.get('hedge_residual_ratio'))} "
            f"前日hedge={_fmt(row.get('previous_hedge_qty'))}"
        )
    return lines


def _build_greeks_explainability_rows(
    product,
    history,
    start_date=None,
    end_date=None,
    abs_tolerance=DEFAULT_ABS_TOLERANCE,
    rel_tolerance=DEFAULT_REL_TOLERANCE,
):
    required = ["日期", "单日GreeksPnL"]
    missing = [column for column in required if column not in history.columns]
    if missing:
        raise ValueError(f"Summary history missing columns: {missing}")

    frame = history.copy()
    frame["_date"] = pd.to_datetime(frame["日期"], errors="coerce")
    frame = frame.dropna(subset=["_date"]).sort_values("_date")
    rows = []
    start = _date_or_none(start_date)
    end = _date_or_none(end_date)

    for i in range(1, len(frame)):
        prev = frame.iloc[i - 1]
        current = frame.iloc[i]
        current_date = current["_date"].date()
        if start is not None and current_date < start:
            continue
        if end is not None and current_date > end:
            continue

        greeks_pnl = _number(current.get("单日GreeksPnL"))
        if greeks_pnl is None:
            continue

        fee_compensation = _daily_fee_compensation(prev, current)
        nav_change = _value_change(prev, current, "估算权益")
        fee_adjusted_nav_change = (
            nav_change + fee_compensation if nav_change is not None else None
        )
        option_daily_pnl = _daily_pnl(current, "期权单日盈亏")
        if option_daily_pnl is None:
            option_daily_pnl = _value_change(prev, current, "期权浮盈亏")
        hedge_daily_pnl = _daily_pnl(current, "对冲单日盈亏")
        if hedge_daily_pnl is None:
            hedge_daily_pnl = _hedge_actual_change(prev, current)
        total_daily_pnl = _daily_pnl(current, "总单日盈亏")
        if total_daily_pnl is None:
            total_daily_pnl = _sum_optional(option_daily_pnl, hedge_daily_pnl)
        if total_daily_pnl is None:
            total_daily_pnl = fee_adjusted_nav_change
        if total_daily_pnl is None:
            continue
        spot_change = _value_change(prev, current, "标的价格")
        option_greeks_pnl = _option_greeks_pnl(prev, current)
        hedge_greeks_pnl = _hedge_greeks_pnl(product, prev, current, spot_change)
        option_residual = _difference(option_daily_pnl, option_greeks_pnl)
        hedge_residual = _difference(hedge_daily_pnl, hedge_greeks_pnl)
        residual = total_daily_pnl - greeks_pnl
        residual_ratio = _safe_ratio(abs(residual), abs(total_daily_pnl))
        tolerance = max(abs_tolerance, abs(total_daily_pnl) * rel_tolerance)

        rows.append(
            {
                "date": str(current_date),
                "previous_date": str(prev["_date"].date()),
                "nav_previous": _number(prev.get("估算权益")),
                "nav_current": _number(current.get("估算权益")),
                "nav_change": nav_change,
                "fee_compensation": fee_compensation,
                "fee_adjusted_nav_change": fee_adjusted_nav_change,
                "option_daily_pnl": option_daily_pnl,
                "hedge_daily_pnl": hedge_daily_pnl,
                "total_daily_pnl": total_daily_pnl,
                "greeks_pnl": greeks_pnl,
                "residual": residual,
                "abs_residual": abs(residual),
                "residual_ratio": residual_ratio,
                "option_actual_change": option_daily_pnl,
                "option_greeks_pnl": option_greeks_pnl,
                "option_residual": option_residual,
                "option_residual_ratio": _safe_ratio(
                    _abs_or_none(option_residual), _abs_or_none(option_daily_pnl)
                ),
                "hedge_actual_change": hedge_daily_pnl,
                "hedge_greeks_pnl": hedge_greeks_pnl,
                "hedge_residual": hedge_residual,
                "hedge_residual_ratio": _safe_ratio(
                    _abs_or_none(hedge_residual), _abs_or_none(hedge_daily_pnl)
                ),
                "spot_change": spot_change,
                "previous_hedge_qty": _number(prev.get("对冲持仓")) or 0.0,
                "tolerance": tolerance,
                "ok": abs(residual) <= tolerance,
            }
        )
    return rows


def _aggregate_metrics(rows):
    if not rows:
        return {
            "row_count": 0,
            "total_daily_pnl": 0.0,
            "total_fee_compensated_nav_change": 0.0,
            "total_greeks_pnl": 0.0,
            "total_residual": 0.0,
            "total_option_daily_pnl": 0.0,
            "total_option_actual_change": 0.0,
            "total_option_greeks_pnl": 0.0,
            "total_option_residual": 0.0,
            "total_option_explained_ratio": None,
            "total_hedge_daily_pnl": 0.0,
            "total_hedge_actual_change": 0.0,
            "total_hedge_greeks_pnl": 0.0,
            "total_hedge_residual": 0.0,
            "total_hedge_explained_ratio": None,
            "mean_abs_residual": None,
            "rmse_residual": None,
            "explained_ratio": None,
        }
    total_actual = sum(row["total_daily_pnl"] for row in rows)
    total_greeks = sum(row["greeks_pnl"] for row in rows)
    total_option_actual = sum(_zero_if_none(row.get("option_daily_pnl")) for row in rows)
    total_option_greeks = sum(_zero_if_none(row.get("option_greeks_pnl")) for row in rows)
    total_hedge_actual = sum(_zero_if_none(row.get("hedge_daily_pnl")) for row in rows)
    total_hedge_greeks = sum(_zero_if_none(row.get("hedge_greeks_pnl")) for row in rows)
    residuals = [row["residual"] for row in rows]
    abs_residuals = [abs(value) for value in residuals]
    return {
        "row_count": len(rows),
        "total_daily_pnl": total_actual,
        "total_fee_compensated_nav_change": total_actual,
        "total_greeks_pnl": total_greeks,
        "total_residual": total_actual - total_greeks,
        "total_option_daily_pnl": total_option_actual,
        "total_option_actual_change": total_option_actual,
        "total_option_greeks_pnl": total_option_greeks,
        "total_option_residual": total_option_actual - total_option_greeks,
        "total_option_explained_ratio": _safe_ratio(total_option_greeks, total_option_actual),
        "total_hedge_daily_pnl": total_hedge_actual,
        "total_hedge_actual_change": total_hedge_actual,
        "total_hedge_greeks_pnl": total_hedge_greeks,
        "total_hedge_residual": total_hedge_actual - total_hedge_greeks,
        "total_hedge_explained_ratio": _safe_ratio(total_hedge_greeks, total_hedge_actual),
        "mean_abs_residual": sum(abs_residuals) / len(abs_residuals),
        "rmse_residual": math.sqrt(
            sum(value * value for value in residuals) / len(residuals)
        ),
        "explained_ratio": _safe_ratio(total_greeks, total_actual),
    }


def _daily_fee_compensation(prev, current):
    daily_fee = _number(current.get("当日手续费"))
    if daily_fee is not None:
        return daily_fee

    prev_fee = _number(prev.get("手续费"))
    current_fee = _number(current.get("手续费"))
    if prev_fee is None or current_fee is None:
        return 0.0
    return current_fee - prev_fee


def _daily_pnl(row, column):
    value = _number(row.get(column))
    return value


def _sum_optional(*values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid)


def _value_change(prev, current, column):
    prev_value = _number(prev.get(column))
    current_value = _number(current.get(column))
    if prev_value is None or current_value is None:
        return None
    return current_value - prev_value


def _option_greeks_pnl(prev, current):
    explicit = _number(current.get("期权单日GreeksPnL"))
    if explicit is not None:
        return explicit

    spot_change = _value_change(prev, current, "标的价格")
    if spot_change is None:
        return None

    call_delta = _number(prev.get("Call Delta")) or 0.0
    put_delta = _number(prev.get("Put Delta")) or 0.0
    option_delta_pnl = (call_delta + put_delta) * spot_change
    gamma_pnl = _number(current.get("单日GammaPnL")) or 0.0
    vega_pnl = _number(current.get("单日VegaPnL")) or 0.0
    theta_pnl = _number(current.get("单日ThetaPnL")) or 0.0
    return option_delta_pnl + gamma_pnl + vega_pnl + theta_pnl


def _hedge_actual_change(prev, current):
    total_change = _value_change(prev, current, "对冲总盈亏")
    if total_change is not None:
        return total_change
    return _value_change(prev, current, "对冲浮盈亏")


def _hedge_greeks_pnl(product, prev, current, spot_change):
    explicit = _number(current.get("对冲单日GreeksPnL"))
    if explicit is not None:
        return explicit

    if spot_change is None:
        return None
    previous_qty = _number(prev.get("对冲持仓")) or 0.0
    start_price = _number(prev.get("对冲最新价"))
    end_price = _number(current.get("对冲最新价"))
    if start_price is None:
        start_price = _number(prev.get("标的价格"))
    if end_price is None:
        end_price = _number(current.get("标的价格"))
    if start_price is None or end_price is None:
        return None
    try:
        from . import account_report

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
    return previous_qty * spot_change


def _difference(left, right):
    if left is None or right is None:
        return None
    return left - right


def _zero_if_none(value):
    return 0.0 if value is None else float(value)


def _abs_or_none(value):
    return None if value is None else abs(value)


def _date_or_none(value):
    if value is None or value == "":
        return None
    return pd.Timestamp(value).date()


def _number(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "待设置", "全部", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


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
