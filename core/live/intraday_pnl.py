from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import account_report
from . import market_data
from . import storage


def calculate_intraday_pnl(product, account_id="default", date=None):
    """Calculate read-only close-to-current PnL from local quote snapshots.

    The position snapshot is the latest account report position history before
    the quote snapshot date. No account state, report history, or output files
    are modified.
    """
    snapshot = market_data.load_latest_quote_snapshot(product, date=date or "latest")
    quote_date = str(snapshot["quote_date"])
    option_snapshot = pd.read_parquet(snapshot["option_snapshot"])
    etf_snapshot = pd.read_parquet(snapshot["etf_snapshot"])
    current_spot = _snapshot_spot(etf_snapshot)
    if current_spot is None:
        raise ValueError(f"ETF snapshot has no usable close price: {snapshot['etf_snapshot']}")

    positions = _load_position_history(product, account_id)
    summaries = _load_summary_history(product, account_id)
    previous_date = _previous_position_date(positions, quote_date, account_id)
    previous_summary = _summary_row(summaries, previous_date, account_id)
    previous_spot = account_report._number(previous_summary.get("标的价格"))
    if previous_spot is None:
        raise ValueError(f"Previous summary row has no 标的价格 for {previous_date}")
    previous_positions = positions.loc[
        positions["日期"].astype(str).eq(previous_date)
        & positions["账户ID"].astype(str).eq(str(account_id))
    ].copy()

    option_rows = []
    for _, row in previous_positions.iterrows():
        if str(row.get("方向") or "").lower() == "hedge":
            continue
        detail = _option_intraday_row(
            product,
            row,
            option_snapshot,
            previous_spot,
            current_spot,
            quote_date,
        )
        if detail is not None:
            option_rows.append(detail)

    hedge_row = _hedge_intraday_row(previous_positions, current_spot)
    summary = _summary(option_rows, hedge_row)
    summary.update(
        {
            "product": product,
            "account_id": account_id,
            "previous_date": previous_date,
            "quote_date": quote_date,
            "snapshot_stamp": snapshot.get("snapshot_stamp"),
            "previous_spot": previous_spot,
            "current_spot": current_spot,
            "metadata_path": snapshot.get("metadata_path"),
        }
    )
    return {
        "summary": summary,
        "option_rows": option_rows,
        "hedge_row": hedge_row,
        "quote_snapshot": snapshot,
    }


def format_intraday_pnl(payload):
    summary = payload["summary"]
    return [
        (
            f"盘中盈亏 {summary['product']} 账户={summary['account_id']} "
            f"{summary['previous_date']}昨收 -> {summary['quote_date']}当前 "
            f"快照={summary.get('snapshot_stamp')}"
        ),
        f"实际盈亏={_fmt(summary['actual_pnl'])}",
        f"Greeks盈亏={_fmt(summary['greeks_pnl'])}",
    ]


def intraday_pnl_json(payload):
    return json.dumps(_localized_payload(payload), ensure_ascii=False, indent=2, default=str)


SUMMARY_KEY_LABELS = {
    "product": "品种",
    "account_id": "账户ID",
    "previous_date": "昨收日期",
    "quote_date": "快照日期",
    "snapshot_stamp": "快照时间戳",
    "previous_spot": "标的昨收价",
    "current_spot": "标的当前价",
    "metadata_path": "快照元数据路径",
    "option_count": "期权合约数",
    "actual_pnl": "实际盈亏",
    "greeks_pnl": "Greeks盈亏",
    "delta_pnl": "DeltaPnL",
    "gamma_pnl": "GammaPnL",
    "vega_pnl": "VegaPnL",
    "theta_pnl": "ThetaPnL",
    "residual": "解释残差",
}


SNAPSHOT_KEY_LABELS = {
    "source": "来源",
    "snapshot_source": "快照来源",
    "snapshot_stamp": "快照时间戳",
    "quote_date": "快照日期",
    "etf_snapshot": "ETF快照路径",
    "option_snapshot": "期权快照路径",
    "metadata_path": "元数据路径",
}


def _localized_payload(payload):
    return {
        "汇总": _localize_dict(
            _display_summary(payload.get("summary") or {}),
            SUMMARY_KEY_LABELS,
        ),
        "行情快照": _localize_dict(payload.get("quote_snapshot") or {}, SNAPSHOT_KEY_LABELS),
    }


def _display_summary(summary):
    keys = [
        "product",
        "account_id",
        "previous_date",
        "quote_date",
        "snapshot_stamp",
        "actual_pnl",
        "greeks_pnl",
    ]
    return {key: summary[key] for key in keys if key in summary}


def _localize_dict(values, labels):
    result = {}
    for key, value in values.items():
        result[labels.get(key, key)] = value
    return result


def _load_position_history(product, account_id):
    path = storage.account_report_position_history_path(product, account_id)
    if not Path(path).exists():
        raise FileNotFoundError(f"Position history not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"Position history is empty: {path}")
    required = {"日期", "账户ID", "方向", "合约代码", "总持仓", "最新价"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Position history missing columns: {sorted(missing)}")
    return frame


def _load_summary_history(product, account_id):
    path = storage.account_report_summary_history_path(product, account_id)
    if not Path(path).exists():
        raise FileNotFoundError(f"Summary history not found: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError(f"Summary history is empty: {path}")
    required = {"日期", "账户ID", "标的价格"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Summary history missing columns: {sorted(missing)}")
    return frame


def _summary_row(summaries, report_date, account_id):
    rows = summaries.loc[
        summaries["日期"].astype(str).eq(str(report_date))
        & summaries["账户ID"].astype(str).eq(str(account_id))
    ]
    if rows.empty:
        raise ValueError(f"No summary history for {account_id} on {report_date}")
    return rows.iloc[-1]


def _previous_position_date(positions, quote_date, account_id):
    quote_ts = pd.Timestamp(quote_date).normalize()
    dates = pd.to_datetime(
        positions.loc[positions["账户ID"].astype(str).eq(str(account_id)), "日期"],
        errors="coerce",
    ).dropna()
    dates = dates.loc[dates.dt.normalize() < quote_ts]
    if dates.empty:
        raise ValueError(f"No previous position history before {quote_date}")
    return dates.max().strftime("%Y-%m-%d")


def _snapshot_spot(etf_snapshot):
    if etf_snapshot is None or etf_snapshot.empty:
        return None
    for column in ["close", "last", "latest", "price"]:
        if column in etf_snapshot.columns:
            value = account_report._number(etf_snapshot.iloc[-1].get(column))
            if value is not None and value > 0:
                return float(value)
    return None


def _option_intraday_row(
    product,
    position_row,
    option_snapshot,
    previous_spot,
    current_spot,
    quote_date,
):
    code = account_report._security_code(position_row.get("合约代码"))
    if code is None:
        return None
    current_price = _snapshot_option_price(option_snapshot, code)
    previous_price = account_report._number(position_row.get("最新价"))
    if current_price is None or previous_price is None:
        return None

    qty = abs(account_report._number(position_row.get("总持仓")) or 0.0)
    direction = -1.0 if str(position_row.get("方向") or "").lower() == "short" else 1.0
    signed_qty = direction * qty
    multiplier = _position_multiplier(product, position_row)

    # Previous greeks already include multiplier and quantity.
    previous_delta = account_report._number(position_row.get("Delta"))
    previous_gamma = account_report._number(position_row.get("Gamma"))
    previous_vega = account_report._number(position_row.get("Vega"))
    previous_theta = account_report._number(position_row.get("Theta"))
    previous_iv = account_report._number(position_row.get("IV"))
    if any(
        value is None
        for value in [previous_delta, previous_gamma, previous_vega, previous_theta, previous_iv]
    ):
        return None

    current_greeks = _current_greeks(product, position_row, current_price, current_spot, signed_qty)
    if current_greeks is None:
        return None
    current_iv = current_greeks["iv"]

    spot_change = float(current_spot) - float(previous_spot)
    step = _trading_day_step(str(position_row.get("日期")), quote_date, product)

    actual_pnl = signed_qty * (float(current_price) - float(previous_price)) * multiplier
    delta_pnl = previous_delta * spot_change
    gamma_pnl = 0.5 * previous_gamma * spot_change * spot_change
    vega_pnl = previous_vega * (current_iv - previous_iv) * 100.0
    theta_pnl = previous_theta * step
    greeks_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    return {
        "code": code,
        "name": position_row.get("合约名称"),
        "direction": position_row.get("方向"),
        "signed_qty": float(signed_qty),
        "previous_price": float(previous_price),
        "current_price": float(current_price),
        "previous_iv": float(previous_iv),
        "current_iv": float(current_iv),
        "previous_spot": float(previous_spot),
        "current_spot": float(current_spot),
        "actual_pnl": float(actual_pnl),
        "delta_pnl": float(delta_pnl),
        "gamma_pnl": float(gamma_pnl),
        "vega_pnl": float(vega_pnl),
        "theta_pnl": float(theta_pnl),
        "greeks_pnl": float(greeks_pnl),
        "residual": float(actual_pnl - greeks_pnl),
    }


def _current_greeks(product, position_row, current_price, current_spot, signed_qty):
    leg = account_report._position_row_leg(position_row)
    flag = "c" if leg == "Call" else "p" if leg == "Put" else None
    if flag is None:
        return None
    return account_report._single_node_option_greeks(
        product,
        {"previous": position_row},
        position_row,
        current_price,
        current_spot,
        flag,
        signed_qty,
        1,
        2,
    )


def _snapshot_option_price(option_snapshot, code):
    if option_snapshot is None or option_snapshot.empty or "order_book_id" not in option_snapshot.columns:
        return None
    rows = option_snapshot.loc[option_snapshot["order_book_id"].apply(account_report._security_code).eq(code)]
    if rows.empty:
        return None
    row = rows.iloc[0]
    for column in ["mid", "close", "last", "latest", "price"]:
        value = account_report._number(row.get(column))
        if value is not None and value > 0:
            return float(value)
    bid = account_report._number(row.get("bid"))
    ask = account_report._number(row.get("ask"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return float((bid + ask) / 2.0)
    return bid if bid is not None and bid > 0 else ask


def _hedge_intraday_row(previous_positions, current_spot):
    hedge_rows = previous_positions.loc[
        previous_positions["方向"].astype(str).str.lower().eq("hedge")
    ]
    if hedge_rows.empty:
        return None
    row = hedge_rows.iloc[-1]
    qty = account_report._number(row.get("总持仓")) or 0.0
    previous_price = account_report._number(row.get("最新价"))
    if previous_price is None:
        return None
    pnl = qty * (float(current_spot) - float(previous_price))
    return {
        "code": account_report._security_code(row.get("合约代码")),
        "qty": float(qty),
        "previous_price": float(previous_price),
        "current_price": float(current_spot),
        "actual_pnl": float(pnl),
        "delta_pnl": float(pnl),
        "greeks_pnl": float(pnl),
    }


def _summary(option_rows, hedge_row):
    option_actual = sum(row["actual_pnl"] for row in option_rows)
    option_greeks = sum(row["greeks_pnl"] for row in option_rows)
    hedge_actual = hedge_row["actual_pnl"] if hedge_row else 0.0
    hedge_greeks = hedge_row["greeks_pnl"] if hedge_row else 0.0
    delta = sum(row["delta_pnl"] for row in option_rows) + hedge_greeks
    gamma = sum(row["gamma_pnl"] for row in option_rows)
    vega = sum(row["vega_pnl"] for row in option_rows)
    theta = sum(row["theta_pnl"] for row in option_rows)
    actual = option_actual + hedge_actual
    greeks = option_greeks + hedge_greeks
    return {
        "option_count": len(option_rows),
        "option_actual_pnl": float(option_actual),
        "option_greeks_pnl": float(option_greeks),
        "hedge_actual_pnl": float(hedge_actual),
        "hedge_greeks_pnl": float(hedge_greeks),
        "actual_pnl": float(actual),
        "greeks_pnl": float(greeks),
        "delta_pnl": float(delta),
        "gamma_pnl": float(gamma),
        "vega_pnl": float(vega),
        "theta_pnl": float(theta),
        "residual": float(actual - greeks),
    }


def _position_multiplier(product, position_row):
    value = account_report._number(position_row.get("合约乘数"))
    if value is not None and value > 0:
        return float(value)
    return float(account_report._contract_multiplier(product))


def _trading_day_step(previous_date, current_date, product):
    dates = pd.Series([previous_date, current_date])
    steps = account_report._trading_day_steps(dates, product=product)
    value = account_report._number(steps.iloc[-1]) if len(steps) else None
    return float(value if value is not None else 1.0)


def _fmt(value):
    if value is None:
        return "nan"
    return f"{float(value):.6f}"
