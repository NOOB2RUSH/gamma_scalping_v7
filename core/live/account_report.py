from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

import core
from . import account as account_store
from . import market_data
from . import signal_engine
from . import storage
from .runtime import PROJECT_ROOT, load_product_config


SUMMARY_COLUMNS = [
    "日期",
    "账户ID",
    "标的价格",
    "现金",
    "期权市值",
    "期权保证金",
    "对冲持仓",
    "对冲成本",
    "对冲保证金",
    "对冲浮盈亏",
    "估算权益",
    "期权浮盈亏",
    "手续费",
    "账户Delta",
    "期权Delta",
    "账户Gamma",
    "账户Vega",
    "账户Theta",
    "持仓IV",
    "Call IV",
    "Put IV",
]

POSITION_COLUMNS = [
    "日期",
    "账户ID",
    "方向",
    "合约代码",
    "合约名称",
    "买卖",
    "持仓类型",
    "总持仓",
    "今持仓",
    "今开仓",
    "今平仓",
    "可平量",
    "最新价",
    "持仓均价",
    "开仓均价",
    "期权市值",
    "占用保证金",
    "持仓盈亏",
    "浮动盈亏",
    "行权价",
    "到期日",
    "剩余天数",
    "IV",
    "Delta",
    "Gamma",
    "Vega",
    "Theta",
]

TRADE_COLUMNS = [
    "序号",
    "投资者账号",
    "交易所",
    "合约代码",
    "合约名称",
    "成交编号",
    "报单编号",
    "开平",
    "买卖",
    "报单价格",
    "成交价格",
    "成交数量",
    "手续费",
    "平仓盈亏",
    "类型",
    "日期",
    "报单时间",
    "成交时间",
    "成交时间(日)",
    "策略名称",
]


def build_live_account_report(
    product,
    account_id="default",
    source="akshare",
    date=None,
    all_trades=False,
):
    config = load_product_config(product)
    snapshot = None
    report_date = date
    if source in {"akshare", "local"}:
        snapshot = market_data.fetch_quote_snapshot(
            product,
            source=source,
            date=date or "latest",
        )
        report_date = snapshot["quote_date"]
    elif source != "none":
        raise ValueError("source must be one of: akshare, local, none")

    market = signal_engine._load_market_context(config, report_date)
    live_account = account_store.load_account(product, account_id=account_id)
    report_date_text = str(market["date"].date())
    spot = float(market["signal_row"]["close"])

    position_rows, account_greeks, option_value, option_margin, option_pnl = (
        _position_rows_from_account(
            live_account,
            market["chain_df"],
            report_date_text,
            account_id,
        )
    )
    holding_rows = _holding_rows_from_export(
        product,
        account_id,
        report_date_text,
        market["chain_df"],
    )
    if holding_rows:
        position_rows = holding_rows
        option_value = _sum_row_values(holding_rows, "期权市值")
        option_margin = _sum_row_values(holding_rows, "占用保证金")
        option_pnl = _sum_row_values(holding_rows, "浮动盈亏")
    trade_rows = _trade_rows_from_export(product, report_date_text)
    trade_fee = _sum_row_values(trade_rows, "手续费")

    hedge_unrealized_pnl = core.hedge.calc_unrealized_pnl(
        live_account.hedge.qty,
        live_account.hedge.entry_price,
        spot,
    )
    nav_estimate = (
        live_account.cash
        + option_value
        + option_margin
        + live_account.hedge.margin
        + hedge_unrealized_pnl
    )
    summary_row = {
        "日期": report_date_text,
        "账户ID": account_id,
        "标的价格": spot,
        "现金": live_account.cash,
        "期权市值": option_value,
        "期权保证金": option_margin,
        "对冲持仓": live_account.hedge.qty,
        "对冲成本": live_account.hedge.entry_price,
        "对冲保证金": live_account.hedge.margin,
        "对冲浮盈亏": hedge_unrealized_pnl,
        "估算权益": nav_estimate,
        "期权浮盈亏": option_pnl,
        "手续费": trade_fee,
        "账户Delta": account_greeks["delta"] + live_account.hedge.qty,
        "期权Delta": account_greeks["delta"],
        "账户Gamma": account_greeks["gamma"],
        "账户Vega": account_greeks["vega"],
        "账户Theta": account_greeks["theta"],
        "持仓IV": account_greeks["position_iv"],
        "Call IV": account_greeks["call_iv"],
        "Put IV": account_greeks["put_iv"],
    }

    summary_history = _update_history_csv(
        storage.account_report_summary_history_path(product, account_id),
        [summary_row],
        SUMMARY_COLUMNS,
        key_columns=["日期", "账户ID"],
    )
    position_history = _update_history_csv(
        storage.account_report_position_history_path(product, account_id),
        position_rows,
        POSITION_COLUMNS,
        key_columns=["日期", "账户ID"],
    )
    return {
        "product": product,
        "account_id": account_id,
        "date": report_date_text,
        "spot": spot,
        "source": source,
        "quote_snapshot": snapshot,
        "summary": summary_row,
        "summary_history": summary_history,
        "position_history": position_history,
        "trade_rows": trade_rows,
        "strategy_state": live_account.strategy_state.to_dict(),
    }


def write_live_account_report(product, payload, output_format="excel"):
    stamp = storage.local_now_stamp()
    out_dir = storage.output_dir(product)
    frames = _report_frames(payload)
    paths = {}

    if output_format not in {"excel", "csv", "both"}:
        raise ValueError("output_format must be one of: excel, csv, both")

    if output_format in {"excel", "both"}:
        excel_path = out_dir / f"{stamp}_account_report.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, frame in frames.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)
        paths["excel"] = excel_path

    if output_format in {"csv", "both"}:
        csv_paths = {}
        csv_names = {
            "账户总体情况": "summary",
            "持仓记录": "positions",
            "当日交易记录": "trades",
        }
        for sheet_name, frame in frames.items():
            csv_path = out_dir / f"{stamp}_account_report_{csv_names[sheet_name]}.csv"
            frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
            csv_paths[sheet_name] = csv_path
        paths["csv"] = csv_paths

    json_path = out_dir / f"{stamp}_account_report.json"
    storage.write_json(json_path, _json_payload(payload))
    paths["json"] = json_path
    return paths


def _json_payload(payload):
    result = dict(payload)
    for key in ["summary_history", "position_history"]:
        value = result.get(key)
        if isinstance(value, pd.DataFrame):
            result[key] = value.to_dict("records")
    return result


def format_terminal_summary(payload):
    summary = payload["summary"]
    lines = [
        (
            f"账户报告={payload['product']}/{payload['account_id']} "
            f"日期={payload['date']} 标的价格={_fmt(payload['spot'])}"
        ),
        (
            f"现金={_fmt(summary['现金'])} 估算权益={_fmt(summary['估算权益'])} "
            f"期权浮盈亏={_fmt(summary['期权浮盈亏'])}"
        ),
        (
            f"账户Delta={_fmt(summary['账户Delta'])} "
            f"Gamma={_fmt(summary['账户Gamma'])} "
            f"Vega={_fmt(summary['账户Vega'])} "
            f"Theta={_fmt(summary['账户Theta'])} "
            f"持仓IV={_fmt(summary['持仓IV'])}"
        ),
        "",
        "持仓记录",
    ]
    lines.extend(
        _plain_table(
            payload["position_history"][
                payload["position_history"]["日期"].astype(str) == str(payload["date"])
            ].to_dict("records"),
            ["方向", "合约代码", "合约名称", "总持仓", "最新价", "开仓均价", "IV", "Delta"],
        )
    )
    lines.extend(["", "当日交易记录"])
    lines.extend(
        _plain_table(
            payload["trade_rows"],
            ["成交编号", "合约代码", "合约名称", "开平", "买卖", "成交价格", "成交数量", "成交时间"],
        )
    )
    return lines


def _report_frames(payload):
    return {
        "账户总体情况": payload["summary_history"],
        "持仓记录": payload["position_history"],
        "当日交易记录": _frame(payload["trade_rows"], TRADE_COLUMNS),
    }


def _position_rows_from_account(live_account, chain_df, report_date, account_id):
    rows = []
    greeks_list = []
    signed_values = []
    margins = []
    option_pnl = 0.0
    for side, position in live_account.positions.items():
        if position is None:
            continue
        try:
            call_row, put_row = core.vol_engine.resolve_position_pair(position, chain_df)
        except IndexError:
            continue

        current_value = core.position.value(position, call_row, put_row)
        signed_value = core.position.signed_value(position, call_row, put_row)
        entry_value = float(position.get("entry_option_value", 0.0) or 0.0)
        pnl = entry_value - current_value if side == "short" else current_value - entry_value
        option_pnl += pnl
        greeks = core.strategy.calc_position_greeks(
            call_row,
            put_row,
            position["call_qty"],
            position["put_qty"],
            side=side,
        )
        greeks_list.append(greeks)
        signed_values.append(signed_value)
        margins.append(core.position.margin_value(position))
        rows.extend(
            [
                _account_leg_row(
                    report_date,
                    account_id,
                    side,
                    position,
                    call_row,
                    "call",
                    greeks,
                ),
                _account_leg_row(
                    report_date,
                    account_id,
                    side,
                    position,
                    put_row,
                    "put",
                    greeks,
                ),
            ]
        )
    return (
        rows,
        core.backtester.combine_greeks(greeks_list),
        sum(signed_values),
        sum(margins),
        option_pnl,
    )


def _account_leg_row(report_date, account_id, side, position, row, leg, greeks):
    qty_key = "call_qty" if leg == "call" else "put_qty"
    price_key = "entry_call_price" if leg == "call" else "entry_put_price"
    iv_key = "call_iv" if leg == "call" else "put_iv"
    delta_key = "call_delta" if leg == "call" else "put_delta"
    gamma_key = "call_gamma" if leg == "call" else "put_gamma"
    vega_key = "call_vega" if leg == "call" else "put_vega"
    theta_key = "call_theta" if leg == "call" else "put_theta"
    return {
        "日期": report_date,
        "账户ID": account_id,
        "方向": side,
        "合约代码": row.get("order_book_id"),
        "合约名称": row.get("contract_symbol"),
        "买卖": "卖" if side == "short" else "买",
        "持仓类型": "义务仓" if side == "short" else "权利仓",
        "总持仓": position.get(qty_key),
        "今持仓": None,
        "今开仓": None,
        "今平仓": None,
        "可平量": None,
        "最新价": row.get("mid"),
        "持仓均价": position.get(price_key),
        "开仓均价": position.get(price_key),
        "期权市值": row.get("mid") * position.get(qty_key) * position.get("contract_multiplier"),
        "占用保证金": position.get("option_margin") if leg == "call" else None,
        "持仓盈亏": None,
        "浮动盈亏": None,
        "行权价": row.get("strike_price"),
        "到期日": str(pd.Timestamp(row.get("maturity_date")).date()),
        "剩余天数": row.get("dte"),
        "IV": greeks.get(iv_key),
        "Delta": greeks.get(delta_key),
        "Gamma": greeks.get(gamma_key),
        "Vega": greeks.get(vega_key),
        "Theta": greeks.get(theta_key),
    }


def _holding_rows_from_export(product, account_id, report_date, chain_df):
    path = _latest_export_file("实时持仓", report_date)
    if path is None:
        return []
    df = _read_export_csv(path)
    if df.empty:
        return []
    chain_meta = _chain_metadata(chain_df)
    rows = []
    for _, item in df.iterrows():
        code = str(item.get("合约代码", "")).strip()
        meta = chain_meta.get(code, {})
        rows.append(
            {
                "日期": report_date,
                "账户ID": account_id,
                "方向": _side_from_holding(item),
                "合约代码": code,
                "合约名称": item.get("合约名称"),
                "买卖": _clean_text(item.get("买卖")),
                "持仓类型": item.get("持仓类型"),
                "总持仓": _number(item.get("总持仓")),
                "今持仓": _number(item.get("今持仓")),
                "今开仓": _number(item.get("今开仓")),
                "今平仓": _number(item.get("今平仓")),
                "可平量": _number(item.get("可平量")),
                "最新价": _number(item.get("最新价")),
                "持仓均价": _number(item.get("持仓均价")),
                "开仓均价": _number(item.get("开仓均价")),
                "期权市值": _number(item.get("期权市值")),
                "占用保证金": _number(item.get("占用保证金")),
                "持仓盈亏": _number(item.get("持仓盈亏")),
                "浮动盈亏": _number(item.get("浮动盈亏")),
                "行权价": meta.get("strike_price"),
                "到期日": meta.get("maturity_date"),
                "剩余天数": meta.get("dte"),
                "IV": meta.get("iv"),
                "Delta": meta.get("delta"),
                "Gamma": meta.get("gamma"),
                "Vega": meta.get("vega"),
                "Theta": meta.get("theta"),
            }
        )
    return rows


def _trade_rows_from_export(product, report_date):
    path = _latest_export_file("成交明细", report_date)
    if path is None:
        return []
    rows = _trade_rows_from_file(path)
    return [row for row in rows if _date8_to_iso(row.get("日期")) == report_date]


def _all_trade_rows_from_exports(product):
    rows = []
    for path in sorted(_live_hold_dir().glob("成交明细*.csv")):
        rows.extend(_trade_rows_from_file(path))
    seen = set()
    unique = []
    for row in rows:
        key = row.get("成交编号") or (row.get("合约代码"), row.get("成交时间(日)"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda row: str(row.get("成交时间(日)") or ""))


def _trade_rows_from_file(path):
    df = _read_export_csv(path)
    rows = []
    for _, item in df.iterrows():
        row = {column: item.get(column) for column in TRADE_COLUMNS}
        row["买卖"] = _clean_text(row.get("买卖"))
        for column in ["报单价格", "成交价格", "成交数量", "手续费", "平仓盈亏"]:
            row[column] = _number(row.get(column))
        row["日期"] = _date8_to_iso(row.get("日期"))
        rows.append(row)
    return rows


def _update_history_csv(path, new_rows, columns, key_columns):
    path = Path(path)
    if path.exists():
        history = pd.read_csv(path, encoding="utf-8-sig")
    else:
        history = pd.DataFrame(columns=columns)
    incoming = _frame(new_rows, columns)
    if not incoming.empty:
        for column in columns:
            if column not in history.columns:
                history[column] = None
        mask = pd.Series(False, index=history.index)
        for _, row in incoming.iterrows():
            row_mask = pd.Series(True, index=history.index)
            for key in key_columns:
                row_mask &= history[key].astype(str) == str(row[key])
            mask |= row_mask
        history = history.loc[~mask]
        if history.empty:
            history = incoming
        else:
            history = pd.concat([history, incoming], ignore_index=True)
    if "日期" in history.columns:
        history = history.sort_values([col for col in ["日期", "账户ID", "合约代码"] if col in history.columns])
    history = history.reindex(columns=columns)
    history.to_csv(path, index=False, encoding="utf-8-sig")
    return history


def _frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def _sum_row_values(rows, column):
    total = 0.0
    for row in rows:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        total += float(value)
    return total


def _chain_metadata(chain_df):
    metadata = {}
    for _, row in chain_df.iterrows():
        code = str(row.get("order_book_id"))
        metadata[code] = {
            "strike_price": row.get("strike_price"),
            "maturity_date": str(pd.Timestamp(row.get("maturity_date")).date()),
            "dte": row.get("dte"),
            "iv": row.get("iv"),
            "delta": row.get("delta"),
            "gamma": row.get("gamma"),
            "vega": row.get("vega"),
            "theta": row.get("theta"),
        }
    return metadata


def _latest_export_file(prefix, report_date=None):
    files = sorted(_live_hold_dir().glob(f"{prefix}*.csv"), key=lambda path: path.stat().st_mtime)
    if report_date is not None:
        matching = [path for path in files if _filename_date(path) == report_date]
        if matching:
            return matching[-1]
    return files[-1] if files else None


def _live_hold_dir():
    return PROJECT_ROOT / "live_hold"


def _read_export_csv(path):
    for encoding in ["utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _filename_date(path):
    match = re.search(r"(20\d{2})_(\d{2})_(\d{2})", Path(path).name)
    if not match:
        return None
    return "-".join(match.groups())


def _date8_to_iso(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"20\d{6}", text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    try:
        return str(pd.Timestamp(text).date())
    except Exception:
        return text


def _side_from_holding(item):
    buy_sell = _clean_text(item.get("买卖"))
    position_type = str(item.get("持仓类型", ""))
    if "卖" in buy_sell or "义务" in position_type:
        return "short"
    return "long"


def _clean_text(value):
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


def _number(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "待设置", "全部"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _plain_table(rows, columns):
    if not rows:
        return ["(none)"]
    widths = {
        column: min(
            22,
            max(len(column), *[len(_fmt(row.get(column))) for row in rows]),
        )
        for column in columns
    }
    output = [
        " | ".join(column.ljust(widths[column]) for column in columns),
        "-+-".join("-" * widths[column] for column in columns),
    ]
    for row in rows:
        output.append(
            " | ".join(
                _fmt(row.get(column))[: widths[column]].ljust(widths[column])
                for column in columns
            )
        )
    return output


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
