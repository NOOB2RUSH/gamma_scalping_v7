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
    "初始资金",
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
    "当日手续费",
    "账户Delta",
    "期权Delta",
    "账户Gamma",
    "账户Vega",
    "账户Theta",
    "持仓IV",
    "Call IV",
    "Put IV",
    "Call Delta",
    "Put Delta",
    "Call Gamma",
    "Put Gamma",
    "Call Vega",
    "Put Vega",
    "Call Theta",
    "Put Theta",
    "单日DeltaPnL",
    "单日GammaPnL",
    "单日VegaPnL",
    "单日ThetaPnL",
    "单日GreeksPnL",
    "累计DeltaPnL",
    "累计GammaPnL",
    "累计VegaPnL",
    "累计ThetaPnL",
    "累计GreeksPnL",
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
    persist_history=True,
):
    _, snapshot, market, _ = prepare_account_report_market(
        product,
        source=source,
        date=date,
    )
    payload = calculate_live_account_report(
        product,
        account_id=account_id,
        source=source,
        snapshot=snapshot,
        market=market,
        all_trades=all_trades,
    )
    if persist_history:
        persist_account_report_history(product, account_id, payload)
    else:
        payload["summary_history"] = _add_summary_greeks_pnl(
            pd.DataFrame(
                [payload["summary"]],
                columns=SUMMARY_COLUMNS,
            ),
            product=product,
        )
        _refresh_current_summary_from_history(payload)
        payload["position_history"] = pd.DataFrame(
            payload.get("position_rows", []),
            columns=POSITION_COLUMNS,
        )
    return payload


def prepare_account_report_market(product, source="akshare", date=None):
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

    market = signal_engine._load_market_context(
        config,
        report_date,
        quote_snapshot=snapshot,
        persist_feature_history=False,
    )
    return config, snapshot, market, report_date


def calculate_live_account_report(
    product,
    account_id="default",
    source="akshare",
    snapshot=None,
    market=None,
    all_trades=False,
):
    if market is None:
        _, snapshot, market, _ = prepare_account_report_market(
            product,
            source=source,
            date=None,
        )
    config = load_product_config(product)
    live_account = account_store.load_account(product, account_id=account_id)
    report_date_text = str(market["date"].date())
    spot = float(market["signal_row"]["close"])
    report_hedge = _hedge_for_report_date(
        live_account,
        product,
        account_id,
        report_date_text,
    )

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
        export_greeks = _greeks_from_holding_export(
            product,
            report_date_text,
            market["chain_df"],
        )
        if export_greeks is not None:
            account_greeks = export_greeks
    position_rows.extend(
        _hedge_rows_from_account(
            product,
            report_hedge,
            account_id,
            report_date_text,
            spot,
        )
    )
    trade_rows = _trade_rows_from_export(product, report_date_text)
    trade_rows.extend(_security_trade_rows_from_export(product, report_date_text))
    daily_fee = _configured_daily_report_fee(
        product,
        account_id,
        report_date_text,
        trade_rows,
    )
    cumulative_fee = _configured_cumulative_report_fee(
        product,
        account_id,
        report_date_text,
    )

    hedge_unrealized_pnl = _hedge_unrealized_pnl_for_report(
        product,
        report_hedge.qty,
        report_hedge.entry_price,
        spot,
        report_date_text,
    )
    initial_cash = float(config.backtest.initial_cash)
    nav_estimate = initial_cash + option_pnl + hedge_unrealized_pnl - cumulative_fee
    summary_row = {
        "日期": report_date_text,
        "账户ID": account_id,
        "初始资金": initial_cash,
        "标的价格": spot,
        "现金": live_account.cash,
        "期权市值": option_value,
        "期权保证金": option_margin,
        "对冲持仓": report_hedge.qty,
        "对冲成本": report_hedge.entry_price,
        "对冲保证金": report_hedge.margin,
        "对冲浮盈亏": hedge_unrealized_pnl,
        "估算权益": nav_estimate,
        "期权浮盈亏": option_pnl,
        "手续费": cumulative_fee,
        "当日手续费": daily_fee,
        "账户Delta": account_greeks["delta"] + report_hedge.qty,
        "期权Delta": account_greeks["delta"],
        "账户Gamma": account_greeks["gamma"],
        "账户Vega": account_greeks["vega"],
        "账户Theta": account_greeks["theta"],
        "持仓IV": account_greeks["position_iv"],
        "Call IV": account_greeks["call_iv"],
        "Put IV": account_greeks["put_iv"],
        "Call Delta": account_greeks["call_delta"],
        "Put Delta": account_greeks["put_delta"],
        "Call Gamma": account_greeks["call_gamma"],
        "Put Gamma": account_greeks["put_gamma"],
        "Call Vega": account_greeks["call_vega"],
        "Put Vega": account_greeks["put_vega"],
        "Call Theta": account_greeks["call_theta"],
        "Put Theta": account_greeks["put_theta"],
    }

    return {
        "product": product,
        "account_id": account_id,
        "date": report_date_text,
        "spot": spot,
        "source": source,
        "quote_snapshot": snapshot,
        "summary": summary_row,
        "summary_history": None,
        "position_history": None,
        "position_rows": position_rows,
        "trade_rows": trade_rows,
        "strategy_state": live_account.strategy_state.to_dict(),
    }


def persist_account_report_history(product, account_id, payload):
    summary_path = storage.account_report_summary_history_path(product, account_id)
    position_path = storage.account_report_position_history_path(product, account_id)
    payload["position_history"] = _update_history_csv(
        position_path,
        payload.get("position_rows", []),
        POSITION_COLUMNS,
        key_columns=["日期", "账户ID"],
    )
    payload["summary_history"] = _update_history_csv(
        summary_path,
        [payload["summary"]],
        SUMMARY_COLUMNS,
        key_columns=["日期", "账户ID"],
    )
    payload["summary_history"] = _add_summary_greeks_pnl(
        payload["summary_history"],
        payload["position_history"],
        product=product,
    )
    payload["summary_history"].to_csv(summary_path, index=False, encoding="utf-8-sig")
    _refresh_current_summary_from_history(payload)
    return payload


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
        (
            f"单日GreeksPnL(昨日Greeks)={_fmt(summary.get('单日GreeksPnL'))} "
            f"Delta={_fmt(summary.get('单日DeltaPnL'))} "
            f"Gamma={_fmt(summary.get('单日GammaPnL'))} "
            f"Vega={_fmt(summary.get('单日VegaPnL'))} "
            f"Theta={_fmt(summary.get('单日ThetaPnL'))}"
        ),
        (
            f"累计GreeksPnL={_fmt(summary.get('累计GreeksPnL'))} "
            f"Delta={_fmt(summary.get('累计DeltaPnL'))} "
            f"Gamma={_fmt(summary.get('累计GammaPnL'))} "
            f"Vega={_fmt(summary.get('累计VegaPnL'))} "
            f"Theta={_fmt(summary.get('累计ThetaPnL'))}"
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


def _greeks_from_holding_export(product, report_date, chain_df):
    path = _latest_export_file("实时持仓", report_date)
    if path is None:
        return None
    df = _read_export_csv(path)
    if df.empty:
        return None

    chain_by_code = {
        str(row.get("order_book_id")): row
        for _, row in chain_df.iterrows()
    }
    grouped = {}
    for _, item in df.iterrows():
        code = str(item.get("合约代码", "")).strip()
        chain_row = chain_by_code.get(code)
        if chain_row is None:
            continue
        qty = int(_number(item.get("总持仓")) or 0)
        if qty <= 0:
            continue
        option_type = str(chain_row.get("option_type", "")).upper()
        leg = "call" if option_type == "C" else "put" if option_type == "P" else None
        if leg is None:
            continue
        key = (
            _side_from_holding(item),
            float(chain_row.get("strike_price")),
            str(pd.Timestamp(chain_row.get("maturity_date")).date()),
        )
        grouped.setdefault(key, {})[leg] = {
            "row": chain_row,
            "qty": qty,
        }

    greeks_list = []
    for (side, _, _), legs in grouped.items():
        if "call" not in legs or "put" not in legs:
            continue
        greeks_list.append(
            core.strategy.calc_position_greeks(
                legs["call"]["row"],
                legs["put"]["row"],
                legs["call"]["qty"],
                legs["put"]["qty"],
                side=side,
            )
        )
    if not greeks_list:
        return None
    return core.backtester.combine_greeks(greeks_list)


def _hedge_for_report_date(live_account, product, account_id, report_date):
    report_ts = pd.Timestamp(report_date).normalize()
    fills = account_store.list_fills(
        product,
        account_id=account_id,
        include_voided=False,
    )
    hedge = None
    has_hedge_fill = False
    saw_hedge_fill = False
    for row in fills:
        payload = account_store.normalize_fill(row["payload"])
        if payload.get("action") not in {"delta_hedge", "rebalance_hedge", "close_hedge"}:
            continue
        has_hedge_fill = True
        fill_date = _date_or_none(payload.get("date"))
        if fill_date is None or fill_date > report_ts:
            continue
        saw_hedge_fill = True
        hedge = account_store.HedgeState(
            qty=float(payload.get("qty", payload.get("new_etf_qty", payload.get("target_hedge_qty", 0.0))) or 0.0),
            entry_price=float(payload.get("entry_price", payload.get("price", 0.0)) or 0.0),
            margin=float(payload.get("margin", 0.0) or 0.0),
            underlying_order_book_id=payload.get("underlying_order_book_id"),
        )
    if saw_hedge_fill:
        return hedge or account_store.HedgeState()
    if has_hedge_fill:
        return account_store.HedgeState()
    return live_account.hedge


def _hedge_rows_from_account(product, hedge, account_id, report_date, spot):
    if abs(float(hedge.qty or 0.0)) <= 1e-9:
        return []

    export_row = _security_holding_export_row(product, report_date)
    latest_price = (
        _number(export_row.get("最新价"))
        if export_row is not None
        else float(spot)
    )
    if latest_price is None:
        latest_price = float(spot)
    qty = float(hedge.qty)
    market_value = (
        _number(export_row.get("市值"))
        if export_row is not None
        else qty * latest_price
    )
    floating_pnl = (
        _number(export_row.get("浮动盈亏"))
        if export_row is not None
        else core.hedge.calc_unrealized_pnl(qty, hedge.entry_price, latest_price)
    )
    security_code = (
        _security_code(export_row.get("证券代码"))
        if export_row is not None
        else _security_code_from_underlying(hedge.underlying_order_book_id)
    )
    security_name = (
        export_row.get("证券名称")
        if export_row is not None
        else hedge.underlying_order_book_id
    )
    return [
        {
            "日期": report_date,
            "账户ID": account_id,
            "方向": "hedge",
            "合约代码": security_code or hedge.underlying_order_book_id,
            "合约名称": security_name,
            "买卖": "买" if qty > 0 else "卖",
            "持仓类型": "ETF对冲",
            "总持仓": qty,
            "今持仓": _number(export_row.get("持有数量")) if export_row is not None else None,
            "今开仓": None,
            "今平仓": None,
            "可平量": _number(export_row.get("可用数量")) if export_row is not None else None,
            "最新价": latest_price,
            "持仓均价": hedge.entry_price,
            "开仓均价": hedge.entry_price,
            "期权市值": market_value,
            "占用保证金": hedge.margin,
            "持仓盈亏": floating_pnl,
            "浮动盈亏": floating_pnl,
            "行权价": None,
            "到期日": None,
            "剩余天数": None,
            "IV": None,
            "Delta": qty,
            "Gamma": 0.0,
            "Vega": 0.0,
            "Theta": 0.0,
        }
    ]


def _security_holding_export_row(product, report_date):
    path = _latest_export_file("证券持仓查询", report_date)
    if path is None:
        return None
    df = _read_export_csv(path)
    target_code = _product_security_code(product)
    rows = []
    for _, row in df.iterrows():
        code = _security_code(row.get("证券代码"))
        if target_code is not None and code != target_code:
            continue
        qty = _number(row.get("持有数量")) or 0.0
        if qty > 0:
            rows.append(row)
    if not rows:
        return None
    return rows[0]


def _hedge_unrealized_pnl_for_report(product, qty, entry_price, spot, report_date):
    export_row = _security_holding_export_row(product, report_date)
    if export_row is not None:
        broker_pnl = _number(export_row.get("浮动盈亏"))
        if broker_pnl is not None:
            return broker_pnl
    return core.hedge.calc_unrealized_pnl(qty, entry_price, spot)


def _trade_rows_from_export(product, report_date):
    path = _latest_export_file("成交明细", report_date)
    if path is None:
        return []
    rows = _trade_rows_from_file(path, product)
    return [row for row in rows if _date8_to_iso(row.get("日期")) == report_date]


def _security_trade_rows_from_export(product, report_date):
    path = _latest_export_file("证券委托查询_实时成交", report_date)
    if path is None:
        return []
    df = _read_export_csv(path)
    target_code = _product_security_code(product)
    rows = []
    for _, item in df.iterrows():
        if _date8_to_iso(item.get("日期")) != report_date:
            continue
        code = _security_code(item.get("证券代码"))
        if target_code is not None and code != target_code:
            continue
        price = _number(item.get("成交价格"))
        qty = _number(item.get("成交数量"))
        rows.append(
            {
                "序号": item.get("序号"),
                "投资者账号": item.get("投资者账号"),
                "交易所": item.get("交易所"),
                "合约代码": code,
                "合约名称": item.get("证券名称"),
                "成交编号": item.get("成交编号"),
                "报单编号": item.get("报单编号"),
                "开平": None,
                "买卖": _clean_text(item.get("买卖")),
                "报单价格": _number(item.get("报单价格")),
                "成交价格": price,
                "成交数量": qty,
                "手续费": _configured_etf_trade_fee(product, price, qty),
                "平仓盈亏": None,
                "类型": "ETF对冲",
                "日期": _date8_to_iso(item.get("日期")),
                "报单时间": item.get("报单时间"),
                "成交时间": item.get("成交时间"),
                "成交时间(日)": item.get("成交时间(日)"),
                "策略名称": item.get("策略名称"),
            }
        )
    return rows


def _all_trade_rows_from_exports(product):
    rows = []
    for path in sorted(_live_hold_dir().glob("成交明细*.csv")):
        rows.extend(_trade_rows_from_file(path, product))
    seen = set()
    unique = []
    for row in rows:
        key = row.get("成交编号") or (row.get("合约代码"), row.get("成交时间(日)"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda row: str(row.get("成交时间(日)") or ""))


def _all_security_trade_rows_from_exports(product):
    rows = []
    for path in sorted(_live_hold_dir().glob("证券委托查询_实时成交*.csv")):
        df = _read_export_csv(path)
        target_code = _product_security_code(product)
        for _, item in df.iterrows():
            code = _security_code(item.get("证券代码"))
            if target_code is not None and code != target_code:
                continue
            price = _number(item.get("成交价格"))
            qty = _number(item.get("成交数量"))
            rows.append(
                {
                    "成交编号": item.get("成交编号"),
                    "合约代码": code,
                    "成交价格": price,
                    "成交数量": qty,
                    "手续费": _configured_etf_trade_fee(product, price, qty),
                    "类型": "ETF对冲",
                    "日期": _date8_to_iso(item.get("日期")),
                    "成交时间(日)": item.get("成交时间(日)") or item.get("成交时间"),
                }
            )
    seen = set()
    unique = []
    for row in rows:
        key = row.get("成交编号") or (row.get("合约代码"), row.get("成交时间(日)"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda row: str(row.get("成交时间(日)") or ""))


def _add_summary_greeks_pnl(summary_history, position_history=None, product=None):
    history = summary_history.copy()
    for column in SUMMARY_COLUMNS:
        if column not in history.columns:
            history[column] = None
    if history.empty:
        return history.reindex(columns=SUMMARY_COLUMNS)

    history = _fill_summary_leg_greeks_from_positions(
        history,
        position_history,
        product,
    )
    history = history.sort_values(["账户ID", "日期"]).reset_index(drop=True)
    groups = []
    for _, group in history.groupby("账户ID", dropna=False, sort=False):
        groups.append(_add_summary_greeks_pnl_for_account(group.copy()))
    result = pd.concat(groups, ignore_index=True) if groups else history
    result = result.sort_values(["日期", "账户ID"]).reset_index(drop=True)
    return result.reindex(columns=SUMMARY_COLUMNS)


def _add_summary_greeks_pnl_for_account(group):
    spot = _numeric_series(group, "标的价格")
    hedge_qty = _numeric_series(group, "对冲持仓").fillna(0.0)
    call_iv = _numeric_series(group, "Call IV")
    put_iv = _numeric_series(group, "Put IV")
    call_delta = _numeric_series(group, "Call Delta")
    put_delta = _numeric_series(group, "Put Delta")
    call_gamma = _numeric_series(group, "Call Gamma")
    put_gamma = _numeric_series(group, "Put Gamma")
    call_vega = _numeric_series(group, "Call Vega")
    put_vega = _numeric_series(group, "Put Vega")
    call_theta = _numeric_series(group, "Call Theta")
    put_theta = _numeric_series(group, "Put Theta")

    spot_chg = spot.diff()
    option_explainable = (
        spot.shift(1).notna()
        & call_iv.shift(1).notna()
        & put_iv.shift(1).notna()
        & call_iv.notna()
        & put_iv.notna()
    )

    option_delta_pnl = (call_delta.shift(1) + put_delta.shift(1)) * spot_chg
    hedge_delta_pnl = hedge_qty.shift(1).fillna(0.0) * spot_chg
    gamma_pnl = (
        0.5
        * (
            (call_gamma.shift(1) + call_gamma) / 2
            + (put_gamma.shift(1) + put_gamma) / 2
        )
        * spot_chg**2
    )
    vega_pnl = (
        ((call_vega.shift(1) + call_vega) / 2) * (call_iv - call_iv.shift(1)) * 100
        + ((put_vega.shift(1) + put_vega) / 2) * (put_iv - put_iv.shift(1)) * 100
    )
    theta_pnl = (
        (call_theta.shift(1) + call_theta) / 2
        + (put_theta.shift(1) + put_theta) / 2
    )

    group["单日DeltaPnL"] = (
        option_delta_pnl.where(option_explainable, 0.0).fillna(0.0)
        + hedge_delta_pnl.fillna(0.0)
    )
    group["单日GammaPnL"] = gamma_pnl.where(option_explainable, 0.0).fillna(0.0)
    group["单日VegaPnL"] = vega_pnl.where(option_explainable, 0.0).fillna(0.0)
    group["单日ThetaPnL"] = theta_pnl.where(option_explainable, 0.0).fillna(0.0)
    group["单日GreeksPnL"] = group[
        ["单日DeltaPnL", "单日GammaPnL", "单日VegaPnL", "单日ThetaPnL"]
    ].sum(axis=1)

    for daily_column, cumulative_column in [
        ("单日DeltaPnL", "累计DeltaPnL"),
        ("单日GammaPnL", "累计GammaPnL"),
        ("单日VegaPnL", "累计VegaPnL"),
        ("单日ThetaPnL", "累计ThetaPnL"),
        ("单日GreeksPnL", "累计GreeksPnL"),
    ]:
        group[cumulative_column] = _numeric_series(group, daily_column).fillna(0.0).cumsum()
    return group


def _fill_summary_leg_greeks_from_positions(summary_history, position_history, product=None):
    if position_history is None or position_history.empty:
        return summary_history
    multiplier = _contract_multiplier(product)
    aggregates = {}
    for _, row in position_history.iterrows():
        leg = _position_row_leg(row)
        if leg is None:
            continue
        key = (str(row.get("日期")), str(row.get("账户ID")))
        target = aggregates.setdefault(
            key,
            {
                "direct": _empty_leg_aggregate(),
                "scaled": _empty_leg_aggregate(),
            },
        )
        qty = abs(_number(row.get("总持仓")) or 0.0)
        scale = qty * multiplier
        direction = -1.0 if str(row.get("方向") or "") == "short" else 1.0
        for metric in ["Delta", "Gamma", "Vega", "Theta"]:
            value = _number(row.get(metric))
            if value is not None:
                target["direct"][f"{leg} {metric}"] += value
                target["scaled"][f"{leg} {metric}"] += value * scale * direction
        iv = _number(row.get("IV"))
        if iv is not None:
            target["direct"][f"{leg} IV"].append(iv)
            target["scaled"][f"{leg} IV"].append(iv)

    if not aggregates:
        return summary_history

    result = summary_history.copy()
    for index, row in result.iterrows():
        values = aggregates.get((str(row.get("日期")), str(row.get("账户ID"))))
        if values is None:
            continue
        selected = values[_select_leg_aggregate_mode(row, values)]
        replace_greeks = _should_replace_leg_greeks(row, selected)
        for column, value in selected.items():
            if column.endswith(" IV"):
                if value:
                    value = sum(value) / len(value)
                else:
                    continue
            if column not in result.columns:
                result[column] = None
            if pd.isna(result.at[index, column]) or (
                replace_greeks and not column.endswith(" IV")
            ):
                result.at[index, column] = value
    return result


def _empty_leg_aggregate():
    return {
        "Call Delta": 0.0,
        "Put Delta": 0.0,
        "Call Gamma": 0.0,
        "Put Gamma": 0.0,
        "Call Vega": 0.0,
        "Put Vega": 0.0,
        "Call Theta": 0.0,
        "Put Theta": 0.0,
        "Call IV": [],
        "Put IV": [],
    }


def _select_leg_aggregate_mode(summary_row, aggregates):
    expected_delta = _number(summary_row.get("期权Delta"))
    if expected_delta is not None:
        direct_delta = (
            aggregates["direct"]["Call Delta"]
            + aggregates["direct"]["Put Delta"]
        )
        scaled_delta = (
            aggregates["scaled"]["Call Delta"]
            + aggregates["scaled"]["Put Delta"]
        )
        if abs(scaled_delta - expected_delta) < abs(direct_delta - expected_delta):
            return "scaled"
        return "direct"

    direct_abs_delta = max(
        abs(aggregates["direct"]["Call Delta"]),
        abs(aggregates["direct"]["Put Delta"]),
    )
    scaled_abs_delta = max(
        abs(aggregates["scaled"]["Call Delta"]),
        abs(aggregates["scaled"]["Put Delta"]),
    )
    return "scaled" if direct_abs_delta <= 5.0 < scaled_abs_delta else "direct"


def _should_replace_leg_greeks(summary_row, selected):
    expected_delta = _number(summary_row.get("期权Delta"))
    current_call_delta = _number(summary_row.get("Call Delta"))
    current_put_delta = _number(summary_row.get("Put Delta"))
    if (
        expected_delta is None
        or current_call_delta is None
        or current_put_delta is None
    ):
        return False
    current_delta = current_call_delta + current_put_delta
    selected_delta = selected["Call Delta"] + selected["Put Delta"]
    current_error = abs(current_delta - expected_delta)
    selected_error = abs(selected_delta - expected_delta)
    tolerance = max(1.0, abs(expected_delta) * 1e-4)
    return current_error > tolerance and selected_error < current_error


def _contract_multiplier(product):
    if product is None:
        return 1.0
    try:
        return float(load_product_config(product).vol.contract_multiplier)
    except Exception:
        return 1.0


def _position_row_leg(row):
    name = str(row.get("合约名称") or "").upper()
    if "购" in name or "CALL" in name:
        return "Call"
    if "沽" in name or "PUT" in name:
        return "Put"
    direction = str(row.get("方向") or "")
    delta = _number(row.get("Delta"))
    if delta is None:
        return None
    if abs(delta) <= 1.5:
        return "Call" if delta > 0 else "Put"
    if direction == "short":
        return "Call" if delta < 0 else "Put"
    if direction == "long":
        return "Call" if delta > 0 else "Put"
    return None


def _numeric_series(frame, column):
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _refresh_current_summary_from_history(payload):
    history = payload.get("summary_history")
    if history is None or history.empty:
        return
    mask = (
        history["日期"].astype(str).eq(str(payload["date"]))
        & history["账户ID"].astype(str).eq(str(payload["account_id"]))
    )
    if mask.any():
        payload["summary"] = history.loc[mask].iloc[-1].to_dict()


def _configured_daily_report_fee(product, account_id, report_date, trade_rows):
    fee = _sum_row_values(trade_rows, "手续费")
    has_option_trade_rows = any(row.get("类型") != "ETF对冲" for row in trade_rows)
    has_etf_trade_rows = any(row.get("类型") == "ETF对冲" for row in trade_rows)

    fills = account_store.list_fills(
        product,
        account_id=account_id,
        include_voided=False,
    )
    for row in fills:
        fill = account_store.normalize_fill(row["payload"])
        if str(fill.get("date")) != str(report_date):
            continue
        action = str(fill.get("action") or "").lower()
        if action in {
            "open_long_straddle",
            "open_short_straddle",
            "close_long_straddle",
            "close_short_straddle",
            "roll_long_straddle",
            "roll_short_straddle",
        }:
            if not has_option_trade_rows and not _is_position_only_holding_import(fill):
                fee += _configured_option_fill_fee(product, fill)
        elif action in {"delta_hedge", "rebalance_hedge", "close_hedge"}:
            if not has_etf_trade_rows:
                fee += _configured_hedge_fill_fee(product, fill)
    return fee


def _configured_cumulative_report_fee(product, account_id, report_date):
    cutoff = _date_or_none(report_date)
    if cutoff is None:
        return 0.0

    option_rows = _all_trade_rows_from_exports(product)
    security_rows = _all_security_trade_rows_from_exports(product)
    fee = 0.0
    option_trade_dates = set()
    security_trade_dates = set()
    for row in option_rows:
        row_date = _date_or_none(row.get("日期"))
        if row_date is None or row_date > cutoff:
            continue
        option_trade_dates.add(str(row_date.date()))
        fee += float(row.get("手续费") or 0.0)
    for row in security_rows:
        row_date = _date_or_none(row.get("日期"))
        if row_date is None or row_date > cutoff:
            continue
        security_trade_dates.add(str(row_date.date()))
        fee += float(row.get("手续费") or 0.0)

    fills = account_store.list_fills(
        product,
        account_id=account_id,
        include_voided=False,
    )
    for row in fills:
        fill = account_store.normalize_fill(row["payload"])
        fill_date = _date_or_none(fill.get("date"))
        if fill_date is None or fill_date > cutoff:
            continue
        date_key = str(fill_date.date())
        action = str(fill.get("action") or "").lower()
        if action in {
            "open_long_straddle",
            "open_short_straddle",
            "close_long_straddle",
            "close_short_straddle",
            "roll_long_straddle",
            "roll_short_straddle",
        }:
            if date_key not in option_trade_dates and not _is_position_only_holding_import(fill):
                fee += _configured_option_fill_fee(product, fill)
        elif action in {"delta_hedge", "rebalance_hedge", "close_hedge"}:
            if date_key not in security_trade_dates:
                fee += _configured_hedge_fill_fee(product, fill)
    return fee


def _configured_option_trade_fee(product, qty):
    qty = _number(qty) or 0.0
    config = load_product_config(product)
    return abs(qty) * float(config.backtest.option_fee_per_contract)


def _configured_etf_trade_fee(product, price, qty):
    price = _number(price) or 0.0
    qty = _number(qty) or 0.0
    config = load_product_config(product)
    return abs(price * qty) * float(config.backtest.etf_fee_rate)


def _configured_option_fill_fee(product, fill):
    config = load_product_config(product)
    call_qty = float(fill.get("call_qty", 0.0) or 0.0)
    put_qty = float(fill.get("put_qty", 0.0) or 0.0)
    return (abs(call_qty) + abs(put_qty)) * float(config.backtest.option_fee_per_contract)


def _configured_hedge_fill_fee(product, fill):
    config = load_product_config(product)
    trades = fill.get("security_trades") or []
    if trades:
        notional = sum(
            abs(float(trade.get("price", 0.0) or 0.0) * float(trade.get("qty", 0.0) or 0.0))
            for trade in trades
        )
    else:
        qty = float(fill.get("trade_etf_qty", fill.get("qty", 0.0)) or 0.0)
        price = float(fill.get("price", fill.get("entry_price", 0.0)) or 0.0)
        notional = abs(qty * price)
    return notional * float(config.backtest.etf_fee_rate)


def _is_position_only_holding_import(fill):
    if fill.get("import_source") != "broker_holding_snapshot":
        return False
    source_file = fill.get("source_file")
    if not source_file:
        return False
    path = Path(source_file)
    if not path.exists():
        path = PROJECT_ROOT / source_file
    if not path.exists():
        return False
    try:
        df = _read_export_csv(path)
    except Exception:
        return False
    if "今开仓" not in df.columns:
        return False
    today_open = pd.to_numeric(df["今开仓"], errors="coerce").fillna(0.0).sum()
    return today_open <= 0


def _trade_rows_from_file(path, product):
    df = _read_export_csv(path)
    rows = []
    for _, item in df.iterrows():
        row = {column: item.get(column) for column in TRADE_COLUMNS}
        row["买卖"] = _clean_text(row.get("买卖"))
        for column in ["报单价格", "成交价格", "成交数量", "手续费", "平仓盈亏"]:
            row[column] = _number(row.get(column))
        row["手续费"] = _configured_option_trade_fee(product, row.get("成交数量"))
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
        return matching[-1] if matching else None
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


def _date_or_none(value):
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception:
        return None


def _side_from_holding(item):
    buy_sell = _clean_text(item.get("买卖"))
    position_type = str(item.get("持仓类型", ""))
    if "卖" in buy_sell or "义务" in position_type:
        return "short"
    return "long"


def _product_security_code(product):
    spec = market_data.SSE_ETF_OPTION_SPECS.get(product)
    if spec is None:
        return None
    return _security_code(spec.etf_symbol)


def _security_code_from_underlying(value):
    if value is None:
        return None
    return _security_code(str(value).split(".", 1)[0])


def _security_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


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
