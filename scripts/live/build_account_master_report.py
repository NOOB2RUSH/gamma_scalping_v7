from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from core.live import storage


REPORT_PATTERN = re.compile(r"(\d{8})_(\d{6})_account_report\.json$")
HOLDING_PATTERN = re.compile(
    r"实时持仓.*_(20\d{2})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})\.csv$"
)
CLOSE_TIME = "150000"

SUMMARY_FIELDS = [
    "初始资金",
    "现金",
    "期权市值",
    "期权保证金",
    "对冲持仓",
    "对冲成本",
    "对冲最新价",
    "对冲保证金",
    "估算权益",
    "期权浮盈亏",
    "期权已实现盈亏",
    "期权总盈亏",
    "对冲浮盈亏",
    "对冲已实现盈亏",
    "对冲总盈亏",
    "手续费",
    "当日手续费",
    "期权单日盈亏",
    "ETF单日盈亏",
    "总单日盈亏",
    "净单日盈亏",
    "账户Delta",
    "期权Delta",
    "账户Gamma",
    "账户Vega",
    "账户Theta",
    "持仓IV",
    "单日DeltaPnL",
    "单日GammaPnL",
    "单日VegaPnL",
    "单日ThetaPnL",
    "单日GreeksPnL",
    "GreeksPnL口径",
    "GreeksPnL说明",
]

POSITION_FIELDS = [
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

TRADE_FIELDS = [
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one workbook from remote/live daily EOD account reports."
    )
    parser.add_argument("--product", default="kc50etf")
    parser.add_argument("--start", default="2026-06-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp.now().normalize()
    report_dir = storage.output_dir(args.product)
    selected = select_eod_reports(report_dir, start, end)
    holding_exports = select_eod_holding_exports(start, end)

    state_rows, position_rows, trade_rows, coverage_rows = build_rows(
        selected,
        holding_exports,
        start,
        end,
    )
    state = pd.DataFrame(state_rows)
    positions = pd.DataFrame(position_rows)
    trades = pd.DataFrame(trade_rows)
    coverage = pd.DataFrame(coverage_rows)
    trade_summary = build_trade_summary(trades, coverage)
    position_reconciliation = build_position_reconciliation(
        trades,
        positions,
        coverage,
    )
    mismatch_dates = set(
        position_reconciliation.loc[
            position_reconciliation["数量对账状态"].eq("不一致"),
            "日期",
        ].astype(str)
    )
    add_quality_warning(
        state,
        mismatch_dates,
        "报告持仓与成交连续推导数量不一致，NAV/盈亏可能失真",
    )
    add_quality_warning(
        coverage,
        mismatch_dates,
        "报告持仓与成交连续推导数量不一致，NAV/盈亏可能失真",
        quality_column="质量说明",
    )
    notes = build_notes(start, end)

    output = (
        Path(args.output)
        if args.output
        else report_dir
        / f"{args.product}_account_master_{start:%Y%m%d}_{end:%Y%m%d}.xlsx"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_workbook(
        output,
        {
            "每日账户状态": state,
            "每日成交汇总": trade_summary,
            "收盘持仓": positions,
            "持仓数量对账": position_reconciliation,
            "当日成交记录": trades,
            "数据覆盖说明": coverage,
            "口径说明": notes,
        },
    )
    print(f"account_master_report={output}")
    print(f"covered_eod_dates={len(selected)}")
    print(
        "missing_weekdays="
        + ",".join(
            coverage.loc[coverage["状态"].eq("缺失"), "日期"].astype(str).tolist()
        )
    )


def select_eod_reports(report_dir, start, end):
    by_date = {}
    for path in report_dir.glob("*_account_report.json"):
        match = REPORT_PATTERN.match(path.name)
        if match is None:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        report_date = pd.Timestamp(
            payload.get("date") or payload.get("summary", {}).get("日期")
        ).normalize()
        if report_date < start or report_date > end:
            continue
        generated_at = pd.Timestamp(f"{match.group(1)} {match.group(2)}")
        by_date.setdefault(report_date, []).append(
            {
                "path": path,
                "payload": payload,
                "generated_at": generated_at,
                "after_close": match.group(2) >= CLOSE_TIME,
            }
        )

    selected = {}
    for report_date, candidates in by_date.items():
        same_day = [
            item
            for item in candidates
            if item["generated_at"].normalize() == report_date
        ]
        after_close = [item for item in same_day if item["after_close"]]
        selected[report_date] = max(
            after_close or same_day or candidates,
            key=lambda item: item["generated_at"],
        )
    return selected


def select_eod_holding_exports(start, end):
    by_date = {}
    for path in Path("live_hold").glob("实时持仓*.csv"):
        match = HOLDING_PATTERN.match(path.name)
        if match is None:
            continue
        generated_at = pd.Timestamp(
            f"{match.group(1)}-{match.group(2)}-{match.group(3)} "
            f"{match.group(4)}:{match.group(5)}:{match.group(6)}"
        )
        report_date = generated_at.normalize()
        if report_date < start or report_date > end:
            continue
        if generated_at.strftime("%H%M%S") < CLOSE_TIME:
            continue
        current = by_date.get(report_date)
        if current is None or generated_at > current["generated_at"]:
            by_date[report_date] = {"path": path, "generated_at": generated_at}
    return by_date


def build_rows(selected, holding_exports, start, end):
    state_rows = []
    position_rows = []
    trade_rows = []
    coverage_rows = []
    prior_nav = None

    for report_date in pd.bdate_range(start, end):
        item = selected.get(report_date)
        if item is None:
            coverage_rows.append(
                {
                    "日期": report_date.date().isoformat(),
                    "状态": "缺失",
                    "收盘后快照": None,
                    "快照生成时间": None,
                    "行情源": None,
                    "持仓来源": None,
                    "质量说明": "远端无该业务日期的账户报告。",
                }
            )
            continue

        payload = item["payload"]
        summary = payload.get("summary", {})
        positions = payload.get("position_rows", []) or []
        trades = payload.get("trade_rows", []) or []
        holding_source = "账户报告"
        quality = []
        if not item["after_close"]:
            quality.append("无15:00后快照，使用当日最后快照")
        if not positions and report_date in holding_exports:
            positions = holding_rows_from_export(holding_exports[report_date]["path"])
            holding_source = holding_exports[report_date]["path"].name
            quality.append("账户报告持仓为空，使用收盘后券商持仓导出补齐")
        if not positions:
            quality.append("无收盘持仓记录，不能据此认定为空仓")
        if summary.get("净单日盈亏") is None:
            quality.append("旧版快照未记录净单日盈亏")
        if summary.get("总单日盈亏") is None:
            quality.append("旧版快照未记录总单日盈亏")

        nav = number(summary.get("估算权益"))
        nav_change = nav - prior_nav if nav is not None and prior_nav is not None else None
        if nav is not None:
            prior_nav = nav
        report_net = number(summary.get("净单日盈亏"))
        trade_fee = sum_present(row.get("手续费") for row in trades)
        broker_close_pnl = sum_present(row.get("平仓盈亏") for row in trades)

        state_row = {
            "日期": report_date.date().isoformat(),
            "收盘快照文件": item["path"].name,
            "快照生成时间": item["generated_at"],
            "行情源": payload.get("source"),
            "持仓来源": holding_source,
            "数据质量": "；".join(quality) if quality else "完整",
            "持仓行数": len(positions),
            "成交行数": len(trades),
            "成交记录手续费合计": trade_fee,
            "成交记录券商平仓盈亏合计": broker_close_pnl,
            "NAV较上个可用收盘变化": nav_change,
            "NAV变化与报告净盈亏差额": (
                nav_change - report_net
                if nav_change is not None and report_net is not None
                else None
            ),
        }
        state_row.update({field: summary.get(field) for field in SUMMARY_FIELDS})
        state_rows.append(state_row)

        for row in positions:
            output = {
                "日期": report_date.date().isoformat(),
                "收盘快照文件": item["path"].name,
                "持仓来源": holding_source,
            }
            output.update({field: row.get(field) for field in POSITION_FIELDS})
            position_rows.append(output)

        for row in trades:
            output = {
                "业务日期": report_date.date().isoformat(),
                "收盘快照文件": item["path"].name,
                "标准类型": (
                    "ETF对冲" if row.get("类型") == "ETF对冲" else "期权"
                ),
            }
            output.update({field: row.get(field) for field in TRADE_FIELDS})
            trade_rows.append(output)

        coverage_rows.append(
            {
                "日期": report_date.date().isoformat(),
                "状态": "已覆盖",
                "收盘后快照": item["path"].name,
                "快照生成时间": item["generated_at"],
                "行情源": payload.get("source"),
                "持仓来源": holding_source,
                "质量说明": "；".join(quality) if quality else "完整",
            }
        )
    return state_rows, position_rows, trade_rows, coverage_rows


def holding_rows_from_export(path):
    frame = read_export_csv(path)
    rows = []
    for _, item in frame.iterrows():
        rows.append(
            {
                "方向": "short" if "卖" in str(item.get("买卖") or "") else "long",
                "合约代码": item.get("合约代码"),
                "合约名称": item.get("合约名称"),
                "买卖": clean_text(item.get("买卖")),
                "持仓类型": item.get("持仓类型"),
                "总持仓": number(item.get("总持仓")),
                "今持仓": number(item.get("今持仓")),
                "今开仓": number(item.get("今开仓")),
                "今平仓": number(item.get("今平仓")),
                "可平量": number(item.get("可平量")),
                "最新价": number(item.get("最新价")),
                "持仓均价": number(item.get("持仓均价")),
                "开仓均价": number(item.get("开仓均价")),
                "期权市值": number(item.get("期权市值")),
                "占用保证金": number(item.get("占用保证金")),
                "持仓盈亏": number(item.get("持仓盈亏")),
                "浮动盈亏": number(item.get("浮动盈亏")),
            }
        )
    return rows


def build_trade_summary(trades, coverage):
    rows = []
    for report_date in coverage["日期"]:
        daily = trades.loc[trades["业务日期"].astype(str).eq(str(report_date))].copy()
        rows.append(
            {
                "日期": report_date,
                "成交笔数": len(daily),
                "期权成交笔数": int(
                    daily.get("标准类型", pd.Series(dtype=object)).eq("期权").sum()
                ),
                "ETF对冲成交笔数": int(
                    daily.get("标准类型", pd.Series(dtype=object)).eq("ETF对冲").sum()
                ),
                "成交数量合计": numeric_sum(daily, "成交数量"),
                "手续费合计": numeric_sum(daily, "手续费"),
                "券商平仓盈亏合计": numeric_sum(daily, "平仓盈亏"),
                "有平仓盈亏记录笔数": int(
                    pd.to_numeric(
                        daily.get("平仓盈亏", pd.Series(dtype=float)),
                        errors="coerce",
                    ).notna().sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_position_reconciliation(trades, positions, coverage):
    ledger = {}
    rows = []
    for report_date in coverage["日期"].astype(str):
        daily_trades = trades.loc[trades["业务日期"].astype(str).eq(report_date)]
        for _, trade in daily_trades.iterrows():
            code = clean_code(trade.get("合约代码"))
            qty = number(trade.get("成交数量"))
            if code is None or qty is None:
                continue
            signed_qty = qty if "买" in str(trade.get("买卖") or "") else -qty
            ledger[code] = ledger.get(code, 0.0) + signed_qty
            if abs(ledger[code]) <= 1e-9:
                ledger[code] = 0.0

        daily_positions = positions.loc[positions["日期"].astype(str).eq(report_date)]
        report_qty = {}
        names = {}
        for _, position in daily_positions.iterrows():
            code = clean_code(position.get("合约代码"))
            qty = number(position.get("总持仓"))
            if code is None or qty is None:
                continue
            direction = str(position.get("方向") or "").lower()
            signed_qty = -abs(qty) if direction == "short" else qty
            report_qty[code] = report_qty.get(code, 0.0) + signed_qty
            names[code] = position.get("合约名称")

        for code in sorted(set(ledger) | set(report_qty)):
            derived = ledger.get(code, 0.0)
            reported = report_qty.get(code, 0.0)
            if abs(derived) <= 1e-9 and abs(reported) <= 1e-9:
                continue
            difference = reported - derived
            rows.append(
                {
                    "日期": report_date,
                    "合约代码": code,
                    "合约名称": names.get(code),
                    "报告收盘数量(带方向)": reported,
                    "成交连续推导数量(带方向)": derived,
                    "数量对账差额": difference,
                    "数量对账状态": "一致" if abs(difference) <= 1e-6 else "不一致",
                }
            )
    return pd.DataFrame(rows)


def add_quality_warning(frame, dates, warning, quality_column="数据质量"):
    if frame.empty or not dates:
        return
    mask = frame["日期"].astype(str).isin(dates)
    current = frame.loc[mask, quality_column].fillna("").astype(str)
    frame.loc[mask, quality_column] = current.apply(
        lambda value: warning if not value or value == "完整" else f"{value}；{warning}"
    )


def build_notes(start, end):
    return pd.DataFrame(
        [
            {
                "项目": "时间范围",
                "说明": f"{start.date().isoformat()} 至 {end.date().isoformat()}，仅列工作日。",
            },
            {
                "项目": "收盘快照选择",
                "说明": "每个业务日期优先选择文件生成时间在15:00之后的最后一份账户报告。",
            },
            {
                "项目": "远端为准",
                "说明": "账户状态和成交均来自远端 output/live 与 live_hold；不使用本地旧文件。",
            },
            {
                "项目": "历史版本差异",
                "说明": "早期账户报告由旧版代码生成；未记录的单日盈亏字段保持为空，不反推为0。",
            },
            {
                "项目": "NAV变化",
                "说明": "NAV较上个可用收盘变化按相邻可用EOD快照计算；缺失工作日会导致跨日累计。",
            },
            {
                "项目": "持仓数量对账",
                "说明": "从零开始按远端每日成交连续推导带方向持仓数量，并与各日报告收盘持仓比较。",
            },
            {
                "项目": "交易盈亏",
                "说明": "当日成交记录中的平仓盈亏为券商原始字段，不等同于系统成本口径交易盈亏。",
            },
        ]
    )


def write_workbook(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.book[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column]
                width = min(max(max(map(len, values), default=0) + 2, 10), 42)
                sheet.column_dimensions[column[0].column_letter].width = width


def read_export_csv(path):
    for encoding in ["utf-8-sig", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def number(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sum_present(values):
    numbers = [number(value) for value in values]
    present = [value for value in numbers if value is not None]
    return sum(present) if present else None


def numeric_sum(frame, column):
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    return float(values.sum()) if values.notna().any() else None


def clean_text(value):
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


def clean_code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


if __name__ == "__main__":
    main()
