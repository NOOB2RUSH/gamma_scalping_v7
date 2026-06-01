from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import akshare as ak
import pandas as pd


ETF_SYMBOL = "510300"
ETF_FILE_PREFIX = "510300.XSHG"
OPTION_FILE_PREFIX = "300ETF_OPTION"
CONTRACT_MULTIPLIER = 10000
CALL_TEXT = "\u770b\u6da8\u671f\u6743"
PUT_TEXT = "\u770b\u8dcc\u671f\u6743"


def ak_call(func, *args, retries=3, sleep=1.0, **kwargs):
    """AKShare 源站偶尔会断开连接；这里做轻量重试，避免批量下载被单次网络波动打断。"""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(
                f"[300etf] retry {func.__name__} {attempt}/{retries}: "
                f"{type(exc).__name__}",
                flush=True,
            )
            time.sleep(sleep)
    raise last_exc


def fetch_etf_history(start, end):
    """优先用东方财富 ETF 日线；网络不稳时退回新浪日线。"""
    try:
        df = ak_call(
            ak.fund_etf_hist_em,
            symbol=ETF_SYMBOL,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        )
        source = "em"
    except Exception as exc:
        print(
            f"[300etf] fund_etf_hist_em failed, fallback to sina: {type(exc).__name__}",
            flush=True,
        )
        df = ak_call(ak.fund_etf_hist_sina, symbol=f"sh{ETF_SYMBOL}")
        source = "sina"

    if df.empty:
        return df

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=rename_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 新浪 ETF 成交量是份，东方财富是手；统一为手，和既有 ETF 数据口径一致。
    if source == "sina":
        df["volume"] = df["volume"] / 100
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "下载华泰柏瑞沪深300ETF(510300)与对应上交所 ETF 期权数据。"
            "期权历史日线没有历史 bid/ask，因此统一使用 close 近似 bid=ask=mid。"
        )
    )
    parser.add_argument("--start", required=True, help="开始日期，例如 20260520")
    parser.add_argument("--end", required=True, help="结束日期，例如 20260521")
    parser.add_argument(
        "--output-root",
        default="data/300etf",
        help="输出根目录，默认 data/300etf。",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.25,
        help="每次 AKShare 请求后的等待秒数，避免触发速率限制。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果目标日期 parquet 已存在，则跳过写入。",
    )
    parser.add_argument(
        "--metadata-source",
        choices=("risk", "current"),
        default="risk",
        help=(
            "risk: 按日期读取上交所风险指标来发现当时活跃合约；"
            "current: 仅下载当前挂牌月份合约。"
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all", "etf", "option"),
        default=("all",),
        help="选择下载任务：all=ETF+期权，etf=只刷新对冲/标的ETF日线，option=只刷新期权链。",
    )
    return parser.parse_args()


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y%m%d")


def download_etf(start, end, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = fetch_etf_history(start, end)
    if df.empty:
        return df

    for date, row in df.groupby("date"):
        payload = row.drop(columns=["date"]).reset_index(drop=True)
        file_path = (
            output_dir
            / f"{ETF_FILE_PREFIX}_{date.strftime('%Y-%m-%d')}_price.parquet"
        )
        payload.to_parquet(file_path, index=False)

    return df


def load_trading_calendar(start, end):
    df = fetch_etf_history(start, end)
    if df.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(df["date"])).sort_values()


def fourth_wednesday(year, month):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[days.month == month]
    wednesdays = [day for day in days if day.weekday() == 2]
    return wednesdays[3]


def adjust_to_trading_day(date, trading_calendar):
    eligible = trading_calendar[trading_calendar >= date]
    if len(eligible) == 0:
        return date
    return eligible[0]


def parse_contract_id(contract_id):
    match = re.fullmatch(r"510300([CP])(\d{2})(\d{2})([A-Z]?)(\d+)", contract_id)
    if match is None:
        raise ValueError(f"无法解析 300ETF 期权合约交易代码: {contract_id}")
    option_type = match.group(1)
    year = 2000 + int(match.group(2))
    month = int(match.group(3))
    raw_strike = float(match.group(5)) / 1000
    return year, month, option_type, raw_strike


def parse_strike_from_symbol(contract_symbol, fallback):
    match = re.search(r"(\d+(?:\.\d+)?)[A-Z]?$", str(contract_symbol))
    if match is None:
        return fallback
    value = float(match.group(1))
    if value > 100:
        value /= 1000
    return value


def metadata_from_risk_row(row, trading_calendar):
    year, month, option_type, raw_strike = parse_contract_id(str(row["CONTRACT_ID"]))
    maturity = adjust_to_trading_day(fourth_wednesday(year, month), trading_calendar)
    return {
        "order_book_id": str(row["SECURITY_ID"]),
        "contract_id": str(row["CONTRACT_ID"]),
        "contract_symbol": str(row["CONTRACT_SYMBOL"]),
        "strike_price": parse_strike_from_symbol(row["CONTRACT_SYMBOL"], raw_strike),
        "maturity_date": maturity,
        "option_type": option_type,
        "contract_multiplier": CONTRACT_MULTIPLIER,
    }


def discover_contracts_from_risk(start, end, trading_calendar, request_sleep):
    metadata = {}
    errors = []
    dates = trading_calendar[
        (trading_calendar >= pd.Timestamp(start)) & (trading_calendar <= pd.Timestamp(end))
    ]
    for idx, date in enumerate(dates, start=1):
        date_text = date.strftime("%Y%m%d")
        try:
            risk_df = ak_call(ak.option_risk_indicator_sse, date=date_text)
            risk_df = risk_df[risk_df["CONTRACT_ID"].astype(str).str.startswith("510300")]
            for _, row in risk_df.iterrows():
                code = str(row["SECURITY_ID"])
                metadata.setdefault(
                    code,
                    metadata_from_risk_row(row, trading_calendar),
                )
            print(
                f"[300etf] risk {idx}/{len(dates)} {date_text}: "
                f"active={len(risk_df)}, unique={len(metadata)}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"date": date_text, "code": "", "error": repr(exc)})
            print(f"[300etf] risk {date_text} failed: {type(exc).__name__}", flush=True)
        time.sleep(request_sleep)
    return metadata, errors


def discover_current_contracts(trading_calendar, request_sleep):
    metadata = {}
    errors = []
    months = ak_call(ak.option_sse_list_sina, symbol="300ETF")
    for month in months:
        for symbol, option_type in [(CALL_TEXT, "C"), (PUT_TEXT, "P")]:
            try:
                codes = ak_call(
                    ak.option_sse_codes_sina,
                    symbol=symbol,
                    trade_date=month,
                    underlying=ETF_SYMBOL,
                )
                code_col = codes.columns[1]
                for code in codes[code_col].astype(str):
                    try:
                        spot_df = ak_call(ak.option_sse_spot_price_sina, symbol=code)
                        value_col = spot_df.columns[1]
                        field_col = spot_df.columns[0]
                        field_map = dict(zip(spot_df[field_col], spot_df[value_col]))
                        contract_symbol = field_map.get("期权合约简称", "")
                        contract_id = (
                            f"510300{option_type}{month[2:]}M"
                            f"{int(float(field_map.get('行权价')) * 1000):05d}"
                        )
                        year = int(month[:4])
                        month_num = int(month[4:])
                        maturity = adjust_to_trading_day(
                            fourth_wednesday(year, month_num),
                            trading_calendar,
                        )
                        metadata[code] = {
                            "order_book_id": code,
                            "contract_id": contract_id,
                            "contract_symbol": contract_symbol,
                            "strike_price": float(field_map.get("行权价")),
                            "maturity_date": maturity,
                            "option_type": option_type,
                            "contract_multiplier": CONTRACT_MULTIPLIER,
                        }
                    except Exception as exc:
                        errors.append({"date": "", "code": code, "error": repr(exc)})
                    time.sleep(request_sleep)
            except Exception as exc:
                errors.append({"date": month, "code": "", "error": repr(exc)})
            time.sleep(request_sleep)
    return metadata, errors


def build_contract_history(code, metadata, start, end):
    daily = ak_call(ak.option_sse_daily_sina, symbol=code)
    if daily.empty:
        return pd.DataFrame()

    daily = daily.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
        }
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[(daily["date"] >= pd.Timestamp(start)) & (daily["date"] <= pd.Timestamp(end))]
    if daily.empty:
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    return pd.DataFrame(
        {
            "date": daily["date"],
            "order_book_id": code,
            "strike_price": metadata["strike_price"],
            "maturity_date": metadata["maturity_date"],
            "option_type": metadata["option_type"],
            "bid": daily["close"],
            "ask": daily["close"],
            # Sina 这里的“成交量”历史上更接近权利金成交额口径，先保留为原始活跃度。
            # 写入每日链前会用上交所当日认购/认沽总成交量缩放成张数。
            "raw_sina_volume": daily["volume"].fillna(0),
            "volume": daily["volume"].fillna(0),
            "open_interest": 0,
            "contract_multiplier": metadata["contract_multiplier"],
            "close": daily["close"],
            "source": "akshare_sse_daily_close_as_bid_ask_volume_scaled",
        }
    )


def fetch_official_side_volume(date):
    stats = ak_call(ak.option_daily_stats_sse, date=date.strftime("%Y%m%d"))
    row = stats[stats.iloc[:, 0].astype(str) == ETF_SYMBOL]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "C": int(float(row.iloc[5])),
        "P": int(float(row.iloc[6])),
    }


def scale_volume_to_official_total(chain, date):
    """把 Sina 原始活跃度按上交所认购/认沽总成交量缩放为合约张数。"""
    official_volume = fetch_official_side_volume(date)
    if official_volume is None:
        return chain

    chain = chain.copy()
    if "raw_sina_volume" not in chain.columns:
        chain["raw_sina_volume"] = chain["volume"]

    chain["volume_source"] = "sse_daily_stats_side_scaled"
    chain["volume"] = 0
    option_type = chain["option_type"].astype(str).str.upper()
    raw_activity = pd.to_numeric(chain["raw_sina_volume"], errors="coerce").fillna(0)

    for side in ["C", "P"]:
        mask = option_type == side
        target_total = official_volume[side]
        side_raw = raw_activity[mask]
        raw_total = side_raw.sum()
        if target_total <= 0 or raw_total <= 0:
            continue

        scaled = (side_raw / raw_total * target_total).round().astype(int)
        diff = target_total - int(scaled.sum())
        if diff != 0 and len(scaled) > 0:
            scaled.loc[side_raw.idxmax()] += diff
        chain.loc[mask, "volume"] = scaled

    return chain


def download_options(start, end, metadata, output_dir, request_sleep, skip_existing):
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    errors = []
    items = sorted(metadata.items())
    for idx, (code, meta) in enumerate(items, start=1):
        try:
            frame = build_contract_history(code, meta, start, end)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            errors.append({"date": "", "code": code, "error": repr(exc)})
        time.sleep(request_sleep)
        if idx % 25 == 0 or idx == len(items):
            print(f"[300etf] contracts {idx}/{len(items)}", flush=True)

    option_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not option_df.empty:
        for date, chain in option_df.groupby("date"):
            file_path = output_dir / f"{OPTION_FILE_PREFIX}_{date.strftime('%Y-%m-%d')}_chain.parquet"
            if skip_existing and file_path.exists():
                continue
            chain = scale_volume_to_official_total(chain, date)
            chain = chain.drop(columns=["date"]).sort_values(
                ["maturity_date", "strike_price", "option_type", "order_book_id"]
            )
            chain.to_parquet(file_path, index=False)
    return option_df, errors


def main():
    args = parse_args()
    start = normalize_date(args.start)
    end = normalize_date(args.end)
    output_root = Path(args.output_root)
    etf_dir = output_root / "etf"
    option_dir = output_root / "option"

    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {"etf", "option"}

    if "etf" in tasks:
        etf_df = download_etf(start, end, etf_dir)
    else:
        etf_df = pd.DataFrame()

    errors = []
    metadata = {}
    option_df = pd.DataFrame()
    if "option" in tasks:
        trading_calendar = load_trading_calendar(start, end)
        if args.metadata_source == "risk":
            metadata, errors = discover_contracts_from_risk(
                start,
                end,
                trading_calendar,
                args.request_sleep,
            )
        else:
            metadata, errors = discover_current_contracts(trading_calendar, args.request_sleep)

        option_df, option_errors = download_options(
            start,
            end,
            metadata,
            option_dir,
            args.request_sleep,
            args.skip_existing,
        )
        errors.extend(option_errors)
    if errors:
        pd.DataFrame(errors).to_csv(
            output_root / "download_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(
        f"[300etf] etf days={etf_df['date'].nunique() if not etf_df.empty else 0}, "
        f"contracts={len(metadata)}, "
        f"option days={option_df['date'].nunique() if not option_df.empty else 0}, "
        f"option rows={len(option_df)}, errors={len(errors)}",
        flush=True,
    )
    print(f"[300etf] output: {output_root}", flush=True)


if __name__ == "__main__":
    main()
