from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import akshare as ak
import pandas as pd


UNDERLYING_SYMBOL = "sh000852"
UNDERLYING_FILE_PREFIX = "000852.XSHG"
OPTION_FILE_PREFIX = "ZZ1000_OPTION"
OPTION_MULTIPLIER = 100

HEDGE_ETF_SYMBOL = "512100"
HEDGE_ETF_FILE_PREFIX = "512100.XSHG"


def ak_call(func, *args, retries=3, sleep=1.0, **kwargs):
    """AKShare 偶发断连时轻量重试，避免批量下载被单次网络波动打断。"""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(
                f"[zz1000] retry {func.__name__} {attempt}/{retries}: "
                f"{type(exc).__name__}",
                flush=True,
            )
            time.sleep(sleep)
    raise last_exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "下载中证1000股指期权、000852指数日线，以及用于 delta 对冲的 "
            "南方中证1000ETF(512100)日线。期权历史日线没有 bid/ask，"
            "因此使用 close 近似 bid=ask=mid。"
        )
    )
    parser.add_argument("--start", required=True, help="开始日期，例如 20220801")
    parser.add_argument("--end", required=True, help="结束日期，例如 20260101")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all", "index", "option", "hedge"),
        default=["all"],
        help="下载任务：all=指数+期权+对冲ETF；也可单独指定 index option hedge。",
    )
    parser.add_argument(
        "--months",
        nargs="*",
        default=None,
        help="要下载的期权月份，例如 mo2606 mo2607；默认使用 AKShare 当前返回的月份。",
    )
    parser.add_argument(
        "--auto-months",
        action="store_true",
        help="自动探测历史月份；会从 --month-start 到 --month-end 逐月尝试 moYYMM。",
    )
    parser.add_argument("--month-start", default=None, help="自动探测月份起点，例如 202207。")
    parser.add_argument("--month-end", default=None, help="自动探测月份终点，例如 202601。")
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.35,
        help="每次 AKShare 请求后的等待秒数，避免触发速率限制。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果目标日期 parquet 已存在，则跳过写入；仍会下载合约以拼接缺失日期。",
    )
    parser.add_argument(
        "--output-root",
        default="data/zz1000",
        help="输出根目录，默认 data/zz1000。",
    )
    return parser.parse_args()


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y%m%d")


def parse_option_code(code):
    match = re.fullmatch(r"mo(\d{2})(\d{2})([CP])(\d+)", code)
    if match is None:
        raise ValueError(f"无法解析中证1000期权代码: {code}")

    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    option_type = match.group(3)
    strike = float(match.group(4))
    return year, month, option_type, strike


def third_friday(year, month):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[days.month == month]
    fridays = [day for day in days if day.weekday() == 4]
    return fridays[2]


def adjust_to_trading_day(date, trading_calendar):
    eligible = trading_calendar[trading_calendar >= date]
    if len(eligible) == 0:
        return date
    return eligible[0]


def load_trading_calendar():
    index_df = ak_call(ak.stock_zh_index_daily, symbol=UNDERLYING_SYMBOL)
    return pd.DatetimeIndex(pd.to_datetime(index_df["date"])).sort_values()


def get_available_months():
    months = ak_call(ak.option_cffex_zz1000_list_sina)
    if isinstance(months, dict):
        result = []
        for values in months.values():
            result.extend(values)
        return result
    return list(months)


def iter_month_codes(month_start, month_end):
    start = pd.Period(str(month_start), freq="M")
    end = pd.Period(str(month_end), freq="M")
    for period in pd.period_range(start, end, freq="M"):
        yield f"mo{period.year % 100:02d}{period.month:02d}"


def discover_historical_months(month_start, month_end, request_sleep):
    months = []
    for month in iter_month_codes(month_start, month_end):
        try:
            snapshot = ak_call(ak.option_cffex_zz1000_spot_sina, symbol=month)
            if not snapshot.empty:
                months.append(month)
                print(f"[zz1000] discovered {month}: {len(snapshot)} strikes", flush=True)
        except Exception as exc:
            print(f"[zz1000] skip {month}: {type(exc).__name__}", flush=True)
        time.sleep(request_sleep)
    return months


def get_month_contract_codes(month):
    snapshot = ak_call(ak.option_cffex_zz1000_spot_sina, symbol=month)
    # AKShare 当前列顺序固定：第 9 列为 call 标识，第 17 列为 put 标识。
    call_code_col = snapshot.columns[8]
    put_code_col = snapshot.columns[16]
    codes = []
    for _, row in snapshot.iterrows():
        codes.append(str(row[call_code_col]))
        codes.append(str(row[put_code_col]))
    return list(dict.fromkeys(codes))


def download_underlying(start, end, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df = ak_call(ak.stock_zh_index_daily, symbol=UNDERLYING_SYMBOL)
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df[
        (index_df["date"] >= pd.Timestamp(start))
        & (index_df["date"] <= pd.Timestamp(end))
    ]
    index_df = index_df[["date", "open", "high", "low", "close", "volume"]].copy()

    for date, row in index_df.groupby("date"):
        payload = row.drop(columns=["date"]).reset_index(drop=True)
        file_path = output_dir / f"{UNDERLYING_FILE_PREFIX}_{date:%Y-%m-%d}_price.parquet"
        payload.to_parquet(file_path, index=False)

    return index_df


def normalize_etf_daily(raw_df):
    """把 AKShare ETF 日线字段转换成项目统一 OHLCV 字段。"""
    df = raw_df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close", "volume"])


def download_hedge_etf(start, end, output_dir):
    """下载南方中证1000ETF(512100)，作为 ZZ1000 delta 对冲标的。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_df = ak_call(
            ak.fund_etf_hist_em,
            symbol=HEDGE_ETF_SYMBOL,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        )
        source = "akshare_fund_etf_hist_em"
    except Exception as exc:
        print(
            f"[zz1000] fund_etf_hist_em failed, fallback to sina: {type(exc).__name__}",
            flush=True,
        )
        raw_df = ak_call(ak.fund_etf_hist_sina, symbol=f"sh{HEDGE_ETF_SYMBOL}")
        source = "akshare_fund_etf_hist_sina"

    df = normalize_etf_daily(raw_df)
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if source.endswith("_sina"):
        # Sina ETF 成交量是份，东方财富是手；统一为手，保持数据口径一致。
        df["volume"] = df["volume"] / 100

    for _, row in df.iterrows():
        date = row["date"]
        payload = pd.DataFrame(
            [
                {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "amount": row["amount"],
                    "order_book_id": HEDGE_ETF_FILE_PREFIX,
                    "source": source,
                }
            ]
        )
        file_path = output_dir / f"{HEDGE_ETF_FILE_PREFIX}_{date:%Y-%m-%d}_price.parquet"
        payload.to_parquet(file_path, index=False)

    return df


def build_contract_history(code, start, end, trading_calendar):
    daily = ak_call(ak.option_cffex_zz1000_daily_sina, symbol=code)
    if daily.empty:
        return pd.DataFrame()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily[
        (daily["date"] >= pd.Timestamp(start))
        & (daily["date"] <= pd.Timestamp(end))
    ]
    if daily.empty:
        return pd.DataFrame()

    year, month, option_type, strike = parse_option_code(code)
    maturity = adjust_to_trading_day(third_friday(year, month), trading_calendar)

    return pd.DataFrame(
        {
            "date": daily["date"],
            "order_book_id": code,
            "strike_price": strike,
            "maturity_date": maturity,
            "option_type": option_type,
            "bid": daily["close"],
            "ask": daily["close"],
            "volume": daily["volume"].astype(int),
            "open_interest": 0,
            "contract_multiplier": OPTION_MULTIPLIER,
            "close": daily["close"],
            "source": "akshare_daily_close_as_bid_ask",
        }
    )


def download_options(
    start,
    end,
    months,
    output_dir,
    trading_calendar,
    request_sleep,
    skip_existing,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    errors = []

    for month in months:
        print(f"[zz1000] download month {month}", flush=True)
        try:
            codes = get_month_contract_codes(month)
            time.sleep(request_sleep)
        except Exception as exc:
            errors.append({"month": month, "code": "", "error": repr(exc)})
            continue

        for idx, code in enumerate(codes, start=1):
            try:
                frame = build_contract_history(code, start, end, trading_calendar)
                time.sleep(request_sleep)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                errors.append({"month": month, "code": code, "error": repr(exc)})
                time.sleep(request_sleep)

            if idx % 20 == 0 or idx == len(codes):
                print(f"  {idx}/{len(codes)} contracts", flush=True)

    option_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not option_df.empty:
        for date, chain in option_df.groupby("date"):
            file_path = output_dir / f"{OPTION_FILE_PREFIX}_{date:%Y-%m-%d}_chain.parquet"
            if skip_existing and file_path.exists():
                continue
            chain = chain.drop(columns=["date"]).sort_values(
                ["maturity_date", "strike_price", "option_type", "order_book_id"]
            )
            chain.to_parquet(file_path, index=False)

    return option_df, pd.DataFrame(errors)


def main():
    args = parse_args()
    start = normalize_date(args.start)
    end = normalize_date(args.end)
    output_root = Path(args.output_root)
    index_dir = output_root / "index"
    option_dir = output_root / "option"
    hedge_etf_dir = output_root / "hedge_etf"
    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {"index", "option", "hedge"}

    if "option" in tasks and args.auto_months:
        if args.month_start is None or args.month_end is None:
            raise ValueError("--auto-months 需要同时指定 --month-start 和 --month-end")
        months = discover_historical_months(
            args.month_start,
            args.month_end,
            args.request_sleep,
        )
    elif "option" in tasks:
        months = args.months if args.months else get_available_months()
        print(f"[zz1000] months: {months}", flush=True)
    else:
        months = []

    trading_calendar = load_trading_calendar()
    index_df = (
        download_underlying(start, end, index_dir)
        if "index" in tasks
        else pd.DataFrame()
    )
    option_df, errors = (
        download_options(
            start,
            end,
            months,
            option_dir,
            trading_calendar,
            args.request_sleep,
            args.skip_existing,
        )
        if "option" in tasks
        else (pd.DataFrame(), pd.DataFrame())
    )
    hedge_df = (
        download_hedge_etf(start, end, hedge_etf_dir)
        if "hedge" in tasks
        else pd.DataFrame()
    )

    if not errors.empty:
        errors.to_csv(output_root / "download_errors.csv", index=False, encoding="utf-8-sig")

    print(
        f"[zz1000] index days={index_df['date'].nunique() if not index_df.empty else 0}, "
        f"option days={option_df['date'].nunique() if not option_df.empty else 0}, "
        f"option rows={len(option_df)}, "
        f"hedge ETF days={hedge_df['date'].nunique() if not hedge_df.empty else 0}, "
        f"errors={len(errors)}",
        flush=True,
    )
    print(f"[zz1000] output: {output_root}", flush=True)


if __name__ == "__main__":
    main()
