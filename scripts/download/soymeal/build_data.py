from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import akshare as ak
import pandas as pd


COMMODITY_OPTION_SYMBOL = "\u8c46\u7c95\u671f\u6743"
UNDERLYING_SYMBOL = "M0"
UNDERLYING_FILE_PREFIX = "M0.DCE"
OPTION_FILE_PREFIX = "SOYMEAL_OPTION"
CONTRACT_MULTIPLIER = 10


def ak_call(func, *args, retries=3, sleep=1.0, **kwargs):
    """轻量重试，避免单次网络波动中断批量下载。"""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == retries:
                break
            print(
                f"[soymeal] retry {func.__name__} {attempt}/{retries}: "
                f"{type(exc).__name__}",
                flush=True,
            )
            time.sleep(sleep)
    raise last_exc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "下载豆粕期权和豆粕主连行情。第一版使用 Sina 商品期权历史日线，"
            "并用豆粕主连 M0 作为现有回测引擎的单一近似标的。"
        )
    )
    parser.add_argument("--start", required=True, help="开始日期，例如 20240101")
    parser.add_argument("--end", required=True, help="结束日期，例如 20240531")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all", "underlying", "option"),
        default=("all",),
        help="all=主连+期权；underlying=只下载 M0；option=只下载期权链。",
    )
    parser.add_argument(
        "--output-root",
        default="data/soymeal",
        help="输出根目录，默认 data/soymeal。",
    )
    parser.add_argument(
        "--contract-months",
        nargs="*",
        default=None,
        help="指定期权合约月份，例如 m2509 m2601；不传则使用当前挂牌月份。",
    )
    parser.add_argument(
        "--auto-months",
        action="store_true",
        help="按日期区间自动生成豆粕常见期权月份，适合历史批量下载。",
    )
    parser.add_argument("--month-start", default=None, help="自动生成合约月份起点，例如 201909。")
    parser.add_argument("--month-end", default=None, help="自动生成合约月份终点，例如 202607。")
    parser.add_argument("--strike-min", type=int, default=1800)
    parser.add_argument("--strike-max", type=int, default=5000)
    parser.add_argument("--strike-step", type=int, default=50)
    parser.add_argument(
        "--auto-strikes-from-underlying",
        action="store_true",
        help="按每个合约月份附近的 M0 历史价格自动生成行权价带，避免全区间蛮力探测。",
    )
    parser.add_argument(
        "--strike-window",
        type=int,
        default=600,
        help="自动行权价带宽度；例如 600 表示价格上下各 600。",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=0.25,
        help="每次请求后的等待秒数，避免触发源站限制。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="目标 parquet 已存在时跳过写入。",
    )
    return parser.parse_args()


def normalize_date(value):
    return pd.Timestamp(value).strftime("%Y%m%d")


def download_underlying(start, end, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = ak_call(ak.futures_zh_daily_sina, symbol=UNDERLYING_SYMBOL)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    df = df[["date", "open", "high", "low", "close", "volume", "settle"]].copy()
    for col in ["open", "high", "low", "close", "volume", "settle"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])

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
                    "settle": row["settle"],
                }
            ]
        )
        file_path = output_dir / f"{UNDERLYING_FILE_PREFIX}_{date:%Y-%m-%d}_price.parquet"
        payload.to_parquet(file_path, index=False)

    return df


def current_contract_months():
    df = ak_call(ak.option_commodity_contract_sina, symbol=COMMODITY_OPTION_SYMBOL)
    if df.empty:
        return []
    return [str(value) for value in df.iloc[:, 1].dropna()]


def auto_contract_months(start, end, month_start=None, month_end=None):
    """生成豆粕常见挂牌月份；实际是否存在由历史行情接口返回结果决定。"""
    start_period = pd.Period(str(month_start), freq="M") if month_start else pd.Period(pd.Timestamp(start), freq="M")
    end_period = pd.Period(str(month_end), freq="M") if month_end else pd.Period(pd.Timestamp(end) + pd.DateOffset(months=2), freq="M")
    listed_months = {1, 3, 5, 7, 8, 9, 11, 12}
    months = []
    for period in pd.period_range(start_period, end_period, freq="M"):
        if period.month in listed_months:
            months.append(f"m{period.year % 100:02d}{period.month:02d}")
    return months


def parse_contract_month(month):
    match = re.fullmatch(r"m(\d{2})(\d{2})", str(month))
    if match is None:
        raise ValueError(f"无法解析豆粕期权月份: {month}")
    return 2000 + int(match.group(1)), int(match.group(2))


def parse_option_code(code):
    match = re.fullmatch(r"m(\d{2})(\d{2})([CP])(\d+)", str(code))
    if match is None:
        raise ValueError(f"无法解析豆粕期权代码: {code}")
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    option_type = match.group(3).lower()
    strike = float(match.group(4))
    contract_month = f"m{year % 100:02d}{month:02d}"
    return year, month, option_type, strike, contract_month


def estimate_dce_option_expiry(contract_month, trading_calendar):
    """按大商所常规商品期权口径近似：标的月份前一月第 12 个交易日。"""
    year, month = parse_contract_month(contract_month)
    first_day = pd.Timestamp(year=year, month=month, day=1)
    prev_month = first_day - pd.DateOffset(months=1)
    month_days = trading_calendar[
        (trading_calendar.year == prev_month.year)
        & (trading_calendar.month == prev_month.month)
    ]
    if len(month_days) >= 12:
        return month_days[11]
    if len(month_days) > 0:
        return month_days[-1]
    return prev_month + pd.offsets.BDay(12)


def round_to_step(value, step, method):
    value = float(value) / step
    if method == "floor":
        return int(value // 1 * step)
    if method == "ceil":
        return int(-(-value // 1) * step)
    return int(round(value) * step)


def load_underlying_history():
    df = ak_call(ak.futures_zh_daily_sina, symbol=UNDERLYING_SYMBOL)
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["date", "close"])


def build_auto_month_strikes(
    months,
    start,
    end,
    strike_step,
    strike_window,
    strike_min,
    strike_max,
    trading_calendar,
):
    """按每个合约月份的相关标的价格区间生成行权价，主要覆盖 ATM 附近。"""
    underlying = load_underlying_history()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    result = {}
    for month in months:
        year, month_num = parse_contract_month(month)
        expiry = estimate_dce_option_expiry(month, trading_calendar)
        active_start = max(
            start_ts,
            pd.Timestamp(year=year, month=month_num, day=1) - pd.DateOffset(months=12),
        )
        active_end = min(end_ts, pd.Timestamp(expiry))
        sample = underlying[
            (underlying["date"] >= active_start) & (underlying["date"] <= active_end)
        ]
        if sample.empty:
            sample = underlying[
                (underlying["date"] >= start_ts) & (underlying["date"] <= end_ts)
            ]
        if sample.empty:
            low = strike_min
            high = strike_max
        else:
            low = max(strike_min, round_to_step(sample["close"].min() - strike_window, strike_step, "floor"))
            high = min(strike_max, round_to_step(sample["close"].max() + strike_window, strike_step, "ceil"))
        result[month] = list(range(low, high + 1, strike_step))
        print(
            f"[soymeal] {month} strike range {low}-{high} "
            f"({len(result[month])} strikes)",
            flush=True,
        )
    return result


def build_contract_codes(months, strike_min, strike_max, strike_step, month_strikes=None):
    strikes = range(strike_min, strike_max + 1, strike_step)
    for month in months:
        strikes_for_month = month_strikes.get(month, strikes) if month_strikes else strikes
        for strike in strikes_for_month:
            yield f"{month}C{strike}"
            yield f"{month}P{strike}"


def fetch_option_history(code, start, end, trading_calendar):
    try:
        # 不存在的历史合约会直接 JSONDecodeError，重试没有意义，会显著拖慢全历史探测。
        df = ak_call(ak.option_commodity_hist_sina, symbol=code, retries=1)
    except Exception as exc:
        print(f"[soymeal] skip {code}: {type(exc).__name__}", flush=True)
        return pd.DataFrame()
    if df.empty:
        return df

    _, _, option_type, strike, contract_month = parse_option_code(code)
    expiry = estimate_dce_option_expiry(contract_month, trading_calendar)

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
    if df.empty:
        return df

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close", "volume"])
    if df.empty:
        return df

    df = df.assign(
        order_book_id=code,
        strike_price=strike,
        maturity_date=pd.Timestamp(expiry),
        option_type=option_type,
        bid=df["close"],
        ask=df["close"],
        contract_multiplier=CONTRACT_MULTIPLIER,
        source="akshare_sina_commodity_option_hist",
    )
    return df[
        [
            "date",
            "order_book_id",
            "strike_price",
            "maturity_date",
            "option_type",
            "bid",
            "ask",
            "volume",
            "contract_multiplier",
            "close",
            "open",
            "high",
            "low",
            "source",
        ]
    ]


def write_option_history(hist, output_dir, skip_existing):
    """按日期增量写入，便于长任务中断后继续。"""
    if hist.empty:
        return 0

    written = 0
    for date, day_df in hist.groupby("date"):
        file_path = output_dir / f"{OPTION_FILE_PREFIX}_{date:%Y-%m-%d}_chain.parquet"
        payload = day_df.drop(columns=["date"]).copy()
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            if skip_existing:
                existing_codes = set(existing["order_book_id"].astype(str))
                payload = payload[
                    ~payload["order_book_id"].astype(str).isin(existing_codes)
                ]
                if payload.empty:
                    continue
            payload = pd.concat([existing, payload], ignore_index=True)
            payload = payload.drop_duplicates(
                subset=["order_book_id", "strike_price", "maturity_date", "option_type"],
                keep="last",
            )
        payload = payload.sort_values(
            ["maturity_date", "strike_price", "option_type", "order_book_id"]
        )
        payload.to_parquet(file_path, index=False)
        written += len(day_df)
    return written


def download_options(
    start,
    end,
    output_dir,
    months,
    strike_min,
    strike_max,
    strike_step,
    auto_strikes_from_underlying,
    strike_window,
    request_sleep,
    skip_existing,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    trading_calendar = pd.DatetimeIndex(
        pd.to_datetime(ak_call(ak.futures_zh_daily_sina, symbol=UNDERLYING_SYMBOL)["date"])
    ).sort_values()

    month_strikes = None
    if auto_strikes_from_underlying:
        month_strikes = build_auto_month_strikes(
            months,
            start,
            end,
            strike_step,
            strike_window,
            strike_min,
            strike_max,
            trading_calendar,
        )

    total_rows = 0
    total_written = 0
    active_contracts = 0
    codes = list(build_contract_codes(months, strike_min, strike_max, strike_step, month_strikes))
    print(f"[soymeal] option codes to probe: {len(codes)}", flush=True)
    for idx, code in enumerate(codes, start=1):
        hist = fetch_option_history(code, start, end, trading_calendar)
        if not hist.empty:
            active_contracts += 1
            total_rows += len(hist)
            written = write_option_history(hist, output_dir, skip_existing)
            total_written += written
            print(
                f"[soymeal] {idx}/{len(codes)} {code}: "
                f"rows={len(hist)}, written={written}",
                flush=True,
            )
        elif idx % 100 == 0:
            print(f"[soymeal] {idx}/{len(codes)} probed", flush=True)
        time.sleep(request_sleep)

    if total_rows == 0:
        print("[soymeal] no option rows downloaded", flush=True)
        return pd.DataFrame()

    print(
        f"[soymeal] active_contracts={active_contracts}, "
        f"rows={total_rows}, written={total_written}",
        flush=True,
    )
    return pd.DataFrame(
        [{"active_contracts": active_contracts, "rows": total_rows, "written": total_written}]
    )


def main():
    args = parse_args()
    start = normalize_date(args.start)
    end = normalize_date(args.end)
    output_root = Path(args.output_root)
    tasks = set(args.tasks)
    if "all" in tasks:
        tasks = {"underlying", "option"}

    if "underlying" in tasks:
        underlying_df = download_underlying(start, end, output_root / "underlying")
        print(
            f"[soymeal] underlying rows={len(underlying_df)}, "
            f"range={underlying_df['date'].min()} - {underlying_df['date'].max()}",
            flush=True,
        )

    if "option" in tasks:
        if args.contract_months:
            months = args.contract_months
        elif args.auto_months:
            months = auto_contract_months(
                start,
                end,
                month_start=args.month_start,
                month_end=args.month_end,
            )
        else:
            months = current_contract_months()
        print(f"[soymeal] contract months: {months}", flush=True)
        options = download_options(
            start,
            end,
            output_root / "option",
            months,
            args.strike_min,
            args.strike_max,
            args.strike_step,
            args.auto_strikes_from_underlying,
            args.strike_window,
            args.request_sleep,
            args.skip_existing,
        )
        if not options.empty:
            if {"date", "order_book_id"}.issubset(options.columns):
                print(
                    f"[soymeal] option rows={len(options)}, "
                    f"dates={options['date'].nunique()}, "
                    f"contracts={options['order_book_id'].nunique()}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
