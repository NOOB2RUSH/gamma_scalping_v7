from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import storage
from .runtime import load_product_config, project_path


CALL_TEXT = "\u770b\u6da8\u671f\u6743"
PUT_TEXT = "\u770b\u8dcc\u671f\u6743"
AKSHARE_OPTION_WORKERS = 8


@dataclass(frozen=True)
class SseEtfOptionSpec:
    etf_symbol: str
    etf_file_prefix: str
    option_file_prefix: str


SSE_ETF_OPTION_SPECS = {
    "50etf": SseEtfOptionSpec("510050", "510050.XSHG", "510050.XSHG"),
    "300etf": SseEtfOptionSpec("510300", "510300.XSHG", "300ETF_OPTION"),
    "500etf": SseEtfOptionSpec("510500", "510500.XSHG", "500ETF_OPTION"),
    "kc50etf": SseEtfOptionSpec("588000", "588000.XSHG", "KC50ETF_OPTION"),
}


def fetch_quote_snapshot(product, source="local", date="latest"):
    """Save one quote snapshot and return its metadata.

    `source=local` snapshots existing parquet files. `source=akshare` pulls the
    latest SSE ETF option quote through AKShare, writes canonical daily parquet
    files, and also writes immutable live quote snapshots.
    """
    if source == "akshare":
        return _fetch_akshare_sse_snapshot(product)
    if source != "local":
        raise ValueError("source must be 'local' or 'akshare'.")

    config = load_product_config(product)
    etf_dir = project_path(config.data.etf_dir)
    opt_dir = project_path(config.data.opt_dir)
    snapshot_date = _resolve_snapshot_date(etf_dir, opt_dir, date)

    etf_file = _find_file_by_date(etf_dir, snapshot_date, "_price")
    opt_file = _find_file_by_date(opt_dir, snapshot_date, "_chain")
    if etf_file is None or opt_file is None:
        raise FileNotFoundError(
            f"Missing local quote files for {product} {snapshot_date.date()}"
        )

    out_dir, time_part, stamp = storage.quote_snapshot_dir(product)
    etf_out = out_dir / f"{time_part}_etf.parquet"
    opt_out = out_dir / f"{time_part}_option_chain.parquet"
    shutil.copy2(etf_file, etf_out)
    shutil.copy2(opt_file, opt_out)

    metadata = {
        "product": product,
        "source": source,
        "snapshot_stamp": stamp,
        "quote_date": snapshot_date.strftime("%Y-%m-%d"),
        "etf_source": str(etf_file),
        "option_source": str(opt_file),
        "etf_snapshot": str(etf_out),
        "option_snapshot": str(opt_out),
    }
    storage.write_json(out_dir / f"{time_part}_metadata.json", metadata)
    return metadata


def _fetch_akshare_sse_snapshot(product):
    if product not in SSE_ETF_OPTION_SPECS:
        raise ValueError(
            f"AKShare live source currently supports: "
            f"{', '.join(sorted(SSE_ETF_OPTION_SPECS))}"
        )

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for source='akshare'.") from exc

    config = load_product_config(product)
    spec = SSE_ETF_OPTION_SPECS[product]
    etf_df, quote_date = _fetch_latest_etf_bar(ak, spec.etf_symbol)
    trading_calendar = _load_local_trading_calendar()
    chain_df = _fetch_sse_option_chain(ak, spec, trading_calendar)
    chain_df = chain_df.sort_values(
        ["maturity_date", "strike_price", "option_type", "order_book_id"]
    ).reset_index(drop=True)

    etf_dir = project_path(config.data.etf_dir)
    opt_dir = project_path(config.data.opt_dir)
    etf_dir.mkdir(parents=True, exist_ok=True)
    opt_dir.mkdir(parents=True, exist_ok=True)

    date_text = quote_date.strftime("%Y-%m-%d")
    etf_canonical = etf_dir / f"{spec.etf_file_prefix}_{date_text}_price.parquet"
    opt_canonical = opt_dir / f"{spec.option_file_prefix}_{date_text}_chain.parquet"
    etf_df.to_parquet(etf_canonical, index=False)
    chain_df.to_parquet(opt_canonical, index=False)

    out_dir, time_part, stamp = storage.quote_snapshot_dir(product)
    etf_out = out_dir / f"{time_part}_etf.parquet"
    opt_out = out_dir / f"{time_part}_option_chain.parquet"
    etf_df.to_parquet(etf_out, index=False)
    chain_df.to_parquet(opt_out, index=False)

    quote_times = [
        value
        for value in chain_df.get("quote_time", pd.Series(dtype=object)).dropna().unique()
    ]
    metadata = {
        "product": product,
        "source": "akshare",
        "snapshot_stamp": stamp,
        "quote_date": date_text,
        "etf_symbol": spec.etf_symbol,
        "etf_canonical": str(etf_canonical),
        "option_canonical": str(opt_canonical),
        "etf_snapshot": str(etf_out),
        "option_snapshot": str(opt_out),
        "option_rows": len(chain_df),
        "option_quote_times": [str(value) for value in sorted(quote_times)],
    }
    storage.write_json(out_dir / f"{time_part}_metadata.json", metadata)
    return metadata


def _fetch_latest_etf_bar(ak, etf_symbol):
    try:
        spot_df = _ak_call(
            ak.option_sse_underlying_spot_price_sina,
            symbol=f"sh{etf_symbol}",
        )
        field_map = _field_map(spot_df)
        quote_date = pd.Timestamp(field_map["行情日期"]).normalize()
        row = pd.DataFrame(
            [
                {
                    "open": _number(field_map.get("今日开盘价")),
                    "high": _number(field_map.get("最高成交价")),
                    "low": _number(field_map.get("最低成交价")),
                    "close": _number(field_map.get("最近成交价")),
                    "volume": _number(field_map.get("成交数量"), 0),
                    "amount": _number(field_map.get("成交金额"), 0),
                }
            ]
        )
        if row[["open", "high", "low", "close"]].notna().all(axis=None):
            return row, quote_date
    except Exception:
        pass

    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y%m%d")
    try:
        raw_df = _ak_call(
            ak.fund_etf_hist_em,
            symbol=etf_symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="",
        )
        source = "akshare_fund_etf_hist_em"
    except Exception:
        raw_df = _ak_call(ak.fund_etf_hist_sina, symbol=f"sh{etf_symbol}")
        source = "akshare_fund_etf_hist_sina"

    if raw_df.empty:
        raise ValueError(f"AKShare ETF history is empty: {etf_symbol}")

    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = raw_df.rename(columns=rename_map).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    row = df.iloc[[-1]][["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        row[col] = pd.to_numeric(row[col], errors="coerce")
    if source.endswith("sina"):
        row["volume"] = row["volume"] / 100
    quote_date = pd.Timestamp(row["date"].iloc[0]).normalize()
    return row.drop(columns=["date"]).reset_index(drop=True), quote_date


def _fetch_sse_option_chain(ak, spec, trading_calendar):
    tasks = []
    months = _ak_call(ak.option_sse_list_sina, symbol=spec.etf_symbol)
    for month in months:
        maturity = _adjust_to_trading_day(
            _fourth_wednesday(int(str(month)[:4]), int(str(month)[4:6])),
            trading_calendar,
        )
        for symbol_text, option_type in [(CALL_TEXT, "C"), (PUT_TEXT, "P")]:
            codes = _ak_call(
                ak.option_sse_codes_sina,
                symbol=symbol_text,
                trade_date=month,
                underlying=spec.etf_symbol,
            )
            if codes.empty:
                continue
            code_col = codes.columns[1]
            for code in codes[code_col].astype(str):
                tasks.append((code, maturity, option_type))

    rows = []
    with ThreadPoolExecutor(max_workers=AKSHARE_OPTION_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_sse_option_row, ak, code, maturity, option_type): (
                code,
                maturity,
                option_type,
            )
            for code, maturity, option_type in tasks
        }
        for future in as_completed(futures):
            code, maturity, option_type = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "order_book_id": code,
                        "strike_price": pd.NA,
                        "maturity_date": maturity,
                        "option_type": option_type,
                        "bid": pd.NA,
                        "ask": pd.NA,
                        "raw_sina_volume": 0,
                        "volume": 0,
                        "open_interest": 0,
                        "contract_multiplier": 10000,
                        "close": pd.NA,
                        "source": "akshare_sse_spot_price_sina_error",
                        "quote_error": repr(exc),
                    }
                )
    chain = pd.DataFrame(rows)
    if chain.empty:
        raise ValueError("AKShare returned empty SSE option chain.")
    return chain


def _fetch_sse_option_row(ak, code, maturity, option_type):
    spot_df = _ak_call(ak.option_sse_spot_price_sina, symbol=code)
    field_map = _field_map(spot_df)
    bid = _number(
        field_map.get("买价"),
        field_map.get("申买价一"),
        field_map.get("最新价"),
    )
    ask = _number(
        field_map.get("卖价"),
        field_map.get("申卖价一"),
        field_map.get("最新价"),
    )
    close = _number(field_map.get("最新价"), bid, ask)
    if (bid is None or bid <= 0) and close is not None:
        bid = close
    if (ask is None or ask <= 0) and close is not None:
        ask = close
    strike = _number(field_map.get("行权价"))
    volume = _number(field_map.get("成交量"), 0)
    open_interest = _number(field_map.get("持仓量"), 0)
    return {
        "order_book_id": str(code),
        "strike_price": strike,
        "maturity_date": pd.Timestamp(maturity),
        "option_type": option_type,
        "bid": bid,
        "ask": ask,
        "raw_sina_volume": volume,
        "volume": volume,
        "open_interest": open_interest,
        "contract_multiplier": 10000,
        "close": close,
        "source": "akshare_sse_spot_price_sina",
        "quote_time": field_map.get("行情时间"),
        "contract_symbol": field_map.get("期权合约简称"),
        "akshare_underlying_symbol": field_map.get("标的股票"),
    }


def _field_map(df):
    field_col = df.columns[0]
    value_col = df.columns[1]
    return {
        str(key).strip(): value
        for key, value in zip(df[field_col], df[value_col])
    }


def _number(*values):
    for value in values:
        if value is None or pd.isna(value):
            continue
        text = str(value).strip().replace(",", "")
        if text == "":
            continue
        try:
            return float(text)
        except ValueError:
            continue
    return None


def _ak_call(func, *args, retries=3, sleep=1.0, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(sleep)
    raise last_exc


def _load_local_trading_calendar():
    try:
        import core

        return core.data_loader.load_etf_trading_calendar()
    except Exception:
        return pd.DatetimeIndex([])


def _fourth_wednesday(year, month):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[days.month == month]
    wednesdays = [day for day in days if day.weekday() == 2]
    return wednesdays[3]


def _adjust_to_trading_day(date, trading_calendar):
    if trading_calendar is None or len(trading_calendar) == 0:
        return pd.Timestamp(date)
    eligible = pd.DatetimeIndex(trading_calendar)
    eligible = eligible[eligible >= pd.Timestamp(date)]
    if len(eligible) == 0:
        return pd.Timestamp(date)
    return eligible[0]


def latest_available_date(product):
    config = load_product_config(product)
    return _resolve_snapshot_date(
        project_path(config.data.etf_dir),
        project_path(config.data.opt_dir),
        "latest",
    )


def _resolve_snapshot_date(etf_dir, opt_dir, date):
    etf_dates = _available_dates(etf_dir, "_price")
    opt_dates = _available_dates(opt_dir, "_chain")
    common_dates = sorted(set(etf_dates) & set(opt_dates))
    if not common_dates:
        raise ValueError("No common ETF/option quote date found.")

    if date == "latest" or date is None:
        return common_dates[-1]

    target = pd.Timestamp(date).normalize()
    if target not in set(common_dates):
        raise ValueError(f"{target.date()} is not available in both ETF and option data.")
    return target


def _available_dates(data_dir, suffix):
    return [
        _parse_date_from_file(path, suffix)
        for path in sorted(Path(data_dir).glob(f"*{suffix}.parquet"))
    ]


def _find_file_by_date(data_dir, date, suffix):
    for path in sorted(Path(data_dir).glob(f"*{suffix}.parquet")):
        if _parse_date_from_file(path, suffix) == pd.Timestamp(date).normalize():
            return path
    return None


def _parse_date_from_file(file_path, suffix):
    base = Path(file_path).stem.rsplit(suffix, 1)
    date_str = base[0].rsplit("_", 1)[1]
    return pd.Timestamp(date_str).normalize()
