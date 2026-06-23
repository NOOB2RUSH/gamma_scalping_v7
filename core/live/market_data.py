from __future__ import annotations

import json
import re
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
LIVE_PRODUCTS = frozenset(SSE_ETF_OPTION_SPECS)


def require_live_product(product):
    if product not in LIVE_PRODUCTS:
        raise ValueError(
            f"Live trading currently supports: {', '.join(sorted(LIVE_PRODUCTS))}"
        )
    return SSE_ETF_OPTION_SPECS[product]


def option_underlying_order_book_id(product):
    return require_live_product(product).etf_file_prefix


def attach_live_underlying_id(product, chain_df):
    result = chain_df.copy()
    result["underlying_order_book_id"] = option_underlying_order_book_id(product)
    return result


def fetch_historical_option_metadata(product, date, codes=None):
    """Return exact-date SSE option metadata from AKShare risk indicators."""
    spec = require_live_product(product)
    target = pd.Timestamp(date).normalize()
    wanted = None if codes is None else {str(code) for code in codes}
    cache_path = storage.historical_option_metadata_cache_path(product)
    cached = _read_historical_option_metadata_cache(cache_path)
    matched = cached[cached["date"].eq(target)] if not cached.empty else cached
    if wanted is not None:
        matched = matched[matched["order_book_id"].astype(str).isin(wanted)]
    cached_metadata = _historical_option_metadata_rows_to_dict(matched)
    if wanted is not None and wanted.issubset(cached_metadata):
        return cached_metadata

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for historical option metadata.") from exc

    risk_df = _ak_call(
        ak.option_risk_indicator_sse,
        date=target.strftime("%Y%m%d"),
    )
    required = {"SECURITY_ID", "CONTRACT_ID", "CONTRACT_SYMBOL"}
    missing = required - set(risk_df.columns)
    if missing:
        raise ValueError(f"AKShare option risk data is missing columns: {sorted(missing)}")

    rows = risk_df[
        risk_df["CONTRACT_ID"].astype(str).str.startswith(spec.etf_symbol)
    ].copy()
    if wanted is not None:
        rows = rows[rows["SECURITY_ID"].astype(str).isin(wanted)]

    fetched_rows = []
    trading_calendar = _load_local_trading_calendar()
    for _, row in rows.iterrows():
        parsed = _parse_sse_option_contract(row["CONTRACT_ID"], spec.etf_symbol)
        if parsed is None:
            continue
        option_type, maturity, raw_strike = parsed
        maturity = _adjust_to_trading_day(maturity, trading_calendar)
        code = str(row["SECURITY_ID"])
        daily = _fetch_historical_option_daily_quote(ak, code, target)
        fetched_rows.append(
            {
                "date": target,
                "order_book_id": code,
                "strike": _parse_contract_symbol_strike(
                    row["CONTRACT_SYMBOL"],
                    raw_strike,
                ),
                "expiry": str(pd.Timestamp(maturity).date()),
                "option_type": option_type,
                "contract_multiplier": 10000,
                "contract_symbol": row["CONTRACT_SYMBOL"],
                "underlying_order_book_id": spec.etf_file_prefix,
                "metadata_source": "akshare_option_risk_indicator_sse",
                "bid": daily.get("close"),
                "ask": daily.get("close"),
                "close": daily.get("close"),
                "volume": daily.get("volume"),
                "daily_data_source": daily.get("source"),
                "fetched_at": storage.utc_now_text(),
            }
        )
    if fetched_rows:
        fetched = pd.DataFrame(fetched_rows)
        updated = fetched if cached.empty else pd.concat([cached, fetched], ignore_index=True)
        updated = updated.drop_duplicates(
            subset=["date", "order_book_id"],
            keep="last",
        ).sort_values(["date", "order_book_id"])
        updated.to_csv(cache_path, index=False)
        cached_metadata.update(_historical_option_metadata_rows_to_dict(fetched))
    return cached_metadata


def _read_historical_option_metadata_cache(path):
    columns = [
        "date",
        "order_book_id",
        "strike",
        "expiry",
        "option_type",
        "contract_multiplier",
        "contract_symbol",
        "underlying_order_book_id",
        "metadata_source",
        "bid",
        "ask",
        "close",
        "volume",
        "daily_data_source",
        "fetched_at",
    ]
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    required = {
        "date",
        "order_book_id",
        "strike",
        "expiry",
        "option_type",
        "contract_multiplier",
        "underlying_order_book_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Historical option metadata cache is missing columns: {sorted(missing)}"
        )
    for column in set(columns) - set(df.columns):
        df[column] = None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[columns]


def _historical_option_metadata_rows_to_dict(rows):
    result = {}
    for _, row in rows.iterrows():
        code = str(row["order_book_id"])
        result[code] = {
            "strike": float(row["strike"]),
            "expiry": str(pd.Timestamp(row["expiry"]).date()),
            "option_type": str(row["option_type"]).upper(),
            "contract_multiplier": int(row["contract_multiplier"]),
            "contract_symbol": row.get("contract_symbol"),
            "underlying_order_book_id": row.get("underlying_order_book_id"),
            "metadata_source": row.get("metadata_source"),
            "bid": _number(row.get("bid")),
            "ask": _number(row.get("ask")),
            "close": _number(row.get("close")),
            "volume": _number(row.get("volume")),
            "daily_data_source": row.get("daily_data_source"),
        }
    return result


def _fetch_historical_option_daily_quote(ak, code, date):
    try:
        daily = _ak_call(ak.option_sse_daily_sina, symbol=code)
    except Exception:
        return {"close": None, "volume": None, "source": "akshare_daily_unavailable"}
    if daily.empty:
        return {"close": None, "volume": None, "source": "akshare_daily_empty"}

    date_col = _first_column(daily, "日期", "date")
    close_col = _first_column(daily, "收盘", "close")
    volume_col = _first_column(daily, "成交量", "volume")
    if date_col is None or close_col is None:
        return {
            "close": None,
            "volume": None,
            "source": "akshare_daily_missing_columns",
        }
    rows = daily.copy()
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.normalize()
    rows = rows[rows[date_col].eq(pd.Timestamp(date).normalize())]
    if rows.empty:
        return {"close": None, "volume": None, "source": "akshare_daily_date_missing"}
    row = rows.iloc[-1]
    return {
        "close": _number(row.get(close_col)),
        "volume": _number(row.get(volume_col)) if volume_col is not None else None,
        "source": "akshare_option_sse_daily_sina",
    }


def fetch_historical_atm_strike(product, date):
    """Return the exact-date ATM strike from AKShare, with a persistent cache."""
    require_live_product(product)

    target = pd.Timestamp(date).normalize()
    cache_path = storage.historical_atm_cache_path(product)
    cached = _read_historical_atm_cache(cache_path)
    matched = cached[cached["date"].eq(target)] if not cached.empty else cached
    if not matched.empty:
        row = matched.iloc[-1]
        return {
            "date": target,
            "spot": float(row["spot"]),
            "strike": float(row["strike"]),
            "source": "akshare_historical_atm_cache",
        }

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for historical ATM fallback.") from exc

    config = load_product_config(product)
    spec = SSE_ETF_OPTION_SPECS[product]
    spot = _fetch_historical_etf_close(ak, spec.etf_symbol, target)
    risk_df = _ak_call(
        ak.option_risk_indicator_sse,
        date=target.strftime("%Y%m%d"),
    )
    strike = _select_historical_atm_strike(risk_df, spec, target, spot, config.vol)
    row = pd.DataFrame(
        [
            {
                "date": target,
                "spot": spot,
                "strike": strike,
                "fetched_at": storage.utc_now_text(),
            }
        ]
    )
    updated = row if cached.empty else pd.concat([cached, row], ignore_index=True)
    updated = updated.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(cache_path, index=False)
    return {
        "date": target,
        "spot": float(spot),
        "strike": float(strike),
        "source": "akshare_historical_market_data",
    }


def _read_historical_atm_cache(path):
    columns = ["date", "spot", "strike", "fetched_at"]
    if not Path(path).exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Historical ATM cache is missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    return df.dropna(subset=["date", "spot", "strike"])


def _fetch_historical_etf_close(ak, etf_symbol, date):
    date_text = pd.Timestamp(date).strftime("%Y%m%d")
    try:
        raw_df = _ak_call(
            ak.fund_etf_hist_em,
            symbol=etf_symbol,
            period="daily",
            start_date=date_text,
            end_date=date_text,
            adjust="",
        )
    except Exception:
        raw_df = _ak_call(ak.fund_etf_hist_sina, symbol=f"sh{etf_symbol}")

    date_col = _first_column(raw_df, "\u65e5\u671f", "date")
    close_col = _first_column(raw_df, "\u6536\u76d8", "close")
    if date_col is None or close_col is None:
        raise ValueError(f"AKShare ETF history has no date/close columns: {etf_symbol}")
    rows = raw_df.copy()
    rows[date_col] = pd.to_datetime(rows[date_col], errors="coerce").dt.normalize()
    rows = rows[rows[date_col].eq(pd.Timestamp(date).normalize())]
    if rows.empty:
        raise ValueError(f"AKShare ETF history is empty for {etf_symbol} {date_text}")
    close = pd.to_numeric(rows[close_col], errors="coerce").dropna()
    if close.empty:
        raise ValueError(f"AKShare ETF close is invalid for {etf_symbol} {date_text}")
    return float(close.iloc[-1])


def _select_historical_atm_strike(risk_df, spec, date, spot, vol_config):
    required = {"CONTRACT_ID", "CONTRACT_SYMBOL"}
    missing = required - set(risk_df.columns)
    if missing:
        raise ValueError(f"AKShare option risk data is missing columns: {sorted(missing)}")

    rows = risk_df[
        risk_df["CONTRACT_ID"].astype(str).str.startswith(spec.etf_symbol)
    ].copy()
    parsed = rows["CONTRACT_ID"].astype(str).map(
        lambda value: _parse_sse_option_contract(value, spec.etf_symbol)
    )
    rows["option_type"] = parsed.map(lambda value: value[0] if value else None)
    rows["maturity_date"] = parsed.map(lambda value: value[1] if value else pd.NaT)
    rows["strike"] = [
        _parse_contract_symbol_strike(symbol, parsed_value[2] if parsed_value else None)
        for symbol, parsed_value in zip(rows["CONTRACT_SYMBOL"], parsed)
    ]
    rows["dte"] = (pd.to_datetime(rows["maturity_date"]) - pd.Timestamp(date)).dt.days
    rows = rows[
        rows["dte"].between(
            int(vol_config.atm_target_dte_min),
            int(vol_config.atm_target_dte_max),
        )
    ].dropna(subset=["strike", "option_type", "maturity_date"])
    pairs = (
        rows.groupby(["strike", "maturity_date"])["option_type"]
        .nunique()
        .reset_index(name="option_type_count")
    )
    pairs = pairs[pairs["option_type_count"].ge(2)].copy()
    if pairs.empty:
        raise ValueError(
            f"AKShare option risk data has no valid call/put ATM pair for "
            f"{spec.etf_symbol} {pd.Timestamp(date).date()}"
        )
    pairs["spot_diff"] = (pairs["strike"] - float(spot)).abs()
    pairs["dte"] = (pairs["maturity_date"] - pd.Timestamp(date)).dt.days
    pairs["target_dte_diff"] = (
        pairs["dte"] - int(vol_config.atm_target_dte)
    ).abs()
    selected = pairs.sort_values(
        ["spot_diff", "target_dte_diff", "dte", "strike"]
    ).iloc[0]
    return float(selected["strike"])


def _parse_sse_option_contract(contract_id, etf_symbol):
    match = re.fullmatch(
        rf"{re.escape(etf_symbol)}([CP])(\d{{2}})(\d{{2}})([A-Z]?)(\d+)",
        str(contract_id),
    )
    if match is None:
        return None
    option_type = match.group(1)
    year = 2000 + int(match.group(2))
    month = int(match.group(3))
    maturity = _fourth_wednesday(year, month)
    raw_strike = float(match.group(5)) / 1000
    return option_type, pd.Timestamp(maturity), raw_strike


def _parse_contract_symbol_strike(contract_symbol, fallback):
    match = re.search(r"(\d+(?:\.\d+)?)[A-Z]?$", str(contract_symbol))
    if match is None:
        return fallback
    value = float(match.group(1))
    return value / 1000 if value > 100 else value


def _first_column(df, *candidates):
    return next((candidate for candidate in candidates if candidate in df.columns), None)


def fetch_quote_snapshot(product, source="local", date="latest"):
    """Save one quote snapshot and return its metadata.

    `source=local` snapshots existing parquet files. `source=akshare` pulls the
    latest SSE ETF option quote through AKShare and writes immutable live quote
    snapshots. Canonical research/backtest parquet files are not modified by
    live quote pulls.
    """
    require_live_product(product)
    if source == "akshare":
        return _fetch_akshare_sse_snapshot(product)
    if source == "snapshot":
        return load_latest_quote_snapshot(product, date=date)
    if source != "local":
        raise ValueError("source must be 'local', 'snapshot', or 'akshare'.")

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


def load_latest_quote_snapshot(product, date="latest"):
    """Return the newest complete saved AKShare snapshot without network I/O."""
    require_live_product(product)
    quote_root = storage.PROJECT_ROOT / "data" / "live" / product / "quotes"
    if date in {None, "", "latest"}:
        day_dirs = sorted(
            (path for path in quote_root.glob("20??????") if path.is_dir()),
            reverse=True,
        )
        expected_date = None
    else:
        expected_date = pd.Timestamp(date).strftime("%Y-%m-%d")
        day_dirs = [quote_root / pd.Timestamp(date).strftime("%Y%m%d")]

    for day_dir in day_dirs:
        for metadata_path in sorted(day_dir.glob("*_metadata.json"), reverse=True):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if metadata.get("source") != "akshare":
                continue
            quote_date = str(metadata.get("quote_date") or "")
            if expected_date is not None and quote_date != expected_date:
                continue
            time_part = metadata_path.name.removesuffix("_metadata.json")
            etf_path = day_dir / f"{time_part}_etf.parquet"
            option_path = day_dir / f"{time_part}_option_chain.parquet"
            if not etf_path.exists() or not option_path.exists():
                continue
            result = dict(metadata)
            result["source"] = "snapshot"
            result["snapshot_source"] = "akshare"
            result["metadata_path"] = str(metadata_path)
            result["etf_snapshot"] = str(etf_path)
            result["option_snapshot"] = str(option_path)
            return result

    date_text = "latest" if expected_date is None else expected_date
    raise FileNotFoundError(
        f"No complete saved AKShare quote snapshot for {product} {date_text}. "
        "Fetch and save an AKShare snapshot first."
    )


def load_previous_quote_snapshot(product, before_date):
    """Return the newest complete AKShare snapshot before a valuation date."""
    require_live_product(product)
    cutoff = pd.Timestamp(before_date).normalize()
    quote_root = storage.PROJECT_ROOT / "data" / "live" / product / "quotes"
    day_dirs = sorted(
        (
            path
            for path in quote_root.glob("20??????")
            if path.is_dir()
            and pd.Timestamp(path.name).normalize() < cutoff
        ),
        reverse=True,
    )
    for day_dir in day_dirs:
        try:
            return load_latest_quote_snapshot(product, date=day_dir.name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"No complete saved AKShare quote snapshot for {product} before "
        f"{cutoff.date()}."
    )


def _fetch_akshare_sse_snapshot(product):
    require_live_product(product)

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for source='akshare'.") from exc

    config = load_product_config(product)
    spec = SSE_ETF_OPTION_SPECS[product]
    etf_df, quote_date = _fetch_latest_etf_bar(ak, spec.etf_symbol)
    trading_calendar = _refresh_akshare_trading_calendar(ak)
    chain_df = _fetch_sse_option_chain(ak, spec, trading_calendar, quote_date)
    chain_df = attach_live_underlying_id(product, chain_df)
    chain_df = chain_df.sort_values(
        ["maturity_date", "strike_price", "option_type", "order_book_id"]
    ).reset_index(drop=True)

    date_text = quote_date.strftime("%Y-%m-%d")
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
        "etf_canonical": None,
        "option_canonical": None,
        "canonical_written": False,
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


def _fetch_sse_option_chain(ak, spec, trading_calendar, quote_date=None):
    try:
        tasks = _sse_option_tasks_from_list_sina(ak, spec, trading_calendar)
    except Exception:
        if quote_date is None:
            raise
        tasks = _sse_option_tasks_from_risk_indicator(
            ak,
            spec,
            trading_calendar,
            quote_date,
        )
    if not tasks:
        raise ValueError(f"AKShare returned no SSE option codes for {spec.etf_symbol}.")

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


def _sse_option_tasks_from_list_sina(ak, spec, trading_calendar):
    tasks = []
    months = [
        str(month)
        for month in _ak_call(ak.option_sse_list_sina, symbol=spec.etf_symbol)
        if month is not None and str(month).strip()
    ]
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
    return tasks


def _sse_option_tasks_from_risk_indicator(ak, spec, trading_calendar, quote_date):
    risk_df = _ak_call(
        ak.option_risk_indicator_sse,
        date=pd.Timestamp(quote_date).strftime("%Y%m%d"),
    )
    required = {"SECURITY_ID", "CONTRACT_ID"}
    missing = required - set(risk_df.columns)
    if missing:
        raise ValueError(
            f"AKShare option risk data is missing columns: {sorted(missing)}"
        )
    tasks = []
    for _, row in risk_df.iterrows():
        contract_id = str(row.get("CONTRACT_ID") or "")
        if not contract_id.startswith(spec.etf_symbol):
            continue
        parsed = _parse_sse_option_contract(contract_id, spec.etf_symbol)
        if parsed is None:
            continue
        option_type, maturity, _ = parsed
        maturity = _adjust_to_trading_day(maturity, trading_calendar)
        tasks.append((str(row["SECURITY_ID"]), maturity, option_type))
    return tasks


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


def load_live_trading_calendar():
    """Load canonical and cached AKShare trading dates without network I/O."""
    local = _load_local_trading_calendar()
    path = storage.live_trading_calendar_path()
    if not path.exists():
        return local
    try:
        cached = pd.read_csv(path, encoding="utf-8-sig")
        dates = pd.to_datetime(cached["trade_date"], errors="coerce").dropna()
    except (OSError, KeyError, ValueError):
        return local
    return pd.DatetimeIndex(sorted(set(local) | set(dates))).normalize()


def _refresh_akshare_trading_calendar(ak):
    try:
        frame = _ak_call(ak.tool_trade_date_hist_sina)
        date_column = "trade_date" if "trade_date" in frame.columns else frame.columns[0]
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
        calendar = pd.DatetimeIndex(dates).normalize().drop_duplicates().sort_values()
        pd.DataFrame({"trade_date": calendar}).to_csv(
            storage.live_trading_calendar_path(),
            index=False,
            encoding="utf-8-sig",
        )
        return calendar
    except Exception:
        return load_live_trading_calendar()


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
