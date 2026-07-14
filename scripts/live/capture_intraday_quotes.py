from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
import core
from core.live import account as account_store
from core.live import market_data, storage
from promote_quote_snapshots import promote_quote_snapshots


DEFAULT_DAILY_SCHEDULE = "10:30,11:35,14:00,15:10"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture intraday ETF and option quotes on an interval without "
            "touching live account state."
        )
    )
    parser.add_argument("--product", choices=core.config.available_live_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=None,
        help=(
            "Polling interval for legacy continuous mode. If omitted, the script "
            "uses the fixed daily schedule."
        ),
    )
    parser.add_argument(
        "--daily-schedule",
        default=DEFAULT_DAILY_SCHEDULE,
        help=(
            "Comma-separated local capture times in HH:MM. "
            f"Default: {DEFAULT_DAILY_SCHEDULE}."
        ),
    )
    parser.add_argument(
        "--option-code",
        action="append",
        default=[],
        help="Option contract code to capture. Can be specified multiple times.",
    )
    parser.add_argument(
        "--no-account-positions",
        action="store_true",
        help="Do not auto-add option codes from the local live account.",
    )
    parser.add_argument("--once", action="store_true", help="Run one capture and exit.")
    parser.add_argument(
        "--target-date",
        default=None,
        help=(
            "Capture minute data for a specific date YYYY-MM-DD. "
            "Default: current local date."
        ),
    )
    parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="Backfill missing available recent minute-data dates and exit.",
    )
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=None,
        help="Limit backfill discovery to the most recent N source-available dates.",
    )
    parser.add_argument(
        "--save-option-greeks-snapshot",
        action="store_true",
        help=(
            "Also save AKShare option Greeks snapshots. Disabled by default "
            "because strategy Greeks are calculated locally."
        ),
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: data/live/<product>/intraday/<YYYYMMDD>.",
    )
    parser.add_argument(
        "--pid-file",
        default=None,
        help="PID file for background operation. Default: output/live/<product>/intraday_capture.pid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval_seconds is not None and args.interval_seconds <= 0:
        raise ValueError("--interval-seconds must be positive")
    if args.backfill_days is not None and args.backfill_days <= 0:
        raise ValueError("--backfill-days must be positive")

    pid_file = _pid_file(args)
    _write_pid(pid_file)
    run_count = 0
    try:
        if args.backfill_missing:
            result = backfill_missing_dates(args)
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
            return
        while True:
            if not args.once and args.interval_seconds is None:
                next_capture = _next_scheduled_capture(args.daily_schedule)
                wait_seconds = max(0.0, (next_capture - pd.Timestamp.now()).total_seconds())
                print(
                    f"next_capture_at={next_capture.replace(microsecond=0).isoformat()} "
                    f"wait_seconds={wait_seconds:.0f}",
                    flush=True,
                )
                time.sleep(wait_seconds)

            run_count += 1
            result = capture_once(args)
            print(_format_result(result), flush=True)
            if args.once:
                break
            if args.max_runs is not None and run_count >= args.max_runs:
                break
            if args.interval_seconds is not None:
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("stopped_by=KeyboardInterrupt", flush=True)
    finally:
        _remove_pid(pid_file)


def capture_once(args):
    captured_at = pd.Timestamp.now().replace(microsecond=0)
    target_date = _target_date(args, captured_at)
    output_dir = _output_dir(args, captured_at, target_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for intraday quote capture.") from exc

    spec = market_data.SSE_ETF_OPTION_SPECS.get(args.product)
    if spec is None:
        raise ValueError(
            "Intraday capture currently supports SSE ETF option products: "
            f"{', '.join(sorted(market_data.SSE_ETF_OPTION_SPECS))}"
        )

    option_codes = _option_codes(args)
    result = {
        "captured_at": captured_at.isoformat(),
        "target_date": str(target_date),
        "output_dir": str(output_dir),
        "etf_symbol": spec.etf_symbol,
        "option_codes": option_codes,
        "quote_snapshot": None,
        "quote_promotion": None,
        "etf_rows": 0,
        "option_minute_rows": {},
        "option_spot_rows": {},
        "option_greeks_rows": {},
        "errors": [],
    }

    if target_date == captured_at.date():
        try:
            result["quote_snapshot"] = market_data.fetch_quote_snapshot(
                args.product,
                source="akshare",
                date="latest",
            )
            if captured_at.hour >= 15:
                result["quote_promotion"] = promote_quote_snapshots(
                    args.product,
                    dates=[captured_at.date()],
                )
        except Exception as exc:
            result["errors"].append(f"quote_snapshot:{type(exc).__name__}:{exc}")

    try:
        etf_frame = _fetch_etf_minute(
            ak,
            spec.etf_symbol,
            captured_at,
            target_date=target_date,
        )
        result["etf_rows"] = _append_dedup_csv(
            output_dir / f"etf_{spec.etf_symbol}_1m.csv",
            etf_frame,
            ["symbol", "timestamp"],
            timestamp_date=target_date,
        )
    except Exception as exc:
        result["errors"].append(f"etf:{type(exc).__name__}:{exc}")

    for code in option_codes:
        try:
            minute_frame = _fetch_option_minute(
                ak,
                code,
                captured_at,
                target_date=target_date,
            )
            result["option_minute_rows"][code] = _append_dedup_csv(
                output_dir / f"option_{code}_1m.csv",
                minute_frame,
                ["symbol", "timestamp"],
                timestamp_date=target_date,
            )
        except Exception as exc:
            result["errors"].append(f"option_minute:{code}:{type(exc).__name__}:{exc}")

        if target_date != captured_at.date():
            continue

        try:
            spot_frame = _fetch_option_snapshot(
                ak.option_sse_spot_price_sina,
                code,
                captured_at,
                source="option_sse_spot_price_sina",
            )
            result["option_spot_rows"][code] = _append_dedup_csv(
                output_dir / f"option_{code}_spot_snapshots.csv",
                spot_frame,
                ["symbol", "captured_at"],
            )
        except Exception as exc:
            result["errors"].append(f"option_spot:{code}:{type(exc).__name__}:{exc}")

        if args.save_option_greeks_snapshot:
            try:
                greeks_frame = _fetch_option_snapshot(
                    ak.option_sse_greeks_sina,
                    code,
                    captured_at,
                    source="option_sse_greeks_sina",
                )
                result["option_greeks_rows"][code] = _append_dedup_csv(
                    output_dir / f"option_{code}_greeks_snapshots.csv",
                    greeks_frame,
                    ["symbol", "captured_at"],
                )
            except Exception as exc:
                result["errors"].append(
                    f"option_greeks:{code}:{type(exc).__name__}:{exc}"
                )

    if not getattr(args, "output_dir", None):
        refresh_intraday_status(args.product, option_codes=option_codes)
    return result


def _option_codes(args):
    codes = [str(code).strip() for code in args.option_code if str(code).strip()]
    if not args.no_account_positions:
        codes.extend(_account_option_codes(args.product, args.account_id))
    return sorted(dict.fromkeys(codes))


def _account_option_codes(product, account_id):
    db_path = storage.account_db_path(product)
    if not Path(db_path).exists():
        return []
    live_account = account_store.load_account(product, account_id=account_id)
    codes = []
    for position in live_account.positions.values():
        if not position:
            continue
        for key in ["call_code", "put_code"]:
            value = position.get(key)
            if value:
                codes.append(str(value))
    return codes


def _fetch_etf_minute(ak, etf_symbol, captured_at):
    source = "stock_zh_a_minute"
    try:
        raw = ak.stock_zh_a_minute(symbol=f"sh{etf_symbol}", period="1", adjust="")
    except Exception:
        source = "fund_etf_hist_min_em"
        raw = ak.fund_etf_hist_min_em(symbol=etf_symbol, period="1", adjust="")

    if raw is None or raw.empty:
        raise ValueError(f"empty ETF minute data: {etf_symbol}")

    frame = raw.copy()
    if "day" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["day"])
    elif "时间" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["时间"])
    elif "日期时间" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["日期时间"])
    else:
        first_column = frame.columns[0]
        frame["timestamp"] = pd.to_datetime(frame[first_column])

    rename_map = {
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
    }
    frame = frame.rename(columns=rename_map)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in frame.columns:
            frame[column] = None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    result = frame[["timestamp", "open", "high", "low", "close", "volume", "amount"]].copy()
    result.insert(0, "symbol", etf_symbol)
    result["source"] = source
    result["captured_at"] = captured_at.isoformat()
    result = result.dropna(subset=["timestamp"])
    timestamp = pd.to_datetime(result["timestamp"], errors="coerce")
    result = result.loc[timestamp.dt.date == pd.Timestamp(captured_at).date()].copy()
    if result.empty:
        raise ValueError(f"no current-date ETF minute data: {etf_symbol}")
    return result


def _fetch_option_minute(ak, option_code, captured_at):
    raw = ak.option_finance_minute_sina(symbol=str(option_code))
    source = "option_finance_minute_sina"
    if not _option_minute_is_current(raw, captured_at):
        raw = ak.option_sse_minute_sina(symbol=str(option_code))
        source = "option_sse_minute_sina"
    if not _option_minute_is_current(raw, captured_at):
        raise ValueError(f"no current-date option minute data: {option_code}")

    frame = raw.copy()
    if {"date", "time"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="coerce",
        )
        rename_map = {
            "price": "price",
            "average_price": "average_price",
            "volume": "volume",
        }
    else:
        frame["timestamp"] = pd.to_datetime(
            frame["日期"].astype(str) + " " + frame["时间"].astype(str),
            errors="coerce",
        )
        rename_map = {
            "价格": "price",
            "均价": "average_price",
            "成交": "volume",
            "持仓": "open_interest",
        }
    frame = frame.rename(columns=rename_map)
    for column in ["price", "average_price", "volume", "open_interest"]:
        if column not in frame.columns:
            frame[column] = None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    result = frame[
        ["timestamp", "price", "average_price", "volume", "open_interest"]
    ].copy()
    result.insert(0, "symbol", str(option_code))
    result["source"] = source
    result["captured_at"] = captured_at.isoformat()
    return result.dropna(subset=["timestamp"])


def _option_minute_is_current(frame, captured_at):
    if frame is None or frame.empty:
        return False
    if {"date", "time"}.issubset(frame.columns):
        timestamp = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="coerce",
        )
    elif {"日期", "时间"}.issubset(frame.columns):
        timestamp = pd.to_datetime(
            frame["日期"].astype(str) + " " + frame["时间"].astype(str),
            errors="coerce",
        )
    elif {"日期", "时间"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["日期"].astype(str) + " " + frame["时间"].astype(str),
            errors="coerce",
        )
        rename_map = {
            "价格": "price",
            "均价": "average_price",
            "成交": "volume",
            "持仓": "open_interest",
        }
    else:
        return False
    latest = timestamp.max()
    return pd.notna(latest) and latest.date() == pd.Timestamp(captured_at).date()


def _fetch_option_snapshot(func, option_code, captured_at, source):
    raw = func(symbol=str(option_code))
    if raw is None or raw.empty:
        raise ValueError(f"empty option snapshot: {option_code}")
    row = _field_value_frame_to_row(raw)
    row.insert(0, "symbol", str(option_code))
    row.insert(1, "captured_at", captured_at.isoformat())
    row["source"] = source
    return row


def _field_value_frame_to_row(frame):
    field_col = "字段" if "字段" in frame.columns else frame.columns[0]
    value_col = "值" if "值" in frame.columns else frame.columns[1]
    payload = {}
    for _, row in frame.iterrows():
        key = str(row.get(field_col)).strip()
        if not key or key.lower() == "nan":
            continue
        payload[key] = row.get(value_col)
    return pd.DataFrame([payload])


def _append_dedup_csv(path, frame, subset, timestamp_date=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _normalize_quote_frame_for_storage(frame)
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        existing = _normalize_quote_frame_for_storage(existing)
        frame = pd.concat([existing, frame], ignore_index=True)
    if timestamp_date is not None and "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.loc[timestamp.dt.date == timestamp_date].copy()
    frame = frame.drop_duplicates(subset=subset, keep="last")
    sort_columns = [column for column in ["timestamp", "captured_at", "symbol"] if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return len(frame)


def _normalize_quote_frame_for_storage(frame):
    frame = frame.copy()
    if "symbol" in frame.columns:
        frame["symbol"] = frame["symbol"].astype(str)
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["timestamp"] = timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
    if "captured_at" in frame.columns:
        captured_at = pd.to_datetime(frame["captured_at"], errors="coerce")
        frame["captured_at"] = captured_at.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return frame


def _output_dir(args, captured_at):
    if args.output_dir:
        return Path(args.output_dir)
    date_part = captured_at.strftime("%Y%m%d")
    return Path(storage.PROJECT_ROOT) / "data" / "live" / args.product / "intraday" / date_part


def _pid_file(args):
    if args.pid_file:
        return Path(args.pid_file)
    return storage.output_dir(args.product) / "intraday_capture.pid"


def _next_scheduled_capture(schedule_text):
    now = pd.Timestamp.now()
    schedule = _parse_daily_schedule(schedule_text)
    for hour, minute in schedule:
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate > now:
            return candidate
    hour, minute = schedule[0]
    tomorrow = now + pd.Timedelta(days=1)
    return tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _parse_daily_schedule(schedule_text):
    result = []
    for item in str(schedule_text or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            hour_text, minute_text = item.split(":", 1)
            hour = int(hour_text)
            minute = int(minute_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --daily-schedule item: {item!r}; expected HH:MM"
            ) from exc
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            raise ValueError(
                f"Invalid --daily-schedule item: {item!r}; expected HH:MM"
            )
        result.append((hour, minute))
    if not result:
        raise ValueError("--daily-schedule must contain at least one HH:MM item")
    return sorted(set(result))


def _write_pid(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")


def _remove_pid(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _format_result(result):
    return (
        f"captured_at={result['captured_at']} "
        f"output_dir={result['output_dir']} "
        f"etf={result['etf_symbol']} etf_rows={result['etf_rows']} "
        f"options={','.join(result['option_codes']) or '-'} "
        f"quote_snapshot={bool(result['quote_snapshot'])} "
        f"quote_promotion={bool(result['quote_promotion'])} "
        f"option_minute_rows={result['option_minute_rows']} "
        f"errors={len(result['errors'])} "
        f"error_detail={result['errors']}"
    )


def _fetch_option_minute(ak, option_code, captured_at):
    raw = ak.option_finance_minute_sina(symbol=str(option_code))
    source = "option_finance_minute_sina"
    if not _option_minute_is_current(raw, captured_at):
        raw = ak.option_sse_minute_sina(symbol=str(option_code))
        source = "option_sse_minute_sina"
    if not _option_minute_is_current(raw, captured_at):
        raise ValueError(f"no current-date option minute data: {option_code}")

    frame = raw.copy()
    parsed = _parse_option_minute_frame(frame)
    if parsed is None:
        raise ValueError(f"unsupported option minute columns: {list(frame.columns)}")
    frame, rename_map = parsed
    frame = frame.rename(columns=rename_map)
    for column in ["price", "average_price", "volume", "open_interest"]:
        if column not in frame.columns:
            frame[column] = None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    result = frame[
        ["timestamp", "price", "average_price", "volume", "open_interest"]
    ].copy()
    result.insert(0, "symbol", str(option_code))
    result["source"] = source
    result["captured_at"] = captured_at.isoformat()
    return result.dropna(subset=["timestamp"])


def _parse_option_minute_frame(frame):
    frame = frame.copy()
    if {"date", "time"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="coerce",
        )
        return frame, {
            "price": "price",
            "average_price": "average_price",
            "volume": "volume",
        }
    if {"日期", "时间"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["日期"].astype(str) + " " + frame["时间"].astype(str),
            errors="coerce",
        )
        return frame, {
            "价格": "price",
            "均价": "average_price",
            "成交": "volume",
            "持仓": "open_interest",
        }
    mojibake_date = "鏃ユ湡"
    mojibake_time = "鏃堕棿"
    if {mojibake_date, mojibake_time}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame[mojibake_date].astype(str) + " " + frame[mojibake_time].astype(str),
            errors="coerce",
        )
        return frame, {
            "浠锋牸": "price",
            "鍧囦环": "average_price",
            "鎴愪氦": "volume",
            "鎸佷粨": "open_interest",
        }
    return None


def _option_minute_is_current(frame, captured_at):
    if frame is None or frame.empty:
        return False
    parsed = _parse_option_minute_frame(frame)
    if parsed is None:
        return False
    parsed_frame, _ = parsed
    latest = parsed_frame["timestamp"].max()
    return pd.notna(latest) and latest.date() == pd.Timestamp(captured_at).date()


def _target_date(args, captured_at):
    value = getattr(args, "target_date", None)
    if value:
        return pd.Timestamp(value).date()
    return pd.Timestamp(captured_at).date()


def _output_dir(args, captured_at, target_date=None):
    if args.output_dir:
        return Path(args.output_dir)
    date_part = pd.Timestamp(target_date or captured_at).strftime("%Y%m%d")
    return Path(storage.PROJECT_ROOT) / "data" / "live" / args.product / "intraday" / date_part


def _fetch_etf_minute(ak, etf_symbol, captured_at, target_date=None):
    target_date = pd.Timestamp(target_date or captured_at).date()
    errors = []
    sources = [
        (
            "stock_zh_a_minute",
            lambda: ak.stock_zh_a_minute(symbol=f"sh{etf_symbol}", period="1", adjust=""),
        ),
        (
            "fund_etf_hist_min_em",
            lambda: ak.fund_etf_hist_min_em(symbol=etf_symbol, period="1", adjust=""),
        ),
    ]
    for source, loader in sources:
        try:
            result = _normalize_etf_minute_frame(
                loader(),
                etf_symbol,
                captured_at,
                source,
                target_date,
            )
            if not result.empty:
                return result
        except Exception as exc:
            errors.append(f"{source}:{type(exc).__name__}:{exc}")
    raise ValueError(
        f"no ETF minute data for {etf_symbol} on {target_date}; errors={errors}"
    )


def _normalize_etf_minute_frame(raw, etf_symbol, captured_at, source, target_date):
    if raw is None or raw.empty:
        raise ValueError(f"empty ETF minute data: {etf_symbol}")

    frame = raw.copy()
    if "day" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["day"], errors="coerce")
    elif "\u65f6\u95f4" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["\u65f6\u95f4"], errors="coerce")
    elif "\u65e5\u671f\u65f6\u95f4" in frame.columns:
        frame["timestamp"] = pd.to_datetime(
            frame["\u65e5\u671f\u65f6\u95f4"],
            errors="coerce",
        )
    elif "鏃堕棿" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["鏃堕棿"], errors="coerce")
    elif "鏃ユ湡鏃堕棿" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["鏃ユ湡鏃堕棿"], errors="coerce")
    else:
        first_column = frame.columns[0]
        frame["timestamp"] = pd.to_datetime(frame[first_column], errors="coerce")

    rename_map = {
        "\u5f00\u76d8": "open",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6536\u76d8": "close",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "寮€鐩?": "open",
        "鏈€楂?": "high",
        "鏈€浣?": "low",
        "鏀剁洏": "close",
        "鎴愪氦閲?": "volume",
        "鎴愪氦棰?": "amount",
    }
    frame = frame.rename(columns=rename_map)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in frame.columns:
            frame[column] = None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    result = frame[
        ["timestamp", "open", "high", "low", "close", "volume", "amount"]
    ].copy()
    result.insert(0, "symbol", etf_symbol)
    result["source"] = source
    result["captured_at"] = captured_at.isoformat()
    result = result.dropna(subset=["timestamp"])
    timestamp = pd.to_datetime(result["timestamp"], errors="coerce")
    return result.loc[timestamp.dt.date == target_date].copy()


def _parse_option_minute_frame(frame):
    if frame is None or frame.empty:
        return None
    frame = frame.copy()
    if {"date", "time"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="coerce",
        )
        return frame, {
            "price": "price",
            "average_price": "average_price",
            "volume": "volume",
        }
    if {"\u65e5\u671f", "\u65f6\u95f4"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["\u65e5\u671f"].astype(str)
            + " "
            + frame["\u65f6\u95f4"].astype(str),
            errors="coerce",
        )
        return frame, {
            "\u4ef7\u683c": "price",
            "\u5747\u4ef7": "average_price",
            "\u6210\u4ea4": "volume",
            "\u6301\u4ed3": "open_interest",
        }
    if {"鏃ユ湡", "鏃堕棿"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["鏃ユ湡"].astype(str) + " " + frame["鏃堕棿"].astype(str),
            errors="coerce",
        )
        return frame, {
            "浠锋牸": "price",
            "鍧囦环": "average_price",
            "鎴愪氦": "volume",
            "鎸佷粨": "open_interest",
        }
    return None


def _fetch_option_minute(ak, option_code, captured_at, target_date=None):
    target_date = pd.Timestamp(target_date or captured_at).date()
    errors = []
    sources = [
        ("option_finance_minute_sina", lambda: ak.option_finance_minute_sina(symbol=str(option_code))),
        ("option_sse_minute_sina", lambda: ak.option_sse_minute_sina(symbol=str(option_code))),
    ]
    for source, loader in sources:
        try:
            raw = loader()
            parsed = _parse_option_minute_frame(raw)
            if parsed is None:
                raise ValueError(
                    f"unsupported option minute columns: {list(raw.columns) if raw is not None else []}"
                )
            frame, rename_map = parsed
            frame = frame.rename(columns=rename_map)
            for column in ["price", "average_price", "volume", "open_interest"]:
                if column not in frame.columns:
                    frame[column] = None
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

            result = frame[
                ["timestamp", "price", "average_price", "volume", "open_interest"]
            ].copy()
            result.insert(0, "symbol", str(option_code))
            result["source"] = source
            result["captured_at"] = captured_at.isoformat()
            result = result.dropna(subset=["timestamp"])
            timestamp = pd.to_datetime(result["timestamp"], errors="coerce")
            result = result.loc[timestamp.dt.date == target_date].copy()
            if not result.empty:
                return result
        except Exception as exc:
            errors.append(f"{source}:{type(exc).__name__}:{exc}")
    if target_date == pd.Timestamp(captured_at).date():
        raise ValueError(
            f"no current-date option minute data: {option_code}; errors={errors}"
        )
    raise ValueError(
        f"no option minute data for {option_code} on {target_date}; errors={errors}"
    )


def _dates_in_minute_frame(frame, parser):
    parsed = parser(frame)
    if parsed is None:
        return set()
    if isinstance(parsed, tuple):
        frame = parsed[0]
    else:
        frame = parsed
    timestamp = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    return {str(value.date()) for value in timestamp}


def discover_backfill_dates(product, account_id="default", option_codes=None, max_days=None):
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required for intraday backfill.") from exc

    spec = market_data.SSE_ETF_OPTION_SPECS.get(product)
    if spec is None:
        raise ValueError(f"unsupported product: {product}")

    args = argparse.Namespace(
        product=product,
        account_id=account_id,
        option_code=option_codes or [],
        no_account_positions=False,
    )
    codes = _option_codes(args)
    etf_dates = set()
    for _, loader in [
        (
            "stock_zh_a_minute",
            lambda: ak.stock_zh_a_minute(symbol=f"sh{spec.etf_symbol}", period="1", adjust=""),
        ),
        (
            "fund_etf_hist_min_em",
            lambda: ak.fund_etf_hist_min_em(symbol=spec.etf_symbol, period="1", adjust=""),
        ),
    ]:
        try:
            frame = _parse_etf_dates_frame(loader())
            if frame is not None:
                timestamp = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
                etf_dates.update(str(value.date()) for value in timestamp)
        except Exception:
            continue

    option_dates_by_code = {}
    for code in codes:
        dates = set()
        for loader in [
            lambda code=code: ak.option_finance_minute_sina(symbol=str(code)),
            lambda code=code: ak.option_sse_minute_sina(symbol=str(code)),
        ]:
            try:
                dates.update(_dates_in_minute_frame(loader(), _parse_option_minute_frame))
            except Exception:
                continue
        option_dates_by_code[code] = dates

    all_dates = etf_dates.copy()
    for dates in option_dates_by_code.values():
        all_dates.update(dates)
    selected = sorted(all_dates)
    if max_days is not None:
        selected = selected[-int(max_days):]

    rows = []
    for date_text in selected:
        missing_options = [
            code for code in codes if date_text not in option_dates_by_code.get(code, set())
        ]
        rows.append(
            {
                "date": date_text,
                "etf_available": date_text in etf_dates,
                "option_codes": codes,
                "missing_option_codes": missing_options,
                "complete": date_text in etf_dates and not missing_options,
            }
        )
    return {
        "product": product,
        "etf_symbol": spec.etf_symbol,
        "option_codes": codes,
        "max_backfill_days": sum(1 for row in rows if row["complete"]),
        "dates": rows,
    }


def _parse_etf_dates_frame(raw):
    if raw is None or raw.empty:
        return None
    frame = raw.copy()
    if "day" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["day"], errors="coerce")
    elif "\u65f6\u95f4" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["\u65f6\u95f4"], errors="coerce")
    elif "\u65e5\u671f\u65f6\u95f4" in frame.columns:
        frame["timestamp"] = pd.to_datetime(
            frame["\u65e5\u671f\u65f6\u95f4"],
            errors="coerce",
        )
    elif "鏃堕棿" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["鏃堕棿"], errors="coerce")
    elif "鏃ユ湡鏃堕棿" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["鏃ユ湡鏃堕棿"], errors="coerce")
    else:
        frame["timestamp"] = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    return frame


def backfill_missing_dates(args):
    product = args.product
    option_codes = [
        str(code).strip()
        for code in getattr(args, "option_code", []) or []
        if str(code).strip()
    ]
    discovery = discover_backfill_dates(
        product,
        account_id=getattr(args, "account_id", "default"),
        option_codes=option_codes,
        max_days=getattr(args, "backfill_days", None),
    )
    status = refresh_intraday_status(
        product,
        option_codes=discovery["option_codes"],
        etf_symbol=discovery["etf_symbol"],
    )
    status_by_date = status.get("dates", {})
    missing = [
        row["date"]
        for row in discovery["dates"]
        if row["complete"] and not status_by_date.get(row["date"], {}).get("complete")
    ]
    results = []
    for date_text in missing:
        run_args = argparse.Namespace(
            product=product,
            account_id=getattr(args, "account_id", "default"),
            interval_seconds=getattr(args, "interval_seconds", None),
            daily_schedule=getattr(args, "daily_schedule", DEFAULT_DAILY_SCHEDULE),
            option_code=option_codes,
            no_account_positions=getattr(args, "no_account_positions", False),
            once=True,
            save_option_greeks_snapshot=False,
            max_runs=None,
            output_dir=None,
            pid_file=getattr(args, "pid_file", None),
            target_date=date_text,
            backfill_missing=False,
            backfill_days=None,
        )
        results.append(capture_once(run_args))
    return {
        "product": product,
        "available_dates": discovery["dates"],
        "missing_dates": missing,
        "results": results,
        "status_path": str(intraday_status_path(product)),
    }


def intraday_root(product):
    return Path(storage.PROJECT_ROOT) / "data" / "live" / product / "intraday"


def intraday_status_path(product):
    return intraday_root(product) / "status.json"


def load_intraday_status(product):
    path = intraday_status_path(product)
    if not path.exists():
        return {"product": product, "dates": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"product": product, "dates": {}}


def save_intraday_status(product, status):
    path = intraday_status_path(product)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def refresh_intraday_status(product, option_codes=None, etf_symbol=None):
    spec = market_data.SSE_ETF_OPTION_SPECS.get(product)
    if etf_symbol is None and spec is not None:
        etf_symbol = spec.etf_symbol
    option_codes = sorted(dict.fromkeys(str(code) for code in (option_codes or [])))
    root = intraday_root(product)
    dates = {}
    if root.exists():
        for day_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            date_text = _normalize_dir_date(day_dir.name)
            if date_text is None:
                continue
            dates[date_text] = scan_intraday_date(
                product,
                date_text,
                option_codes=option_codes,
                etf_symbol=etf_symbol,
            )
    status = {
        "product": product,
        "etf_symbol": etf_symbol,
        "option_codes": option_codes,
        "updated_at": pd.Timestamp.now().replace(microsecond=0).isoformat(),
        "dates": dates,
    }
    save_intraday_status(product, status)
    return status


def scan_intraday_date(product, date_text, option_codes=None, etf_symbol=None):
    date_part = pd.Timestamp(date_text).strftime("%Y%m%d")
    day_dir = intraday_root(product) / date_part
    etf_rows = 0
    if etf_symbol:
        etf_rows = _csv_rows(day_dir / f"etf_{etf_symbol}_1m.csv")
    option_rows = {}
    for code in option_codes or []:
        option_rows[str(code)] = _csv_rows(day_dir / f"option_{code}_1m.csv")
    missing_options = [code for code, rows in option_rows.items() if rows <= 0]
    complete = etf_rows > 0 and not missing_options
    if not option_rows and option_codes:
        complete = False
    return {
        "date": str(pd.Timestamp(date_text).date()),
        "dir": str(day_dir),
        "etf_rows": etf_rows,
        "option_rows": option_rows,
        "missing_option_codes": missing_options,
        "complete": complete,
    }


def _normalize_dir_date(name):
    try:
        return str(pd.Timestamp(name).date())
    except Exception:
        return None


def _csv_rows(path):
    path = Path(path)
    if not path.exists():
        return 0
    try:
        return int(pd.read_csv(path).shape[0])
    except Exception:
        return 0


def _format_result(result):
    return (
        f"captured_at={result['captured_at']} "
        f"target_date={result.get('target_date', '-')} "
        f"output_dir={result['output_dir']} "
        f"etf={result['etf_symbol']} etf_rows={result['etf_rows']} "
        f"options={','.join(result['option_codes']) or '-'} "
        f"quote_snapshot={bool(result['quote_snapshot'])} "
        f"quote_promotion={bool(result['quote_promotion'])} "
        f"option_minute_rows={result['option_minute_rows']} "
        f"errors={len(result['errors'])} "
        f"error_detail={result['errors']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
