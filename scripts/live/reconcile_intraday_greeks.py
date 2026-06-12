from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import core
from core import vol_engine
from core.live import account_report, market_data, storage
from core.live.runtime import load_product_config


COL_DATE = "\u65e5\u671f"
COL_ACCOUNT = "\u8d26\u6237ID"
COL_SIDE = "\u65b9\u5411"
COL_CODE = "\u5408\u7ea6\u4ee3\u7801"
COL_NAME = "\u5408\u7ea6\u540d\u79f0"
COL_QTY = "\u603b\u6301\u4ed3"
COL_LAST = "\u6700\u65b0\u4ef7"
COL_STRIKE = "\u884c\u6743\u4ef7"
COL_DTE = "\u5269\u4f59\u5929\u6570"
COL_SPOT = "\u6807\u7684\u4ef7\u683c"
COL_OPT_DAILY = "\u671f\u6743\u5355\u65e5\u76c8\u4e8f"
COL_HEDGE_DAILY = "\u5bf9\u51b2\u5355\u65e5\u76c8\u4e8f"
COL_TOTAL_DAILY = "\u603b\u5355\u65e5\u76c8\u4e8f"
COL_OPTION_PNL = "\u671f\u6743\u6d6e\u76c8\u4e8f"
COL_HEDGE_TOTAL = "\u5bf9\u51b2\u603b\u76c8\u4e8f"
COL_TOTAL_GREEKS = "\u5355\u65e5GreeksPnL"
COL_HEDGE_QTY = "\u5bf9\u51b2\u6301\u4ed3"
COL_HEDGE_PRICE = "\u5bf9\u51b2\u6700\u65b0\u4ef7"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconcile daily PnL with 5-minute intraday Greeks integration."
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--account-id", default="default")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--freq", default="5min")
    parser.add_argument(
        "--intraday-dir",
        default=None,
        help="Directory containing etf_*_1m.csv and option_*_1m.csv. Default: latest data/live/<product>/intraday/*.",
    )
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_product_config(args.product)
    rows = reconcile_intraday_greeks(
        args.product,
        account_id=args.account_id,
        start_date=args.start_date,
        end_date=args.end_date,
        freq=args.freq,
        intraday_dir=args.intraday_dir,
    )
    frame = pd.DataFrame(rows)
    if args.output_csv:
        path = Path(args.output_csv)
    else:
        path = storage.output_dir(args.product) / (
            f"{storage.local_now_stamp()}_intraday_greeks_reconcile.csv"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"intraday_reconcile_csv={path}")
    if frame.empty:
        print("no_rows")
        return
    print(
        frame[
            [
                "date",
                "nodes",
                "actual_total_pnl",
                "old_total_greeks_pnl",
                "old_total_residual",
                "intraday_total_greeks_pnl",
                "intraday_total_residual",
                "old_abs_residual",
                "intraday_abs_residual",
            ]
        ].to_string(index=False)
    )
    old_abs = frame["old_abs_residual"].dropna()
    new_abs = frame["intraday_abs_residual"].dropna()
    if not old_abs.empty and not new_abs.empty:
        print(
            "summary "
            f"old_mean_abs_residual={old_abs.mean():.6f} "
            f"intraday_mean_abs_residual={new_abs.mean():.6f} "
            f"improvement={old_abs.mean() - new_abs.mean():.6f}"
        )


def reconcile_intraday_greeks(
    product,
    account_id="default",
    start_date=None,
    end_date=None,
    freq="5min",
    intraday_dir=None,
):
    summary = pd.read_csv(
        storage.account_report_summary_history_path(product, account_id),
        encoding="utf-8-sig",
    )
    positions = pd.read_csv(
        storage.account_report_position_history_path(product, account_id),
        encoding="utf-8-sig",
    )
    summary = summary[summary[COL_ACCOUNT].astype(str).eq(str(account_id))].copy()
    summary["_date"] = pd.to_datetime(summary[COL_DATE], errors="coerce")
    summary = summary.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
    start = _date_or_none(start_date)
    end = _date_or_none(end_date)
    intraday_path = _resolve_intraday_dir(product, intraday_dir)
    etf_symbol = market_data.SSE_ETF_OPTION_SPECS[product].etf_symbol
    etf_minute = _load_etf_minute(intraday_path, etf_symbol, freq)

    rows = []
    for index in range(1, len(summary)):
        prev = summary.iloc[index - 1]
        current = summary.iloc[index]
        current_date = current["_date"].date()
        if start is not None and current_date < start:
            continue
        if end is not None and current_date > end:
            continue
        rows.append(
            _reconcile_day(
                product,
                intraday_path,
                etf_minute,
                positions,
                prev,
                current,
                freq,
            )
        )
    return rows


def _reconcile_day(product, intraday_path, etf_minute, positions, prev, current, freq):
    prev_date = str(prev[COL_DATE])
    current_date = str(current[COL_DATE])
    prev_positions = _option_position_rows(positions, prev_date)
    current_positions = _option_position_rows(positions, current_date)
    pair = _same_straddle_pair(prev_positions, current_positions)
    if pair is None:
        return {
            "date": current_date,
            "previous_date": prev_date,
            "status": "skipped",
            "reason": "option position changed or cannot identify one call/put pair",
        }

    call_row, put_row = pair
    call_code = str(call_row[COL_CODE])
    put_code = str(put_row[COL_CODE])
    option_minute = _load_option_pair_minute(intraday_path, call_code, put_code, freq)
    path = _build_path(
        etf_minute,
        option_minute,
        prev,
        current,
        prev_positions,
        current_positions,
        call_code,
        put_code,
    )
    if len(path) < 2:
        return {
            "date": current_date,
            "previous_date": prev_date,
            "status": "skipped",
            "reason": "not enough intraday path nodes",
        }

    path = _add_intraday_greeks(path, call_row, put_row)
    if len(path) < 2:
        return {
            "date": current_date,
            "previous_date": prev_date,
            "status": "skipped",
            "reason": "not enough valid IV/Greeks nodes",
        }

    option_parts = _integrate_option_greeks(path)
    hedge_greeks = _hedge_greeks_pnl(product, prev, current)
    actual_option = _actual_option_pnl(prev, current)
    actual_hedge = _actual_hedge_pnl(prev, current)
    actual_total = _actual_total_pnl(prev, current, actual_option, actual_hedge)
    old_total = _number(current.get(COL_TOTAL_GREEKS))
    old_option = old_total - hedge_greeks if old_total is not None and hedge_greeks is not None else None
    intraday_option = option_parts["option_greeks_pnl"]
    intraday_total = _sum_optional(intraday_option, hedge_greeks)

    return {
        "date": current_date,
        "previous_date": prev_date,
        "status": "ok",
        "freq": freq,
        "nodes": len(path),
        "intervals": len(path) - 1,
        "call_code": call_code,
        "put_code": put_code,
        "actual_option_pnl": actual_option,
        "actual_hedge_pnl": actual_hedge,
        "actual_total_pnl": actual_total,
        "old_option_greeks_pnl": old_option,
        "old_option_residual": _difference(actual_option, old_option),
        "old_total_greeks_pnl": old_total,
        "old_total_residual": _difference(actual_total, old_total),
        "old_abs_residual": _abs_or_none(_difference(actual_total, old_total)),
        "intraday_option_delta_pnl": option_parts["delta_pnl"],
        "intraday_option_gamma_pnl": option_parts["gamma_pnl"],
        "intraday_option_vega_pnl": option_parts["vega_pnl"],
        "intraday_option_theta_pnl": option_parts["theta_pnl"],
        "intraday_option_greeks_pnl": intraday_option,
        "intraday_option_residual": _difference(actual_option, intraday_option),
        "hedge_greeks_pnl": hedge_greeks,
        "hedge_residual": _difference(actual_hedge, hedge_greeks),
        "intraday_total_greeks_pnl": intraday_total,
        "intraday_total_residual": _difference(actual_total, intraday_total),
        "intraday_abs_residual": _abs_or_none(_difference(actual_total, intraday_total)),
        "path_start": str(path["timestamp"].iloc[0]),
        "path_end": str(path["timestamp"].iloc[-1]),
    }


def _option_position_rows(positions, date_text):
    rows = positions[
        positions[COL_DATE].astype(str).eq(str(date_text))
        & ~positions[COL_SIDE].astype(str).eq("hedge")
    ].copy()
    if rows.empty:
        return rows
    rows["_leg"] = rows.apply(_leg_from_position_row, axis=1)
    return rows


def _same_straddle_pair(prev_positions, current_positions):
    if prev_positions.empty or current_positions.empty:
        return None
    prev_codes = set(prev_positions[COL_CODE].astype(str))
    current_codes = set(current_positions[COL_CODE].astype(str))
    if prev_codes != current_codes:
        return None
    calls = current_positions[current_positions["_leg"].eq("call")]
    puts = current_positions[current_positions["_leg"].eq("put")]
    if len(calls) != 1 or len(puts) != 1:
        return None
    call = calls.iloc[0]
    put = puts.iloc[0]
    prev_call = prev_positions[prev_positions[COL_CODE].astype(str).eq(str(call[COL_CODE]))]
    prev_put = prev_positions[prev_positions[COL_CODE].astype(str).eq(str(put[COL_CODE]))]
    if prev_call.empty or prev_put.empty:
        return None
    if _number(prev_call.iloc[0].get(COL_QTY)) != _number(call.get(COL_QTY)):
        return None
    if _number(prev_put.iloc[0].get(COL_QTY)) != _number(put.get(COL_QTY)):
        return None
    return call, put


def _leg_from_position_row(row):
    name = str(row.get(COL_NAME) or "")
    if "\u8d2d" in name:
        return "call"
    if "\u6cbd" in name:
        return "put"
    delta = _number(row.get("Delta"))
    if delta is None:
        return None
    return "call" if delta > 0 else "put"


def _build_path(
    etf_minute,
    option_minute,
    prev,
    current,
    prev_positions,
    current_positions,
    call_code,
    put_code,
):
    prev_date = str(prev[COL_DATE])
    current_date = str(current[COL_DATE])
    start_ts = pd.Timestamp(prev_date + " 15:00:00")
    current_day = pd.Timestamp(current_date).date()
    merged = etf_minute.merge(option_minute, on="timestamp", how="inner")
    path = merged[
        merged["timestamp"].eq(start_ts) | merged["timestamp"].dt.date.eq(current_day)
    ].copy()
    path = path.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    path = path.reset_index(drop=True)
    if path.empty:
        return path

    prev_call = prev_positions[prev_positions[COL_CODE].astype(str).eq(str(call_code))].iloc[0]
    prev_put = prev_positions[prev_positions[COL_CODE].astype(str).eq(str(put_code))].iloc[0]
    current_call = current_positions[
        current_positions[COL_CODE].astype(str).eq(str(call_code))
    ].iloc[0]
    current_put = current_positions[
        current_positions[COL_CODE].astype(str).eq(str(put_code))
    ].iloc[0]
    path.loc[0, "spot"] = _number(prev.get(COL_SPOT))
    path.loc[0, "call_px"] = _number(prev_call.get(COL_LAST))
    path.loc[0, "put_px"] = _number(prev_put.get(COL_LAST))
    path.loc[path.index[-1], "spot"] = _number(current.get(COL_SPOT))
    path.loc[path.index[-1], "call_px"] = _number(current_call.get(COL_LAST))
    path.loc[path.index[-1], "put_px"] = _number(current_put.get(COL_LAST))

    start_dte = _number(prev_call.get(COL_DTE))
    end_dte = _number(current_call.get(COL_DTE))
    if start_dte is None or end_dte is None:
        return path.iloc[0:0]
    progress = np.linspace(0.0, 1.0, len(path))
    path["dte"] = start_dte - progress * (start_dte - end_dte)
    path["ttm"] = path["dte"] / float(core.config.CONFIG.vol.annual_days)
    return path


def _add_intraday_greeks(path, call_row, put_row):
    call_greeks = _leg_intraday_greeks(path, call_row, "call_px", "c")
    put_greeks = _leg_intraday_greeks(path, put_row, "put_px", "p")
    result = pd.concat(
        [path, call_greeks.add_prefix("call_"), put_greeks.add_prefix("put_")],
        axis=1,
    )
    return result.dropna(
        subset=[
            "call_iv",
            "put_iv",
            "call_delta",
            "put_delta",
            "call_gamma",
            "put_gamma",
            "call_vega",
            "put_vega",
            "call_theta",
            "put_theta",
        ]
    ).reset_index(drop=True)


def _leg_intraday_greeks(path, row, price_col, flag):
    vollib = vol_engine._load_vollib_funcs()
    strike = _number(row.get(COL_STRIKE))
    qty = _number(row.get(COL_QTY)) or 0.0
    side = str(row.get(COL_SIDE) or "").lower()
    direction = -1.0 if side == "short" else 1.0
    multiplier = float(core.config.CONFIG.vol.contract_multiplier)
    valid = (
        path[price_col].astype(float).gt(0)
        & path["spot"].astype(float).gt(0)
        & path["ttm"].astype(float).gt(0)
        & pd.notna(strike)
    )
    iv = pd.Series(np.nan, index=path.index, dtype=float)
    iv.loc[valid] = vollib["implied_volatility"](
        price=path.loc[valid, price_col],
        S=path.loc[valid, "spot"],
        t=path.loc[valid, "ttm"],
        K=strike,
        r=float(core.config.CONFIG.vol.risk_free_rate),
        flag=flag,
        model="black_scholes",
        return_as="series",
        on_error="ignore",
    ).to_numpy()

    delta = pd.Series(np.nan, index=path.index, dtype=float)
    gamma = pd.Series(np.nan, index=path.index, dtype=float)
    vega = pd.Series(np.nan, index=path.index, dtype=float)
    theta = pd.Series(np.nan, index=path.index, dtype=float)
    valid_greeks = valid & iv.notna() & iv.gt(0)
    if valid_greeks.any():
        kwargs = {
            "flag": flag,
            "S": path.loc[valid_greeks, "spot"],
            "K": strike,
            "t": path.loc[valid_greeks, "ttm"],
            "r": float(core.config.CONFIG.vol.risk_free_rate),
            "model": "black_scholes",
            "sigma": iv.loc[valid_greeks],
            "return_as": "series",
        }
        delta.loc[valid_greeks] = vollib["delta"](**kwargs).to_numpy()
        gamma.loc[valid_greeks] = vollib["gamma"](**kwargs).to_numpy()
        vega.loc[valid_greeks] = vollib["vega"](**kwargs).to_numpy()
        theta_365 = vollib["theta"](**kwargs).to_numpy()
        theta.loc[valid_greeks] = theta_365 * (
            365.0 / float(core.config.CONFIG.vol.annual_days)
        )

    scale = direction * qty * multiplier
    return pd.DataFrame(
        {
            "iv": iv,
            "delta": delta * scale,
            "gamma": gamma * scale,
            "vega": vega * scale,
            "theta": theta * scale,
        }
    )


def _integrate_option_greeks(path):
    intervals = len(path) - 1
    prev = path.iloc[:-1]
    spot_change = path["spot"].diff().iloc[1:].to_numpy()
    call_iv_change = path["call_iv"].diff().iloc[1:].to_numpy()
    put_iv_change = path["put_iv"].diff().iloc[1:].to_numpy()
    delta_pnl = (
        (prev["call_delta"].to_numpy() + prev["put_delta"].to_numpy())
        * spot_change
    ).sum()
    gamma_pnl = (
        0.5
        * (
            prev["call_gamma"].to_numpy()
            + prev["put_gamma"].to_numpy()
        )
        * spot_change
        * spot_change
    ).sum()
    vega_pnl = (
        prev["call_vega"].to_numpy() * call_iv_change * 100.0
        + prev["put_vega"].to_numpy() * put_iv_change * 100.0
    ).sum()
    theta_pnl = (
        (prev["call_theta"].to_numpy() + prev["put_theta"].to_numpy())
        * (1.0 / intervals)
    ).sum()
    return {
        "delta_pnl": float(delta_pnl),
        "gamma_pnl": float(gamma_pnl),
        "vega_pnl": float(vega_pnl),
        "theta_pnl": float(theta_pnl),
        "option_greeks_pnl": float(delta_pnl + gamma_pnl + vega_pnl + theta_pnl),
    }
def _hedge_greeks_pnl(product, prev, current):
    start_price = _number(prev.get(COL_HEDGE_PRICE))
    end_price = _number(current.get(COL_HEDGE_PRICE))
    if start_price is None:
        start_price = _number(prev.get(COL_SPOT))
    if end_price is None:
        end_price = _number(current.get(COL_SPOT))
    previous_qty = _number(prev.get(COL_HEDGE_QTY)) or 0.0
    if start_price is None or end_price is None:
        return None
    trade_rows = account_report._security_trade_rows_by_date(product).get(
        str(current.get(COL_DATE)),
        [],
    )
    if trade_rows:
        return account_report._segmented_hedge_delta_pnl(
            previous_qty,
            start_price,
            end_price,
            trade_rows,
        )
    return previous_qty * (end_price - start_price)


def _load_etf_minute(intraday_path, etf_symbol, freq):
    path = Path(intraday_path) / f"etf_{etf_symbol}_1m.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return _resample_last(
        frame[["timestamp", "close"]].rename(columns={"close": "spot"}).dropna(),
        "spot",
        freq,
    )


def _load_option_pair_minute(intraday_path, call_code, put_code, freq):
    call = _load_option_minute(intraday_path, call_code, "call_px", freq)
    put = _load_option_minute(intraday_path, put_code, "put_px", freq)
    return call.merge(put, on="timestamp", how="inner")


def _load_option_minute(intraday_path, code, column, freq):
    path = Path(intraday_path) / f"option_{code}_1m.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    return _resample_last(
        frame[["timestamp", "price"]].rename(columns={"price": column}).dropna(),
        column,
        freq,
    )


def _resample_last(frame, value_column, freq):
    return (
        frame.set_index("timestamp")
        .sort_index()[[value_column]]
        .resample(freq)
        .last()
        .dropna()
        .reset_index()
    )


def _resolve_intraday_dir(product, intraday_dir):
    if intraday_dir:
        return Path(intraday_dir)
    root = storage.PROJECT_ROOT / "data" / "live" / product / "intraday"
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No intraday directories under {root}")
    return candidates[-1]


def _actual_option_pnl(prev, current):
    value = _number(current.get(COL_OPT_DAILY))
    if value is not None:
        return value
    return _value_change(prev, current, COL_OPTION_PNL)


def _actual_hedge_pnl(prev, current):
    value = _number(current.get(COL_HEDGE_DAILY))
    if value is not None:
        return value
    return _value_change(prev, current, COL_HEDGE_TOTAL)


def _actual_total_pnl(prev, current, option_pnl, hedge_pnl):
    value = _number(current.get(COL_TOTAL_DAILY))
    if value is not None:
        return value
    return _sum_optional(option_pnl, hedge_pnl)


def _value_change(prev, current, column):
    prev_value = _number(prev.get(column))
    current_value = _number(current.get(column))
    if prev_value is None or current_value is None:
        return None
    return current_value - prev_value


def _number(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", "").replace("%", "")
    if text in {"", "--", "nan", "None"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def _difference(left, right):
    if left is None or right is None:
        return None
    return left - right


def _sum_optional(*values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid)


def _abs_or_none(value):
    return None if value is None else abs(value)


def _date_or_none(value):
    if value is None or value == "":
        return None
    return pd.Timestamp(value).date()


if __name__ == "__main__":
    main()
