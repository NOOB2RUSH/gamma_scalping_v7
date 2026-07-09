from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core  # noqa: E402
from core import position as opt_position  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "research" / "pair_backtest"
PRODUCT_LABELS = {
    "50etf": "50ETF",
    "300etf": "300ETF",
    "500etf": "500ETF",
    "kc50etf": "KC50ETF",
}


@dataclass
class ProductBundle:
    product: str
    config: object
    etf_by_date: dict
    opt_by_date: dict
    hedge_by_date: dict | None
    trading_calendar: pd.DatetimeIndex
    enriched_by_date: dict
    features: pd.DataFrame
    daily_ohlc: pd.DataFrame


@dataclass
class PairBacktestState:
    cash: float
    positions: dict = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)
    daily_records: list[dict] = field(default_factory=list)
    entry_signal: str | None = None
    entry_date: pd.Timestamp | None = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Backtest a minimal pair ATM-IV-ratio straddle strategy using the "
            "existing single-product data, IV, pricing, margin, and fee logic."
        )
    )
    parser.add_argument("--left", default="50etf", help="Left product in IV ratio.")
    parser.add_argument("--right", default="300etf", help="Right product in IV ratio.")
    parser.add_argument("--start", default=None, help="Inclusive start date.")
    parser.add_argument("--end", default=None, help="Inclusive end date.")
    parser.add_argument(
        "--ratio-window",
        type=int,
        default=252,
        help="Rolling window for IV-ratio z-score.",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=60,
        help="Minimum observations for IV-ratio rolling mean/std.",
    )
    parser.add_argument(
        "--entry-z",
        type=float,
        default=1.5,
        help="Open when z-score is above this or below its negative.",
    )
    parser.add_argument(
        "--exit-z",
        type=float,
        default=0.3,
        help="Close when z-score mean-reverts inside +/- this threshold.",
    )
    parser.add_argument(
        "--base-short-qty",
        type=int,
        default=10,
        help=(
            "Fixed quantity for the rich-IV short straddle when "
            "--target-margin-to-nav is set to 0."
        ),
    )
    parser.add_argument(
        "--target-margin-to-nav",
        type=float,
        default=0.10,
        help=(
            "Target short-leg margin as a fraction of account NAV. "
            "Default 0.10 is the neutral sizing profile. Set to 0 for fixed qty."
        ),
    )
    parser.add_argument(
        "--sizing-mode",
        choices=("fixed_target", "dynamic_z"),
        default="fixed_target",
        help=(
            "fixed_target uses --target-margin-to-nav. dynamic_z uses "
            "z-score tiers to scale the short-leg margin target."
        ),
    )
    parser.add_argument(
        "--dynamic-margin-base",
        type=float,
        default=0.20,
        help="dynamic_z margin target when abs(z) is below --dynamic-z-mid.",
    )
    parser.add_argument(
        "--dynamic-z-mid",
        type=float,
        default=2.0,
        help="dynamic_z threshold for the mid margin tier.",
    )
    parser.add_argument(
        "--dynamic-margin-mid",
        type=float,
        default=0.25,
        help="dynamic_z margin target when abs(z) reaches --dynamic-z-mid.",
    )
    parser.add_argument(
        "--dynamic-z-high",
        type=float,
        default=2.5,
        help="dynamic_z threshold for the high margin tier.",
    )
    parser.add_argument(
        "--dynamic-margin-high",
        type=float,
        default=0.30,
        help="dynamic_z margin target when abs(z) reaches --dynamic-z-high.",
    )
    parser.add_argument(
        "--max-leg-qty",
        type=int,
        default=500,
        help="Maximum straddle quantity per product leg.",
    )
    parser.add_argument(
        "--max-margin-to-nav",
        type=float,
        default=0.50,
        help=(
            "Hard cap for total pair option margin divided by NAV. "
            "Entry sizing is clipped to this cap; existing pairs are closed "
            "if refreshed margin breaches it."
        ),
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000_000.0,
        help="Initial shared account cash.",
    )
    parser.add_argument(
        "--min-cash-reserve",
        type=float,
        default=50_000.0,
        help="Minimum cash after opening both legs.",
    )
    parser.add_argument(
        "--min-exit-dte",
        type=int,
        default=3,
        help="Close both legs when either leg reaches this DTE.",
    )
    parser.add_argument(
        "--max-holding-days",
        type=int,
        default=60,
        help="Maximum calendar holding days for a pair trade.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for daily/trades CSV outputs.",
    )
    return parser.parse_args()


def _set_product_config(product):
    cfg = core.config.load_config(product)
    core.config.CONFIG = cfg
    core.vol_engine.CONFIG = cfg
    core.position.CONFIG = cfg
    core.strategy.CONFIG = cfg
    return cfg


def _common_start_end(left, right, start, end):
    left_cfg = core.config.load_config(left)
    right_cfg = core.config.load_config(right)
    start_ts = pd.Timestamp(start) if start else max(
        pd.Timestamp(left_cfg.backtest.start),
        pd.Timestamp(right_cfg.backtest.start),
    )
    end_ts = pd.Timestamp(end) if end else min(
        pd.Timestamp(left_cfg.backtest.end),
        pd.Timestamp(right_cfg.backtest.end),
    )
    return start_ts, end_ts


def load_product_bundle(product, start, end):
    cfg = _set_product_config(product)
    etf_by_date = core.data_loader.load_etf_series(start, end)
    hedge_by_date = core.data_loader.load_hedge_series(start, end)
    opt_by_date = core.data_loader.load_opt_series(start, end)
    opt_by_date = core.data_loader.attach_underlying_prices(opt_by_date, hedge_by_date)
    trading_calendar = core.data_loader.load_etf_trading_calendar()
    enriched = core.cache.get_enriched_option_chains(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        start,
        end,
    )
    features = core.cache.get_vol_features(
        etf_by_date,
        opt_by_date,
        trading_calendar,
        enriched,
        start,
        end,
    ).sort_index()
    daily_ohlc = core.vol_engine.build_daily_ohlc_df(etf_by_date)
    return ProductBundle(
        product=product,
        config=cfg,
        etf_by_date=etf_by_date,
        opt_by_date=opt_by_date,
        hedge_by_date=hedge_by_date,
        trading_calendar=pd.DatetimeIndex(trading_calendar),
        enriched_by_date=enriched,
        features=features,
        daily_ohlc=daily_ohlc,
    )


def build_pair_features(left_bundle, right_bundle, ratio_window, min_periods):
    left = left_bundle.product
    right = right_bundle.product
    frame = left_bundle.features[["atm_iv"]].rename(
        columns={"atm_iv": f"atm_iv_{left}"}
    ).join(
        right_bundle.features[["atm_iv"]].rename(
            columns={"atm_iv": f"atm_iv_{right}"}
        ),
        how="inner",
    )
    left_iv = pd.to_numeric(frame[f"atm_iv_{left}"], errors="coerce")
    right_iv = pd.to_numeric(frame[f"atm_iv_{right}"], errors="coerce")
    ratio_col = f"iv_ratio_{left}_over_{right}"
    frame[ratio_col] = left_iv / right_iv
    frame.loc[~left_iv.gt(0) | ~right_iv.gt(0), ratio_col] = pd.NA
    ratio = pd.to_numeric(frame[ratio_col], errors="coerce")
    frame["iv_ratio_mean"] = ratio.rolling(
        ratio_window,
        min_periods=min_periods,
    ).mean()
    frame["iv_ratio_std"] = ratio.rolling(
        ratio_window,
        min_periods=min_periods,
    ).std()
    frame["iv_ratio_z"] = (ratio - frame["iv_ratio_mean"]) / frame["iv_ratio_std"]
    frame.loc[~frame["iv_ratio_std"].gt(0), "iv_ratio_z"] = pd.NA
    return frame


def _chain_row(chain_df, order_book_id):
    rows = chain_df[chain_df["order_book_id"].astype(str) == str(order_book_id)]
    if rows.empty:
        raise IndexError(order_book_id)
    return rows.iloc[0]


def _spot(bundle, date):
    return float(bundle.daily_ohlc.loc[date, "close"])


def _atm(bundle, date):
    _set_product_config(bundle.product)
    if date not in bundle.enriched_by_date or date not in bundle.daily_ohlc.index:
        return None
    return core.vol_engine.select_atm_from_chain(
        bundle.enriched_by_date[date],
        _spot(bundle, date),
    )


def _position_rows(bundle, date, position):
    chain_df = bundle.enriched_by_date.get(date)
    if chain_df is None:
        raise IndexError(position["call_code"])
    return (
        _chain_row(chain_df, position["call_code"]),
        _chain_row(chain_df, position["put_code"]),
    )


def _position_underlying_price(bundle, atm, fallback):
    value = atm.get("underlying_price")
    if pd.notna(value):
        return float(value)
    return fallback


def _project_open_cash(cash, bundle, atm, qty, side):
    _set_product_config(bundle.product)
    trade_value = opt_position.calc_trade_value(atm["call"], atm["put"], qty, qty)
    fee = opt_position.calc_option_fee(qty, qty)
    if side == "short":
        margin = opt_position.calc_short_margin(
            atm["call"],
            atm["put"],
            qty,
            qty,
            _position_underlying_price(bundle, atm, _spot(bundle, atm["call"]["date"])),
        )
        return cash + trade_value - fee - margin
    return cash - trade_value - fee


def _straddle_vega(atm):
    greeks = core.strategy.calc_position_greeks(
        atm["call"],
        atm["put"],
        1,
        1,
        side="long",
    )
    return abs(float(greeks.get("vega", 0.0) or 0.0))


def _short_margin_per_straddle(bundle, atm):
    _set_product_config(bundle.product)
    return opt_position.calc_short_margin(
        atm["call"],
        atm["put"],
        1,
        1,
        _position_underlying_price(bundle, atm, _spot(bundle, atm["call"]["date"])),
    )


def _target_margin_to_nav_for_signal(
    zscore,
    sizing_mode,
    target_margin_to_nav,
    dynamic_margin_base,
    dynamic_z_mid,
    dynamic_margin_mid,
    dynamic_z_high,
    dynamic_margin_high,
):
    if sizing_mode == "fixed_target":
        return target_margin_to_nav
    abs_z = abs(float(zscore))
    if abs_z >= dynamic_z_high:
        return dynamic_margin_high
    if abs_z >= dynamic_z_mid:
        return dynamic_margin_mid
    return dynamic_margin_base


def _entry_plan(
    left_bundle,
    right_bundle,
    date,
    signal,
    zscore,
    base_short_qty,
    max_leg_qty,
    sizing_mode,
    target_margin_to_nav,
    dynamic_margin_base,
    dynamic_z_mid,
    dynamic_margin_mid,
    dynamic_z_high,
    dynamic_margin_high,
    max_margin_to_nav,
    account_nav,
):
    left_atm = _atm(left_bundle, date)
    right_atm = _atm(right_bundle, date)
    if left_atm is None or right_atm is None:
        return None, "missing_atm"

    if signal == "short_left_long_right":
        short_product, long_product = left_bundle.product, right_bundle.product
        short_bundle, long_bundle = left_bundle, right_bundle
        short_atm, long_atm = left_atm, right_atm
    elif signal == "long_left_short_right":
        short_product, long_product = right_bundle.product, left_bundle.product
        short_bundle, long_bundle = right_bundle, left_bundle
        short_atm, long_atm = right_atm, left_atm
    else:
        return None, "no_signal"

    short_vega = _straddle_vega(short_atm)
    long_vega = _straddle_vega(long_atm)
    if short_vega <= 0 or long_vega <= 0:
        return None, "missing_vega"

    short_margin_per_straddle = _short_margin_per_straddle(short_bundle, short_atm)
    if short_margin_per_straddle <= 0:
        return None, "missing_short_margin"
    effective_target_margin_to_nav = _target_margin_to_nav_for_signal(
        zscore,
        sizing_mode,
        target_margin_to_nav,
        dynamic_margin_base,
        dynamic_z_mid,
        dynamic_margin_mid,
        dynamic_z_high,
        dynamic_margin_high,
    )
    if max_margin_to_nav > 0:
        effective_target_margin_to_nav = min(
            effective_target_margin_to_nav,
            max_margin_to_nav,
        )
    if effective_target_margin_to_nav > 0:
        target_margin = max(
            0.0,
            float(account_nav) * float(effective_target_margin_to_nav),
        )
        short_qty = int(math.floor(target_margin / short_margin_per_straddle))
    else:
        target_margin = None
        short_qty = int(base_short_qty)
    short_qty = min(max(short_qty, 0), int(max_leg_qty))
    if short_qty <= 0:
        return None, "target_size_too_small"
    long_qty = int(round(short_qty * short_vega / long_vega))
    long_qty = max(1, min(long_qty, int(max_leg_qty)))

    return {
        short_product: {
            "bundle": short_bundle,
            "atm": short_atm,
            "side": "short",
            "qty": short_qty,
            "sizing_margin_per_straddle": short_margin_per_straddle,
            "sizing_target_margin": target_margin,
            "sizing_target_margin_to_nav": effective_target_margin_to_nav,
            "sizing_mode": sizing_mode,
        },
        long_product: {
            "bundle": long_bundle,
            "atm": long_atm,
            "side": "long",
            "qty": long_qty,
            "sizing_short_vega": short_vega,
            "sizing_long_vega": long_vega,
            "sizing_mode": sizing_mode,
        },
    }, None


def _annotate_new_trades(trades, start_idx, product, pair_signal, ratio, zscore):
    for trade in trades[start_idx:]:
        trade["product"] = product
        trade["pair_signal"] = pair_signal
        trade["iv_ratio"] = ratio
        trade["iv_ratio_z"] = zscore


def _open_pair(state, date, plan, signal, ratio, zscore, min_cash_reserve):
    projected_cash = state.cash
    for product, leg in plan.items():
        projected_cash = _project_open_cash(
            projected_cash,
            leg["bundle"],
            leg["atm"],
            leg["qty"],
            leg["side"],
        )
    if projected_cash < min_cash_reserve:
        return False, "cash_reserve"

    for product, leg in plan.items():
        _set_product_config(product)
        start_idx = len(state.trades)
        spot = _position_underlying_price(
            leg["bundle"],
            leg["atm"],
            _spot(leg["bundle"], date),
        )
        state.cash, position, option_value = opt_position.open_trade(
            date,
            state.cash,
            leg["atm"],
            leg["qty"],
            leg["qty"],
            state.trades,
            trade_type=f"pair_open_{leg['side']}_straddle",
            side=leg["side"],
            spot=spot,
            short_entry_regime="pair_iv_ratio" if leg["side"] == "short" else None,
        )
        position["product"] = product
        position["pair_signal"] = signal
        position["sizing_target_margin_to_nav"] = leg.get(
            "sizing_target_margin_to_nav"
        )
        position["sizing_mode"] = leg.get("sizing_mode")
        position["last_option_value"] = option_value
        state.positions[product] = position
        _annotate_new_trades(state.trades, start_idx, product, signal, ratio, zscore)
        for trade in state.trades[start_idx:]:
            trade["sizing_target_margin_to_nav"] = leg.get(
                "sizing_target_margin_to_nav"
            )
            trade["sizing_target_margin"] = leg.get("sizing_target_margin")
            trade["sizing_margin_per_straddle"] = leg.get(
                "sizing_margin_per_straddle"
            )
            trade["sizing_short_vega"] = leg.get("sizing_short_vega")
            trade["sizing_long_vega"] = leg.get("sizing_long_vega")
            trade["sizing_mode"] = leg.get("sizing_mode")
    state.entry_signal = signal
    state.entry_date = pd.Timestamp(date)
    return True, None


def _close_position(state, bundle, date, position, reason, ratio, zscore):
    product = bundle.product
    _set_product_config(product)
    start_idx = len(state.trades)
    try:
        call_row, put_row = _position_rows(bundle, date, position)
        state.cash, close_value = opt_position.close_trade(
            date,
            state.cash,
            position,
            call_row,
            put_row,
            state.trades,
            trade_type="pair_close_straddle",
            exit_reason=reason,
        )
    except IndexError:
        state.cash, close_value = opt_position.close_at_last_value(
            date,
            state.cash,
            position,
            state.trades,
            exit_reason=f"{reason}_missing_contract_last_value",
        )
    _annotate_new_trades(
        state.trades,
        start_idx,
        product,
        position.get("pair_signal"),
        ratio,
        zscore,
    )
    return close_value


def _close_pair(state, bundles, date, reason, ratio, zscore):
    for product, position in list(state.positions.items()):
        if position is None:
            continue
        _close_position(state, bundles[product], date, position, reason, ratio, zscore)
        state.positions[product] = None
    state.entry_signal = None
    state.entry_date = None


def _mark_position(state, bundle, date, position):
    product = bundle.product
    _set_product_config(product)
    try:
        call_row, put_row = _position_rows(bundle, date, position)
    except IndexError:
        return {
            "option_value": float(position.get("last_option_value", 0.0) or 0.0),
            "margin": float(position.get("option_margin", 0.0) or 0.0),
            "dte": None,
            "vega": None,
            "warning": f"{product}:missing_position_contracts",
        }

    if position.get("side") == "short":
        old_margin = float(position.get("option_margin", 0.0) or 0.0)
        spot = call_row.get("underlying_close")
        if pd.isna(spot):
            spot = _spot(bundle, date)
        new_margin = opt_position.calc_short_margin(
            call_row,
            put_row,
            position["call_qty"],
            position["put_qty"],
            spot,
        )
        state.cash -= new_margin - old_margin
        position["option_margin"] = new_margin

    option_value = opt_position.signed_value(position, call_row, put_row)
    position["last_option_value"] = option_value
    greeks = core.strategy.calc_position_greeks(
        call_row,
        put_row,
        position["call_qty"],
        position["put_qty"],
        side=position.get("side", "long"),
    )
    return {
        "option_value": option_value,
        "margin": float(position.get("option_margin", 0.0) or 0.0),
        "dte": int(call_row["dte"]),
        "vega": greeks.get("vega"),
        "warning": None,
    }


def _mark_all_positions(state, bundles, date):
    marks = {}
    for product, position in state.positions.items():
        if position is None:
            continue
        marks[product] = _mark_position(state, bundles[product], date, position)
    return marks


def _entry_signal(zscore, entry_z):
    if pd.isna(zscore):
        return None
    if zscore > entry_z:
        return "short_left_long_right"
    if zscore < -entry_z:
        return "long_left_short_right"
    return None


def _exit_reason(
    state,
    marks,
    zscore,
    exit_z,
    min_exit_dte,
    max_holding_days,
    date,
    max_margin_to_nav,
):
    if not state.entry_signal:
        return None
    if any(mark.get("warning") for mark in marks.values()):
        return "missing_position_contracts"
    option_value = sum(float(mark.get("option_value", 0.0) or 0.0) for mark in marks.values())
    option_margin = sum(float(mark.get("margin", 0.0) or 0.0) for mark in marks.values())
    nav = state.cash + option_value + option_margin
    if max_margin_to_nav > 0 and nav > 0 and option_margin / nav > max_margin_to_nav:
        return "margin_to_nav_cap"
    dtes = [mark.get("dte") for mark in marks.values() if mark.get("dte") is not None]
    if dtes and min(dtes) <= min_exit_dte:
        return "near_expiry"
    if state.entry_date is not None and (pd.Timestamp(date) - state.entry_date).days >= max_holding_days:
        return "max_holding_days"
    if pd.isna(zscore):
        return None
    if state.entry_signal == "short_left_long_right" and zscore <= exit_z:
        return "ratio_mean_reversion"
    if state.entry_signal == "long_left_short_right" and zscore >= -exit_z:
        return "ratio_mean_reversion"
    return None


def _daily_record(date, state, marks, pair_row, left, right):
    option_value = sum(mark["option_value"] for mark in marks.values())
    option_margin = sum(mark["margin"] for mark in marks.values())
    nav = state.cash + option_value + option_margin
    record = {
        "date": date,
        "cash": state.cash,
        "option_value": option_value,
        "option_margin": option_margin,
        "margin_to_nav": option_margin / nav if nav > 0 else None,
        "nav": nav,
        "position_count": sum(1 for position in state.positions.values() if position),
        "entry_signal": state.entry_signal,
        "entry_date": state.entry_date,
        f"atm_iv_{left}": pair_row.get(f"atm_iv_{left}"),
        f"atm_iv_{right}": pair_row.get(f"atm_iv_{right}"),
        f"iv_ratio_{left}_over_{right}": pair_row.get(
            f"iv_ratio_{left}_over_{right}"
        ),
        "iv_ratio_mean": pair_row.get("iv_ratio_mean"),
        "iv_ratio_std": pair_row.get("iv_ratio_std"),
        "iv_ratio_z": pair_row.get("iv_ratio_z"),
        "data_warning_reasons": ",".join(
            mark["warning"] for mark in marks.values() if mark.get("warning")
        ),
    }
    for product, mark in marks.items():
        position = state.positions.get(product) or {}
        record[f"{product}_side"] = position.get("side")
        record[f"{product}_qty"] = position.get("call_qty")
        record[f"{product}_option_value"] = mark.get("option_value")
        record[f"{product}_option_margin"] = mark.get("margin")
        record[f"{product}_dte"] = mark.get("dte")
        record[f"{product}_vega"] = mark.get("vega")
    return record


def run_pair_backtest(
    left_bundle,
    right_bundle,
    pair_features,
    initial_cash,
    min_cash_reserve,
    entry_z,
    exit_z,
    base_short_qty,
    max_leg_qty,
    sizing_mode,
    target_margin_to_nav,
    dynamic_margin_base,
    dynamic_z_mid,
    dynamic_margin_mid,
    dynamic_z_high,
    dynamic_margin_high,
    max_margin_to_nav,
    min_exit_dte,
    max_holding_days,
):
    left = left_bundle.product
    right = right_bundle.product
    bundles = {left: left_bundle, right: right_bundle}
    state = PairBacktestState(cash=float(initial_cash), positions={left: None, right: None})
    ratio_col = f"iv_ratio_{left}_over_{right}"

    for date, pair_row in pair_features.iterrows():
        if date not in left_bundle.enriched_by_date or date not in right_bundle.enriched_by_date:
            continue

        ratio = pair_row.get(ratio_col)
        zscore = pair_row.get("iv_ratio_z")
        marks = _mark_all_positions(state, bundles, date)
        reason = _exit_reason(
            state,
            marks,
            zscore,
            exit_z,
            min_exit_dte,
            max_holding_days,
            date,
            max_margin_to_nav,
        )
        if reason is not None:
            _close_pair(state, bundles, date, reason, ratio, zscore)
            marks = _mark_all_positions(state, bundles, date)

        if not any(state.positions.values()):
            signal = _entry_signal(zscore, entry_z)
            if signal is not None and pd.notna(ratio):
                plan, skip_reason = _entry_plan(
                    left_bundle,
                    right_bundle,
                    date,
                    signal,
                    zscore,
                    base_short_qty,
                    max_leg_qty,
                    sizing_mode,
                    target_margin_to_nav,
                    dynamic_margin_base,
                    dynamic_z_mid,
                    dynamic_margin_mid,
                    dynamic_z_high,
                    dynamic_margin_high,
                    max_margin_to_nav,
                    state.cash,
                )
                if plan is not None:
                    opened, skip_reason = _open_pair(
                        state,
                        date,
                        plan,
                        signal,
                        ratio,
                        zscore,
                        min_cash_reserve,
                    )
                    if opened:
                        marks = _mark_all_positions(state, bundles, date)
                if skip_reason is not None:
                    state.trades.append(
                        {
                            "date": date,
                            "type": "pair_open_skip",
                            "pair_signal": signal,
                            "skip_reason": skip_reason,
                            "iv_ratio": ratio,
                            "iv_ratio_z": zscore,
                            "cash": state.cash,
                        }
                    )

        state.daily_records.append(_daily_record(date, state, marks, pair_row, left, right))

    if any(state.positions.values()):
        last_date = pair_features.index[-1]
        pair_row = pair_features.loc[last_date]
        _close_pair(
            state,
            bundles,
            last_date,
            "end_of_backtest",
            pair_row.get(ratio_col),
            pair_row.get("iv_ratio_z"),
        )
        final_record = _daily_record(last_date, state, {}, pair_row, left, right)
        if state.daily_records and state.daily_records[-1].get("date") == last_date:
            state.daily_records[-1] = final_record
        else:
            state.daily_records.append(final_record)

    daily = pd.DataFrame(state.daily_records)
    trades = pd.DataFrame(state.trades)
    return daily, trades


def _summary(daily, trades):
    if daily.empty:
        return {"days": 0, "trades": len(trades)}
    nav = pd.to_numeric(daily["nav"], errors="coerce")
    return {
        "days": len(daily),
        "trades": len(trades),
        "start_nav": nav.iloc[0],
        "end_nav": nav.iloc[-1],
        "total_return": nav.iloc[-1] / nav.iloc[0] - 1 if nav.iloc[0] else math.nan,
        "min_nav": nav.min(),
        "max_nav": nav.max(),
    }


def write_nav_chart(daily, output_path, left, right):
    if daily.empty:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    zscore = pd.to_numeric(frame["iv_ratio_z"], errors="coerce")
    position_count = pd.to_numeric(frame["position_count"], errors="coerce").fillna(0)
    invested = position_count.gt(0)

    fig, (ax_nav, ax_dd, ax_z) = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 0.8, 1.0]},
    )
    left_label = PRODUCT_LABELS.get(left, left.upper())
    right_label = PRODUCT_LABELS.get(right, right.upper())
    fig.suptitle(
        f"{left_label} vs {right_label} Pair IV-Ratio Backtest",
        fontsize=14,
    )

    ax_nav.plot(frame["date"], nav, label="NAV", linewidth=1.4, color="tab:blue")
    ax_nav.set_ylabel("NAV")
    ax_nav.grid(True, alpha=0.25)
    ax_nav.legend(loc="best")

    ax_dd.fill_between(
        frame["date"],
        drawdown,
        0.0,
        color="tab:red",
        alpha=0.18,
        linewidth=0,
    )
    ax_dd.plot(frame["date"], drawdown, color="tab:red", linewidth=1.0)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_dd.grid(True, alpha=0.25)

    ax_z.plot(
        frame["date"],
        zscore,
        label="IV ratio z-score",
        linewidth=1.1,
        color="tab:purple",
    )
    ax_z.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_z.axhline(1.5, color="tab:red", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_z.axhline(-1.5, color="tab:green", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_z.set_ylabel("Z-score")
    ax_z.set_xlabel("Date")
    ax_z.grid(True, alpha=0.25)
    ax_z.legend(loc="best")

    for ax in (ax_nav, ax_dd, ax_z):
        start = None
        for date, is_invested in zip(frame["date"], invested):
            if is_invested and start is None:
                start = date
            elif not is_invested and start is not None:
                ax.axvspan(start, date, color="tab:gray", alpha=0.08, linewidth=0)
                start = None
        if start is not None:
            ax.axvspan(
                start,
                frame["date"].iloc[-1],
                color="tab:gray",
                alpha=0.08,
                linewidth=0,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    if args.left == args.right:
        raise ValueError("--left and --right must be different products.")
    if args.ratio_window <= 1:
        raise ValueError("--ratio-window must be greater than 1.")
    if args.min_periods <= 1:
        raise ValueError("--min-periods must be greater than 1.")
    if args.entry_z <= args.exit_z:
        raise ValueError("--entry-z should be greater than --exit-z.")
    if args.dynamic_z_high < args.dynamic_z_mid:
        raise ValueError("--dynamic-z-high must be greater than or equal to --dynamic-z-mid.")
    for name in [
        "target_margin_to_nav",
        "dynamic_margin_base",
        "dynamic_margin_mid",
        "dynamic_margin_high",
    ]:
        value = getattr(args, name)
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    if args.max_margin_to_nav < 0:
        raise ValueError("--max-margin-to-nav must be non-negative.")

    start, end = _common_start_end(args.left, args.right, args.start, args.end)
    left_bundle = load_product_bundle(args.left, start, end)
    right_bundle = load_product_bundle(args.right, start, end)
    pair_features = build_pair_features(
        left_bundle,
        right_bundle,
        args.ratio_window,
        args.min_periods,
    )
    daily, trades = run_pair_backtest(
        left_bundle,
        right_bundle,
        pair_features,
        args.initial_cash,
        args.min_cash_reserve,
        args.entry_z,
        args.exit_z,
        args.base_short_qty,
        args.max_leg_qty,
        args.sizing_mode,
        args.target_margin_to_nav,
        args.dynamic_margin_base,
        args.dynamic_z_mid,
        args.dynamic_margin_mid,
        args.dynamic_z_high,
        args.dynamic_margin_high,
        args.max_margin_to_nav,
        args.min_exit_dte,
        args.max_holding_days,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{timestamp}_{args.left}_vs_{args.right}_pair_iv_ratio_backtest"
    run_dir.mkdir(parents=True, exist_ok=True)
    daily_path = run_dir / "daily.csv"
    trades_path = run_dir / "trades.csv"
    features_path = run_dir / "features.csv"
    nav_chart_path = run_dir / "nav_curve.png"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    pair_features.reset_index(names="date").to_csv(
        features_path,
        index=False,
        encoding="utf-8-sig",
    )
    write_nav_chart(daily, nav_chart_path, args.left, args.right)

    summary = _summary(daily, trades)
    print(f"left={args.left} right={args.right} start={start.date()} end={end.date()}")
    for key, value in summary.items():
        print(f"{key}={value}")
    print(f"run_dir={run_dir}")
    print(f"features_csv={features_path}")
    print(f"daily_csv={daily_path}")
    print(f"trades_csv={trades_path}")
    print(f"nav_chart={nav_chart_path}")


if __name__ == "__main__":
    main()
