import math
from dataclasses import dataclass, field

import pandas as pd

from . import hedge, position as opt_position, strategy, vol_engine
from .config import CONFIG


POSITION_SIDES = ("long", "short")
NUMERIC_GREEK_KEYS = (
    "delta",
    "gamma",
    "vega",
    "theta",
    "call_delta",
    "put_delta",
    "call_gamma",
    "put_gamma",
    "call_vega",
    "put_vega",
    "call_theta",
    "put_theta",
)
IV_GREEK_KEYS = ("call_iv", "put_iv", "position_iv")


def empty_greeks():
    """无持仓时使用的希腊值占位，保证日报字段完整。"""
    return {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "theta": 0.0,
        "call_iv": None,
        "put_iv": None,
        "position_iv": None,
        "call_delta": 0.0,
        "put_delta": 0.0,
        "call_gamma": 0.0,
        "put_gamma": 0.0,
        "call_vega": 0.0,
        "put_vega": 0.0,
        "call_theta": 0.0,
        "put_theta": 0.0,
    }


def combine_greeks(greeks_list):
    """把多个仓位 Greeks 汇总到账户口径；IV 字段仅保留非空值的简单均值。"""
    combined = empty_greeks()
    valid_ivs = {key: [] for key in IV_GREEK_KEYS}
    for greeks in greeks_list:
        if greeks is None:
            continue
        for key in NUMERIC_GREEK_KEYS:
            combined[key] += greeks.get(key, 0.0) or 0.0
        for key in IV_GREEK_KEYS:
            value = greeks.get(key)
            if pd.notna(value):
                valid_ivs[key].append(value)

    for key, values in valid_ivs.items():
        combined[key] = sum(values) / len(values) if values else None
    return combined


def single_call_greeks(row, qty, side):
    """按单腿 Call 的方向、张数和合约乘数计算 Greeks。"""
    direction = -1 if side == "short" else 1
    scale = float(qty) * float(row["contract_multiplier"])
    result = empty_greeks()
    for greek in ["delta", "gamma", "vega", "theta"]:
        value = direction * float(row[greek]) * scale
        result[greek] = value
        result[f"call_{greek}"] = value
    result["call_iv"] = row.get("iv")
    result["position_iv"] = row.get("iv")
    return result


def empty_side_record():
    return {
        "option_value": 0.0,
        "greeks": empty_greeks(),
        "pnl_position_iv": None,
        "pnl_call_iv": None,
        "pnl_put_iv": None,
        "pnl_greeks": empty_greeks(),
        "eod_position_dte": None,
    }


@dataclass
class BacktestState:
    """回测账户状态：现金、期权仓位、ETF 对冲仓位和输出记录。"""

    cash: float
    positions: dict = field(
        default_factory=lambda: {
            "long": None,
            "short": None,
        }
    )
    option_hedges: list[dict] = field(default_factory=list)
    hedge_etf_qty: float = 0.0
    hedge_entry_price: float = 0.0
    hedge_margin: float = 0.0
    hedge_underlying_order_book_id: str | None = None
    strike_mismatch_days: dict = field(
        default_factory=lambda: {
            "long": 0,
            "short": 0,
        }
    )
    roll_cooldown_left: dict = field(
        default_factory=lambda: {
            "long": 0,
            "short": 0,
        }
    )
    trades: list[dict] = field(default_factory=list)
    daily_records: list[dict] = field(default_factory=list)


def execute_delta_hedge(
    date,
    cash,
    greeks,
    hedge_etf_qty,
    hedge_entry_price,
    hedge_margin,
    spot,
    trades,
    target_qty=None,
    trade_type="delta_hedge",
    etf_fee_rate=None,
    underlying_order_book_id=None,
    current_price=None,
    current_underlying_order_book_id=None,
    daily_volume=None,
):
    """把 ETF 对冲仓位调整到目标数量，并记录交易和手续费。"""
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate

    if target_qty is None:
        target_qty = -greeks["delta"]
    target_qty = strategy.round_etf_hedge_target(target_qty)
    if target_qty == 0:
        underlying_order_book_id = current_underlying_order_book_id
    if current_price is None:
        current_price = spot

    if (
        hedge_etf_qty == target_qty
        and current_underlying_order_book_id == underlying_order_book_id
    ):
        return (
            cash,
            hedge_etf_qty,
            hedge_entry_price,
            hedge_margin,
            current_underlying_order_book_id,
        )

    old_qty = hedge_etf_qty
    old_underlying = current_underlying_order_book_id
    hedge_pnl = 0.0
    fee_notional = 0.0

    underlying_changed = (
        old_qty != 0
        and target_qty != 0
        and old_underlying != underlying_order_book_id
    )
    if underlying_changed:
        cash, hedge_pnl = hedge.close_etf_hedge(
            cash,
            hedge_etf_qty,
            hedge_entry_price,
            hedge_margin,
            current_price,
        )
        fee_notional += abs(old_qty) * current_price
        cash, hedge_etf_qty, hedge_entry_price, hedge_margin, open_pnl = (
            hedge.rebalance_etf_hedge(
                cash,
                0.0,
                0.0,
                0.0,
                target_qty,
                spot,
            )
        )
        hedge_pnl += open_pnl
        fee_notional += abs(target_qty) * spot
    else:
        cash, hedge_etf_qty, hedge_entry_price, hedge_margin, hedge_pnl = (
            hedge.rebalance_etf_hedge(
                cash,
                hedge_etf_qty,
                hedge_entry_price,
                hedge_margin,
                target_qty,
                spot,
            )
        )
        fee_notional += abs(hedge_etf_qty - old_qty) * spot

    trade_qty = hedge_etf_qty - old_qty
    liquidity_trade_qty = (
        abs(old_qty) + abs(hedge_etf_qty)
        if underlying_changed
        else abs(trade_qty)
    )
    liquidity_ratio = CONFIG.backtest.liquidity_warning_volume_ratio
    liquidity_limit_qty = (
        float(daily_volume) * liquidity_ratio
        if daily_volume is not None and pd.notna(daily_volume)
        else None
    )
    etf_fee = fee_notional * etf_fee_rate
    cash -= etf_fee
    new_underlying = underlying_order_book_id if hedge_etf_qty != 0 else None

    trades.append(
        {
            "date": date,
            "type": trade_type,
            "old_hedge_underlying_order_book_id": old_underlying,
            "hedge_underlying_order_book_id": new_underlying,
            "old_etf_qty": old_qty,
            "new_etf_qty": hedge_etf_qty,
            "trade_etf_qty": trade_qty,
            "etf_qty": trade_qty,
            "price": spot,
            "hedge_pnl": hedge_pnl,
            "fee": etf_fee,
            "liquidity_warning_ratio": liquidity_ratio,
            "liquidity_check_available": liquidity_limit_qty is not None,
            "liquidity_warning": (
                liquidity_trade_qty > liquidity_limit_qty
                if liquidity_limit_qty is not None
                else False
            ),
            "liquidity_warning_legs": (
                "etf"
                if liquidity_limit_qty is not None
                and liquidity_trade_qty > liquidity_limit_qty
                else ""
            ),
            "liquidity_volume_missing_legs": (
                "" if liquidity_limit_qty is not None else "etf"
            ),
            "etf_volume": daily_volume,
            "etf_liquidity_trade_qty": liquidity_trade_qty,
            "etf_liquidity_limit_qty": liquidity_limit_qty,
        }
    )
    return cash, hedge_etf_qty, hedge_entry_price, hedge_margin, new_underlying


def build_daily_record(
    date,
    spot,
    cash,
    option_value,
    hedge_etf_qty,
    hedge_underlying_order_book_id,
    hedge_price,
    hedge_entry_price,
    hedge_margin,
    hedge_unrealized_pnl,
    nav,
    etf_fee,
    option_fee,
    positions,
    side_records,
    greeks,
    feature_row,
    pnl_position_iv,
    pnl_call_iv,
    pnl_put_iv,
    pnl_greeks,
    eod_position_dte,
):
    """生成单日账户快照：账户字段汇总，long/short 字段分别展开。"""
    active_sides = [
        side for side in POSITION_SIDES if positions.get(side) is not None
    ]
    eod_has_position = len(active_sides) > 0
    if len(active_sides) == 1:
        eod_position_side = active_sides[0]
    elif len(active_sides) > 1:
        eod_position_side = "both"
    else:
        eod_position_side = None

    account_delta = greeks["delta"] + hedge_etf_qty
    option_margin = sum(
        opt_position.margin_value(positions[side])
        for side in POSITION_SIDES
        if positions.get(side) is not None
    )

    record = {
        "date": date,
        "spot": spot,
        "cash": cash,
        "cash_negative_warning": cash < 0,
        "option_value": option_value,
        "option_margin": option_margin,
        "hedge_etf_qty": hedge_etf_qty,
        "hedge_underlying_order_book_id": hedge_underlying_order_book_id,
        "hedge_price": hedge_price,
        "hedge_entry_price": hedge_entry_price,
        "hedge_margin": hedge_margin,
        "hedge_unrealized_pnl": hedge_unrealized_pnl,
        "nav": nav,
        "etf_fee": etf_fee,
        "option_fee": option_fee,
        "eod_has_position": eod_has_position,
        "eod_position_side": eod_position_side,
        "eod_position_call_code": None,
        "eod_position_put_code": None,
        "eod_position_underlying_order_book_id": None,
        "eod_position_strike": None,
        "eod_position_expiry": None,
        "eod_position_dte": eod_position_dte,
        "eod_position_call_qty": sum(
            positions[side]["call_qty"]
            for side in POSITION_SIDES
            if positions.get(side) is not None
        ),
        "eod_position_put_qty": sum(
            positions[side]["put_qty"]
            for side in POSITION_SIDES
            if positions.get(side) is not None
        ),
        "account_delta": account_delta,
        "account_gamma": greeks["gamma"],
        "account_vega": greeks["vega"],
        "account_theta": greeks["theta"],
        "eod_position_iv": greeks["position_iv"],
        "eod_call_iv": greeks["call_iv"],
        "eod_put_iv": greeks["put_iv"],
        "eod_call_delta": greeks["call_delta"],
        "eod_put_delta": greeks["put_delta"],
        "eod_call_gamma": greeks["call_gamma"],
        "eod_put_gamma": greeks["put_gamma"],
        "eod_call_vega": greeks["call_vega"],
        "eod_put_vega": greeks["put_vega"],
        "eod_call_theta": greeks["call_theta"],
        "eod_put_theta": greeks["put_theta"],
        "pnl_position_iv": pnl_position_iv,
        "pnl_call_iv": pnl_call_iv,
        "pnl_put_iv": pnl_put_iv,
        "pnl_call_delta": pnl_greeks["call_delta"],
        "pnl_put_delta": pnl_greeks["put_delta"],
        "pnl_call_gamma": pnl_greeks["call_gamma"],
        "pnl_put_gamma": pnl_greeks["put_gamma"],
        "pnl_call_vega": pnl_greeks["call_vega"],
        "pnl_put_vega": pnl_greeks["put_vega"],
        "pnl_call_theta": pnl_greeks["call_theta"],
        "pnl_put_theta": pnl_greeks["put_theta"],
        "atm_iv": feature_row["atm_iv"],
        "long_open_signal": feature_row.get("long_open_signal", False),
        "short_open_signal": feature_row.get("short_open_signal", False),
    }

    if len(active_sides) == 1:
        position = positions[active_sides[0]]
        record["eod_position_call_code"] = position["call_code"]
        record["eod_position_put_code"] = position["put_code"]
        record["eod_position_underlying_order_book_id"] = position.get(
            "underlying_order_book_id"
        )
        record["eod_position_strike"] = position["strike"]
        record["eod_position_expiry"] = position["expiry"]

    for side in POSITION_SIDES:
        position = positions.get(side)
        side_record = side_records[side]
        side_greeks = side_record["greeks"]
        side_pnl_greeks = side_record["pnl_greeks"]
        prefix = f"{side}_"
        record.update(
            {
                f"{prefix}has_position": position is not None,
                f"{prefix}position_call_code": (
                    position["call_code"] if position is not None else None
                ),
                f"{prefix}position_put_code": (
                    position["put_code"] if position is not None else None
                ),
                f"{prefix}position_underlying_order_book_id": (
                    position.get("underlying_order_book_id")
                    if position is not None
                    else None
                ),
                f"{prefix}position_strike": (
                    position["strike"] if position is not None else None
                ),
                f"{prefix}position_expiry": (
                    position["expiry"] if position is not None else None
                ),
                f"{prefix}position_dte": side_record["eod_position_dte"],
                f"{prefix}position_call_qty": (
                    position["call_qty"] if position is not None else 0
                ),
                f"{prefix}position_put_qty": (
                    position["put_qty"] if position is not None else 0
                ),
                f"{prefix}option_value": side_record["option_value"],
                f"{prefix}option_margin": (
                    opt_position.margin_value(position) if position is not None else 0.0
                ),
                f"{prefix}eod_position_iv": side_greeks["position_iv"],
                f"{prefix}eod_call_iv": side_greeks["call_iv"],
                f"{prefix}eod_put_iv": side_greeks["put_iv"],
                f"{prefix}eod_call_delta": side_greeks["call_delta"],
                f"{prefix}eod_put_delta": side_greeks["put_delta"],
                f"{prefix}eod_call_gamma": side_greeks["call_gamma"],
                f"{prefix}eod_put_gamma": side_greeks["put_gamma"],
                f"{prefix}eod_call_vega": side_greeks["call_vega"],
                f"{prefix}eod_put_vega": side_greeks["put_vega"],
                f"{prefix}eod_call_theta": side_greeks["call_theta"],
                f"{prefix}eod_put_theta": side_greeks["put_theta"],
                f"{prefix}pnl_position_iv": side_record["pnl_position_iv"],
                f"{prefix}pnl_call_iv": side_record["pnl_call_iv"],
                f"{prefix}pnl_put_iv": side_record["pnl_put_iv"],
                f"{prefix}pnl_call_delta": side_pnl_greeks["call_delta"],
                f"{prefix}pnl_put_delta": side_pnl_greeks["put_delta"],
                f"{prefix}pnl_call_gamma": side_pnl_greeks["call_gamma"],
                f"{prefix}pnl_put_gamma": side_pnl_greeks["put_gamma"],
                f"{prefix}pnl_call_vega": side_pnl_greeks["call_vega"],
                f"{prefix}pnl_put_vega": side_pnl_greeks["put_vega"],
                f"{prefix}pnl_call_theta": side_pnl_greeks["call_theta"],
                f"{prefix}pnl_put_theta": side_pnl_greeks["put_theta"],
            }
        )

    return record

def add_greeks_pnl(daily_df):
    """用昨日 EOD 持仓解释今日 PnL；long/short 可同时贡献，ETF 对冲单独按账户口径加入。"""
    df = daily_df.copy()
    spot_chg = df["spot"].diff()

    for side in POSITION_SIDES:
        prefix = f"{side}_"
        current_has_position = df[f"{prefix}has_position"].eq(True)
        prev_has_position = df[f"{prefix}has_position"].shift(1).eq(True)
        same_position = (
            current_has_position
            & prev_has_position
            & (
                df[f"{prefix}position_call_code"]
                == df[f"{prefix}position_call_code"].shift(1)
            )
            & (
                df[f"{prefix}position_put_code"]
                == df[f"{prefix}position_put_code"].shift(1)
            )
        )
        explainable_day = prev_has_position & df[f"{prefix}pnl_position_iv"].notna()
        position_changed = prev_has_position & ~same_position

        call_iv_chg = df[f"{prefix}pnl_call_iv"] - df[f"{prefix}eod_call_iv"].shift(1)
        put_iv_chg = df[f"{prefix}pnl_put_iv"] - df[f"{prefix}eod_put_iv"].shift(1)
        avg_call_gamma = (
            df[f"{prefix}eod_call_gamma"].shift(1)
            + df[f"{prefix}pnl_call_gamma"]
        ) / 2
        avg_put_gamma = (
            df[f"{prefix}eod_put_gamma"].shift(1)
            + df[f"{prefix}pnl_put_gamma"]
        ) / 2
        avg_call_vega = (
            df[f"{prefix}eod_call_vega"].shift(1)
            + df[f"{prefix}pnl_call_vega"]
        ) / 2
        avg_put_vega = (
            df[f"{prefix}eod_put_vega"].shift(1)
            + df[f"{prefix}pnl_put_vega"]
        ) / 2
        avg_call_theta = (
            df[f"{prefix}eod_call_theta"].shift(1)
            + df[f"{prefix}pnl_call_theta"]
        ) / 2
        avg_put_theta = (
            df[f"{prefix}eod_put_theta"].shift(1)
            + df[f"{prefix}pnl_put_theta"]
        ) / 2

        df[f"{prefix}delta_pnl"] = (
            df[f"{prefix}eod_call_delta"].shift(1)
            + df[f"{prefix}eod_put_delta"].shift(1)
        ) * spot_chg
        df[f"{prefix}gamma_pnl"] = 0.5 * (avg_call_gamma + avg_put_gamma) * spot_chg**2
        df[f"{prefix}vega_pnl"] = (
            avg_call_vega * call_iv_chg * 100
            + avg_put_vega * put_iv_chg * 100
        )
        df[f"{prefix}theta_pnl"] = avg_call_theta + avg_put_theta

        side_cols = [
            f"{prefix}delta_pnl",
            f"{prefix}gamma_pnl",
            f"{prefix}vega_pnl",
            f"{prefix}theta_pnl",
        ]
        df.loc[~explainable_day, side_cols] = 0.0
        df[f"{prefix}greeks_pnl"] = df[side_cols].sum(axis=1, min_count=len(side_cols))
        df[f"{prefix}greeks_explainable_day"] = explainable_day
        df[f"{prefix}greeks_position_changed_day"] = position_changed

    if "hedge_price" in df.columns:
        hedge_price_chg = df["hedge_price"].diff()
    else:
        hedge_price_chg = spot_chg
    if "hedge_underlying_order_book_id" in df.columns:
        hedge_underlying = df["hedge_underlying_order_book_id"]
        prev_hedge_underlying = hedge_underlying.shift(1)
        same_hedge_underlying = (
            hedge_underlying.eq(prev_hedge_underlying)
            | (hedge_underlying.isna() & prev_hedge_underlying.isna())
        )
    else:
        same_hedge_underlying = pd.Series(True, index=df.index)
    df["hedge_delta_pnl"] = df["hedge_etf_qty"].shift(1) * hedge_price_chg
    df.loc[~same_hedge_underlying.fillna(False), "hedge_delta_pnl"] = 0.0
    df["delta_pnl"] = (
        df["long_delta_pnl"] + df["short_delta_pnl"] + df["hedge_delta_pnl"]
    )

    prev_side_delta = {}
    for side in POSITION_SIDES:
        prefix = f"{side}_"
        prev_side_delta[side] = (
            df[f"{prefix}eod_call_delta"].shift(1)
            + df[f"{prefix}eod_put_delta"].shift(1)
        )
    prev_option_delta = sum(prev_side_delta.values())
    for side in POSITION_SIDES:
        prefix = f"{side}_"
        hedge_share = prev_side_delta[side] / prev_option_delta
        hedge_share = hedge_share.where(prev_option_delta != 0, 0.0).fillna(0.0)
        df[f"{prefix}hedge_delta_pnl"] = df["hedge_delta_pnl"] * hedge_share
        df[f"{prefix}total_delta_pnl"] = (
            df[f"{prefix}delta_pnl"] + df[f"{prefix}hedge_delta_pnl"]
        )

    df["gamma_pnl"] = df["long_gamma_pnl"] + df["short_gamma_pnl"]
    df["vega_pnl"] = df["long_vega_pnl"] + df["short_vega_pnl"]
    df["theta_pnl"] = df["long_theta_pnl"] + df["short_theta_pnl"]
    pnl_cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]
    df["greeks_pnl"] = df[pnl_cols].sum(axis=1, min_count=len(pnl_cols))

    df["daily_nav_pnl"] = df["nav"].diff()
    df["daily_fee"] = df["etf_fee"].fillna(0.0) + df["option_fee"].fillna(0.0)
    df["daily_nav_pnl_before_fee"] = df["daily_nav_pnl"] + df["daily_fee"]
    df["greeks_unexplained_pnl"] = df["daily_nav_pnl"] - df["greeks_pnl"]
    df["greeks_unexplained_pnl_before_fee"] = (
        df["daily_nav_pnl_before_fee"] - df["greeks_pnl"]
    )
    df["greeks_explainable_day"] = (
        df["long_greeks_explainable_day"] | df["short_greeks_explainable_day"]
    )
    df["greeks_position_changed_day"] = (
        df["long_greeks_position_changed_day"]
        | df["short_greeks_position_changed_day"]
    )
    df["pnl_position_side"] = None
    long_only = df["long_greeks_explainable_day"] & ~df["short_greeks_explainable_day"]
    short_only = df["short_greeks_explainable_day"] & ~df["long_greeks_explainable_day"]
    both = df["long_greeks_explainable_day"] & df["short_greeks_explainable_day"]
    df.loc[long_only, "pnl_position_side"] = "long"
    df.loc[short_only, "pnl_position_side"] = "short"
    df.loc[both, "pnl_position_side"] = "both"

    actual_abs = df["daily_nav_pnl"].abs()
    residual_abs = df["greeks_unexplained_pnl"].abs()
    actual_before_fee_abs = df["daily_nav_pnl_before_fee"].abs()
    residual_before_fee_abs = df["greeks_unexplained_pnl_before_fee"].abs()
    df["greeks_explain_ratio"] = pd.NA
    df["greeks_explain_ratio_before_fee"] = pd.NA
    actual_mask = actual_abs != 0
    actual_before_fee_mask = actual_before_fee_abs != 0
    df.loc[actual_mask, "greeks_explain_ratio"] = (
        1 - residual_abs.loc[actual_mask] / actual_abs.loc[actual_mask]
    )
    df.loc[actual_before_fee_mask, "greeks_explain_ratio_before_fee"] = (
        1
        - residual_before_fee_abs.loc[actual_before_fee_mask]
        / actual_before_fee_abs.loc[actual_before_fee_mask]
    )
    return df

class BacktestEngine:
    """按交易日推进回测；long/short 独立持仓，现金和 ETF 对冲为账户级。"""

    def __init__(
        self,
        etf_by_date,
        opt_by_date,
        signals_df,
        config,
        trading_calendar=None,
        enriched_opt_by_date=None,
        hedge_by_date=None,
    ):
        self.etf_by_date = etf_by_date
        self.opt_by_date = opt_by_date
        self.enriched_opt_by_date = enriched_opt_by_date
        self.hedge_by_date = hedge_by_date
        self.signals_df = signals_df
        self.config = config
        self.daily_ohlc = vol_engine.build_daily_ohlc_df(etf_by_date)
        if trading_calendar is None:
            trading_calendar = self.daily_ohlc.index
        self.trading_calendar = pd.DatetimeIndex(trading_calendar)

    def run(self):
        state = BacktestState(cash=self.config["initial_cash"])

        for date in self.signals_df.index:
            if date not in self.daily_ohlc.index:
                continue

            spot = self.daily_ohlc.loc[date, "close"]
            feature_row = self.signals_df.loc[date]

            if date not in self.opt_by_date:
                self._handle_missing_option_day(date, spot, feature_row, state)
                continue

            has_cached_chain = (
                self.enriched_opt_by_date is not None
                and date in self.enriched_opt_by_date
            )
            if has_cached_chain:
                chain_df = self.enriched_opt_by_date[date]
            else:
                chain_df = vol_engine.add_iv_for_day(
                    self.opt_by_date[date],
                    spot,
                    trading_calendar=self.trading_calendar,
                )
                chain_df = vol_engine.add_greeks_for_day(chain_df, spot)

            day = self._new_day(date, spot, feature_row, chain_df)

            if pd.isna(feature_row["atm_iv"]) and not self._has_any_position(state):
                continue

            if self._has_any_position(state):
                self._mark_current_positions_for_capacity(day, state)
                if day["defer_delta_hedge"] or self._enforce_margin_limit(day, state):
                    self._record_day(day, state)
                    continue

            for side in POSITION_SIDES:
                if state.positions[side] is not None:
                    self._handle_existing_position(day, state, side)

            for side, signal_col, trade_type in [
                ("long", "long_open_signal", "open_straddle"),
                ("short", "short_open_signal", "open_short_straddle"),
            ]:
                if (
                    not self._has_any_position(state)
                    and state.positions[side] is None
                    and feature_row.get(signal_col, False)
                    and side not in day["skip_new_entry_by_side"]
                    and state.roll_cooldown_left[side] <= 0
                ):
                    self._open_new_position(day, state, trade_type=trade_type, side=side)

            self._tick_flat_cooldowns(day, state)
            self._update_day_aggregates(day, state)
            if not day["defer_delta_hedge"]:
                self._hedge_to(date, spot, state, day, day["greeks"])
                if self.config.get("dynamic_position_control_enabled", False):
                    self._enforce_margin_limit(day, state)
            self._record_day(day, state)

        daily_df = pd.DataFrame(state.daily_records).set_index("date")
        daily_df = add_greeks_pnl(daily_df)
        trades_df = pd.DataFrame(state.trades)
        return daily_df, trades_df

    def _new_day(self, date, spot, feature_row, chain_df):
        return {
            "date": date,
            "spot": spot,
            "feature_row": feature_row,
            "chain_df": chain_df,
            "side_records": {side: empty_side_record() for side in POSITION_SIDES},
            "option_value": 0.0,
            "core_option_value": 0.0,
            "option_hedge_value": 0.0,
            "option_hedge_margin": 0.0,
            "option_hedge_greeks": empty_greeks(),
            "greeks": empty_greeks(),
            "pnl_position_iv": None,
            "pnl_call_iv": None,
            "pnl_put_iv": None,
            "pnl_greeks": empty_greeks(),
            "eod_position_dte": None,
            "daily_etf_fee": 0.0,
            "daily_option_fee": 0.0,
            "skip_new_entry_by_side": set(),
            "cooldown_started_by_side": set(),
            "defer_delta_hedge": False,
            "data_warnings": [],
        }

    def _has_any_position(self, state):
        return any(state.positions.get(side) is not None for side in POSITION_SIDES)

    def _handle_missing_option_day(self, date, spot, feature_row, state):
        """缺少期权链时，所有已有期权仓位按末值离场，ETF 对冲归零。"""
        if not self._has_any_position(state):
            print(
                f"[missing option chain] date={date.date()}, "
                f"spot={spot}, no position, skip day"
            )
            return

        day = self._new_day(date, spot, feature_row, None)
        for side in POSITION_SIDES:
            position = state.positions[side]
            if position is None:
                continue
            self._record_data_warning(day, state, side, "missing_option_chain")
            self._set_side_eod(
                day,
                state,
                side,
                float(position.get("last_option_value", 0.0) or 0.0),
                empty_greeks(),
                None,
            )
        day["defer_delta_hedge"] = True
        self._update_day_aggregates(day, state)
        self._record_day(day, state)

    def _handle_existing_position(self, day, state, side):
        """处理单侧已有跨式仓位：估值、平仓、展期或继续持有。"""
        date = day["date"]
        spot = day["spot"]
        feature_row = day["feature_row"]
        position = state.positions[side]
        side_record = day["side_records"][side]

        try:
            call_row, put_row = self._get_position_rows(day, state, side)
        except IndexError:
            self._record_data_warning(
                day,
                state,
                side,
                "missing_position_contracts",
            )
            self._set_side_eod(
                day,
                state,
                side,
                float(position.get("last_option_value", 0.0) or 0.0),
                empty_greeks(),
                None,
            )
            day["defer_delta_hedge"] = True
            return

        position_dte = int(call_row["dte"])
        pnl_greeks = strategy.calc_position_greeks(
            call_row,
            put_row,
            position["call_qty"],
            position["put_qty"],
            side=side,
        )
        side_record["pnl_position_iv"] = pnl_greeks["position_iv"]
        side_record["pnl_call_iv"] = pnl_greeks["call_iv"]
        side_record["pnl_put_iv"] = pnl_greeks["put_iv"]
        side_record["pnl_greeks"] = pnl_greeks.copy()

        if pd.isna(pnl_greeks["position_iv"]):
            self._record_data_warning(
                day,
                state,
                side,
                "missing_position_iv",
            )
            self._set_side_eod(
                day,
                state,
                side,
                opt_position.signed_value(position, call_row, put_row),
                empty_greeks(),
                position_dte,
            )
            day["defer_delta_hedge"] = True
            return

        if side == "short" and opt_position.has_short_volume_spike(
            position,
            call_row,
            put_row,
        ):
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_trade(
                date,
                state.cash,
                position,
                call_row,
                put_row,
                state.trades,
                exit_reason="short_volume_spike",
            )
            self._add_new_option_fees(day, state, trade_count)
            state.positions[side] = None
            self._start_cooldown(day, state, side)
            day["skip_new_entry_by_side"].add(side)
            return

        if side == "short":
            current_market_value = opt_position.value(position, call_row, put_row)
            if strategy.is_short_stop_loss(position, current_market_value):
                close_reason = "short_stop_loss"
            else:
                close_reason = strategy.get_short_close_reason(
                    feature_row,
                    position_dte,
                    position,
                )
        else:
            close_reason = strategy.get_close_reason(feature_row, position_dte)

        self._update_roll_buffer(feature_row, state, side)
        roll_signal = self._should_roll_position(day, state, side, position_dte)

        if close_reason is not None:
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_trade(
                date,
                state.cash,
                position,
                call_row,
                put_row,
                state.trades,
                exit_reason=close_reason,
            )
            self._add_new_option_fees(day, state, trade_count)
            state.positions[side] = None
            self._start_cooldown(day, state, side)
            day["skip_new_entry_by_side"].add(side)
            if side == "long" and close_reason == "iv_high":
                short_cooldown_days = (
                    CONFIG.strategy.short_cooldown_after_long_iv_high_exit_days
                )
                self._start_cooldown_for_days(
                    day,
                    state,
                    "short",
                    short_cooldown_days,
                )
                if short_cooldown_days > 0:
                    day["skip_new_entry_by_side"].add("short")
            return

        if roll_signal:
            self._roll_position(day, state, side, call_row, put_row, pnl_greeks)
            return

        self._set_existing_position_eod(
            day,
            state,
            side,
            call_row,
            put_row,
            pnl_greeks,
            position_dte,
        )
        position["last_option_value"] = day["side_records"][side]["option_value"]
        self._tick_roll_cooldown(state, side)

    def _mark_current_positions_for_capacity(self, day, state):
        for side in POSITION_SIDES:
            position = state.positions.get(side)
            if position is None:
                continue
            try:
                call_row, put_row = self._get_position_rows(day, state, side)
            except IndexError:
                self._record_data_warning(
                    day,
                    state,
                    side,
                    "missing_position_contracts",
                )
                self._set_side_eod(
                    day,
                    state,
                    side,
                    float(position.get("last_option_value", 0.0) or 0.0),
                    empty_greeks(),
                    None,
                )
                day["defer_delta_hedge"] = True
                continue
            greeks = strategy.calc_position_greeks(
                call_row,
                put_row,
                position["call_qty"],
                position["put_qty"],
                side=side,
            )
            if pd.isna(greeks["position_iv"]):
                self._record_data_warning(day, state, side, "missing_position_iv")
                greeks = empty_greeks()
                day["defer_delta_hedge"] = True
            record = day["side_records"][side]
            record["pnl_position_iv"] = greeks["position_iv"]
            record["pnl_call_iv"] = greeks["call_iv"]
            record["pnl_put_iv"] = greeks["put_iv"]
            record["pnl_greeks"] = greeks.copy()
            self._refresh_short_margin(day, state, side, call_row, put_row)
            self._set_side_eod(
                day,
                state,
                side,
                opt_position.signed_value(position, call_row, put_row),
                greeks,
                int(call_row["dte"]),
            )
        self._update_day_aggregates(day, state)

    def _refresh_short_margin(self, day, state, side, call_row, put_row):
        position = state.positions.get(side)
        if position is None or position.get("side", "long") != "short":
            return
        underlying_price = call_row.get("underlying_close")
        if pd.isna(underlying_price):
            underlying_price = self._position_underlying_price(
                day,
                position,
                day["spot"],
            )
        old_margin = float(position.get("option_margin", 0.0) or 0.0)
        new_margin = opt_position.calc_short_margin(
            call_row,
            put_row,
            position["call_qty"],
            position["put_qty"],
            underlying_price,
        )
        state.cash -= new_margin - old_margin
        position["option_margin"] = new_margin

    def _entry_target_qty(self, feature_row, max_qty, side):
        if side == "short":
            return strategy.calc_short_entry_target_qty(feature_row, max_qty)
        return strategy.calc_entry_target_qty(feature_row, max_qty)

    def _side_max_qty(self, side):
        """按方向读取每腿张数；跨式组合默认 call/put 等量。"""
        return self.config[f"{side}_qty"]

    def _proportional_side_max_qty(self, day, state, side):
        base_qty = self._side_max_qty(side)
        if not self.config.get("proportional_position_sizing_enabled", False):
            return base_qty

        base_nav = float(self.config["position_sizing_base_nav"])
        if base_nav <= 0:
            raise ValueError("position_sizing_base_nav must be positive")
        nav = max(0.0, float(self._current_nav_and_margin(day, state)[0]))
        return int(base_qty * nav // base_nav)

    def _dynamic_target_qty(self, day, state, atm, requested_qty, side, replacing_side=None):
        if not self.config.get("dynamic_position_control_enabled", False):
            return requested_qty
        nav = self._current_nav_and_margin(day, state)[0]
        occupation_limit = max(0.0, nav * self.config["max_margin_to_nav_ratio"])
        other_occupation = sum(
            self._position_capital_occupation(position)
            for candidate_side, position in state.positions.items()
            if position is not None and candidate_side != replacing_side
        )
        other_occupation += sum(
            self._option_hedge_capital_occupation(position)
            for position in state.option_hedges
        )
        other_greeks = combine_greeks(
            [
                day["side_records"][candidate_side]["greeks"]
                for candidate_side in POSITION_SIDES
                if candidate_side != replacing_side
            ]
        )
        for qty in range(int(requested_qty), 0, -1):
            projected = strategy.calc_position_greeks(
                atm["call"],
                atm["put"],
                qty,
                qty,
                side=side,
            )
            if side == "short":
                option_occupation = opt_position.calc_short_margin(
                    atm["call"],
                    atm["put"],
                    qty,
                    qty,
                    self._atm_underlying_price(atm, day["spot"]),
                )
            else:
                option_occupation = opt_position.calc_trade_value(
                    atm["call"],
                    atm["put"],
                    qty,
                    qty,
                )
            projected_delta = float(other_greeks["delta"]) + float(projected["delta"])
            hedge_occupation = abs(projected_delta) * float(day["spot"])
            if (
                other_occupation + option_occupation + hedge_occupation
                <= occupation_limit + 1e-6
            ):
                return qty
        return 0

    def _should_roll_position(self, day, state, side, position_dte):
        if state.roll_cooldown_left[side] > 0 or pd.isna(day["feature_row"]["atm_strike"]):
            return False

        position = state.positions[side]
        dte_too_low = position_dte <= CONFIG.strategy.roll_dte_threshold
        strike_roll_ready = (
            position["strike"] != day["feature_row"]["atm_strike"]
            and state.strike_mismatch_days[side]
            >= CONFIG.strategy.roll_strike_mismatch_days
        )
        if not (dte_too_low or strike_roll_ready):
            return False

        target_qty = self._entry_target_qty(
            day["feature_row"],
            self._proportional_side_max_qty(day, state, side),
            side,
        )
        return target_qty > 0

    def _roll_position(self, day, state, side, call_row, put_row, pnl_greeks):
        date = day["date"]
        spot = day["spot"]
        position = state.positions[side]
        atm = vol_engine.select_atm_from_chain(
            day["chain_df"],
            spot,
            target_dte_min=CONFIG.strategy.roll_dte_threshold + 1,
        )

        if atm is None:
            self._set_existing_position_eod(
                day,
                state,
                side,
                call_row,
                put_row,
                pnl_greeks,
                int(call_row["dte"]),
            )
            self._start_cooldown(day, state, side)
            return

        call_qty = self._entry_target_qty(
            day["feature_row"],
            self._proportional_side_max_qty(day, state, side),
            side,
        )
        put_qty = self._entry_target_qty(
            day["feature_row"],
            self._proportional_side_max_qty(day, state, side),
            side,
        )
        call_qty = put_qty = self._dynamic_target_qty(
            day,
            state,
            atm,
            min(call_qty, put_qty),
            side,
            replacing_side=side,
        )
        if call_qty <= 0 or put_qty <= 0:
            self._set_existing_position_eod(
                day,
                state,
                side,
                call_row,
                put_row,
                pnl_greeks,
                int(call_row["dte"]),
            )
            self._start_cooldown(day, state, side)
            return

        projected_cash = self._project_cash_after_option_close(
            state.cash,
            position,
            call_row,
            put_row,
        )
        projected_cash = self._project_cash_after_option_open(
            projected_cash,
            atm,
            call_qty,
            put_qty,
            side,
            self._atm_underlying_price(atm, spot),
        )
        projected_greeks = strategy.calc_position_greeks(
            atm["call"],
            atm["put"],
            call_qty,
            put_qty,
            side=side,
        )
        cash_would_decrease = projected_cash < state.cash
        if cash_would_decrease and not self._has_cash_reserve(projected_cash):
            self._record_cash_reserve_skip(
                date,
                state,
                "skip_roll_cash_reserve",
                projected_cash,
                side,
            )
            self._set_existing_position_eod(
                day,
                state,
                side,
                call_row,
                put_row,
                pnl_greeks,
                int(call_row["dte"]),
            )
            self._start_cooldown(day, state, side)
            return

        trade_count = len(state.trades)
        state.cash, _ = opt_position.close_trade(
            date,
            state.cash,
            position,
            call_row,
            put_row,
            state.trades,
            trade_type="roll_close_straddle",
        )
        state.cash, new_position, option_value = opt_position.open_trade(
            date,
            state.cash,
            atm,
            call_qty,
            put_qty,
            state.trades,
            trade_type="roll_open_straddle",
            side=side,
            spot=self._atm_underlying_price(atm, spot),
            short_entry_regime=(
                position.get("short_entry_regime", CONFIG.strategy.short_signal_mode)
                if side == "short"
                else None
            ),
        )
        state.positions[side] = new_position
        self._add_new_option_fees(day, state, trade_count)
        self._set_side_eod(day, state, side, option_value, projected_greeks, atm["dte"])
        self._start_cooldown(day, state, side)

    def _open_new_position(self, day, state, trade_type, side="long"):
        date = day["date"]
        spot = day["spot"]
        atm = vol_engine.select_atm_from_chain(day["chain_df"], spot)
        if atm is None:
            return

        call_qty = self._entry_target_qty(
            day["feature_row"],
            self._proportional_side_max_qty(day, state, side),
            side,
        )
        put_qty = self._entry_target_qty(
            day["feature_row"],
            self._proportional_side_max_qty(day, state, side),
            side,
        )
        call_qty = put_qty = self._dynamic_target_qty(
            day,
            state,
            atm,
            min(call_qty, put_qty),
            side,
        )
        if call_qty <= 0 or put_qty <= 0:
            return

        projected_cash = self._project_cash_after_option_open(
            state.cash,
            atm,
            call_qty,
            put_qty,
            side,
            self._atm_underlying_price(atm, spot),
        )
        projected_greeks = strategy.calc_position_greeks(
            atm["call"],
            atm["put"],
            call_qty,
            put_qty,
            side=side,
        )
        if not self._has_cash_reserve(projected_cash):
            self._record_cash_reserve_skip(
                date,
                state,
                "skip_open_cash_reserve",
                projected_cash,
                side,
            )
            return

        trade_count = len(state.trades)
        short_entry_regime = None
        if side == "short":
            short_entry_regime = strategy.get_short_open_regime(day["feature_row"])
        state.cash, new_position, option_value = opt_position.open_trade(
            date,
            state.cash,
            atm,
            call_qty,
            put_qty,
            state.trades,
            trade_type=trade_type,
            side=side,
            spot=self._atm_underlying_price(atm, spot),
            short_entry_regime=short_entry_regime,
        )
        state.positions[side] = new_position
        self._add_new_option_fees(day, state, trade_count)
        self._set_side_eod(day, state, side, option_value, projected_greeks, atm["dte"])
        self._reset_roll_state(state, side)

    def _set_side_eod(self, day, state, side, option_value, greeks, dte):
        record = day["side_records"][side]
        record["option_value"] = option_value
        record["greeks"] = greeks.copy()
        record["eod_position_dte"] = dte
        position = state.positions[side]
        if position is not None:
            position["last_option_value"] = option_value

    def _set_existing_position_eod(
        self,
        day,
        state,
        side,
        call_row,
        put_row,
        greeks,
        dte,
    ):
        self._set_side_eod(
            day,
            state,
            side,
            opt_position.signed_value(state.positions[side], call_row, put_row),
            greeks,
            dte,
        )

    def _chain_row(self, chain_df, order_book_id):
        if chain_df is None:
            raise IndexError(order_book_id)
        rows = chain_df[chain_df["order_book_id"].astype(str) == str(order_book_id)]
        if rows.empty:
            raise IndexError(order_book_id)
        return rows.iloc[0]

    def _single_option_signed_value(self, hedge_position, row):
        direction = -1.0 if hedge_position.get("side") == "short" else 1.0
        return (
            direction
            * float(row["mid"])
            * int(hedge_position.get("qty", 0) or 0)
            * float(row["contract_multiplier"])
        )

    def _update_option_hedge_marks(self, day, state):
        option_value = 0.0
        margin = 0.0
        greeks_list = []
        for hedge_position in state.option_hedges:
            try:
                row = self._chain_row(day["chain_df"], hedge_position["order_book_id"])
            except (IndexError, TypeError):
                option_value += hedge_position.get("last_option_value", 0.0)
                margin += hedge_position.get("option_margin", 0.0)
                continue
            qty = int(hedge_position.get("qty", 0) or 0)
            side = hedge_position.get("side", "long")
            signed_value = self._single_option_signed_value(hedge_position, row)
            option_value += signed_value
            if side == "short":
                old_margin = float(hedge_position.get("option_margin", 0.0) or 0.0)
                underlying_price = row.get("underlying_close")
                if pd.isna(underlying_price):
                    underlying_price = day["spot"]
                new_margin = opt_position.margin_call(
                    float(underlying_price),
                    float(row["strike_price"]),
                    float(row["mid"]),
                    float(row["contract_multiplier"]),
                ) * qty
                state.cash -= new_margin - old_margin
                hedge_position["option_margin"] = new_margin
                margin += new_margin
            hedge_position["last_price"] = float(row["mid"])
            hedge_position["last_option_value"] = signed_value
            greeks_list.append(single_call_greeks(row, qty, side))
        day["option_hedge_value"] = option_value
        day["option_hedge_margin"] = margin
        day["option_hedge_greeks"] = combine_greeks(greeks_list)

    def _update_day_aggregates(self, day, state=None):
        if state is not None:
            self._update_option_hedge_marks(day, state)
            for side in POSITION_SIDES:
                if state.positions.get(side) is None:
                    record = day["side_records"][side]
                    record["option_value"] = 0.0
                    record["greeks"] = empty_greeks()
                    record["eod_position_dte"] = None
        side_records = day["side_records"]
        day["core_option_value"] = sum(
            side_records[side]["option_value"] for side in POSITION_SIDES
        )
        day["option_value"] = day["core_option_value"] + day.get(
            "option_hedge_value",
            0.0,
        )
        day["greeks"] = combine_greeks(
            [
                *[side_records[side]["greeks"] for side in POSITION_SIDES],
                day.get("option_hedge_greeks"),
            ]
        )
        day["pnl_greeks"] = combine_greeks(
            [side_records[side]["pnl_greeks"] for side in POSITION_SIDES]
        )
        day["pnl_position_iv"] = day["pnl_greeks"]["position_iv"]
        day["pnl_call_iv"] = day["pnl_greeks"]["call_iv"]
        day["pnl_put_iv"] = day["pnl_greeks"]["put_iv"]
        dtes = [
            side_records[side]["eod_position_dte"]
            for side in POSITION_SIDES
            if side_records[side]["eod_position_dte"] is not None
        ]
        day["eod_position_dte"] = min(dtes) if dtes else None

    def _min_cash_reserve(self):
        return self.config.get("min_cash_reserve", CONFIG.backtest.min_cash_reserve)

    def _has_cash_reserve(self, projected_cash):
        return projected_cash >= self._min_cash_reserve()

    def _project_cash_after_option_open(self, cash, atm, call_qty, put_qty, side, spot):
        trade_value = opt_position.calc_trade_value(
            atm["call"],
            atm["put"],
            call_qty,
            put_qty,
        )
        fee = opt_position.calc_option_fee(call_qty, put_qty)
        if side == "short":
            option_margin = opt_position.calc_short_margin(
                atm["call"],
                atm["put"],
                call_qty,
                put_qty,
                spot,
            )
            return cash + trade_value - fee - option_margin
        return cash - trade_value - fee

    def _atm_underlying_price(self, atm, fallback_price):
        value = atm.get("underlying_price")
        if pd.notna(value):
            return float(value)
        return fallback_price

    def _position_underlying_price(self, day, position, fallback_price):
        return self._get_hedge_price(
            day["date"],
            position.get("underlying_order_book_id"),
            fallback_price,
        )

    def _project_cash_after_option_close(self, cash, position, call_row, put_row):
        close_value = opt_position.value(position, call_row, put_row)
        fee = opt_position.calc_option_fee(position["call_qty"], position["put_qty"])
        if position.get("side", "long") == "short":
            return cash + position.get("option_margin", 0.0) - close_value - fee
        return cash + close_value - fee

    def _get_hedge_price(self, date, underlying_order_book_id, fallback_price):
        if underlying_order_book_id is None:
            return fallback_price
        if self.hedge_by_date is None:
            raise ValueError(
                "启用 delta hedge 时缺少 hedge 标的数据；"
                f"无法为 {underlying_order_book_id} 取价"
            )
        hedge_df = self.hedge_by_date.get(pd.Timestamp(date))
        if hedge_df is None:
            raise ValueError(f"{date} 缺少 hedge 标的数据")
        rows = hedge_df[
            hedge_df["order_book_id"].astype(str) == str(underlying_order_book_id)
        ]
        if rows.empty:
            raise ValueError(f"{date} 缺少 hedge 标的 {underlying_order_book_id}")
        return float(rows.iloc[0]["close"])

    def _get_hedge_volume(self, date, underlying_order_book_id):
        if self.hedge_by_date is None:
            return None
        hedge_df = self.hedge_by_date.get(pd.Timestamp(date))
        if hedge_df is None or hedge_df.empty or "volume" not in hedge_df.columns:
            return None
        if underlying_order_book_id is None and len(hedge_df) == 1:
            return float(hedge_df.iloc[0]["volume"])
        rows = hedge_df[
            hedge_df["order_book_id"].astype(str) == str(underlying_order_book_id)
        ]
        if rows.empty:
            return None
        return float(rows.iloc[0]["volume"])

    def _active_hedge_underlying_order_book_id(self, state):
        ids = {
            position.get("underlying_order_book_id")
            for position in state.positions.values()
            if position is not None and position.get("underlying_order_book_id")
        }
        if len(ids) > 1:
            raise ValueError(f"当前账户持有多个 hedge 标的，暂不支持账户级单腿对冲: {ids}")
        return next(iter(ids), None)

    def _project_cash_after_hedge(
        self,
        cash,
        state,
        date,
        fallback_price,
        target_qty,
        target_underlying_order_book_id=None,
    ):
        if not self.config.get("allow_etf_short_hedge", True):
            target_qty = max(0.0, float(target_qty))
        if target_qty == 0:
            target_underlying_order_book_id = state.hedge_underlying_order_book_id
        target_price = self._get_hedge_price(
            date,
            target_underlying_order_book_id,
            fallback_price,
        )
        current_price = self._get_hedge_price(
            date,
            state.hedge_underlying_order_book_id,
            fallback_price,
        )

        fee_notional = 0.0
        underlying_changed = (
            state.hedge_etf_qty != 0
            and target_qty != 0
            and state.hedge_underlying_order_book_id
            != target_underlying_order_book_id
        )
        if underlying_changed:
            projected_cash, _ = hedge.close_etf_hedge(
                cash,
                state.hedge_etf_qty,
                state.hedge_entry_price,
                state.hedge_margin,
                current_price,
            )
            fee_notional += abs(state.hedge_etf_qty) * current_price
            projected_cash, _, _, _, _ = hedge.rebalance_etf_hedge(
                projected_cash,
                0.0,
                0.0,
                0.0,
                target_qty,
                target_price,
            )
            fee_notional += abs(target_qty) * target_price
        else:
            projected_cash, _, _, _, _ = hedge.rebalance_etf_hedge(
                cash,
                state.hedge_etf_qty,
                state.hedge_entry_price,
                state.hedge_margin,
                target_qty,
                target_price,
            )
            fee_notional += abs(target_qty - state.hedge_etf_qty) * target_price

        etf_fee = fee_notional * self.config["etf_fee_rate"]
        return projected_cash - etf_fee

    def _record_cash_reserve_skip(self, date, state, trade_type, projected_cash, side=None):
        state.trades.append(
            {
                "date": date,
                "type": trade_type,
                "side": side,
                "projected_cash": projected_cash,
                "min_cash_reserve": self._min_cash_reserve(),
            }
        )

    def _update_roll_buffer(self, feature_row, state, side):
        position = state.positions[side]
        if (
            pd.notna(feature_row["atm_strike"])
            and position is not None
            and position["strike"] != feature_row["atm_strike"]
        ):
            state.strike_mismatch_days[side] += 1
            return

        state.strike_mismatch_days[side] = 0

    def _tick_roll_cooldown(self, state, side):
        if state.roll_cooldown_left[side] > 0:
            state.roll_cooldown_left[side] -= 1

    def _tick_flat_cooldowns(self, day, state):
        for side in POSITION_SIDES:
            if state.positions.get(side) is not None:
                continue
            if side in day["cooldown_started_by_side"]:
                continue
            self._tick_roll_cooldown(state, side)

    def _start_cooldown_for_days(self, day, state, side, days):
        if days <= 0:
            return
        state.strike_mismatch_days[side] = 0
        state.roll_cooldown_left[side] = max(state.roll_cooldown_left[side], days)
        day["cooldown_started_by_side"].add(side)

    def _start_cooldown(self, day, state, side):
        self._start_cooldown_for_days(
            day,
            state,
            side,
            CONFIG.strategy.roll_cooldown_days,
        )

    def _reset_roll_state(self, state, side=None):
        sides = POSITION_SIDES if side is None else (side,)
        for item in sides:
            state.strike_mismatch_days[item] = 0
            state.roll_cooldown_left[item] = 0

    def _execute_etf_target(self, date, spot, state, day, greeks, target_qty, trade_type):
        target_qty = float(target_qty)
        if not self.config.get("allow_etf_short_hedge", True):
            target_qty = max(0.0, target_qty)
        target_underlying_order_book_id = (
            state.hedge_underlying_order_book_id
            if target_qty == 0
            else self._active_hedge_underlying_order_book_id(state)
        )
        projected_cash = self._project_cash_after_hedge(
            state.cash,
            state,
            date,
            spot,
            target_qty,
            target_underlying_order_book_id,
        )
        if projected_cash < state.cash and not self._has_cash_reserve(projected_cash):
            self._record_cash_reserve_skip(
                date,
                state,
                "skip_delta_hedge_cash_reserve",
                projected_cash,
            )
            return False

        trade_count = len(state.trades)
        (
            state.cash,
            state.hedge_etf_qty,
            state.hedge_entry_price,
            state.hedge_margin,
            state.hedge_underlying_order_book_id,
        ) = execute_delta_hedge(
            date,
            state.cash,
            greeks,
            state.hedge_etf_qty,
            state.hedge_entry_price,
            state.hedge_margin,
            self._get_hedge_price(date, target_underlying_order_book_id, spot),
            state.trades,
            target_qty=target_qty,
            trade_type=trade_type,
            etf_fee_rate=self.config["etf_fee_rate"],
            underlying_order_book_id=target_underlying_order_book_id,
            current_price=self._get_hedge_price(
                date,
                state.hedge_underlying_order_book_id,
                spot,
            ),
            current_underlying_order_book_id=state.hedge_underlying_order_book_id,
            daily_volume=self._get_hedge_volume(
                date,
                target_underlying_order_book_id
                or state.hedge_underlying_order_book_id,
            ),
        )
        if day is not None and len(state.trades) > trade_count:
            day["daily_etf_fee"] += state.trades[-1].get("fee", 0.0)
        return True

    def _current_nav_and_margin(self, day, state):
        hedge_price = self._get_hedge_price(
            day["date"],
            state.hedge_underlying_order_book_id,
            day["spot"],
        )
        hedge_unrealized_pnl = hedge.calc_unrealized_pnl(
            state.hedge_etf_qty,
            state.hedge_entry_price,
            hedge_price,
        )
        option_margin = sum(
            opt_position.margin_value(position)
            for position in state.positions.values()
            if position is not None
        ) + day.get("option_hedge_margin", 0.0)
        total_margin = option_margin + state.hedge_margin
        nav = (
            state.cash
            + day.get("option_value", 0.0)
            + option_margin
            + state.hedge_margin
            + hedge_unrealized_pnl
        )
        return nav, total_margin

    def _position_capital_occupation(self, position):
        if position is None:
            return 0.0
        if position.get("side", "long") == "short":
            return float(position.get("option_margin", 0.0) or 0.0)
        return max(0.0, float(position.get("last_option_value", 0.0) or 0.0))

    def _option_hedge_capital_occupation(self, position):
        if position.get("side", "long") == "short":
            return float(position.get("option_margin", 0.0) or 0.0)
        return max(0.0, float(position.get("last_option_value", 0.0) or 0.0))

    def _current_nav_and_occupation(self, day, state):
        nav = self._current_nav_and_margin(day, state)[0]
        occupation = sum(
            self._position_capital_occupation(position)
            for position in state.positions.values()
            if position is not None
        )
        occupation += sum(
            self._option_hedge_capital_occupation(position)
            for position in state.option_hedges
        )
        hedge_price = self._get_hedge_price(
            day["date"],
            state.hedge_underlying_order_book_id,
            day["spot"],
        )
        occupation += abs(float(state.hedge_etf_qty)) * float(hedge_price)
        return nav, occupation

    def _reduce_position_for_margin(self, day, state, side, target_qty):
        position = state.positions.get(side)
        if position is None:
            return False
        call_row, put_row = self._get_position_rows(day, state, side)
        current_qty = min(int(position["call_qty"]), int(position["put_qty"]))
        target_qty = max(0, min(int(target_qty), current_qty - 1))
        close_qty = current_qty - target_qty
        if close_qty <= 0:
            return False

        multiplier = float(position["contract_multiplier"])
        close_value = (
            float(call_row["mid"]) + float(put_row["mid"])
        ) * close_qty * multiplier
        fee = opt_position.calc_option_fee(close_qty, close_qty)
        old_margin = float(position.get("option_margin", 0.0) or 0.0)
        if side == "short":
            new_margin = old_margin * target_qty / current_qty if target_qty > 0 else 0.0
            state.cash += old_margin - new_margin - close_value - fee
        else:
            new_margin = 0.0
            state.cash += close_value - fee

        state.trades.append(
            {
                "date": day["date"],
                "type": "reduce_short_straddle_for_capacity",
                "side": side,
                "call_code": position["call_code"],
                "put_code": position["put_code"],
                "old_qty": current_qty,
                "new_qty": target_qty,
                "trade_call_qty": -close_qty,
                "trade_put_qty": -close_qty,
                "fee": fee,
                "cash": state.cash,
                "margin_before": old_margin,
                "margin_after": new_margin,
                "margin_limit_ratio": self.config["max_margin_to_nav_ratio"],
            }
        )

        day["daily_option_fee"] += fee
        if target_qty == 0:
            state.positions[side] = None
            day["side_records"][side] = empty_side_record()
            return True

        position["call_qty"] = target_qty
        position["put_qty"] = target_qty
        position["option_margin"] = new_margin
        position["last_option_value"] = opt_position.signed_value(
            position,
            call_row,
            put_row,
        )
        greeks = strategy.calc_position_greeks(
            call_row,
            put_row,
            target_qty,
            target_qty,
            side=side,
        )
        self._set_side_eod(
            day,
            state,
            side,
            position["last_option_value"],
            greeks,
            int(call_row["dte"]),
        )
        return True

    def _record_data_warning(self, day, state, side, reason):
        warning = {
            "date": day["date"],
            "type": "data_warning",
            "side": side,
            "reason": reason,
        }
        day["data_warnings"].append(warning)
        state.trades.append(warning)

    def _enforce_margin_limit(self, day, state):
        if self.config.get("dynamic_position_control_enabled", False):
            return self._enforce_dynamic_occupation_limit(day, state)

        short_position = state.positions.get("short")
        if short_position is None:
            return False
        ratio_limit = float(self.config["max_margin_to_nav_ratio"])
        self._update_day_aggregates(day, state)
        nav, total_margin = self._current_nav_and_margin(day, state)
        margin_limit = max(0.0, nav * ratio_limit)
        cash_shortfall = max(0.0, self._min_cash_reserve() - state.cash)
        margin_excess = max(0.0, total_margin - margin_limit)
        if cash_shortfall <= 1e-6 and margin_excess <= 1e-6:
            return False

        current_qty = min(
            int(short_position.get("call_qty", 0) or 0),
            int(short_position.get("put_qty", 0) or 0),
        )
        if current_qty <= 0:
            return False
        call_row, put_row = self._get_position_rows(day, state, "short")
        option_margin = float(short_position.get("option_margin", 0.0) or 0.0)
        margin_relief_per_contract = option_margin / current_qty
        multiplier = float(short_position["contract_multiplier"])
        close_cost_per_contract = (
            float(call_row["mid"]) + float(put_row["mid"])
        ) * multiplier + opt_position.calc_option_fee(1, 1)
        cash_relief_per_contract = margin_relief_per_contract - close_cost_per_contract
        close_for_cash = (
            int(math.ceil(cash_shortfall / cash_relief_per_contract))
            if cash_shortfall > 0 and cash_relief_per_contract > 0
            else 0
        )
        close_for_margin = (
            int(math.ceil(margin_excess / margin_relief_per_contract))
            if margin_excess > 0 and margin_relief_per_contract > 0
            else 0
        )
        close_qty = min(current_qty, max(close_for_cash, close_for_margin))
        if close_qty <= 0:
            self._record_data_warning(
                day,
                state,
                "short",
                "capacity_reduction_not_feasible",
            )
            day["defer_delta_hedge"] = True
            return False
        reduced = self._reduce_position_for_margin(
            day,
            state,
            "short",
            current_qty - close_qty,
        )
        day["defer_delta_hedge"] = bool(reduced)
        return reduced

    def _enforce_dynamic_occupation_limit(self, day, state):
        ratio_limit = float(self.config["max_margin_to_nav_ratio"])
        changed = False
        for _ in range(100):
            self._update_day_aggregates(day, state)
            nav, occupation = self._current_nav_and_occupation(day, state)
            occupation_limit = max(0.0, nav * ratio_limit)
            if nav > 0 and occupation <= occupation_limit + 1e-6:
                return changed

            if state.option_hedges:
                self._close_option_delta_hedges(day["date"], state, day)
                self._update_day_aggregates(day, state)
                nav, occupation = self._current_nav_and_occupation(day, state)
                occupation_limit = max(0.0, nav * ratio_limit)
                if nav > 0 and occupation <= occupation_limit + 1e-6:
                    self._hedge_to(
                        day["date"],
                        day["spot"],
                        state,
                        day,
                        day["greeks"],
                        target_qty=strategy.round_etf_hedge_target(
                            -float(day["greeks"]["delta"])
                        ),
                    )
                    changed = True
                    continue

            candidates = [
                (self._position_capital_occupation(position), side, position)
                for side, position in state.positions.items()
                if position is not None
            ]
            if not candidates:
                break
            _, side, position = max(candidates, key=lambda item: item[0])
            current_qty = min(int(position["call_qty"]), int(position["put_qty"]))
            if current_qty <= 0:
                break
            target_qty = self._dynamic_reduction_target_qty(
                day,
                state,
                side,
                position,
                current_qty,
                occupation_limit,
            )
            if not self._reduce_position_for_margin(day, state, side, target_qty):
                break
            changed = True
            self._update_day_aggregates(day, state)
            self._hedge_to(
                day["date"],
                day["spot"],
                state,
                day,
                day["greeks"],
                target_qty=strategy.round_etf_hedge_target(
                    -float(day["greeks"]["delta"])
                ),
            )

        nav, occupation = self._current_nav_and_occupation(day, state)
        state.trades.append(
            {
                "date": day["date"],
                "type": "capital_occupation_limit_unresolved",
                "nav": nav,
                "capital_occupation": occupation,
                "capital_occupation_ratio": occupation / nav if nav > 0 else None,
                "capital_occupation_limit_ratio": ratio_limit,
            }
        )
        return changed

    def _dynamic_reduction_target_qty(
        self,
        day,
        state,
        side,
        position,
        current_qty,
        occupation_limit,
    ):
        position_occupation = self._position_capital_occupation(position)
        other_occupation = sum(
            self._position_capital_occupation(candidate)
            for candidate_side, candidate in state.positions.items()
            if candidate is not None and candidate_side != side
        )
        other_occupation += sum(
            self._option_hedge_capital_occupation(candidate)
            for candidate in state.option_hedges
        )

        position_delta = float(
            day["side_records"].get(side, {}).get("greeks", {}).get("delta", 0.0)
            or 0.0
        )
        other_delta = float(day["greeks"]["delta"]) - position_delta
        for target_qty in range(current_qty - 1, -1, -1):
            remaining_ratio = target_qty / current_qty
            projected_option_delta = other_delta + position_delta * remaining_ratio
            projected_hedge_qty = strategy.round_etf_hedge_target(
                -projected_option_delta
            )
            projected_occupation = (
                other_occupation
                + position_occupation * remaining_ratio
                + abs(projected_hedge_qty) * float(day["spot"])
            )
            if projected_occupation <= occupation_limit + 1e-6:
                return target_qty
        return 0

    def _close_option_delta_hedges(self, date, state, day):
        if not state.option_hedges:
            return
        fee = 0.0
        legs = []
        for hedge_position in state.option_hedges:
            try:
                row = self._chain_row(day["chain_df"], hedge_position["order_book_id"])
                price = float(row["mid"])
            except (IndexError, TypeError):
                price = float(hedge_position.get("last_price", 0.0) or 0.0)
            qty = int(hedge_position.get("qty", 0) or 0)
            multiplier = float(hedge_position.get("contract_multiplier", 10000) or 10000)
            leg_fee = opt_position.calc_option_fee(qty, 0)
            value = price * qty * multiplier
            if hedge_position.get("side") == "short":
                state.cash += (
                    float(hedge_position.get("option_margin", 0.0) or 0.0)
                    - value
                    - leg_fee
                )
            else:
                state.cash += value - leg_fee
            fee += leg_fee
            legs.append(
                {
                    "order_book_id": hedge_position["order_book_id"],
                    "side": hedge_position["side"],
                    "qty": qty,
                    "price": price,
                }
            )
        state.option_hedges = []
        day["daily_option_fee"] += fee
        state.trades.append(
            {
                "date": date,
                "type": "close_option_delta_hedge",
                "fee": fee,
                "cash": state.cash,
                "legs": legs,
            }
        )

    def _gamma_neutral_solution(self, state, day, residual_delta):
        short_position = state.positions.get("short")
        if short_position is None or int(short_position.get("call_qty", 0) or 0) <= 1:
            return None
        try:
            source_row = self._chain_row(day["chain_df"], short_position["call_code"])
        except IndexError:
            return None
        source = {
            "max_qty": int(short_position["call_qty"]),
            "row": source_row,
        }
        candidates = day["chain_df"].copy()
        candidates = candidates[
            candidates["option_type"].astype(str).str.lower().eq("c")
            & (pd.to_numeric(candidates["mid"], errors="coerce") > 0)
            & (pd.to_numeric(candidates["delta"], errors="coerce") > 0)
            & (pd.to_numeric(candidates["gamma"], errors="coerce") > 0)
            & (
                pd.to_numeric(candidates["contract_multiplier"], errors="coerce")
                == float(source_row["contract_multiplier"])
            )
            & pd.to_datetime(candidates["maturity_date"]).dt.normalize().eq(
                pd.Timestamp(source_row["maturity_date"]).normalize()
            )
            & ~candidates["order_book_id"].astype(str).eq(str(source_row["order_book_id"]))
        ]
        candidates = self._light_itm_call_candidates(candidates, day["spot"])
        solution = opt_position.solve_liquid_call_delta_hedge(
            source,
            [row for _, row in candidates.iterrows()],
            float(residual_delta),
        )
        if solution is None:
            return None
        return {
            **solution,
            "source_row": source_row,
        }

    def _light_itm_call_candidates(self, candidates, spot):
        if candidates.empty:
            return candidates
        candidates = vol_engine.filter_standard_option_contracts(candidates).copy()
        candidates["_strike"] = pd.to_numeric(
            candidates["strike_price"],
            errors="coerce",
        )
        candidates["_volume"] = pd.to_numeric(
            candidates.get("volume"),
            errors="coerce",
        ).fillna(-1.0)
        max_itm_ratio = float(
            getattr(CONFIG.strategy, "option_delta_hedge_max_itm_ratio", 0.10)
        )
        candidates = candidates[
            (candidates["_strike"] < float(spot))
            & (candidates["_strike"] >= float(spot) * (1.0 - max_itm_ratio))
        ]
        candidates["_liquidity_capacity"] = candidates.apply(
            opt_position.liquidity_capacity,
            axis=1,
        )
        return candidates[candidates["_liquidity_capacity"] > 0].sort_values(
            "_volume",
            ascending=False,
        )

    def _open_gamma_neutral_option_hedge(self, date, state, day, solution):
        source_row = solution["source_row"]
        open_legs = solution["open_legs"]
        primary_open_row = open_legs[0]["row"]
        close_qty = int(solution["close_qty"])
        source_multiplier = float(source_row["contract_multiplier"])
        source_price = float(source_row["mid"])
        source_fee = opt_position.calc_option_fee(close_qty, 0)
        open_fee = opt_position.calc_option_fee(solution["open_qty"], 0)
        open_value = 0.0
        margin = 0.0
        for leg in open_legs:
            row = leg["row"]
            qty = int(leg["qty"])
            multiplier = float(row["contract_multiplier"])
            price = float(row["mid"])
            open_value += price * qty * multiplier
            margin += opt_position.margin_call(
                float(day["spot"]),
                float(row["strike_price"]),
                price,
                multiplier,
            ) * qty
        projected_cash = (
            state.cash
            - source_price * close_qty * source_multiplier
            - source_fee
            + open_value
            - open_fee
            - margin
        )
        if not self._has_cash_reserve(projected_cash):
            self._record_cash_reserve_skip(
                date,
                state,
                "skip_option_delta_hedge_cash_reserve",
                projected_cash,
            )
            return False
        state.cash = projected_cash
        state.option_hedges = []
        if close_qty > 0:
            state.option_hedges.append({
                "order_book_id": source_row["order_book_id"],
                "side": "long",
                "qty": close_qty,
                "entry_price": source_price,
                "last_price": source_price,
                "contract_multiplier": source_multiplier,
                "option_margin": 0.0,
            })
        for leg in open_legs:
            row = leg["row"]
            state.option_hedges.append({
                "order_book_id": row["order_book_id"],
                "side": "short",
                "qty": int(leg["qty"]),
                "entry_price": float(row["mid"]),
                "last_price": float(row["mid"]),
                "contract_multiplier": float(row["contract_multiplier"]),
                "option_margin": opt_position.margin_call(
                    float(day["spot"]),
                    float(row["strike_price"]),
                    float(row["mid"]),
                    float(row["contract_multiplier"]),
                ) * int(leg["qty"]),
            })
        fee = source_fee + open_fee
        day["daily_option_fee"] += fee
        state.trades.append(
            {
                "date": date,
                "type": "option_delta_hedge_combination",
                "fee": fee,
                "cash": state.cash,
                "close_call_code": source_row["order_book_id"],
                "close_call_qty": close_qty,
                "close_call_price": source_price,
                "open_call_code": primary_open_row["order_book_id"],
                "open_call_qty": solution["open_qty"],
                "open_call_price": float(primary_open_row["mid"]),
                "open_call_strike": float(primary_open_row["strike_price"]),
                "open_legs": [
                    {
                        "order_book_id": leg["row"]["order_book_id"],
                        "qty": int(leg["qty"]),
                        "price": float(leg["row"]["mid"]),
                        "strike": float(leg["row"]["strike_price"]),
                        "volume": float(leg["row"]["volume"]),
                        "liquidity_capacity": int(leg["liquidity_capacity"]),
                    }
                    for leg in open_legs
                ],
                "open_liquidity_capacity": sum(
                    int(leg["liquidity_capacity"]) for leg in open_legs
                ),
                "close_liquidity_capacity": solution["close_liquidity_capacity"],
                "close_call_volume": source_row.get("volume"),
                "solver_priority": "liquidity_then_delta_then_gamma",
                "residual_delta_before_option_hedge": solution[
                    "residual_delta_before"
                ],
                "delta_neutral_achieved": solution["delta_neutral_achieved"],
                "liquidity_capacity_exhausted": solution[
                    "liquidity_capacity_exhausted"
                ],
                "option_margin": margin,
                "liquidity_warning_ratio": CONFIG.backtest.liquidity_warning_volume_ratio,
                "liquidity_check_available": True,
                "liquidity_warning": False,
                "liquidity_warning_legs": "",
                "liquidity_volume_missing_legs": "",
                "delta_effect": solution["delta_effect"],
                "gamma_effect": solution["gamma_effect"],
                "projected_option_delta": solution["projected_delta"],
                "etf_buy_qty": solution["etf_buy_qty"],
                "projected_account_delta": solution["combined_delta"],
            }
        )
        return True

    def _hedge_to(
        self,
        date,
        spot,
        state,
        day=None,
        greeks=None,
        target_qty=None,
        trade_type="delta_hedge",
    ):
        if not self.config.get("enable_delta_hedge", CONFIG.strategy.enable_delta_hedge):
            return

        if greeks is None:
            greeks = empty_greeks()
        if not self._has_any_position(state):
            if state.option_hedges and day is not None:
                self._close_option_delta_hedges(date, state, day)
                self._update_day_aggregates(day, state)
                greeks = day["greeks"]
            self._execute_etf_target(
                date,
                spot,
                state,
                day,
                greeks,
                0.0,
                "close_hedge",
            )
            return
        option_delta = float(greeks["delta"])
        account_delta = option_delta + float(state.hedge_etf_qty)
        normalized_delta, delta_capacity = strategy.normalized_account_delta(
            account_delta,
            state.positions,
            option_hedges=state.option_hedges,
            default_multiplier=CONFIG.vol.contract_multiplier,
        )
        tolerance_ratio = float(
            self.config.get("delta_hedge_tolerance_ratio", 0.05)
        )
        tolerance = delta_capacity * tolerance_ratio
        if (
            target_qty is None
            and delta_capacity > 0
            and abs(normalized_delta) <= tolerance_ratio
        ):
            return

        if target_qty is None and state.option_hedges and day is not None:
            self._close_option_delta_hedges(date, state, day)
            self._update_day_aggregates(day, state)
            greeks = day["greeks"]
            option_delta = float(greeks["delta"])
            account_delta = option_delta + float(state.hedge_etf_qty)
            normalized_delta, delta_capacity = strategy.normalized_account_delta(
                account_delta,
                state.positions,
                option_hedges=state.option_hedges,
                default_multiplier=CONFIG.vol.contract_multiplier,
            )
            tolerance = delta_capacity * tolerance_ratio
            if delta_capacity > 0 and abs(normalized_delta) <= tolerance_ratio:
                return

        projected_target_qty = strategy.round_etf_hedge_target(
            -option_delta if target_qty is None else float(target_qty)
        )
        if (
            self.config.get("allow_etf_short_hedge", True)
            or projected_target_qty >= 0
            or target_qty is not None
        ):
            if target_qty is not None and target_qty == 0:
                self._close_option_delta_hedges(date, state, day)
            self._execute_etf_target(
                date,
                spot,
                state,
                day,
                greeks,
                projected_target_qty,
                trade_type,
            )
            return

        if not (
            self.config.get("enable_option_delta_hedge", False)
            and self.config.get("option_delta_hedge_combination_enabled", False)
            and day is not None
        ):
            state.trades.append(
                {
                    "date": date,
                    "type": "skip_delta_hedge_etf_short_disabled",
                    "account_delta": account_delta,
                    "tolerance": tolerance,
                }
            )
            return

        self._close_option_delta_hedges(date, state, day)
        self._execute_etf_target(
            date,
            spot,
            state,
            day,
            greeks,
            0.0,
            "reduce_etf_before_option_delta_hedge",
        )
        core_greeks = combine_greeks(
            [day["side_records"][side]["greeks"] for side in POSITION_SIDES]
        )
        residual_delta = float(core_greeks["delta"])
        if residual_delta <= tolerance:
            return
        solution = self._gamma_neutral_solution(state, day, residual_delta)
        if solution is None:
            state.trades.append(
                {
                    "date": date,
                    "type": "skip_option_delta_hedge_combination",
                    "reason": "no_liquid_light_itm_option_delta_hedge_solution",
                    "residual_delta": residual_delta,
                    "tolerance": tolerance,
                }
            )
            return
        if not self._open_gamma_neutral_option_hedge(date, state, day, solution):
            return
        self._update_day_aggregates(day, state)
        self._execute_etf_target(
            date,
            spot,
            state,
            day,
            day["greeks"],
            solution["etf_buy_qty"],
            "option_delta_hedge_etf_correction",
        )

    def _add_new_option_fees(self, day, state, trade_count):
        for trade in state.trades[trade_count:]:
            if "straddle" in trade.get("type", ""):
                day["daily_option_fee"] += trade.get("fee", 0.0)

    def _get_position_rows(self, day, state, side):
        return vol_engine.resolve_position_pair(
            state.positions[side],
            day["chain_df"],
        )

    def _record_day(self, day, state):
        self._update_day_aggregates(day, state)
        hedge_price = self._get_hedge_price(
            day["date"],
            state.hedge_underlying_order_book_id,
            day["spot"],
        )
        hedge_unrealized_pnl = hedge.calc_unrealized_pnl(
            state.hedge_etf_qty,
            state.hedge_entry_price,
            hedge_price,
        )
        option_margin = sum(
            opt_position.margin_value(state.positions[side])
            for side in POSITION_SIDES
            if state.positions.get(side) is not None
        ) + day.get("option_hedge_margin", 0.0)
        nav = (
            state.cash
            + day["option_value"]
            + option_margin
            + state.hedge_margin
            + hedge_unrealized_pnl
        )
        record = build_daily_record(
                day["date"],
                day["spot"],
                state.cash,
                day["option_value"],
                state.hedge_etf_qty,
                state.hedge_underlying_order_book_id,
                hedge_price,
                state.hedge_entry_price,
                state.hedge_margin,
                hedge_unrealized_pnl,
                nav,
                day["daily_etf_fee"],
                day["daily_option_fee"],
                state.positions,
                day["side_records"],
                day["greeks"],
                day["feature_row"],
                day["pnl_position_iv"],
                day["pnl_call_iv"],
                day["pnl_put_iv"],
                day["pnl_greeks"],
                day["eod_position_dte"],
        )
        record["option_margin"] = option_margin
        record["option_hedge_value"] = day.get("option_hedge_value", 0.0)
        record["option_hedge_margin"] = day.get("option_hedge_margin", 0.0)
        record["option_hedge_count"] = len(state.option_hedges)
        record["option_hedge_delta"] = day.get("option_hedge_greeks", {}).get(
            "delta",
            0.0,
        )
        record["option_hedge_gamma"] = day.get("option_hedge_greeks", {}).get(
            "gamma",
            0.0,
        )
        total_margin = option_margin + state.hedge_margin
        record["total_margin"] = total_margin
        record["margin_to_nav_ratio"] = total_margin / nav if nav > 0 else None
        _, capital_occupation = self._current_nav_and_occupation(day, state)
        record["capital_occupation"] = capital_occupation
        record["capital_occupation_ratio"] = (
            capital_occupation / nav if nav > 0 else None
        )
        record["capital_occupation_limit_ratio"] = self.config.get(
            "max_margin_to_nav_ratio"
        )
        record["capital_occupation_limit_breach"] = (
            nav > 0
            and capital_occupation
            > nav * self.config["max_margin_to_nav_ratio"] + 1e-6
        )
        record["margin_limit_ratio"] = self.config.get("max_margin_to_nav_ratio")
        record["margin_limit_breach"] = (
            nav > 0
            and total_margin > nav * self.config["max_margin_to_nav_ratio"] + 1e-6
        )
        record["data_warning_count"] = len(day.get("data_warnings", []))
        record["data_warning_reasons"] = ",".join(
            str(item.get("reason"))
            for item in day.get("data_warnings", [])
            if item.get("reason")
        )
        state.daily_records.append(record)


def _backtest_config(
    initial_cash=None,
    min_cash_reserve=None,
    long_qty=None,
    short_qty=None,
    etf_fee_rate=None,
    enable_delta_hedge=None,
):
    return {
        "initial_cash": (
            CONFIG.backtest.initial_cash if initial_cash is None else initial_cash
        ),
        "min_cash_reserve": (
            CONFIG.backtest.min_cash_reserve
            if min_cash_reserve is None
            else min_cash_reserve
        ),
        "long_qty": CONFIG.backtest.long_qty if long_qty is None else long_qty,
        "short_qty": CONFIG.backtest.short_qty if short_qty is None else short_qty,
        "etf_fee_rate": (
            CONFIG.backtest.etf_fee_rate if etf_fee_rate is None else etf_fee_rate
        ),
        "enable_delta_hedge": (
            CONFIG.strategy.enable_delta_hedge
            if enable_delta_hedge is None
            else enable_delta_hedge
        ),
        "delta_hedge_tolerance_ratio": CONFIG.strategy.delta_hedge_tolerance_ratio,
        "allow_etf_short_hedge": CONFIG.strategy.allow_etf_short_hedge,
        "enable_option_delta_hedge": CONFIG.strategy.enable_option_delta_hedge,
        "option_delta_hedge_combination_enabled": (
            CONFIG.strategy.option_delta_hedge_combination_enabled
        ),
        "dynamic_position_control_enabled": (
            CONFIG.backtest.dynamic_position_control_enabled
        ),
        "proportional_position_sizing_enabled": (
            CONFIG.backtest.proportional_position_sizing_enabled
        ),
        "position_sizing_base_nav": CONFIG.backtest.position_sizing_base_nav,
        "max_margin_to_nav_ratio": CONFIG.backtest.max_margin_to_nav_ratio,
    }


def run_backtest(
    etf_by_date,
    opt_by_date,
    signals_df,
    initial_cash=None,
    min_cash_reserve=None,
    long_qty=None,
    short_qty=None,
    etf_fee_rate=None,
    enable_delta_hedge=None,
    trading_calendar=None,
    enriched_opt_by_date=None,
    hedge_by_date=None,
):
    engine = BacktestEngine(
        etf_by_date,
        opt_by_date,
        signals_df,
        config=_backtest_config(
            initial_cash=initial_cash,
            min_cash_reserve=min_cash_reserve,
            long_qty=long_qty,
            short_qty=short_qty,
            etf_fee_rate=etf_fee_rate,
            enable_delta_hedge=enable_delta_hedge,
        ),
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
        hedge_by_date=hedge_by_date,
    )
    return engine.run()
