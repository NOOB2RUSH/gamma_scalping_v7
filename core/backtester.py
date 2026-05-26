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
    hedge_etf_qty: float = 0.0
    hedge_entry_price: float = 0.0
    hedge_margin: float = 0.0
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
):
    """把 ETF 对冲仓位调整到目标数量，并记录交易和手续费。"""
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate

    if target_qty is None:
        target_qty = -greeks["delta"]

    if hedge_etf_qty == target_qty:
        return cash, hedge_etf_qty, hedge_entry_price, hedge_margin

    old_qty = hedge_etf_qty
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
    trade_qty = hedge_etf_qty - old_qty
    etf_fee = abs(trade_qty) * spot * etf_fee_rate
    cash -= etf_fee

    trades.append(
        {
            "date": date,
            "type": trade_type,
            "old_etf_qty": old_qty,
            "new_etf_qty": hedge_etf_qty,
            "trade_etf_qty": trade_qty,
            "etf_qty": trade_qty,
            "price": spot,
            "hedge_pnl": hedge_pnl,
            "fee": etf_fee,
        }
    )
    return cash, hedge_etf_qty, hedge_entry_price, hedge_margin


def build_daily_record(
    date,
    spot,
    cash,
    option_value,
    hedge_etf_qty,
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
    side_greeks_cols = []

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
        side_greeks_cols.extend(side_cols)

    df["hedge_delta_pnl"] = df["hedge_etf_qty"].shift(1) * spot_chg
    df["delta_pnl"] = (
        df["long_delta_pnl"] + df["short_delta_pnl"] + df["hedge_delta_pnl"]
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
    ):
        self.etf_by_date = etf_by_date
        self.opt_by_date = opt_by_date
        self.enriched_opt_by_date = enriched_opt_by_date
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
            self._update_day_aggregates(day)
            self._hedge_to(date, spot, state, day, day["greeks"])
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
            print(
                f"[missing option chain] date={date.date()}, spot={spot}, "
                f"side={side}, position={position['call_code']}/{position['put_code']}, "
                f"strike={position['strike']}, expiry={position['expiry'].date()}, "
                "action=close_at_last_value"
            )
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_at_last_value(
                date,
                state.cash,
                position,
                state.trades,
                exit_reason="missing_option_chain_last_price",
            )
            self._add_new_option_fees(day, state, trade_count)
            state.positions[side] = None
            self._start_cooldown(day, state, side)
            day["skip_new_entry_by_side"].add(side)

        self._update_day_aggregates(day)
        self._hedge_to(
            date,
            spot,
            state,
            day,
            greeks=empty_greeks(),
            target_qty=0.0,
            trade_type="close_hedge",
        )
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
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_at_last_value(
                date,
                state.cash,
                position,
                state.trades,
            )
            self._add_new_option_fees(day, state, trade_count)
            state.positions[side] = None
            self._start_cooldown(day, state, side)
            day["skip_new_entry_by_side"].add(side)
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
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_trade(
                date,
                state.cash,
                position,
                call_row,
                put_row,
                state.trades,
                exit_reason="missing_position_iv",
            )
            self._add_new_option_fees(day, state, trade_count)
            state.positions[side] = None
            self._start_cooldown(day, state, side)
            day["skip_new_entry_by_side"].add(side)
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
                close_reason = strategy.get_short_close_reason(feature_row, position_dte)
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

    def _entry_target_qty(self, feature_row, max_qty, side):
        if side == "short":
            return strategy.calc_short_entry_target_qty(feature_row, max_qty)
        return strategy.calc_entry_target_qty(feature_row, max_qty)

    def _side_max_qty(self, side):
        """按方向读取每腿张数；跨式组合默认 call/put 等量。"""
        return self.config[f"{side}_qty"]

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
            self._side_max_qty(side),
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
            self._side_max_qty(side),
            side,
        )
        put_qty = self._entry_target_qty(
            day["feature_row"],
            self._side_max_qty(side),
            side,
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
            spot,
        )
        projected_greeks = strategy.calc_position_greeks(
            atm["call"],
            atm["put"],
            call_qty,
            put_qty,
            side=side,
        )
        account_greeks = self._project_account_greeks_with_side(
            day,
            side,
            projected_greeks,
        )
        projected_cash = self._project_cash_after_hedge(
            projected_cash,
            state,
            spot,
            -account_greeks["delta"],
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
            spot=spot,
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
            self._side_max_qty(side),
            side,
        )
        put_qty = self._entry_target_qty(
            day["feature_row"],
            self._side_max_qty(side),
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
            spot,
        )
        projected_greeks = strategy.calc_position_greeks(
            atm["call"],
            atm["put"],
            call_qty,
            put_qty,
            side=side,
        )
        account_greeks = self._project_account_greeks_with_side(
            day,
            side,
            projected_greeks,
        )
        projected_cash = self._project_cash_after_hedge(
            projected_cash,
            state,
            spot,
            -account_greeks["delta"],
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
        state.cash, new_position, option_value = opt_position.open_trade(
            date,
            state.cash,
            atm,
            call_qty,
            put_qty,
            state.trades,
            trade_type=trade_type,
            side=side,
            spot=spot,
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

    def _refresh_short_margin(self, state, side, call_row, put_row, spot):
        position = state.positions[side]
        if position is None or position.get("side", "long") != "short":
            return 0.0

        old_margin = position.get("option_margin", 0.0)
        new_margin = opt_position.calc_short_margin(
            call_row,
            put_row,
            position["call_qty"],
            position["put_qty"],
            spot,
        )
        margin_change = new_margin - old_margin
        if margin_change != 0:
            state.cash -= margin_change
            position["option_margin"] = new_margin
        return margin_change

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
        self._refresh_short_margin(state, side, call_row, put_row, day["spot"])
        self._set_side_eod(
            day,
            state,
            side,
            opt_position.signed_value(state.positions[side], call_row, put_row),
            greeks,
            dte,
        )

    def _update_day_aggregates(self, day):
        side_records = day["side_records"]
        day["option_value"] = sum(
            side_records[side]["option_value"] for side in POSITION_SIDES
        )
        day["greeks"] = combine_greeks(
            [side_records[side]["greeks"] for side in POSITION_SIDES]
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

    def _project_account_greeks_with_side(self, day, side, projected_greeks):
        greeks_list = []
        for candidate_side in POSITION_SIDES:
            if candidate_side == side:
                greeks_list.append(projected_greeks)
            else:
                greeks_list.append(day["side_records"][candidate_side]["greeks"])
        return combine_greeks(greeks_list)

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

    def _project_cash_after_option_close(self, cash, position, call_row, put_row):
        close_value = opt_position.value(position, call_row, put_row)
        fee = opt_position.calc_option_fee(position["call_qty"], position["put_qty"])
        if position.get("side", "long") == "short":
            return cash + position.get("option_margin", 0.0) - close_value - fee
        return cash + close_value - fee

    def _project_cash_after_hedge(self, cash, state, spot, target_qty):
        projected_cash, _, _, _, _ = hedge.rebalance_etf_hedge(
            cash,
            state.hedge_etf_qty,
            state.hedge_entry_price,
            state.hedge_margin,
            target_qty,
            spot,
        )
        etf_fee = abs(target_qty - state.hedge_etf_qty) * spot * self.config["etf_fee_rate"]
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

        projected_target_qty = -greeks["delta"] if target_qty is None else target_qty
        projected_cash = self._project_cash_after_hedge(
            state.cash,
            state,
            spot,
            projected_target_qty,
        )
        cash_would_decrease = projected_cash < state.cash
        if cash_would_decrease and not self._has_cash_reserve(projected_cash):
            self._record_cash_reserve_skip(
                date,
                state,
                "skip_delta_hedge_cash_reserve",
                projected_cash,
            )
            return

        trade_count = len(state.trades)
        (
            state.cash,
            state.hedge_etf_qty,
            state.hedge_entry_price,
            state.hedge_margin,
        ) = execute_delta_hedge(
            date,
            state.cash,
            greeks,
            state.hedge_etf_qty,
            state.hedge_entry_price,
            state.hedge_margin,
            spot,
            state.trades,
            target_qty=target_qty,
            trade_type=trade_type,
            etf_fee_rate=self.config["etf_fee_rate"],
        )

        if day is not None and len(state.trades) > trade_count:
            day["daily_etf_fee"] += state.trades[-1].get("fee", 0.0)

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
        self._update_day_aggregates(day)
        hedge_unrealized_pnl = hedge.calc_unrealized_pnl(
            state.hedge_etf_qty,
            state.hedge_entry_price,
            day["spot"],
        )
        option_margin = sum(
            opt_position.margin_value(state.positions[side])
            for side in POSITION_SIDES
            if state.positions.get(side) is not None
        )
        nav = (
            state.cash
            + day["option_value"]
            + option_margin
            + state.hedge_margin
            + hedge_unrealized_pnl
        )
        state.daily_records.append(
            build_daily_record(
                day["date"],
                day["spot"],
                state.cash,
                day["option_value"],
                state.hedge_etf_qty,
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
        )


class AlwaysAtmBenchmarkEngine(BacktestEngine):
    """独立基准账户：不看进出场阈值，始终持有指定方向的 ATM 跨式。"""

    def __init__(
        self,
        etf_by_date,
        opt_by_date,
        signals_df,
        config,
        benchmark_side,
        trading_calendar=None,
        enriched_opt_by_date=None,
    ):
        super().__init__(
            etf_by_date,
            opt_by_date,
            signals_df,
            config,
            trading_calendar=trading_calendar,
            enriched_opt_by_date=enriched_opt_by_date,
        )
        if benchmark_side not in POSITION_SIDES:
            raise ValueError("benchmark_side 只能是 'long' 或 'short'")
        self.benchmark_side = benchmark_side

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
            side = self.benchmark_side

            if state.positions[side] is not None:
                self._handle_existing_position(day, state, side)

            if (
                state.positions[side] is None
                and side not in day["skip_new_entry_by_side"]
                and state.roll_cooldown_left[side] <= 0
            ):
                self._open_new_position(
                    day,
                    state,
                    trade_type="always_atm_open_straddle",
                    side=side,
                )

            self._tick_flat_cooldowns(day, state)
            self._update_day_aggregates(day)
            self._hedge_to(date, spot, state, day, day["greeks"])
            self._record_day(day, state)

        daily_df = pd.DataFrame(state.daily_records).set_index("date")
        daily_df = add_greeks_pnl(daily_df)
        trades_df = pd.DataFrame(state.trades)
        return daily_df, trades_df

    def _entry_target_qty(self, feature_row, max_qty, side):
        return max_qty

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
        return dte_too_low or strike_roll_ready

    def _handle_existing_position(self, day, state, side):
        """基准账户只因 DTE/ATM 偏离展期，不因进出场阈值平仓。"""
        date = day["date"]
        position = state.positions[side]

        try:
            call_row, put_row = self._get_position_rows(day, state, side)
        except IndexError:
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_at_last_value(
                date,
                state.cash,
                position,
                state.trades,
            )
            self._add_new_option_fees(day, state, trade_count)
            state.positions[side] = None
            self._start_cooldown(day, state, side)
            return

        position_dte = int(call_row["dte"])
        pnl_greeks = strategy.calc_position_greeks(
            call_row,
            put_row,
            position["call_qty"],
            position["put_qty"],
            side=side,
        )
        side_record = day["side_records"][side]
        side_record["pnl_position_iv"] = pnl_greeks["position_iv"]
        side_record["pnl_call_iv"] = pnl_greeks["call_iv"]
        side_record["pnl_put_iv"] = pnl_greeks["put_iv"]
        side_record["pnl_greeks"] = pnl_greeks.copy()

        if side == "short":
            current_market_value = opt_position.value(position, call_row, put_row)
            if strategy.is_short_stop_loss(position, current_market_value):
                trade_count = len(state.trades)
                state.cash, _ = opt_position.close_trade(
                    date,
                    state.cash,
                    position,
                    call_row,
                    put_row,
                    state.trades,
                    exit_reason="short_stop_loss",
                )
                self._add_new_option_fees(day, state, trade_count)
                state.positions[side] = None
                self._start_cooldown(day, state, side)
                day["skip_new_entry_by_side"].add(side)
                return

        self._update_roll_buffer(day["feature_row"], state, side)
        if self._should_roll_position(day, state, side, position_dte):
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


class NoDeltaHedgeBacktestEngine(BacktestEngine):
    """裸 short vega 对比账户：保留进出场/展期，但不做 ETF delta hedge。"""

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
        return


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
):
    if initial_cash is None:
        initial_cash = CONFIG.backtest.initial_cash
    if min_cash_reserve is None:
        min_cash_reserve = CONFIG.backtest.min_cash_reserve
    if long_qty is None:
        long_qty = CONFIG.backtest.long_qty
    if short_qty is None:
        short_qty = CONFIG.backtest.short_qty
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate
    if enable_delta_hedge is None:
        enable_delta_hedge = CONFIG.strategy.enable_delta_hedge

    engine = BacktestEngine(
        etf_by_date,
        opt_by_date,
        signals_df,
        config={
            "initial_cash": initial_cash,
            "min_cash_reserve": min_cash_reserve,
            "long_qty": long_qty,
            "short_qty": short_qty,
            "etf_fee_rate": etf_fee_rate,
            "enable_delta_hedge": enable_delta_hedge,
        },
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
    )
    return engine.run()


def run_no_delta_hedge_backtest(
    etf_by_date,
    opt_by_date,
    signals_df,
    initial_cash=None,
    min_cash_reserve=None,
    long_qty=None,
    short_qty=None,
    etf_fee_rate=None,
    trading_calendar=None,
    enriched_opt_by_date=None,
):
    if initial_cash is None:
        initial_cash = CONFIG.backtest.initial_cash
    if min_cash_reserve is None:
        min_cash_reserve = CONFIG.backtest.min_cash_reserve
    if long_qty is None:
        long_qty = CONFIG.backtest.long_qty
    if short_qty is None:
        short_qty = CONFIG.backtest.short_qty
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate

    engine = NoDeltaHedgeBacktestEngine(
        etf_by_date,
        opt_by_date,
        signals_df,
        config={
            "initial_cash": initial_cash,
            "min_cash_reserve": min_cash_reserve,
            "long_qty": long_qty,
            "short_qty": short_qty,
            "etf_fee_rate": etf_fee_rate,
        },
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
    )
    return engine.run()


def run_always_atm_benchmark(
    etf_by_date,
    opt_by_date,
    signals_df,
    benchmark_side=None,
    initial_cash=None,
    min_cash_reserve=None,
    always_atm_qty=None,
    etf_fee_rate=None,
    enable_delta_hedge=None,
    trading_calendar=None,
    enriched_opt_by_date=None,
):
    if benchmark_side is None:
        benchmark_side = CONFIG.reference.always_atm_side
    if initial_cash is None:
        initial_cash = CONFIG.backtest.initial_cash
    if min_cash_reserve is None:
        min_cash_reserve = CONFIG.backtest.min_cash_reserve
    if always_atm_qty is None:
        always_atm_qty = CONFIG.reference.always_atm_qty
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate
    if enable_delta_hedge is None:
        enable_delta_hedge = CONFIG.strategy.enable_delta_hedge

    engine = AlwaysAtmBenchmarkEngine(
        etf_by_date,
        opt_by_date,
        signals_df,
        config={
            "initial_cash": initial_cash,
            "min_cash_reserve": min_cash_reserve,
            "long_qty": always_atm_qty,
            "short_qty": always_atm_qty,
            "etf_fee_rate": etf_fee_rate,
            "enable_delta_hedge": enable_delta_hedge,
        },
        benchmark_side=benchmark_side,
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
    )
    return engine.run()
