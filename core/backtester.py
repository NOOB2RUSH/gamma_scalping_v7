from dataclasses import dataclass, field

import pandas as pd

from . import hedge, position as opt_position, strategy, vol_engine
from .config import CONFIG


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


@dataclass
class BacktestState:
    """回测账户状态：现金、期权仓位、ETF 对冲仓位和输出记录。"""

    cash: float
    position: dict | None = None
    hedge_etf_qty: float = 0.0
    hedge_entry_price: float = 0.0
    hedge_margin: float = 0.0
    strike_mismatch_days: int = 0
    roll_cooldown_left: int = 0
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
    position,
    greeks,
    feature_row,
    pnl_position_iv,
    pnl_call_iv,
    pnl_put_iv,
    pnl_greeks,
    eod_position_dte,
):
    """生成单日账户快照，字段名保持为报表输出口径。"""
    eod_has_position = position is not None
    account_delta = greeks["delta"] + hedge_etf_qty

    return {
        "date": date,
        "spot": spot,
        "cash": cash,
        "option_value": option_value,
        "hedge_etf_qty": hedge_etf_qty,
        "hedge_entry_price": hedge_entry_price,
        "hedge_margin": hedge_margin,
        "hedge_unrealized_pnl": hedge_unrealized_pnl,
        "nav": nav,
        "etf_fee": etf_fee,
        "option_fee": option_fee,
        "eod_has_position": eod_has_position,
        "eod_position_call_code": (
            position["call_code"] if eod_has_position else None
        ),
        "eod_position_put_code": position["put_code"] if eod_has_position else None,
        "eod_position_strike": position["strike"] if eod_has_position else None,
        "eod_position_expiry": position["expiry"] if eod_has_position else None,
        "eod_position_dte": eod_position_dte,
        "eod_position_call_qty": position["call_qty"] if eod_has_position else 0,
        "eod_position_put_qty": position["put_qty"] if eod_has_position else 0,
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
        "open_signal": feature_row["open_signal"],
    }


def add_greeks_pnl(daily_df):
    """用前一交易日收盘 Greeks 解释下一交易日 NAV 变化。"""
    df = daily_df.copy()
    prev_has_position = df["eod_has_position"].shift(1).fillna(False).astype(bool)
    same_position = (
        df["eod_has_position"]
        & prev_has_position
        & (df["eod_position_call_code"] == df["eod_position_call_code"].shift(1))
        & (df["eod_position_put_code"] == df["eod_position_put_code"].shift(1))
    )
    explainable_day = prev_has_position & df["pnl_position_iv"].notna()
    position_changed = prev_has_position & ~same_position

    spot_chg = df["spot"].diff()
    call_iv_chg = df["pnl_call_iv"] - df["eod_call_iv"].shift(1)
    put_iv_chg = df["pnl_put_iv"] - df["eod_put_iv"].shift(1)
    avg_call_gamma = (
        df["eod_call_gamma"].shift(1) + df["pnl_call_gamma"]
    ) / 2
    avg_put_gamma = (
        df["eod_put_gamma"].shift(1) + df["pnl_put_gamma"]
    ) / 2
    avg_call_vega = (
        df["eod_call_vega"].shift(1) + df["pnl_call_vega"]
    ) / 2
    avg_put_vega = (
        df["eod_put_vega"].shift(1) + df["pnl_put_vega"]
    ) / 2
    avg_call_theta = (
        df["eod_call_theta"].shift(1) + df["pnl_call_theta"]
    ) / 2
    avg_put_theta = (
        df["eod_put_theta"].shift(1) + df["pnl_put_theta"]
    ) / 2

    # Delta 使用昨日 EOD 账户 delta；Gamma/Vega/Theta 使用同一旧仓的昨日和今日平均值。
    df["delta_pnl"] = (
        df["eod_call_delta"].shift(1)
        + df["eod_put_delta"].shift(1)
        + df["hedge_etf_qty"].shift(1)
    ) * spot_chg
    df["gamma_pnl"] = 0.5 * (avg_call_gamma + avg_put_gamma) * spot_chg**2
    df["vega_pnl"] = (
        avg_call_vega * call_iv_chg * 100
        + avg_put_vega * put_iv_chg * 100
    )
    df["theta_pnl"] = avg_call_theta + avg_put_theta

    pnl_cols = ["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]
    df.loc[~explainable_day, pnl_cols] = 0.0
    df["greeks_pnl"] = df[pnl_cols].sum(axis=1, min_count=len(pnl_cols))

    df["daily_nav_pnl"] = df["nav"].diff()
    df["daily_fee"] = df["etf_fee"].fillna(0.0) + df["option_fee"].fillna(0.0)
    df["daily_nav_pnl_before_fee"] = df["daily_nav_pnl"] + df["daily_fee"]
    df["greeks_unexplained_pnl"] = df["daily_nav_pnl"] - df["greeks_pnl"]
    df["greeks_unexplained_pnl_before_fee"] = (
        df["daily_nav_pnl_before_fee"] - df["greeks_pnl"]
    )
    df["greeks_explainable_day"] = explainable_day
    df["greeks_position_changed_day"] = position_changed

    actual_abs = df["daily_nav_pnl"].abs()
    residual_abs = df["greeks_unexplained_pnl"].abs()
    actual_before_fee_abs = df["daily_nav_pnl_before_fee"].abs()
    residual_before_fee_abs = df["greeks_unexplained_pnl_before_fee"].abs()
    df["greeks_explain_ratio"] = 1 - residual_abs / actual_abs
    df["greeks_explain_ratio_before_fee"] = (
        1 - residual_before_fee_abs / actual_before_fee_abs
    )
    df.loc[actual_abs == 0, "greeks_explain_ratio"] = pd.NA
    df.loc[actual_before_fee_abs == 0, "greeks_explain_ratio_before_fee"] = pd.NA
    return df


class BacktestEngine:
    """按交易日推进回测，集中管理状态，减少主流程里的隐式变量传递。"""

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

            day = {
                "date": date,
                "spot": spot,
                "feature_row": feature_row,
                "chain_df": chain_df,
                "option_value": 0.0,
                "greeks": empty_greeks(),
                "pnl_position_iv": None,
                "pnl_call_iv": None,
                "pnl_put_iv": None,
                "pnl_greeks": empty_greeks(),
                "eod_position_dte": None,
                "daily_etf_fee": 0.0,
                "daily_option_fee": 0.0,
            }

            if pd.isna(feature_row["atm_iv"]) and state.position is None:
                continue

            if state.position is not None:
                self._handle_existing_position(day, state)

            if day.get("stop_after_record"):
                continue

            if state.position is None and feature_row["open_signal"]:
                self._open_new_position(day, state, trade_type="open_straddle")

            self._record_day(day, state)

        daily_df = pd.DataFrame(state.daily_records).set_index("date")
        daily_df = add_greeks_pnl(daily_df)
        trades_df = pd.DataFrame(state.trades)
        return daily_df, trades_df

    def _handle_missing_option_day(self, date, spot, feature_row, state):
        """缺少期权链时打印明确信息；已有仓位按最后估值离场并记录日报。"""
        if state.position is None:
            print(
                f"[missing option chain] date={date.date()}, "
                f"spot={spot}, no position, skip day"
            )
            return

        print(
            f"[missing option chain] date={date.date()}, spot={spot}, "
            f"position={state.position['call_code']}/{state.position['put_code']}, "
            f"strike={state.position['strike']}, expiry={state.position['expiry'].date()}, "
            "action=close_at_last_value_and_close_hedge"
        )

        day = {
            "date": date,
            "spot": spot,
            "feature_row": feature_row,
            "chain_df": None,
            "option_value": 0.0,
            "greeks": empty_greeks(),
            "pnl_position_iv": None,
            "pnl_call_iv": None,
            "pnl_put_iv": None,
            "pnl_greeks": empty_greeks(),
            "eod_position_dte": None,
            "daily_etf_fee": 0.0,
            "daily_option_fee": 0.0,
        }
        trade_count = len(state.trades)
        state.cash, _ = opt_position.close_at_last_value(
            date,
            state.cash,
            state.position,
            state.trades,
            exit_reason="missing_option_chain_last_price",
        )
        self._add_new_option_fees(day, state, trade_count)
        self._hedge_to(
            date,
            spot,
            state,
            day,
            greeks={"delta": 0.0},
            target_qty=0.0,
            trade_type="close_hedge",
        )
        state.position = None
        self._reset_roll_state(state)
        self._record_day(day, state)

    def _handle_existing_position(self, day, state):
        """处理已有跨式仓位：缺数据、止损/到期、移仓或继续持有。"""
        date = day["date"]
        spot = day["spot"]
        feature_row = day["feature_row"]
        chain_df = day["chain_df"]

        try:
            call_row, put_row = self._get_position_rows(day, state)
        except IndexError:
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_at_last_value(
                date,
                state.cash,
                state.position,
                state.trades,
            )
            self._add_new_option_fees(day, state, trade_count)
            self._hedge_to(
                date,
                spot,
                state,
                day,
                empty_greeks(),
                target_qty=0.0,
                trade_type="close_hedge",
            )
            state.position = None
            self._reset_roll_state(state)
            day["option_value"] = 0.0
            day["greeks"] = empty_greeks()
            self._record_day(day, state)
            day["stop_after_record"] = True
            return

        position_dte = int(call_row["dte"])
        day["greeks"] = strategy.calc_position_greeks(
            call_row,
            put_row,
            state.position["call_qty"],
            state.position["put_qty"],
        )
        day["pnl_position_iv"] = day["greeks"]["position_iv"]
        day["pnl_call_iv"] = day["greeks"]["call_iv"]
        day["pnl_put_iv"] = day["greeks"]["put_iv"]
        day["pnl_greeks"] = day["greeks"].copy()
        day["eod_position_dte"] = position_dte

        if pd.isna(day["greeks"]["position_iv"]):
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_trade(
                date,
                state.cash,
                state.position,
                call_row,
                put_row,
                state.trades,
                exit_reason="missing_position_iv",
            )
            self._add_new_option_fees(day, state, trade_count)
            self._hedge_to(
                date,
                spot,
                state,
                day,
                empty_greeks(),
                target_qty=0.0,
                trade_type="close_hedge",
            )
            state.position = None
            self._reset_roll_state(state)
            day["option_value"] = 0.0
            day["greeks"] = empty_greeks()
            day["eod_position_dte"] = None
            self._record_day(day, state)
            day["stop_after_record"] = True
            return

        close_reason = strategy.get_close_reason(
            day["greeks"]["position_iv"],
            position_dte,
        )
        self._update_roll_buffer(feature_row, state)
        roll_signal = (
            pd.notna(feature_row["atm_iv"])
            and strategy.should_roll_position(
                feature_row,
                position_dte,
                state.position["strike"],
                state.strike_mismatch_days,
                state.roll_cooldown_left > 0,
            )
        )

        if close_reason is not None:
            trade_count = len(state.trades)
            state.cash, _ = opt_position.close_trade(
                date,
                state.cash,
                state.position,
                call_row,
                put_row,
                state.trades,
                exit_reason=close_reason,
            )
            self._add_new_option_fees(day, state, trade_count)
            self._hedge_to(
                date,
                spot,
                state,
                day,
                day["greeks"],
                target_qty=0.0,
                trade_type="close_hedge",
            )
            state.position = None
            self._reset_roll_state(state)
            day["option_value"] = 0.0
            day["greeks"] = empty_greeks()
            day["eod_position_dte"] = None
            return

        if roll_signal:
            self._roll_position(day, state, call_row, put_row)
            return

        day["option_value"] = opt_position.value(state.position, call_row, put_row)
        state.position["last_option_value"] = day["option_value"]
        self._hedge_to(date, spot, state, day, day["greeks"])
        self._tick_roll_cooldown(state)

    def _roll_position(
        self,
        day,
        state,
        call_row,
        put_row,
    ):
        date = day["date"]
        spot = day["spot"]
        atm = vol_engine.select_atm_from_chain(
            day["chain_df"],
            spot,
        )

        if atm is None:
            day["option_value"] = opt_position.value(state.position, call_row, put_row)
            state.position["last_option_value"] = day["option_value"]
            self._hedge_to(date, spot, state, day, day["greeks"])
            self._tick_roll_cooldown(state)
            return

        trade_count = len(state.trades)
        state.cash, _ = opt_position.close_trade(
            date,
            state.cash,
            state.position,
            call_row,
            put_row,
            state.trades,
            trade_type="roll_close_straddle",
        )
        state.cash, state.position, day["option_value"] = opt_position.open_trade(
            date,
            state.cash,
            atm,
            self.config["call_qty"],
            self.config["put_qty"],
            state.trades,
            trade_type="roll_open_straddle",
        )
        self._add_new_option_fees(day, state, trade_count)
        day["greeks"] = strategy.calc_position_greeks(
            atm["call"],
            atm["put"],
            self.config["call_qty"],
            self.config["put_qty"],
        )
        day["eod_position_dte"] = atm["dte"]
        self._hedge_to(date, spot, state, day, day["greeks"])
        state.strike_mismatch_days = 0
        state.roll_cooldown_left = CONFIG.strategy.roll_cooldown_days

    def _open_new_position(self, day, state, trade_type):
        date = day["date"]
        spot = day["spot"]
        atm = vol_engine.select_atm_from_chain(
            day["chain_df"],
            spot,
        )
        if atm is None:
            return

        trade_count = len(state.trades)
        state.cash, state.position, day["option_value"] = opt_position.open_trade(
            date,
            state.cash,
            atm,
            self.config["call_qty"],
            self.config["put_qty"],
            state.trades,
            trade_type=trade_type,
        )
        self._add_new_option_fees(day, state, trade_count)
        day["greeks"] = strategy.calc_position_greeks(
            atm["call"],
            atm["put"],
            self.config["call_qty"],
            self.config["put_qty"],
        )
        day["eod_position_dte"] = atm["dte"]
        self._hedge_to(date, spot, state, day, day["greeks"])
        self._reset_roll_state(state)

    def _update_roll_buffer(self, feature_row, state):
        """记录当前持仓行权价连续几天不在当日 ATM 档位。"""
        if (
            pd.notna(feature_row["atm_strike"])
            and state.position is not None
            and state.position["strike"] != feature_row["atm_strike"]
        ):
            state.strike_mismatch_days += 1
            return

        state.strike_mismatch_days = 0

    def _tick_roll_cooldown(self, state):
        """成功 roll 后，下一个持仓日开始逐日扣减冷却天数。"""
        if state.roll_cooldown_left > 0:
            state.roll_cooldown_left -= 1

    def _reset_roll_state(self, state):
        """开仓或离场后，roll 缓冲状态从 0 开始计算。"""
        state.strike_mismatch_days = 0
        state.roll_cooldown_left = 0

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
        if greeks is None:
            greeks = empty_greeks()

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
        """把本日新增的期权交易手续费汇总到日报。"""
        for trade in state.trades[trade_count:]:
            if "straddle" in trade.get("type", ""):
                day["daily_option_fee"] += trade.get("fee", 0.0)

    def _get_position_rows(self, day, state):
        return vol_engine.resolve_position_pair(
            state.position,
            day["chain_df"],
        )

    def _record_day(self, day, state):
        hedge_unrealized_pnl = hedge.calc_unrealized_pnl(
            state.hedge_etf_qty,
            state.hedge_entry_price,
            day["spot"],
        )
        nav = (
            state.cash
            + day["option_value"]
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
                state.position,
                day["greeks"],
                day["feature_row"],
                day["pnl_position_iv"],
                day["pnl_call_iv"],
                day["pnl_put_iv"],
                day["pnl_greeks"],
                day["eod_position_dte"],
            )
        )


def run_backtest(
    etf_by_date,
    opt_by_date,
    signals_df,
    initial_cash=None,
    call_qty=None,
    put_qty=None,
    etf_fee_rate=None,
    trading_calendar=None,
    enriched_opt_by_date=None,
):
    if initial_cash is None:
        initial_cash = CONFIG.backtest.initial_cash
    if call_qty is None:
        call_qty = CONFIG.backtest.call_qty
    if put_qty is None:
        put_qty = CONFIG.backtest.put_qty
    if etf_fee_rate is None:
        etf_fee_rate = CONFIG.backtest.etf_fee_rate

    engine = BacktestEngine(
        etf_by_date,
        opt_by_date,
        signals_df,
        config={
            "initial_cash": initial_cash,
            "call_qty": call_qty,
            "put_qty": put_qty,
            "etf_fee_rate": etf_fee_rate,
        },
        trading_calendar=trading_calendar,
        enriched_opt_by_date=enriched_opt_by_date,
    )
    return engine.run()
