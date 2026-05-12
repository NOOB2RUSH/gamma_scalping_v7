import os
from pathlib import Path

NUMBA_CACHE_DIR = Path(r"C:\tmp\numba_cache")
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import numpy as np
import pandas as pd
from py_vollib_vectorized import (
    vectorized_delta,
    vectorized_gamma,
    vectorized_implied_volatility,
    vectorized_theta,
    vectorized_vega,
)

from .config import CONFIG


def build_daily_ohlc_df(etf_by_date: dict[str, pd.DataFrame]):
    """把 data_loader 返回的 ETF 数据转换为按日期索引的日频 OHLC。"""
    if not etf_by_date:
        raise ValueError("ETF 数据为空")

    rows = []
    for date in etf_by_date:
        if etf_by_date[date].shape[0] != 1:
            raise ValueError(f"{date} 的 ETF 数据行数异常")

        row = etf_by_date[date].iloc[0].copy()
        row["date"] = date
        rows.append(row)

    daily_ohlc_df = pd.DataFrame(rows).set_index("date").sort_index()
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(daily_ohlc_df.columns)
    if missing:
        raise ValueError(f"ETF 数据缺少字段: {missing}")

    return daily_ohlc_df[required_cols]


def calculate_yz_hv(
    daily_ohlc_df: pd.DataFrame, rolling_windows=None, annual_days=None
):
    """基于 Yang-Zhang 方法计算年化历史波动率。"""
    if rolling_windows is None:
        rolling_windows = CONFIG.vol.hv_windows
    if annual_days is None:
        annual_days = CONFIG.vol.annual_days

    result: pd.DataFrame = daily_ohlc_df.copy()
    curr_open = result["open"]
    curr_high = result["high"]
    curr_low = result["low"]
    curr_close = result["close"]
    prev_close = curr_close.shift(1)

    # 隔夜收益。
    overnight_ret: pd.Series = np.log(curr_open / prev_close)

    # 日内开收收益。
    intraday_ret: pd.Series = np.log(curr_close / curr_open)

    # Rogers-Satchell 波动率项。
    rs_ret: pd.Series = np.log(curr_high / curr_open) * np.log(
        curr_high / curr_close
    ) + np.log(curr_low / curr_open) * np.log(curr_low / curr_close)

    result["overnight_ret"] = overnight_ret
    result["intraday_ret"] = intraday_ret
    result["rs_ret"] = rs_ret

    for window in rolling_windows:
        overnight_var = overnight_ret.rolling(window).var()
        intraday_var = intraday_ret.rolling(window).var()
        rs_mean = rs_ret.rolling(window).mean()

        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz_var = overnight_var + k * intraday_var + (1 - k) * rs_mean
        result[f"yz_var{window}"] = yz_var
        result[f"yz_hv{window}"] = np.sqrt(yz_var) * np.sqrt(annual_days)

    return result


def add_iv_for_day(chain_df: pd.DataFrame, spot, r=None, q=None, annual_days=None):
    """为单日期权链计算 mid、DTE、TTM 和隐含波动率。"""
    if r is None:
        r = CONFIG.vol.risk_free_rate
    if q is None:
        q = CONFIG.vol.dividend_yield
    if annual_days is None:
        annual_days = CONFIG.vol.annual_days

    chain_df = chain_df.copy()
    chain_df["date"] = pd.to_datetime(chain_df["date"])
    chain_df["maturity_date"] = pd.to_datetime(chain_df["maturity_date"])
    chain_df["dte"] = (chain_df["maturity_date"] - chain_df["date"]).dt.days
    chain_df["ttm"] = chain_df["dte"] / annual_days
    chain_df["mid"] = (chain_df["bid"] + chain_df["ask"]) / 2
    chain_df["option_type"] = chain_df["option_type"].str.lower()

    valid = (
        (chain_df["dte"] > 0)
        & (chain_df["strike_price"] > 0)
        & (chain_df["mid"] > 0)
        & chain_df["option_type"].notna()
        & chain_df["option_type"].isin(["c", "p"])
    )

    chain_df["iv"] = np.nan
    iv_values = vectorized_implied_volatility(
        price=chain_df.loc[valid, "mid"],
        S=spot,
        t=chain_df.loc[valid, "ttm"],
        K=chain_df.loc[valid, "strike_price"],
        r=r,
        flag=chain_df.loc[valid, "option_type"],
        model="black_scholes",
        return_as="series",
        on_error="ignore",
    )
    # vectorized_* 返回的 Series 索引从 0 开始；写回原 DataFrame 时必须按位置赋值。
    chain_df.loc[valid, "iv"] = iv_values.to_numpy()

    return chain_df


def add_greeks_for_day(chain_df: pd.DataFrame, spot, r=None, annual_days=None):
    """为单日期权链计算 Delta、Gamma、Theta、Vega。"""
    if r is None:
        r = CONFIG.vol.risk_free_rate
    if annual_days is None:
        annual_days = CONFIG.vol.annual_days

    chain_df = chain_df.copy()
    chain_df["delta"] = vectorized_delta(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()
    chain_df["gamma"] = vectorized_gamma(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()

    theta_365 = vectorized_theta(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()
    # py_vollib 返回的是自然日 theta，这里换算为交易日口径。
    chain_df["theta"] = theta_365 * (365 / annual_days)

    chain_df["vega"] = vectorized_vega(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()
    return chain_df


def select_call_put_pair(pair_chain, spot):
    """取同一行权价和到期日下可确认的真实 call/put 对"""
    calls = pair_chain[pair_chain["option_type"] == "c"]
    puts = pair_chain[pair_chain["option_type"] == "p"]

    if len(calls) == 0 or len(puts) == 0:
        return None, None

    if len(calls) != 1 or len(puts) != 1:
        candidates = []

        for _, call_row in calls.iterrows():
            for _, put_row in puts.iterrows():
                call_iv_valid = pd.notna(call_row["iv"]) and call_row["iv"] > 0
                put_iv_valid = pd.notna(put_row["iv"]) and put_row["iv"] > 0
                if call_iv_valid and put_iv_valid:
                    candidates.append(
                        (
                            call_row["volume"] + put_row["volume"],
                            call_row,
                            put_row,
                        )
                    )

        if not candidates:
            return None, None

        _, call_row, put_row = max(candidates, key=lambda item: item[0])
        return call_row, put_row

    call_row = calls.iloc[0]
    put_row = puts.iloc[0]
    call_iv_valid = pd.notna(call_row["iv"]) and call_row["iv"] > 0
    put_iv_valid = pd.notna(put_row["iv"]) and put_row["iv"] > 0
    if not (call_iv_valid and put_iv_valid):
        return None, None

    return call_row, put_row


def resolve_position_pair(position, chain_df):
    """按原 order_book_id 找回已有持仓合约。"""
    call_rows = chain_df[chain_df["order_book_id"] == position["call_code"]]
    put_rows = chain_df[chain_df["order_book_id"] == position["put_code"]]
    if call_rows.empty or put_rows.empty:
        raise IndexError("position contract not found in option chain")

    return call_rows.iloc[0], put_rows.iloc[0]


def _make_atm_result(strike, expiry, call_row, put_row):
    return {
        "strike": strike,
        "expiry": expiry,
        "dte": int(call_row["dte"]),
        "call_iv": call_row["iv"],
        "put_iv": put_row["iv"],
        "atm_iv": (call_row["iv"] + put_row["iv"]) / 2,
        "call": call_row,
        "put": put_row,
    }


def _select_nearest_atm(
    chain_df,
    spot,
    target_dte=None,
    target_dte_min=None,
    target_dte_max=None,
    atm_moneyness_tol=None,
):
    """优先选择最接近现价、期限最接近目标 DTE 的 ATM call/put 对。"""
    if target_dte is None:
        target_dte = CONFIG.vol.atm_target_dte
    if target_dte_min is None:
        target_dte_min = CONFIG.vol.atm_target_dte_min
    if target_dte_max is None:
        target_dte_max = CONFIG.vol.atm_target_dte_max
    if atm_moneyness_tol is None:
        atm_moneyness_tol = CONFIG.vol.atm_moneyness_tol

    chain_df = chain_df[
        (chain_df["dte"] >= target_dte_min) & (chain_df["dte"] <= target_dte_max)
    ]
    if chain_df.empty:
        return None

    strike_order = (
        chain_df[["strike_price"]]
        .drop_duplicates()
        .assign(strike_diff=lambda x: (x["strike_price"] - spot).abs())
        .sort_values("strike_diff")
    )

    for strike in strike_order["strike_price"]:
        if abs(strike / spot - 1) > atm_moneyness_tol:
            continue

        strike_chain = chain_df[chain_df["strike_price"] == strike]
        expiry_order = (
            strike_chain[["maturity_date", "dte"]]
            .drop_duplicates()
            .assign(dte_diff=lambda x: (x["dte"] - target_dte).abs())
            .sort_values(["dte_diff", "dte"])
        )

        for _, expiry_row in expiry_order.iterrows():
            expiry = expiry_row["maturity_date"]
            pair_chain = strike_chain[strike_chain["maturity_date"] == expiry]
            call_row, put_row = select_call_put_pair(pair_chain, spot)
            if call_row is not None:
                return _make_atm_result(strike, expiry, call_row, put_row)

    return None


def calc_atm_iv_for_day(
    daily_opt_chain: pd.DataFrame,
    spot,
    target_dte=None,
    target_dte_min=None,
    target_dte_max=None,
    atm_moneyness_tol=None,
):
    """按策略交易口径选择当日 ATM 跨式合约并返回 IV 与 Greeks。"""
    if target_dte is None:
        target_dte = CONFIG.vol.atm_target_dte
    if target_dte_min is None:
        target_dte_min = CONFIG.vol.atm_target_dte_min
    if target_dte_max is None:
        target_dte_max = CONFIG.vol.atm_target_dte_max
    if atm_moneyness_tol is None:
        atm_moneyness_tol = CONFIG.vol.atm_moneyness_tol

    chain_df = add_iv_for_day(daily_opt_chain, spot)
    chain_df = add_greeks_for_day(chain_df, spot)
    chain_df = chain_df[
        chain_df["contract_multiplier"] == CONFIG.vol.contract_multiplier
    ]
    # 仅保留 DTE 在策略要求范围内的期权数据。
    chain_df = chain_df[
        (chain_df["dte"] >= target_dte_min) & (chain_df["dte"] <= target_dte_max)
    ]

    return _select_nearest_atm(
        chain_df,
        spot,
        target_dte=target_dte,
        target_dte_min=target_dte_min,
        target_dte_max=target_dte_max,
        atm_moneyness_tol=atm_moneyness_tol,
    )


def calc_feature_atm_iv_for_day(daily_opt_chain: pd.DataFrame, spot):
    """按特征计算口径获取真实 ATM 合约 IV；找不到有效合约时返回 None。"""
    chain_df = add_iv_for_day(daily_opt_chain, spot)
    chain_df = add_greeks_for_day(chain_df, spot)
    chain_df = chain_df[
        (chain_df["contract_multiplier"] == CONFIG.vol.contract_multiplier)
        & (chain_df["dte"] >= CONFIG.vol.atm_target_dte_min)
        & (chain_df["dte"] <= CONFIG.vol.atm_target_dte_max)
    ]

    if chain_df.empty:
        return None

    return _select_nearest_atm(chain_df, spot)


def build_vol_features(etf_by_date, opt_by_date):
    """生成策略信号需要的波动率特征表。"""
    daily_etf_ohlc = build_daily_ohlc_df(etf_by_date)
    hv_df = calculate_yz_hv(daily_etf_ohlc)
    rows = []

    for date in hv_df.index:
        if date not in opt_by_date:
            continue

        spot = hv_df.loc[date, "close"]
        chain = opt_by_date[date]
        atm = calc_feature_atm_iv_for_day(chain, spot)

        if atm is None:
            atm_iv = np.nan
            atm_strike = np.nan
            atm_expiry = pd.NaT
            atm_dte = np.nan
        else:
            atm_iv = atm["atm_iv"]
            atm_strike = atm["strike"]
            atm_expiry = atm["expiry"]
            atm_dte = atm["dte"]

        rows.append(
            {
                "date": date,
                "close": spot,
                "yz_hv60": hv_df.loc[date, "yz_hv60"],
                "atm_iv": atm_iv,
                "atm_strike": atm_strike,
                "atm_expiry": atm_expiry,
                "atm_dte": atm_dte,
            }
        )

    return pd.DataFrame(rows).set_index("date").sort_index()
