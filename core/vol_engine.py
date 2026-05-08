import os
from pathlib import Path

NUMBA_CACHE_DIR = Path(r"C:\tmp\numba_cache")
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))

import pandas as pd
import numpy as np
from py_vollib_vectorized import (
    vectorized_implied_volatility,
    vectorized_delta,
    vectorized_gamma,
    vectorized_vega,
    vectorized_theta,
)

from .config import CONFIG


def build_daily_ohlc_df(etf_by_date: dict[str, pd.DataFrame]):
    """将 etf_by_date 转为按日期索引的日频 OHLC DataFrame

    Args:
        etf_by_date (dict): data_loader返回的字典
    """
    if not etf_by_date:
        raise ValueError("ETF数据为空")

    rows = []
    for date in etf_by_date:
        if etf_by_date[date].shape[0] != 1:
            raise ValueError(f"{date}的ETF数据行数异常")
        row = etf_by_date[date].iloc[0].copy()
        row["date"] = date
        rows.append(row)

    daily_ohlc_df = pd.DataFrame(rows)

    daily_ohlc_df = daily_ohlc_df.set_index("date").sort_index()
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = set(required_cols) - set(daily_ohlc_df.columns)
    if missing:
        raise ValueError(f"ETF数据缺失:{missing}")
    daily_ohlc_df = daily_ohlc_df[required_cols]

    return daily_ohlc_df


def calculate_yz_hv(
    daily_ohlc_df: pd.DataFrame, rolling_windows=None, annual_days=None
):
    """基于yang-zhang法计算历史波动率

    Args:

    """
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

    # 隔夜收益
    overnight_ret: pd.Series = np.log(curr_open / prev_close)

    # 日内收益
    intraday_ret: pd.Series = np.log(curr_close / curr_open)

    # R-S项
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

    chain_df.loc[valid, "iv"] = vectorized_implied_volatility(
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

    return chain_df


def add_greeks_for_day(chain_df: pd.DataFrame, spot, r=None, annual_days=None):
    if r is None:
        r = CONFIG.vol.risk_free_rate
    if annual_days is None:
        annual_days = CONFIG.vol.annual_days

    chain_df = chain_df.copy()

    chain_df["delta"] = np.nan
    chain_df["delta"] = vectorized_delta(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    )

    chain_df["gamma"] = np.nan
    chain_df["gamma"] = vectorized_gamma(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    )

    chain_df["theta"] = np.nan
    theta_365 = vectorized_theta(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    )
    # py_vollib returns theta per calendar day. Convert it to per trading day.
    chain_df["theta"] = theta_365 * (365 / annual_days)

    chain_df["vega"] = np.nan
    chain_df["vega"] = vectorized_vega(
        flag=chain_df["option_type"],
        S=spot,
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    )

    return chain_df


def calc_atm_iv_for_day(
    daily_opt_chain: pd.DataFrame,
    spot,
    target_dte=None,
    target_dte_min=None,
    target_dte_max=None,
    atm_moneyness_tol=None,
):
    """atm选约

    Args:
        daily_opt_chain (pd.DataFrame): _description_
        spot (_type_): _description_
        target_dte (int, optional): _description_. Defaults to 20.
        target_dte_min (int, optional): _description_. Defaults to 5.
        target_dte_max (int, optional): _description_. Defaults to 35.
    """

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
    # 仅保留dte在要求范围内的期权数据
    chain_df = chain_df[
        (chain_df["dte"] >= target_dte_min) & (chain_df["dte"] <= target_dte_max)
    ]

    # 按strike离spot距离排序
    strike_order = (
        chain_df[["strike_price"]]
        .drop_duplicates()
        .assign(strike_diff=lambda x: (x["strike_price"] - spot).abs())
        .sort_values("strike_diff")
    )

    # 对每个strike，找到第一个call+put都有且到期日合适的
    for strike in strike_order["strike_price"]:
        if abs(strike / spot - 1) > atm_moneyness_tol:
            continue
        strike_chain = chain_df[chain_df["strike_price"] == strike].copy()
        expiry_order = (
            strike_chain[["maturity_date", "dte"]]
            .drop_duplicates()
            .assign(dte_diff=lambda x: (x["dte"] - target_dte).abs())
            .sort_values("dte_diff")
        )

        # 按到期日遍历
        for _, expiry_row in expiry_order.iterrows():
            expiry = expiry_row["maturity_date"]
            pair_chain = strike_chain[strike_chain["maturity_date"] == expiry]
            call = pair_chain[pair_chain["option_type"] == "c"]
            put = pair_chain[pair_chain["option_type"] == "p"]
            if not call.empty and not put.empty:
                call_row = call.iloc[0]
                put_row = put.iloc[0]

                # iv 不能nan
                if (
                    pd.notna(call_row["iv"])
                    and pd.notna(put_row["iv"])
                    and call_row["iv"] > 0
                    and put_row["iv"] > 0
                ):

                    return {
                        "strike": strike,
                        "expiry": expiry,
                        "dte": int(expiry_row["dte"]),
                        "call_iv": call_row["iv"],
                        "put_iv": put_row["iv"],
                        "atm_iv": (call_row["iv"] + put_row["iv"]) / 2,
                        "call": call_row,
                        "put": put_row,
                    }


def build_vol_features(etf_by_date, opt_by_date):
    daily_etf_ohlc = build_daily_ohlc_df(etf_by_date)

    hv_df = calculate_yz_hv(daily_etf_ohlc)

    rows = []

    for date in hv_df.index:
        if date not in opt_by_date:
            continue

        spot = hv_df.loc[date, "close"]
        chain = opt_by_date[date]

        atm = calc_atm_iv_for_day(chain, spot)

        if atm is None:
            atm_iv = np.nan
        else:
            atm_iv = atm["atm_iv"]

        rows.append(
            {
                "date": date,
                "close": spot,
                # "yz_hv5": hv_df.loc[date, "yz_hv5"],
                # "yz_hv20": hv_df.loc[date, "yz_hv20"],
                "yz_hv60": hv_df.loc[date, "yz_hv60"],
                "atm_iv": atm_iv,
            }
        )

    features = pd.DataFrame(rows).set_index("date").sort_index()
    return features
