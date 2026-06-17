import numpy as np
import pandas as pd

from .config import CONFIG
from . import vol_surface


_VOLLIB_FUNCS = None
IV_OBSERVATION_MODES = {"legacy", "simple_atm_absolute", "surface_percentile"}


def _iv_observation_mode():
    mode = getattr(CONFIG.vol, "iv_observation_mode", "legacy")
    if mode not in IV_OBSERVATION_MODES:
        raise ValueError(
            "CONFIG.vol.iv_observation_mode 只能是 "
            "'legacy'、'simple_atm_absolute' 或 'surface_percentile'"
        )
    return mode


def _surface_signal_enabled():
    mode = _iv_observation_mode()
    if mode == "surface_percentile":
        return True
    if mode == "simple_atm_absolute":
        return False
    return CONFIG.vol.surface_atm_iv_enabled


def _load_vollib_funcs():
    """懒加载 py_vollib_vectorized，缓存命中时避免启动阶段触发 numba。"""
    global _VOLLIB_FUNCS
    if _VOLLIB_FUNCS is None:
        from py_vollib_vectorized import (
            vectorized_delta,
            vectorized_gamma,
            vectorized_implied_volatility,
            vectorized_theta,
            vectorized_vega,
        )

        _VOLLIB_FUNCS = {
            "delta": vectorized_delta,
            "gamma": vectorized_gamma,
            "implied_volatility": vectorized_implied_volatility,
            "theta": vectorized_theta,
            "vega": vectorized_vega,
        }
    return _VOLLIB_FUNCS


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


def _count_trading_dte(date, maturity_date, trading_calendar=None):
    """计算从当前交易日收盘后到到期日之间的剩余交易日数。"""
    date = pd.Timestamp(date).normalize()
    maturity_date = pd.Timestamp(maturity_date).normalize()

    if maturity_date <= date:
        return 0

    if trading_calendar is not None:
        calendar = (
            pd.DatetimeIndex(trading_calendar)
            .normalize()
            .drop_duplicates()
            .sort_values()
        )
        if len(calendar) > 0 and maturity_date <= calendar.max():
            return int(((calendar > date) & (calendar <= maturity_date)).sum())

    return len(pd.bdate_range(date + pd.offsets.BDay(1), maturity_date))


def add_iv_for_day(
    chain_df: pd.DataFrame,
    spot,
    r=None,
    q=None,
    annual_days=None,
    trading_calendar=None,
):
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
    chain_df["dte"] = [
        _count_trading_dte(date, maturity_date, trading_calendar)
        for date, maturity_date in zip(chain_df["date"], chain_df["maturity_date"])
    ]
    chain_df["ttm"] = chain_df["dte"] / annual_days
    chain_df["mid"] = (chain_df["bid"] + chain_df["ask"]) / 2
    chain_df["option_type"] = chain_df["option_type"].str.lower()
    if "underlying_close" in chain_df.columns:
        chain_df["pricing_spot"] = pd.to_numeric(
            chain_df["underlying_close"],
            errors="coerce",
        )
    else:
        chain_df["pricing_spot"] = spot

    valid = (
        (chain_df["dte"] > 0)
        & (chain_df["strike_price"] > 0)
        & (chain_df["mid"] > 0)
        & (chain_df["pricing_spot"] > 0)
        & chain_df["option_type"].notna()
        & chain_df["option_type"].isin(["c", "p"])
    )

    chain_df["iv"] = np.nan
    vollib = _load_vollib_funcs()
    iv_values = vollib["implied_volatility"](
        price=chain_df.loc[valid, "mid"],
        S=chain_df.loc[valid, "pricing_spot"],
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
    if "pricing_spot" not in chain_df.columns:
        if "underlying_close" in chain_df.columns:
            chain_df["pricing_spot"] = pd.to_numeric(
                chain_df["underlying_close"],
                errors="coerce",
            )
        else:
            chain_df["pricing_spot"] = spot
    vollib = _load_vollib_funcs()
    chain_df["delta"] = vollib["delta"](
        flag=chain_df["option_type"],
        S=chain_df["pricing_spot"],
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()
    chain_df["gamma"] = vollib["gamma"](
        flag=chain_df["option_type"],
        S=chain_df["pricing_spot"],
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()

    theta_365 = vollib["theta"](
        flag=chain_df["option_type"],
        S=chain_df["pricing_spot"],
        K=chain_df["strike_price"],
        t=chain_df["ttm"],
        r=r,
        model="black_scholes",
        sigma=chain_df["iv"],
        return_as="series",
    ).to_numpy()
    # py_vollib 返回的是自然日 theta，这里换算为交易日口径。
    chain_df["theta"] = theta_365 * (365 / annual_days)

    chain_df["vega"] = vollib["vega"](
        flag=chain_df["option_type"],
        S=chain_df["pricing_spot"],
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


def filter_standard_option_contracts(chain_df):
    """Exclude dividend-adjusted SSE ETF options from new contract selection."""
    if chain_df is None or chain_df.empty or "contract_symbol" not in chain_df.columns:
        return chain_df

    symbols = chain_df["contract_symbol"].fillna("").astype(str).str.strip()
    adjusted = symbols.str.upper().str.endswith("A") | symbols.str.contains(
        "调整",
        regex=False,
    )
    return chain_df.loc[~adjusted]


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
        "underlying_order_book_id": call_row.get("underlying_order_book_id"),
        "underlying_price": call_row.get("underlying_close", call_row.get("pricing_spot")),
    }


def _atm_pair_total_volume(atm):
    """返回 ATM call+put 成交量；缺失时视为 0，便于低流动性过滤。"""
    if atm is None:
        return 0.0
    call_volume = atm["call"].get("volume", 0) or 0
    put_volume = atm["put"].get("volume", 0) or 0
    return float(call_volume) + float(put_volume)


def _search_near_month_atm_by_volume(candidate_atms, primary_atm):
    """目标 DTE 合约成交量不足时，在同 strike 更近到期月中寻找更活跃的 ATM。"""
    min_total_volume = CONFIG.vol.atm_min_total_volume
    if (
        not CONFIG.vol.atm_low_volume_search_near_month
        or min_total_volume <= 0
        or primary_atm is None
        or _atm_pair_total_volume(primary_atm) >= min_total_volume
    ):
        return primary_atm

    primary_volume = _atm_pair_total_volume(primary_atm)
    near_month_atms = [
        atm
        for atm in candidate_atms
        if atm["dte"] < primary_atm["dte"]
        and _atm_pair_total_volume(atm) > primary_volume
    ]
    if not near_month_atms:
        return primary_atm

    # 从原目标月向前找，优先使用最接近原候选的近月，而不是直接跳到最短 DTE。
    return sorted(near_month_atms, key=lambda atm: atm["dte"], reverse=True)[0]


def _has_contract_specific_underlying(chain_df):
    """商品期权链带有逐合约对应期货价，需要按到期月内的期货价选 ATM。"""
    return (
        "underlying_order_book_id" in chain_df.columns
        and "underlying_close" in chain_df.columns
        and chain_df["underlying_order_book_id"].notna().any()
    )


def _select_contract_month_atm(
    chain_df,
    spot,
    target_dte,
    atm_moneyness_tol,
    preferred_expiry=None,
    preferred_underlying_order_book_id=None,
):
    """商品期权：先定到期月/标的期货，再在该组内选最 ATM 的 call/put 对。"""
    chain_df = chain_df.copy()
    chain_df["_pricing_spot"] = pd.to_numeric(
        chain_df.get("pricing_spot", chain_df["underlying_close"]),
        errors="coerce",
    )
    chain_df["_strike_diff"] = (
        pd.to_numeric(chain_df["strike_price"], errors="coerce")
        - chain_df["_pricing_spot"]
    ).abs()

    if preferred_expiry is not None:
        preferred_chain = chain_df[
            chain_df["maturity_date"].eq(pd.Timestamp(preferred_expiry))
        ]
        if preferred_underlying_order_book_id is not None:
            preferred_chain = preferred_chain[
                preferred_chain["underlying_order_book_id"].astype(str)
                == str(preferred_underlying_order_book_id)
            ]
        if preferred_chain.empty:
            return None
        expiry_order = preferred_chain[
            ["maturity_date", "underlying_order_book_id", "dte"]
        ].drop_duplicates()
    else:
        expiry_order = (
            chain_df[["maturity_date", "underlying_order_book_id", "dte"]]
            .dropna(subset=["maturity_date", "underlying_order_book_id", "dte"])
            .drop_duplicates()
        )
        if (
            getattr(CONFIG.vol, "atm_selection_mode", "target_dte")
            == "near_month_min_dte"
        ):
            # 商品期权的离散 ATM IV 用近月合约滚动：先排除过近到期月，
            # 再取剩余 DTE 最短的一组；跨日跟踪状态在 build_vol_features 中维护。
            expiry_order = expiry_order.sort_values(
                ["dte", "maturity_date", "underlying_order_book_id"]
            )
        else:
            expiry_order = (
                expiry_order.assign(dte_diff=lambda x: (x["dte"] - target_dte).abs())
                .sort_values(
                    ["dte_diff", "dte", "maturity_date", "underlying_order_book_id"]
                )
            )

    for _, expiry_row in expiry_order.iterrows():
        expiry = expiry_row["maturity_date"]
        underlying_order_book_id = expiry_row["underlying_order_book_id"]
        expiry_chain = chain_df[
            (chain_df["maturity_date"] == expiry)
            & (
                chain_df["underlying_order_book_id"].astype(str)
                == str(underlying_order_book_id)
            )
        ]
        if expiry_chain.empty:
            continue

        strike_order = (
            expiry_chain[["strike_price", "_strike_diff"]]
            .dropna(subset=["_strike_diff"])
            .groupby("strike_price", as_index=False)["_strike_diff"]
            .min()
            .sort_values(["_strike_diff", "strike_price"])
        )
        for _, strike_row in strike_order.iterrows():
            strike = strike_row["strike_price"]
            if strike_row["_strike_diff"] > atm_moneyness_tol:
                continue

            pair_chain = expiry_chain[expiry_chain["strike_price"] == strike]
            call_row, put_row = select_call_put_pair(pair_chain, spot)
            if call_row is not None:
                return _make_atm_result(strike, expiry, call_row, put_row)

    return None


def _select_nearest_atm(
    chain_df,
    spot,
    target_dte=None,
    target_dte_min=None,
    target_dte_max=None,
    atm_moneyness_tol=None,
    preferred_expiry=None,
    preferred_underlying_order_book_id=None,
):
    """选择策略使用的 ATM call/put 对。"""
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

    if _has_contract_specific_underlying(chain_df):
        return _select_contract_month_atm(
            chain_df,
            spot,
            target_dte,
            atm_moneyness_tol,
            preferred_expiry=preferred_expiry,
            preferred_underlying_order_book_id=preferred_underlying_order_book_id,
        )

    chain_df = chain_df.copy()
    if "pricing_spot" in chain_df.columns:
        ref_spot = chain_df["pricing_spot"]
    elif "underlying_close" in chain_df.columns:
        ref_spot = pd.to_numeric(chain_df["underlying_close"], errors="coerce")
    else:
        ref_spot = spot
    chain_df["_strike_diff"] = (chain_df["strike_price"] - ref_spot).abs()

    strike_order = (
        chain_df[["strike_price", "_strike_diff"]]
        .dropna(subset=["_strike_diff"])
        .groupby("strike_price", as_index=False)["_strike_diff"]
        .min()
        .sort_values("_strike_diff")
    )

    for _, strike_row in strike_order.iterrows():
        strike = strike_row["strike_price"]
        if strike_row["_strike_diff"] > atm_moneyness_tol:
            continue

        strike_chain = chain_df[chain_df["strike_price"] == strike]
        expiry_order = (
            strike_chain[["maturity_date", "dte"]]
            .drop_duplicates()
            .assign(dte_diff=lambda x: (x["dte"] - target_dte).abs())
            .sort_values(["dte_diff", "dte"])
        )

        candidate_atms = []
        for _, expiry_row in expiry_order.iterrows():
            expiry = expiry_row["maturity_date"]
            pair_chain = strike_chain[strike_chain["maturity_date"] == expiry]
            call_row, put_row = select_call_put_pair(pair_chain, spot)
            if call_row is not None:
                candidate_atms.append(
                    _make_atm_result(strike, expiry, call_row, put_row)
                )

        if candidate_atms:
            primary_atm = candidate_atms[0]
            return _search_near_month_atm_by_volume(candidate_atms, primary_atm)

    return None


def select_atm_from_chain(
    chain_df: pd.DataFrame,
    spot,
    target_dte=None,
    target_dte_min=None,
    target_dte_max=None,
    atm_moneyness_tol=None,
    preferred_expiry=None,
    preferred_underlying_order_book_id=None,
):
    """从已计算 IV/Greeks 的期权链中选择策略使用的 ATM 跨式合约。"""
    if target_dte is None:
        target_dte = CONFIG.vol.atm_target_dte
    if target_dte_min is None:
        target_dte_min = CONFIG.vol.atm_target_dte_min
    if target_dte_max is None:
        target_dte_max = CONFIG.vol.atm_target_dte_max
    if atm_moneyness_tol is None:
        atm_moneyness_tol = CONFIG.vol.atm_moneyness_tol

    selection_mode = getattr(CONFIG.vol, "atm_selection_mode", "target_dte")
    if selection_mode not in {"target_dte", "near_month_min_dte"}:
        raise ValueError(
            "CONFIG.vol.atm_selection_mode 只能是 "
            "'target_dte' 或 'near_month_min_dte'"
        )

    chain_df = filter_standard_option_contracts(chain_df)
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
        preferred_expiry=preferred_expiry,
        preferred_underlying_order_book_id=preferred_underlying_order_book_id,
    )


def calc_atm_pool_volume(
    chain_df,
    spot,
    target_dte_min=None,
    target_dte_max=None,
    atm_moneyness_tol=None,
):
    """汇总 ATM 附近合约池成交量，用于观察真实流动性而非单一合约跳变。"""
    if target_dte_min is None:
        target_dte_min = CONFIG.vol.atm_target_dte_min
    if target_dte_max is None:
        target_dte_max = CONFIG.vol.atm_target_dte_max
    if atm_moneyness_tol is None:
        atm_moneyness_tol = CONFIG.vol.atm_moneyness_tol

    pool = filter_standard_option_contracts(chain_df)
    pool = pool[
        (pool["contract_multiplier"] == CONFIG.vol.contract_multiplier)
        & (pool["dte"] >= target_dte_min)
        & (pool["dte"] <= target_dte_max)
    ]
    if "pricing_spot" in pool.columns:
        ref_spot = pool["pricing_spot"]
    elif "underlying_close" in pool.columns:
        ref_spot = pd.to_numeric(pool["underlying_close"], errors="coerce")
    else:
        ref_spot = spot
    pool = pool[(pool["strike_price"] - ref_spot).abs() <= atm_moneyness_tol]
    if pool.empty:
        return {
            "atm_pool_call_volume": np.nan,
            "atm_pool_put_volume": np.nan,
            "atm_pool_total_volume": np.nan,
            "atm_pool_min_leg_volume": np.nan,
        }

    call_volume = pool.loc[pool["option_type"] == "c", "volume"].sum()
    put_volume = pool.loc[pool["option_type"] == "p", "volume"].sum()
    return {
        "atm_pool_call_volume": call_volume,
        "atm_pool_put_volume": put_volume,
        "atm_pool_total_volume": call_volume + put_volume,
        "atm_pool_min_leg_volume": min(call_volume, put_volume),
    }


def _calc_atm_iv_percentile(atm_iv_series, window):
    """计算当日 ATM IV 在过去固定交易日窗口中的历史百分位。"""
    if window is None or window <= 0:
        return pd.Series(np.nan, index=atm_iv_series.index)

    def percentile(values):
        current = values[-1]
        return (values <= current).mean()

    # ATM IV 偶尔会因为合约筛选缺口缺失；百分位用最近 window 个有效样本计算，
    # 避免单个缺失日把后续 252 天的百分位全部打成 NaN。
    valid_atm_iv = atm_iv_series.dropna()
    percentile_series = valid_atm_iv.rolling(window, min_periods=window).apply(
        percentile,
        raw=True,
    )
    return percentile_series.reindex(atm_iv_series.index)


def _surface_config_from_runtime():
    filters = vol_surface.SurfaceFilters(
        min_dte=CONFIG.vol.surface_min_dte,
        min_volume=CONFIG.vol.surface_min_volume,
        max_spread_pct=CONFIG.vol.surface_max_spread_pct,
        min_abs_delta=CONFIG.vol.surface_min_abs_delta,
        max_abs_delta=CONFIG.vol.surface_max_abs_delta,
    )
    return vol_surface.SurfaceConfig(
        standard_dtes=CONFIG.vol.surface_standard_dtes,
        annual_days=CONFIG.vol.annual_days,
        risk_free_rate=CONFIG.vol.risk_free_rate,
        filters=filters,
    )


def _calc_surface_atm_for_day(chain_df):
    """提取固定期限 ATM IV；曲面全样本图只作诊断，日频信号逐日计算。"""
    if not _surface_signal_enabled():
        return {
            "surface_atm_iv": np.nan,
            "surface_atm_iv_method": None,
            "surface_iv_point_count": np.nan,
        }

    surface_config = _surface_config_from_runtime()
    surface = vol_surface.build_daily_surface(
        chain_df,
        config=surface_config,
        standard_dtes=(CONFIG.vol.surface_atm_target_dte,),
        k_grid_mode=CONFIG.vol.surface_k_grid_mode,
        allow_term_extrapolate=CONFIG.vol.surface_allow_term_extrapolate,
        term_extrapolate_mode=CONFIG.vol.surface_term_extrapolate_mode,
    )
    atm_df = surface["atm"]
    if atm_df.empty:
        surface_atm_iv = np.nan
        method = "empty"
    else:
        atm_row = atm_df.iloc[0]
        surface_atm_iv = atm_row.get("surface_iv", np.nan)
        method = atm_row.get("method")
    return {
        "surface_atm_iv": surface_atm_iv,
        "surface_atm_iv_method": method,
        "surface_iv_point_count": len(surface["points"]),
    }


def build_enriched_option_chains(etf_by_date, opt_by_date, trading_calendar=None):
    """预先计算每日全链 IV 和 Greeks，供特征生成和回测主流程复用。"""
    daily_etf_ohlc = build_daily_ohlc_df(etf_by_date)
    if trading_calendar is None:
        trading_calendar = daily_etf_ohlc.index

    enriched_by_date = {}
    for date in daily_etf_ohlc.index:
        if date not in opt_by_date:
            continue

        spot = daily_etf_ohlc.loc[date, "close"]
        chain_df = add_iv_for_day(
            opt_by_date[date],
            spot,
            trading_calendar=trading_calendar,
        )
        enriched_by_date[date] = add_greeks_for_day(chain_df, spot)

    return enriched_by_date


def build_vol_features(
    etf_by_date,
    opt_by_date,
    trading_calendar=None,
    enriched_opt_by_date=None,
):
    """生成策略信号需要的波动率特征表。"""
    daily_etf_ohlc = build_daily_ohlc_df(etf_by_date)
    if trading_calendar is None:
        trading_calendar = daily_etf_ohlc.index

    hv_df = calculate_yz_hv(daily_etf_ohlc)
    rows = []
    active_atm_expiry = None
    active_atm_underlying_order_book_id = None
    track_near_month = (
        getattr(CONFIG.vol, "atm_selection_mode", "target_dte")
        == "near_month_min_dte"
    )

    for date in hv_df.index:
        if date not in opt_by_date:
            continue

        spot = hv_df.loc[date, "close"]
        if enriched_opt_by_date is not None and date in enriched_opt_by_date:
            chain_df = enriched_opt_by_date[date]
        else:
            chain_df = add_iv_for_day(
                opt_by_date[date],
                spot,
                trading_calendar=trading_calendar,
            )

        if track_near_month and _has_contract_specific_underlying(chain_df):
            active_dte = None
            if active_atm_expiry is not None:
                active_dte = _count_trading_dte(
                    date,
                    active_atm_expiry,
                    trading_calendar,
                )
                if active_dte < CONFIG.vol.atm_target_dte_min:
                    active_atm_expiry = None
                    active_atm_underlying_order_book_id = None

            if active_atm_expiry is not None:
                atm = select_atm_from_chain(
                    chain_df,
                    spot,
                    preferred_expiry=active_atm_expiry,
                    preferred_underlying_order_book_id=(
                        active_atm_underlying_order_book_id
                    ),
                )
            else:
                atm = select_atm_from_chain(chain_df, spot)
                if atm is not None:
                    active_atm_expiry = atm["expiry"]
                    active_atm_underlying_order_book_id = atm.get(
                        "underlying_order_book_id"
                    )
        else:
            atm = select_atm_from_chain(chain_df, spot)
        atm_pool_volume = calc_atm_pool_volume(chain_df, spot)
        surface_atm = _calc_surface_atm_for_day(chain_df)

        if atm is None:
            atm_iv = np.nan
            atm_strike = np.nan
            atm_expiry = pd.NaT
            atm_dte = np.nan
            atm_call_volume = np.nan
            atm_put_volume = np.nan
            atm_call_code = None
            atm_put_code = None
            atm_underlying_order_book_id = None
            atm_underlying_price = np.nan
        else:
            atm_iv = atm["atm_iv"]
            atm_strike = atm["strike"]
            atm_expiry = atm["expiry"]
            atm_dte = atm["dte"]
            atm_call_volume = atm["call"].get("volume", np.nan)
            atm_put_volume = atm["put"].get("volume", np.nan)
            atm_call_code = atm["call"].get("order_book_id")
            atm_put_code = atm["put"].get("order_book_id")
            atm_underlying_order_book_id = atm.get("underlying_order_book_id")
            atm_underlying_price = atm.get("underlying_price", np.nan)

        rows.append(
            {
                "date": date,
                "close": spot,
                "yz_hv60": hv_df.loc[date, "yz_hv60"],
                "atm_iv": atm_iv,
                "atm_strike": atm_strike,
                "atm_expiry": atm_expiry,
                "atm_dte": atm_dte,
                "atm_call_code": atm_call_code,
                "atm_put_code": atm_put_code,
                "atm_underlying_order_book_id": atm_underlying_order_book_id,
                "atm_underlying_price": atm_underlying_price,
                "atm_call_volume": atm_call_volume,
                "atm_put_volume": atm_put_volume,
                "atm_total_volume": atm_call_volume + atm_put_volume,
                **atm_pool_volume,
                **surface_atm,
            }
        )

    features_df = pd.DataFrame(rows).set_index("date").sort_index()
    features_df["atm_iv_percentile"] = _calc_atm_iv_percentile(
        features_df["atm_iv"],
        CONFIG.vol.atm_iv_percentile_window,
    )
    observation_mode = _iv_observation_mode()
    if _surface_signal_enabled():
        features_df["surface_atm_iv_percentile"] = _calc_atm_iv_percentile(
            features_df["surface_atm_iv"],
            CONFIG.vol.atm_iv_percentile_window,
        )
        features_df["signal_iv"] = features_df["surface_atm_iv"]
        features_df["signal_iv_percentile"] = features_df[
            "surface_atm_iv_percentile"
        ]
    elif observation_mode == "simple_atm_absolute":
        features_df["surface_atm_iv_percentile"] = np.nan
        features_df["signal_iv"] = features_df["atm_iv"]
        features_df["signal_iv_percentile"] = np.nan
    else:
        features_df["signal_iv"] = features_df["atm_iv"]
        features_df["signal_iv_percentile"] = features_df["atm_iv_percentile"]
    return features_df
