from pathlib import Path

import pandas as pd

from . import config


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_data_dir(path):
    """把配置里的相对数据目录解析到项目根目录下。"""
    data_dir = Path(path)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir


def load_etf_series(start, end):
    """加载指定日期区间内的 ETF 日线数据，返回 {date: DataFrame}。"""
    etf_by_date = {}
    data_dir = _resolve_data_dir(config.CONFIG.data.etf_dir)
    required_cols = {"open", "high", "low", "close", "volume"}

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    for file_path in sorted(data_dir.glob("*price.parquet")):
        date = _parse_date_from_file(file_path, "_price")
        if start <= date <= end:
            df = _read_parquet(file_path)
            etf_by_date[date] = _validate_df(df, required_cols, date)

    if not etf_by_date:
        raise ValueError("ETF 数据为空")
    return etf_by_date


def load_hedge_series(start, end):
    """加载 delta hedge 标的日线；支持同一日期多只合约。"""
    hedge_dir = config.CONFIG.data.hedge_etf_dir
    if hedge_dir is None:
        return None

    hedge_by_date = {}
    data_dir = _resolve_data_dir(hedge_dir)
    required_cols = {"open", "high", "low", "close", "volume"}

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    for file_path in sorted(data_dir.glob("*price.parquet")):
        date = _parse_date_from_file(file_path, "_price")
        if not start <= date <= end:
            continue

        df = _read_parquet(file_path).copy()
        if "order_book_id" not in df.columns:
            df.insert(0, "order_book_id", _parse_symbol_from_price_file(file_path))
        else:
            df["order_book_id"] = df["order_book_id"].map(_normalize_price_order_book_id)
        df = _validate_df(df, required_cols | {"order_book_id"}, date)
        if date in hedge_by_date:
            hedge_by_date[date] = pd.concat([hedge_by_date[date], df], ignore_index=True)
        else:
            hedge_by_date[date] = df

    if not hedge_by_date:
        raise ValueError(f"delta hedge 标的数据为空: {data_dir}")
    return hedge_by_date


def load_etf_trading_calendar():
    """从 ETF 日线文件名提取完整交易日历，用于按真实交易日计算 DTE。"""
    data_dir = _resolve_data_dir(config.CONFIG.data.etf_dir)
    dates = [
        _parse_date_from_file(file_path, "_price")
        for file_path in sorted(data_dir.glob("*price.parquet"))
    ]
    if not dates:
        raise ValueError("ETF 交易日历为空")

    return pd.DatetimeIndex(dates).sort_values()


def load_opt_series(start, end):
    """加载指定日期区间内的期权链数据，返回 {date: DataFrame}。"""
    opt_by_date = {}
    data_dir = _resolve_data_dir(config.CONFIG.data.opt_dir)
    required_cols = {
        "order_book_id",
        "strike_price",
        "maturity_date",
        "option_type",
        "bid",
        "ask",
        "volume",
        "contract_multiplier",
        "close",
    }

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    for file_path in sorted(data_dir.glob("*chain.parquet")):
        date = _parse_date_from_file(file_path, "_chain")
        if start <= date <= end:
            df = _read_parquet(file_path)
            df.insert(0, "date", date)
            df = _ensure_option_underlying_id(df)
            opt_by_date[date] = _validate_df(df, required_cols, date)

    if not opt_by_date:
        raise ValueError("期权数据为空")
    return opt_by_date


def attach_underlying_prices(opt_by_date, hedge_by_date):
    """把期权对应期货合约的当日价格贴到期权链，供 IV/Greeks 和 hedge 使用。"""
    if hedge_by_date is None:
        return opt_by_date

    result = {}
    price_cols = ["open", "high", "low", "close", "volume"]
    for date, chain_df in opt_by_date.items():
        df = chain_df.copy()
        if "underlying_order_book_id" not in df.columns or date not in hedge_by_date:
            result[date] = df
            continue

        hedge_df = hedge_by_date[date]
        available_cols = [
            col for col in price_cols if col in hedge_df.columns
        ]
        price_df = hedge_df[["order_book_id", *available_cols]].drop_duplicates(
            subset=["order_book_id"],
            keep="last",
        )
        rename_map = {
            col: f"underlying_{col}"
            for col in available_cols
        }
        price_df = price_df.rename(
            columns={
                "order_book_id": "underlying_order_book_id",
                **rename_map,
            }
        )
        df = df.merge(price_df, on="underlying_order_book_id", how="left")
        result[date] = df

    return result


def _ensure_option_underlying_id(df):
    return df


def _validate_df(df, required_cols, date):
    if df.empty:
        raise ValueError(f"{date} 数据为空")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{date} 缺少字段: {missing}")

    return df


def _read_parquet(file_path):
    """读取 parquet，并在远程环境/文件损坏时明确指出问题文件。"""
    try:
        return pd.read_parquet(file_path)
    except Exception as exc:
        raise OSError(f"读取 parquet 失败: {file_path}") from exc


def _parse_date_from_file(file_path, suffix):
    base = file_path.stem.rsplit(suffix, 1)
    date_str = base[0].rsplit("_", 1)[1]
    return pd.Timestamp(date_str)


def _parse_symbol_from_price_file(file_path):
    return _normalize_price_order_book_id(file_path.stem.rsplit("_", 1)[0])


def _normalize_price_order_book_id(value):
    return str(value).rsplit("_", 1)[0]
