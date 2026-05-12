from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).parent


def load_etf_series(start, end):
    """加载指定日期区间内的 ETF 日线数据，返回 {date: DataFrame}。"""
    etf_by_date = {}
    data_dir = SCRIPT_DIR.parent / "data" / "etf"
    required_cols = {"open", "high", "low", "close", "volume"}

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    for file_path in sorted(data_dir.glob("*price.parquet")):
        date = _parse_date_from_file(file_path, "_price")
        if start <= date <= end:
            df = pd.read_parquet(file_path)
            etf_by_date[date] = _validate_df(df, required_cols, date)

    if not etf_by_date:
        raise ValueError("ETF 数据为空")
    return etf_by_date


def load_opt_series(start, end):
    """加载指定日期区间内的期权链数据，返回 {date: DataFrame}。"""
    opt_by_date = {}
    data_dir = SCRIPT_DIR.parent / "data" / "opt"
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
            df = pd.read_parquet(file_path)
            df.insert(0, "date", date)
            opt_by_date[date] = _validate_df(df, required_cols, date)

    if not opt_by_date:
        raise ValueError("期权数据为空")
    return opt_by_date


def _validate_df(df, required_cols, date):
    if df.empty:
        raise ValueError(f"{date} 数据为空")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{date} 缺少字段: {missing}")

    return df


def _parse_date_from_file(file_path, suffix):
    base = file_path.stem.rsplit(suffix, 1)
    date_str = base[0].split("_", 1)[1]
    return pd.Timestamp(date_str)
