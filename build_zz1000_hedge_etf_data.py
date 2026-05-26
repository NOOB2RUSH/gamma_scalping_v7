from __future__ import annotations

import argparse
from pathlib import Path

import akshare as ak
import pandas as pd


ETF_SYMBOL = "512100"
ETF_FILE_PREFIX = "512100.XSHG"


def parse_args():
    parser = argparse.ArgumentParser(
        description="下载南方中证1000ETF(512100)日线，作为中证1000 gamma scalping 对冲标的。"
    )
    parser.add_argument("--start", required=True, help="开始日期，例如 20220801")
    parser.add_argument("--end", required=True, help="结束日期，例如 20260101")
    parser.add_argument(
        "--output-dir",
        default="data/zz1000/hedge_etf",
        help="输出目录，默认 data/zz1000/hedge_etf。",
    )
    return parser.parse_args()


def normalize_etf_daily(raw_df):
    """把 AKShare 中文字段转换成项目统一 OHLCV 字段。"""
    df = raw_df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close", "volume"])


def write_daily_files(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        date = row["date"]
        daily_df = pd.DataFrame(
            [
                {
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                    "amount": row["amount"],
                    "order_book_id": ETF_FILE_PREFIX,
                    "source": "akshare_fund_etf_hist_em",
                }
            ]
        )
        file_path = output_dir / f"{ETF_FILE_PREFIX}_{date:%Y-%m-%d}_price.parquet"
        daily_df.to_parquet(file_path, index=False)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_df = ak.fund_etf_hist_em(
        symbol=ETF_SYMBOL,
        period="daily",
        start_date=args.start,
        end_date=args.end,
        adjust="",
    )
    df = normalize_etf_daily(raw_df)
    write_daily_files(df, output_dir)

    print(f"[512100] days={len(df)}")
    if not df.empty:
        print(f"[512100] range={df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"[512100] output={output_dir}")


if __name__ == "__main__":
    main()
