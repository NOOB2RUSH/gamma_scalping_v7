from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
import core
from core.live import market_data, storage
from core.live.runtime import load_product_config, project_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Promote the latest live quote snapshots into canonical strategy "
            "ETF/option parquet files."
        )
    )
    parser.add_argument("--product", choices=core.config.available_products(), required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date", action="append", default=[])
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip canonical files that already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = promote_quote_snapshots(
        args.product,
        start_date=args.start_date,
        end_date=args.end_date,
        dates=args.date,
        overwrite=not args.no_overwrite,
    )
    for row in rows:
        print(
            " ".join(
                f"{key}={value}"
                for key, value in row.items()
                if value is not None
            )
        )


def promote_quote_snapshots(
    product,
    start_date=None,
    end_date=None,
    dates=None,
    overwrite=True,
):
    config = load_product_config(product)
    spec = market_data.SSE_ETF_OPTION_SPECS.get(product)
    if spec is None:
        raise ValueError(f"Unsupported product for quote promotion: {product}")

    quote_root = storage.PROJECT_ROOT / "data" / "live" / product / "quotes"
    etf_out = project_path(config.data.etf_dir)
    opt_out = project_path(config.data.opt_dir)
    etf_out.mkdir(parents=True, exist_ok=True)
    opt_out.mkdir(parents=True, exist_ok=True)

    selected_dates = _selected_quote_dates(
        quote_root,
        start_date=start_date,
        end_date=end_date,
        dates=dates,
    )
    rows = []
    for date_text in selected_dates:
        day_dir = quote_root / date_text.replace("-", "")
        etf_src, opt_src = _latest_snapshot_pair(day_dir, date_text)
        if etf_src is None or opt_src is None:
            rows.append(
                {
                    "date": date_text,
                    "status": "skipped",
                    "reason": "missing_quote_snapshot_pair",
                }
            )
            continue

        etf_dst = etf_out / f"{spec.etf_file_prefix}_{date_text}_price.parquet"
        opt_dst = opt_out / f"{spec.option_file_prefix}_{date_text}_chain.parquet"
        if not overwrite and etf_dst.exists() and opt_dst.exists():
            rows.append(
                {
                    "date": date_text,
                    "status": "skipped",
                    "reason": "canonical_files_exist",
                    "etf": etf_dst,
                    "option": opt_dst,
                }
            )
            continue

        etf_rows, opt_rows = _validate_snapshot_pair(etf_src, opt_src)
        shutil.copy2(etf_src, etf_dst)
        shutil.copy2(opt_src, opt_dst)
        rows.append(
            {
                "date": date_text,
                "status": "ok",
                "etf_rows": etf_rows,
                "option_rows": opt_rows,
                "etf_source": etf_src,
                "option_source": opt_src,
                "etf": etf_dst,
                "option": opt_dst,
            }
        )
    return rows


def _selected_quote_dates(quote_root, start_date=None, end_date=None, dates=None):
    if dates:
        selected = {_normalize_date_text(value) for value in dates}
    else:
        selected = {
            _normalize_date_text(path.name)
            for path in quote_root.glob("20??????")
            if path.is_dir()
        }
    start = _date_or_none(start_date)
    end = _date_or_none(end_date)
    result = []
    for date_text in selected:
        date = pd.Timestamp(date_text).normalize()
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        result.append(date_text)
    return sorted(result)


def _latest_snapshot_pair(day_dir, expected_date):
    if not day_dir.exists():
        return None, None
    expected_date = _normalize_date_text(expected_date)
    for metadata_path in reversed(sorted(day_dir.glob("*_metadata.json"))):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _normalize_date_text(metadata.get("quote_date")) != expected_date:
            continue
        stem = metadata_path.name.removesuffix("_metadata.json")
        etf_path = day_dir / f"{stem}_etf.parquet"
        option_path = day_dir / f"{stem}_option_chain.parquet"
        if etf_path.exists() and option_path.exists():
            return etf_path, option_path
    return None, None


def _validate_snapshot_pair(etf_src, opt_src):
    etf_df = pd.read_parquet(etf_src)
    opt_df = pd.read_parquet(opt_src)
    required_etf = {"open", "high", "low", "close", "volume"}
    required_opt = {
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
    missing_etf = required_etf - set(etf_df.columns)
    missing_opt = required_opt - set(opt_df.columns)
    if missing_etf or missing_opt:
        raise ValueError(
            f"Invalid quote snapshot pair: etf_missing={sorted(missing_etf)} "
            f"option_missing={sorted(missing_opt)}"
        )
    return len(etf_df), len(opt_df)


def _normalize_date_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return str(pd.Timestamp(text).date())


def _date_or_none(value):
    if value is None or value == "":
        return None
    return pd.Timestamp(value).normalize()


if __name__ == "__main__":
    main()
