from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


DEFAULT_ACTIONS_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "live_corporate_actions.csv"
)


@lru_cache(maxsize=8)
def _load_actions_cached(path_text: str, modified_ns: int):
    del modified_ns
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for column in ("ex_date", "record_date", "payment_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime(
                "%Y-%m-%d"
            )
    if "product" in frame.columns:
        frame["product"] = frame["product"].astype(str).str.lower().str.strip()
    return frame


def load_corporate_actions(path=None):
    """Load source-controlled live corporate actions.

    The file is deliberately separate from strategy parameters: these rows are
    dated accounting facts, not assumptions used for option valuation.
    """
    path = Path(path or DEFAULT_ACTIONS_PATH)
    modified_ns = path.stat().st_mtime_ns if path.exists() else -1
    return _load_actions_cached(str(path.resolve()), modified_ns).copy()


def cash_distribution_per_share(product, ex_date, path=None):
    frame = load_corporate_actions(path)
    required = {"product", "ex_date", "cash_dividend_per_share"}
    if frame.empty or not required.issubset(frame.columns):
        return 0.0
    target_date = pd.Timestamp(ex_date).strftime("%Y-%m-%d")
    rows = frame.loc[
        frame["product"].eq(str(product).lower().strip())
        & frame["ex_date"].eq(target_date)
    ]
    if rows.empty:
        return 0.0
    values = pd.to_numeric(rows["cash_dividend_per_share"], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else 0.0


def cash_distribution_details(product, ex_date, path=None):
    frame = load_corporate_actions(path)
    required = {"product", "ex_date", "cash_dividend_per_share"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    target_date = pd.Timestamp(ex_date).strftime("%Y-%m-%d")
    rows = frame.loc[
        frame["product"].eq(str(product).lower().strip())
        & frame["ex_date"].eq(target_date)
    ]
    return rows.to_dict("records")


def cash_distribution_series(product, dates, path=None):
    return pd.Series(
        [cash_distribution_per_share(product, value, path=path) for value in dates],
        index=getattr(dates, "index", None),
        dtype="float64",
    )
