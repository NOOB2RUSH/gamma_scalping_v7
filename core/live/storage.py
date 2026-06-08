from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .runtime import PROJECT_ROOT


def utc_now_text():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def local_now_stamp():
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def quote_snapshot_dir(product, stamp=None):
    stamp = stamp or local_now_stamp()
    date_part, time_part = stamp.split("_", 1)
    path = PROJECT_ROOT / "data" / "live" / product / "quotes" / date_part
    path.mkdir(parents=True, exist_ok=True)
    return path, time_part, stamp


def account_db_path(product):
    path = PROJECT_ROOT / "state" / "live" / product
    path.mkdir(parents=True, exist_ok=True)
    return path / "account.sqlite"


def feature_history_path(product):
    path = PROJECT_ROOT / "state" / "live" / product
    path.mkdir(parents=True, exist_ok=True)
    return path / "feature_history.parquet"


def account_report_summary_history_path(product, account_id="default"):
    path = PROJECT_ROOT / "state" / "live" / product
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{account_id}_account_summary_history.csv"


def account_report_position_history_path(product, account_id="default"):
    path = PROJECT_ROOT / "state" / "live" / product
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{account_id}_position_history.csv"


def clear_account_report_history(product, account_id="default"):
    for path in [
        account_report_summary_history_path(product, account_id),
        account_report_position_history_path(product, account_id),
    ]:
        path.unlink(missing_ok=True)


def output_dir(product):
    path = PROJECT_ROOT / "output" / "live" / product
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))
