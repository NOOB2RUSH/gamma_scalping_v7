import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from . import config, vol_engine


CACHE_VERSION = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "backtest"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _hash_key(payload):
    payload = {"cache_version": CACHE_VERSION, **payload}
    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _parse_date_from_file(file_path, suffix):
    base = file_path.stem.rsplit(suffix, 1)
    date_str = base[0].rsplit("_", 1)[1]
    return pd.Timestamp(date_str)


def _file_manifest(data_dir, pattern, suffix, start=None, end=None):
    """生成数据文件指纹；文件名、大小或修改时间变化都会让缓存失效。"""
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None
    rows = []

    for file_path in sorted(data_dir.glob(pattern)):
        date = _parse_date_from_file(file_path, suffix)
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue

        stat = file_path.stat()
        rows.append(
            {
                "name": file_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    return rows


def _resolve_project_path(path):
    data_path = Path(path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    return data_path


def _path_for_signature(path):
    """缓存签名使用稳定路径；外部绝对路径则保留原样。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_data_signature(start, end):
    """生成本次回测数据签名，覆盖 ETF 区间、期权区间和完整交易日历。"""
    data_cfg = config.CONFIG.data
    etf_dir = _resolve_project_path(data_cfg.etf_dir)
    opt_dir = _resolve_project_path(data_cfg.opt_dir)
    return {
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "product": data_cfg.product,
        "etf_dir": _path_for_signature(etf_dir),
        "opt_dir": _path_for_signature(opt_dir),
        "etf_range": _file_manifest(
            etf_dir,
            "*price.parquet",
            "_price",
            start,
            end,
        ),
        "opt_range": _file_manifest(
            opt_dir,
            "*chain.parquet",
            "_chain",
            start,
            end,
        ),
        # DTE 使用完整 ETF 交易日历，因此完整日历变化也需要让缓存失效。
        "etf_calendar": _file_manifest(
            etf_dir,
            "*price.parquet",
            "_price",
        ),
    }


def _load_or_build(cache_name, key_payload, builder):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = _hash_key(key_payload)
    cache_path = CACHE_DIR / f"{cache_name}_{cache_key}.pkl"

    if cache_path.exists():
        print(f"[cache hit] {cache_name}: {cache_path}")
        try:
            with cache_path.open("rb") as file:
                return pickle.load(file)
        except Exception as exc:
            print(
                f"[cache invalid] {cache_name}: {type(exc).__name__}: {exc}. "
                "rebuild cache."
            )
            cache_path.unlink(missing_ok=True)

    print(f"[cache miss] {cache_name}: {cache_path}")
    result = builder()
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    return result


def _enriched_config_signature():
    cfg = config.CONFIG
    return {
        "annual_days": cfg.vol.annual_days,
        "risk_free_rate": cfg.vol.risk_free_rate,
        "dividend_yield": cfg.vol.dividend_yield,
        "contract_multiplier": cfg.vol.contract_multiplier,
    }


def _feature_config_signature():
    cfg = config.CONFIG
    return {
        "annual_days": cfg.vol.annual_days,
        "hv_windows": cfg.vol.hv_windows,
        "atm_iv_percentile_window": cfg.vol.atm_iv_percentile_window,
        "atm_target_dte": cfg.vol.atm_target_dte,
        "atm_target_dte_min": cfg.vol.atm_target_dte_min,
        "atm_target_dte_max": cfg.vol.atm_target_dte_max,
        "atm_moneyness_tol_mode": "absolute_price_diff",
        "atm_moneyness_tol": cfg.vol.atm_moneyness_tol,
        "contract_multiplier": cfg.vol.contract_multiplier,
    }


def get_enriched_option_chains(
    etf_by_date,
    opt_by_date,
    trading_calendar,
    start,
    end,
):
    """读取或计算每日 IV/Greeks 全链缓存。"""
    key_payload = {
        "kind": "enriched_option_chains",
        "data": build_data_signature(start, end),
        "config": _enriched_config_signature(),
    }
    return _load_or_build(
        "enriched_option_chains",
        key_payload,
        lambda: vol_engine.build_enriched_option_chains(
            etf_by_date,
            opt_by_date,
            trading_calendar=trading_calendar,
        ),
    )


def get_vol_features(
    etf_by_date,
    opt_by_date,
    trading_calendar,
    enriched_opt_by_date,
    start,
    end,
):
    """读取或计算回测所需的波动率 features 缓存。"""
    key_payload = {
        "kind": "vol_features",
        "schema": "atm_pool_volume_valid_iv_percentile_v1",
        "data": build_data_signature(start, end),
        "enriched_config": _enriched_config_signature(),
        "feature_config": _feature_config_signature(),
    }

    def build_features():
        return vol_engine.build_vol_features(
            etf_by_date,
            opt_by_date,
            trading_calendar=trading_calendar,
            enriched_opt_by_date=enriched_opt_by_date,
        )

    return _load_or_build("vol_features", key_payload, build_features)
