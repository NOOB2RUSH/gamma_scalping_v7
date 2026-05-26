import os
from pathlib import Path


def _setup_numba_cache():
    """把 numba 缓存放到项目目录，避免写入 Python 安装目录失败。"""
    project_root = Path(__file__).resolve().parent.parent
    cache_dir = project_root / ".cache" / "numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))


_setup_numba_cache()

from . import (
    config,
    data_loader,
    vol_engine,
    analytics,
    backtester,
    cache,
    hedge,
    position,
    strategy,
)
