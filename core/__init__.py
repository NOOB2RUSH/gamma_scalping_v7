import os
import tempfile
from pathlib import Path


def _setup_numba_cache():
    """把 numba 缓存放到项目目录，避免写入 Python 安装目录失败。"""
    configured_cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if configured_cache_dir:
        cache_dir = Path(configured_cache_dir)
    elif os.name == "nt":
        # py_vollib_vectorized generates very long cache filenames. Keeping the
        # cache under the repo can hit Windows' 260-character path limit.
        cache_dir = Path(tempfile.gettempdir()) / "nb"
    else:
        project_root = Path(__file__).resolve().parent.parent
        cache_dir = project_root / ".cache" / "numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


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
    vol_surface,
)
