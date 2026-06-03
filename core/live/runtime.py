from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import core


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sync_config(config):
    """Point existing strategy/valuation modules at a selected product config."""
    core.config.CONFIG = config
    core.vol_engine.CONFIG = config
    core.strategy.CONFIG = config
    core.backtester.CONFIG = config
    core.position.CONFIG = config
    core.hedge.CONFIG = config
    core.analytics.CONFIG = config
    return config


def load_product_config(product, start=None, end=None):
    config = core.config.load_config(product)
    updates = {
        key: value
        for key, value in {"start": start, "end": end}.items()
        if value is not None
    }
    if updates:
        config = replace(config, backtest=replace(config.backtest, **updates))
    return sync_config(config)


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
