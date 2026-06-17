"""配置入口。

默认仍然指向 50ETF，保证旧命令 `python run.py` 行为不变。
需要切换品种时，使用 `load_config(product)` 读取对应的独立配置。
"""

from importlib import import_module

from .configs.config_50etf import CONFIG
from .configs.config_schema import (
    AppConfig,
    BacktestConfig,
    DataConfig,
    ReportConfig,
    StrategyConfig,
    VolConfig,
)


PRODUCT_CONFIG_MODULES = {
    "300etf": "core.configs.config_300etf",
    "50etf": "core.configs.config_50etf",
    "500etf": "core.configs.config_500etf",
    "kc50etf": "core.configs.config_kc50etf",
    "zz1000": "core.configs.config_zz1000",
}


def available_products():
    return tuple(PRODUCT_CONFIG_MODULES)


def available_live_products():
    return tuple(
        product
        for product in available_products()
        if product in {"50etf", "300etf", "500etf", "kc50etf"}
    )


def load_config(product):
    """按品种名称读取独立配置。"""
    product_key = str(product).lower()
    if product_key not in PRODUCT_CONFIG_MODULES:
        raise ValueError(
            f"未知交易品种: {product}，可选品种: {', '.join(available_products())}"
        )

    module = import_module(PRODUCT_CONFIG_MODULES[product_key])
    return module.CONFIG
