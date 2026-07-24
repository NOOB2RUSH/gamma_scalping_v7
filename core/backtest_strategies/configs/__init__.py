"""Sparse, per-product configuration overrides for backtest strategy plugins."""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
from typing import Any, Mapping

from core.configs.config_schema import AppConfig


_STRATEGY_CONFIG_MODULES = {
    "dynamic_atm_iv_straddle": (
        "core.backtest_strategies.configs.dynamic_atm_iv_straddle"
    ),
    "original_atm_iv_straddle": (
        "core.backtest_strategies.configs.original_atm_iv_straddle"
    ),
}

_OVERRIDABLE_SECTIONS = ("backtest", "strategy", "vol", "report")


def available_strategy_config_ids() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGY_CONFIG_MODULES))


def load_strategy_config_overrides(
    strategy_id: str,
    product: str,
) -> Mapping[str, Mapping[str, Any]]:
    """Return the sparse overrides declared for one strategy/product pair."""
    module_name = _STRATEGY_CONFIG_MODULES.get(str(strategy_id))
    if module_name is None:
        return {}
    module = import_module(module_name)
    product_overrides = getattr(module, "PRODUCT_OVERRIDES", {})
    return product_overrides.get(str(product).lower(), {})


def resolve_strategy_config(base_config: AppConfig, strategy_id: str) -> AppConfig:
    """Merge plugin overrides over a product config, field by field.

    Fields omitted by the plugin profile retain their values from ``base_config``.
    Explicit ``None`` values remain valid overrides because presence in the sparse
    mapping, rather than truthiness, controls precedence.
    """
    overrides = load_strategy_config_overrides(
        strategy_id,
        base_config.data.product,
    )
    if not overrides:
        return base_config

    invalid_sections = set(overrides) - set(_OVERRIDABLE_SECTIONS)
    if invalid_sections:
        names = ", ".join(sorted(invalid_sections))
        raise ValueError(
            f"Unsupported strategy config override sections for {strategy_id}: {names}"
        )

    app_updates = {}
    for section_name, section_updates in overrides.items():
        section = getattr(base_config, section_name)
        valid_fields = {item.name for item in fields(section)}
        invalid_fields = set(section_updates) - valid_fields
        if invalid_fields:
            names = ", ".join(sorted(invalid_fields))
            raise ValueError(
                f"Unsupported {section_name} config overrides for "
                f"{strategy_id}/{base_config.data.product}: {names}"
            )
        app_updates[section_name] = replace(section, **dict(section_updates))

    return replace(base_config, **app_updates)


__all__ = [
    "available_strategy_config_ids",
    "load_strategy_config_overrides",
    "resolve_strategy_config",
]
