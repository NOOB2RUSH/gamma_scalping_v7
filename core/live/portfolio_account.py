from __future__ import annotations

import core

from . import account as account_store


PORTFOLIO_INITIAL_CASH = 10_000_000.0


def shared_cash(account_id="default", products=None):
    products = tuple(products or core.config.available_live_products())
    cash = PORTFOLIO_INITIAL_CASH
    for product in products:
        state = account_store.load_account(product, account_id=account_id)
        cash += float(state.cash) - product_initial_cash(product)
    return cash


def product_initial_cash(product):
    return float(core.config.load_config(product).backtest.initial_cash)


def shared_nav_from_subaccounts(payloads):
    nav = PORTFOLIO_INITIAL_CASH
    for product, payload in payloads.items():
        summary = payload.get("summary") or {}
        subaccount_nav = _number(summary.get("估算权益"))
        initial_cash = _number(summary.get("初始资金"))
        if subaccount_nav is None:
            continue
        nav += subaccount_nav - (
            initial_cash
            if initial_cash is not None
            else product_initial_cash(product)
        )
    return nav


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
