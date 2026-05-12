def calc_unrealized_pnl(etf_qty, entry_price, current_price):
    """计算当前 ETF 对冲仓位的未实现盈亏。"""
    if etf_qty > 0:
        return etf_qty * (current_price - entry_price)
    if etf_qty < 0:
        return abs(etf_qty) * (entry_price - current_price)
    return 0.0


def close_etf_hedge(cash, etf_qty, entry_price, margin, price):
    """平掉 ETF 对冲仓位，释放保证金并结算盈亏。"""
    pnl = calc_unrealized_pnl(etf_qty, entry_price, price)
    return cash + margin + pnl, pnl


def rebalance_etf_hedge(cash, current_qty, entry_price, margin, target_qty, price):
    """把 ETF 对冲仓位从 current_qty 调整到 target_qty。"""
    realized_pnl = 0.0

    if current_qty == target_qty:
        return cash, current_qty, entry_price, margin, realized_pnl

    # 没有旧仓位：直接开目标 ETF 仓位，并冻结对应保证金。
    if current_qty == 0:
        new_margin = abs(target_qty) * price
        return cash - new_margin, target_qty, price, new_margin, realized_pnl

    # 目标为 0：全平 ETF 仓位，释放保证金并结算盈亏。
    if target_qty == 0:
        cash, realized_pnl = close_etf_hedge(
            cash,
            current_qty,
            entry_price,
            margin,
            price,
        )
        return cash, 0.0, 0.0, 0.0, realized_pnl

    # 方向反转：先全平旧方向，再开新方向。
    if current_qty * target_qty < 0:
        cash, realized_pnl = close_etf_hedge(
            cash,
            current_qty,
            entry_price,
            margin,
            price,
        )
        new_margin = abs(target_qty) * price
        return cash - new_margin, target_qty, price, new_margin, realized_pnl

    # 同方向加仓：只冻结新增仓位保证金，并用数量加权更新入场价。
    if abs(target_qty) > abs(current_qty):
        add_qty = target_qty - current_qty
        add_margin = abs(add_qty) * price
        new_entry_price = (entry_price * abs(current_qty) + price * abs(add_qty)) / abs(
            target_qty
        )
        return (
            cash - add_margin,
            target_qty,
            new_entry_price,
            margin + add_margin,
            realized_pnl,
        )

    # 同方向减仓：释放减掉部分的保证金，并结算这部分盈亏。
    close_qty = abs(current_qty) - abs(target_qty)
    close_ratio = close_qty / abs(current_qty)
    released_margin = margin * close_ratio

    if current_qty > 0:
        realized_pnl = close_qty * (price - entry_price)
    else:
        realized_pnl = close_qty * (entry_price - price)

    new_margin = margin - released_margin
    cash += released_margin + realized_pnl
    return cash, target_qty, entry_price, new_margin, realized_pnl
