from __future__ import annotations


SHORT_CLOSE_OPEN_PAIR = ("买入平仓", "卖出开仓")
LONG_CLOSE_OPEN_PAIR = ("卖出平仓", "买入开仓")


def net_exact_report_rows(rows):
    """Net no-op option close/open pairs from display rows."""
    result = []
    for row in rows:
        direction = row.get("方向")
        opposite = _opposite_direction(direction)
        if opposite is None:
            result.append(row)
            continue

        residual = dict(row)
        residual_qty = _qty(residual.get("数量"))
        if residual_qty <= 0:
            result.append(row)
            continue

        index = 0
        while index < len(result) and residual_qty > 1e-9:
            other = result[index]
            if (
                str(other.get("合约代码")) != str(residual.get("合约代码"))
                or other.get("方向") != opposite
            ):
                index += 1
                continue
            other_qty = _qty(other.get("数量"))
            matched_qty = min(other_qty, residual_qty)
            other_qty -= matched_qty
            residual_qty -= matched_qty
            if other_qty <= 1e-9:
                result.pop(index)
            else:
                other = dict(other)
                other["数量"] = _display_qty(other_qty)
                result[index] = other
                index += 1

        if residual_qty > 1e-9:
            residual["数量"] = _display_qty(residual_qty)
            result.append(residual)
    return result


def _opposite_direction(direction):
    if direction == SHORT_CLOSE_OPEN_PAIR[0]:
        return SHORT_CLOSE_OPEN_PAIR[1]
    if direction == SHORT_CLOSE_OPEN_PAIR[1]:
        return SHORT_CLOSE_OPEN_PAIR[0]
    if direction == LONG_CLOSE_OPEN_PAIR[0]:
        return LONG_CLOSE_OPEN_PAIR[1]
    if direction == LONG_CLOSE_OPEN_PAIR[1]:
        return LONG_CLOSE_OPEN_PAIR[0]
    return None


def _qty(value):
    try:
        return abs(float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _display_qty(value):
    value = float(value)
    return int(value) if value.is_integer() else value
