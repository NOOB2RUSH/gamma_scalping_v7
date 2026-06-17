# Live Account Reconciliation Columns

This document fixes the accounting meaning of the live account reconciliation columns. These definitions are the source of truth for account reports and portfolio reports.

## Position-Level Columns

`持仓盈亏`

Daily mark-to-market PnL from positions held at the previous close:

```text
abs(previous_position) * (current_close - previous_close) * sign(previous_position) * multiplier
```

For a short option position, `sign(previous_position) = -1`. For a long ETF position, `sign(previous_position) = 1`.

This column is not cumulative floating PnL versus entry cost. New positions opened during the report date have `持仓盈亏 = 0` until they become previous-close positions on the next report date.

`交易盈亏`

Realized PnL for trades that close an existing position, measured against the position cost basis:

```text
closed_quantity * (trade_price - cost_basis) * sign(closed_position) * multiplier
```

This is a cost-basis realized PnL field, not same-day mark-to-market PnL for newly opened positions.

`当日盯市交易盈亏`

Mark-to-market PnL caused by trades during the report date when the trade can be compared with the previous close or an intraday opening cost. This is separate from `交易盈亏` because it is a daily PnL attribution field, not a cost-basis realized PnL field.

`当日盈亏分解合计`

Daily explainable PnL at the position-row level:

```text
持仓盈亏 + 当日盯市交易盈亏
```

`今日变化`

Position quantity change versus the previous report date for the same contract/security:

```text
current_position - previous_position
```

## Summary-Level Columns

`期权单日盈亏`

Daily option PnL for the product. When all required position-level attribution is available, it should reconcile to the sum of `当日盈亏分解合计` across the product's call and put rows.

`ETF单日盈亏`

Daily ETF hedge PnL for the product. When all required position-level attribution is available, it should reconcile to the sum of `当日盈亏分解合计` across the product's ETF rows.

`总单日盈亏(手续费前)`

```text
期权单日盈亏 + ETF单日盈亏
```

`净单日盈亏`

```text
总单日盈亏(手续费前) - 当日手续费
```

`当日盈亏对账差额`

Difference between the summary-level daily PnL and the position-level explainable PnL:

```text
总单日盈亏(手续费前) - sum(当日盈亏分解合计)
```

A non-zero value means the visible position rows do not fully explain the summary PnL for that date. It must not be hidden by redefining `持仓盈亏` as cumulative floating PnL.

## Summary Greeks PnL

When intraday Greeks path data is unavailable and the report falls back to EOD summary rows, `GreeksPnL口径 = previous_close`.

The EOD fallback uses only the previous report date's closing Greeks:

```text
期权单日DeltaPnL = previous_delta * (current_underlying - previous_underlying)
期权单日GammaPnL = 0.5 * previous_gamma * (current_underlying - previous_underlying)^2
期权单日VegaPnL  = previous_vega  * (current_iv - previous_iv) * 100
期权单日ThetaPnL = previous_theta
```

It must not average previous-date and current-date Greeks. Current-date Greeks are end-of-period exposures, not the exposure held over the interval being explained.
