# Live Account Reconciliation Columns

This document fixes the accounting meaning of the live account reconciliation columns. These definitions are the source of truth for account reports and portfolio reports.

## Notation and Core Identity

For a single contract or security on report date `t`:

- `pos(t-1)`: signed position at the previous report date. Long is positive, short is negative.
- `pos(t)`: signed position at the current report date. Long is positive, short is negative.
- `c(t-1)`: previous report mark price.
- `c(t)`: current report mark price.
- `Vb`: total buy quantity during `(t-1, t]`, always non-negative.
- `Vs`: total sell quantity during `(t-1, t]`, always non-negative.
- `Pb`: volume-weighted average buy price during `(t-1, t]`.
- `Ps`: volume-weighted average sell price during `(t-1, t]`.
- `M`: contract multiplier. For ETF shares, `M = 1`; for options, use the contract multiplier.

If there are multiple buy or sell fills, interpret `Vb * Pb` and `Vs * Ps` as notional sums:

```text
Vb * Pb = sum(buy_qty_i * buy_price_i)
Vs * Ps = sum(sell_qty_j * sell_price_j)
```

The signed position identity is:

```text
pos(t) = pos(t-1) + Vb - Vs
```

The daily mark-to-market PnL identity is:

```text
单行总单日盈亏
= 持仓盈亏 + 当日盯市交易盈亏
= pos(t-1) * [c(t) - c(t-1)] * M
  + [Vb * (c(t) - Pb) - Vs * (c(t) - Ps)] * M
```

Equivalently:

```text
单行总单日盈亏
= [pos(t) * c(t) - pos(t-1) * c(t-1) - Vb * Pb + Vs * Ps] * M
```

Report exports may show position quantity as an absolute number plus a direction column. For these identities, that quantity must first be converted to signed `pos`.

## Position-Level Columns

`持仓盈亏`

Daily mark-to-market PnL from the previous report date's position:

```text
previous_signed_position * (current_mark - previous_mark) * multiplier
```

For a short option position, `previous_signed_position` is negative. For a long ETF hedge position, it is positive.

This column is not cumulative floating PnL versus entry cost. New positions opened during the report date have `持仓盈亏 = 0` until they become previous-close positions on the next report date.

`交易盈亏`

Current live reports use this as the mark-to-current PnL of trades executed during the report date:

```text
trade_signed_quantity * (current_mark - trade_price) * multiplier
```

For a buy trade, `trade_signed_quantity` is positive. For a sell trade, it is negative. This makes newly opened intraday positions contribute daily PnL immediately, measured from fill price to the report mark.

Despite the column name, this field is not used as cumulative cost-basis realized PnL in the daily PnL identity.

`当日盯市交易盈亏`

Same daily attribution concept as `交易盈亏`: mark-to-current PnL caused by trades during the report date. In the current implementation it is equal to `交易盈亏` for daily decomposition.

`当日盈亏分解合计`

Daily explainable PnL at the position-row level:

```text
持仓盈亏 + 当日盯市交易盈亏
```

Because the current implementation sets `交易盈亏 = 当日盯市交易盈亏` for this purpose, summary checks may also phrase the same identity as `持仓盈亏 + 交易盈亏`.

`今日变化`

Position quantity change versus the previous report date for the same contract/security:

```text
current_position - previous_position
```

## Summary-Level Columns

These columns are daily, not cumulative. The valuation interval is from the previous available report date to the current report date. The first history row is a base row and should not be interpreted as a fully explainable daily PnL interval.

`期权单日盈亏`

Daily option PnL for the product. When all required position-level attribution is available, it should reconcile to the sum of `当日盈亏分解合计` across the product's option rows.

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

`单日盈亏/AUM`

Gross daily PnL ratio:

```text
总单日盈亏(手续费前) / AUM
```

This ratio intentionally uses the pre-fee PnL, matching the current report output.

`当日盈亏对账差额`

Difference between the summary-level daily PnL and the position-level explainable PnL:

```text
总单日盈亏(手续费前) - sum(当日盈亏分解合计)
```

A non-zero value means the visible position rows do not fully explain the summary PnL for that date. It must not be hidden by redefining `持仓盈亏` as cumulative floating PnL.

## Product and Portfolio Daily PnL

For each product:

```text
期权单日盈亏 = sum(option row 当日盈亏分解合计)
ETF单日盈亏  = sum(ETF row 当日盈亏分解合计)
总单日盈亏(手续费前) = 期权单日盈亏 + ETF单日盈亏
净单日盈亏 = 总单日盈亏(手续费前) - 当日手续费
```

For the unified portfolio report:

```text
组合本地净单日盈亏 = sum(各产品 净单日盈亏)
```

This portfolio net value is the local PnL number used for broker fund reconciliation. It is not the same as `sum(总单日盈亏(手续费前))` unless total daily fees are zero.

## Broker Fund Reconciliation

Fund reconciliation is produced by the account reconciliation workflow before
the per-product checks. It reconciles broker account equity snapshots against
local portfolio PnL. It reads the two daily `live_hold/实时资金*.csv` exports:

- option account row: uses `总资产` from the account identified by `投资者账号` starting with `期权`, or by option-specific fields such as `期权市值` / `保证金`.
- securities account row: uses `总资产` from the account identified by `投资者账号` starting with `证券`, or by `证券市值`.
- option margin row: uses `保证金` from the option account and removes margin
  occupancy changes from the broker asset change before comparing to local PnL.

The broker-side identity is:

```text
券商合并总资产 = 期权账户总资产 + 证券账户总资产
券商总资产变化 = 今日券商合并总资产 - 上一资金日券商合并总资产
期权保证金变化 = 今日期权保证金 - 上一资金日期权保证金
剔除保证金后券商资产变化 = 券商总资产变化 + 期权保证金变化
资金对账差额 = 剔除保证金后券商资产变化 - 组合本地净单日盈亏
```

The expected equality is:

```text
剔除保证金后券商资产变化 ≈ 组合本地净单日盈亏
```

The current absolute tolerance is 1.0 currency unit. If there are external cash transfers, interest, broker adjustments, missing exports, mismatched report dates, or fee/settlement differences, the equality can fail or remain uncheckable.

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
