# Simulated Live Trading Design

## Scope

The first live module targets one product, one account, and manual fill
confirmation. It does not place broker orders. It produces actionable advice,
keeps a local shadow account, and reconciles that account against a broker
snapshot supplied by the user.

Supported live products are the four SSE ETF option products: `50etf`,
`300etf`, `500etf`, and `kc50etf`. `zz1000` remains available to the backtest
layer but is intentionally excluded from live commands until a dedicated
index-option quote and hedge workflow is implemented.

To add another ETF option product to the live system, follow
[`add_live_product.md`](add_live_product.md). That checklist is the source of
truth for required config, quote, broker-import, report, and test mappings.

The live layer is intentionally separate from the backtest layer. It reuses
product config, option selection, IV/Greeks, strategy predicates, and position
valuation helpers, but it does not reuse or mutate the backtest state machine.

## Workflow

1. Fetch or load the latest product quotes.
2. Save an immutable quote snapshot under `data/live/<product>/quotes/`.
3. Build latest features and signals from the existing daily data set.
4. Read the local shadow account from SQLite.
5. Generate advice:
   - open long or short straddle
   - close long or short straddle
   - roll existing straddle
   - rebalance delta hedge
   - no action / data warning
6. Write a signal report under `output/live/<product>/`.
7. User executes any real trades manually.
8. User confirms fills back into the shadow account.
9. End of day, user imports a broker snapshot and runs reconciliation.
10. If a confirmed fill is wrong, void it, optionally insert a replacement fill,
    and rebuild the shadow account from the active fill log.

## Storage

Quote snapshots are append-only parquet files:

```text
data/live/<product>/quotes/YYYYMMDD/HHMMSS_etf.parquet
data/live/<product>/quotes/YYYYMMDD/HHMMSS_option_chain.parquet
data/live/<product>/quotes/YYYYMMDD/HHMMSS_metadata.json
```

Account state is SQLite:

```text
state/live/<product>/account.sqlite
state/live/<product>/feature_history.parquet
```

Reports are plain text/Markdown:

```text
output/live/<product>/YYYYMMDD_HHMMSS_signal.md
output/live/<product>/YYYYMMDD_HHMMSS_reconcile.md
```

## Coupling Rules

- `core/live/*` may import existing `core.config`, `core.data_loader`,
  `core.cache`, `core.vol_engine`, `core.strategy`, and `core.position`.
- Existing backtest modules should not import `core.live`.
- Signal generation must not mutate cash, positions, hedge, fills, or strategy
  state. It is a read-only execution-plan generator.
- Broker import of option and ETF exports is the source of truth for position
  and trade changes in the shadow account.
- Reconciliation validates how well daily Greeks PnL explains fee-adjusted NAV
  changes (`NAV change + daily fee`) and does not auto-adjust state.

## Quote Sources

`source=local` snapshots the latest existing parquet files and is useful for
repeatable tests.

`source=akshare` currently supports SSE ETF options: `50etf`, `300etf`,
`500etf`, and `kc50etf`. It pulls the latest ETF daily bar and the current SSE
option board through AKShare and writes immutable live snapshots without
modifying the canonical research/backtest parquet files.

New straddle entries use `backtest.long_qty` / `backtest.short_qty` from the
active product configuration; all four current live products default to 10
call contracts and 10 put contracts. Existing
positions retain their actual imported quantities for close, roll, reduction,
and hedge calculations.

Each live option chain is tagged with its executable ETF hedge:
`510050.XSHG`, `510300.XSHG`, `510500.XSHG`, or `588000.XSHG`. When a broker
holding import references a contract absent from local historical chains, the
importer queries the exact trade date through AKShare's SSE risk indicator and
caches the resulting daily contract metadata, close, and volume under
`state/live/<product>/`. This cache is intentionally separate from the complete
canonical research option chain.

Live signal generation is read-only. It enriches the latest option chain with
IV/Greeks and builds the latest signal row without mutating the shadow account
or the live feature-history store. Account state changes are driven by broker
holding/trade imports and explicit account rebuild/amend tools.

## Roll State

Rolls have no cooldown. A held position can roll whenever its DTE reaches the
configured threshold or spot moves more than one strike interval away from the
held strike.

ATM-strike mismatch is not stored in strategy state. For each signal run the
engine reconstructs consecutive mismatch days from broker option holding
snapshots under `live_hold/实时持仓*.csv` and the historical `atm_strike` signal
rows. If any required holding snapshot or ATM row is missing during the current
holding period, signal generation raises an error instead of falling back to a
stored counter.

## Fill Audit And Corrections

Confirmed fills are append-only records in the SQLite `fills` table. Corrections
do not overwrite old records. `amend_fill.py` marks the wrong fill as voided,
optionally inserts a corrected replacement fill, then rebuilds the account from
non-voided fills.

```bash
python scripts/live/show_account_log.py --product kc50etf
python scripts/live/amend_fill.py --product kc50etf --fill-id 12 --replacement-fill fills/fill_12_fixed.json --reason wrong_price
python scripts/live/rebuild_account.py --product kc50etf
```

`init_account.py --reset` starts a new shadow account for that account id and
clears its fill and reconciliation history.

## Broker Holding Snapshot Import

`import_holdings.py` reads the newest CSV under `live_hold/` by default and can
auto-confirm supported straddle openings from a broker holding snapshot.

```bash
python scripts/live/import_holdings.py --product kc50etf --dry-run
python scripts/live/import_holdings.py --product kc50etf
python scripts/live/import_holdings.py --product kc50etf --include-existing
```

By default it imports only rows with today's open quantity (`今开仓 > 0`) so an
old overnight holding is not repeatedly confirmed. `--include-existing` imports
the total holding quantity and is intended for one-time shadow-account seeding.

`实时持仓*.csv` is the only supported broker position snapshot, and
`成交明细*.csv` is the only supported broker execution export. The importer
uses the holding snapshot for current option quantities and the matching trade
detail for opens, closes, and leg quantity changes. Legacy `成交汇总` is
intentionally unsupported.

ETF hedge imports have their own two standard files:

- `证券持仓查询(信息导出)*.csv`: authoritative current ETF holding snapshot.
- `证券委托查询_实时成交(信息导出)*.csv`: report-date ETF execution details.

```bash
python scripts/live/import_etf.py --product kc50etf --dry-run
python scripts/live/import_etf.py --product kc50etf
```

No other ETF holding or trade export filename is accepted. When a matching
report-date ETF execution file is unavailable, the holding snapshot may update
the target quantity and mark, but no cash movement is invented.

## Live Account Report

`live_account_report.py` produces the operator-facing account status report. It
has three spreadsheet sections: cumulative account summary, cumulative position
snapshots, and report-date broker trade details.

```bash
python scripts/live/live_account_report.py --product kc50etf
python scripts/live/live_account_report.py --product kc50etf --mode diagnose
python scripts/live/live_account_report.py --product kc50etf --source none
```

By default it fetches the latest AKShare quotes before reporting and writes an
timestamped cumulative Excel workbook with separate sheets:

- `账户总体情况`: one row per report date since the account report history began.
- `持仓记录`: position snapshots accumulated by report date. Contract names are
  read from the broker holding export when available.
- `交易记录`: cumulative trade rows from `live_hold/成交明细*.csv`.

The default report mode shows the operator-facing account, daily PnL, Greeks,
position, and trade fields. `--mode diagnose` adds internal PnL decomposition,
broker reconciliation, and Greeks-explanation fields. Current option positions
are revalued against current mid prices, with IV and Greeks calculated through
the same project valuation functions used by live signals.

`live_portfolio_report.py` keeps the four product shadow accounts independent
but combines their output into one cumulative workbook:

```bash
python scripts/live/live_portfolio_report.py
python scripts/live/live_portfolio_report.py --source none
```

The workbook is written under `output/live/portfolio` with `组合汇总`,
`子账户汇总`, `持仓记录`, and `交易记录` sheets. Positions and trades include a
`品种` column. Cash, NAV, margin, fees, and PnL are summed at portfolio level;
Greeks remain separated by product because their raw units are not directly
comparable across underlyings.

`live_console.py` is the portfolio-level operator interface. It always operates
on all four live products using their independent `default` subaccounts, so the
operator no longer selects a product or account before generating signals,
viewing positions, importing holdings, reporting, rebuilding, or reconciling.

## Planned Hedge Advice

Live signals only mutate cash and positions after manual fill confirmation.
When open, close, roll, or option-leg rebalance advice is generated, live mode
treats all actions as one ordered execution plan. Signal generation follows
this fixed flow:

1. Capture the current option positions, ETF hedge, and account Greeks.
2. Generate option lifecycle actions such as open, close, roll, and leg
   rebalance.
3. Keep the existing ETF hedge in place while option actions execute. A normal
   roll does not require the ETF position to be reset to zero.
4. Project option positions from the generic `position_target` carried by each
   action. A new option action must not require registration in another
   action-name whitelist to participate in state projection.
5. Project every ETF `target_hedge_qty`, then combine all ETF adjustments for
   the same underlying into one trade from the current quantity to the final
   target. The net ETF trade executes after the option plan.
6. Evaluate the post-plan account delta. Option legs are preferred for a
   material structural imbalance, and ETF is then used for the remaining
   fine-tuning that the product constraints permit.
7. Render the same ordered plan to signal JSON, Markdown, terminal output, and
   InfiniTrader orders. Account state changes only after confirmed fills are
   imported.

The following fields define the two states unambiguously:

- `current_account_delta` is the delta before any generated action.
- `planned_hedge_qty` is the ETF position after all generated actions.
- `planned_account_delta` is the final projected option delta plus
  `planned_hedge_qty`.
- `account_delta_after_hedge` is retained as a compatibility alias for
  `planned_account_delta`; it must not contain the pre-plan delta.

The default absolute Delta trigger is 5,000. It is an entry condition, not the
post-hedge target band. If neither the current account nor the projected option
plan breaches `[-5,000, +5,000]`, no option-leg rebalance or ETF hedge is
generated, even when a small ETF trade could move Delta closer to zero.

Once either state breaches the trigger, Delta control remains active for the
whole execution plan: first complete the required roll or option-leg balance,
then set the final ETF target against the projected option Delta with a target
account Delta of zero. A roll that reduces Delta back below 5,000 must not
cancel this final ETF step. The final achievable residual may still be non-zero
because of ETF board-lot rounding, option-contract granularity, the short-ETF
restriction, liquidity, cash reserve, or capacity constraints; those are
execution constraints rather than a tolerance target.

`core.live.etf_netting` is the single registry and extraction layer for ETF
actions. Reports and execution adapters must consume it instead of maintaining
their own ETF action-name sets. Historical `CLOSE_HEDGE_BEFORE_ROLL` advice is
also treated as nettable, so an old signal cannot generate an unnecessary ETF
sell-and-buy round trip. Any new executable action must add regression coverage
for final-state projection, displayed order and quantities, generated broker
orders, and fill import.

## First Version Limitations

- Online quote support currently covers SSE ETF options only.
- Intraday quotes are treated as reference data. The optimized strategies are
  daily/EOD strategies, so live reports mark advice as EOD-style unless a
  product-specific intraday source is later added.

## Historical Mirror And Immutable Snapshot Replay

The registered backtest plugin `live_straddle` is the historical mirror of the
current live policy. It delegates signal predicates to the same `core.strategy`
implementation and uses the live target-state contracts for roll, adjusted
contract liquidation, option-leg balancing and final ETF delta control. The
historical executor calls the same live delta-plan builder and the same ETF
netting layer, then fills only the final ETF target in simulated state.

Snapshot replay calls `generate_signal_from_context()` with a detached in-memory
account and explicit immutable market context. It does not load or write the
real shadow account and does not update live feature history. Actual comparison
opens the broker-derived SQLite account in read-only mode.

Commands, output columns, attribution limits and parity gates are documented in
[`live_straddle_replay.md`](live_straddle_replay.md).
