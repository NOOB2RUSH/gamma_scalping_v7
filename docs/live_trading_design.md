# Simulated Live Trading Design

## Scope

The first live module targets one product, one account, and manual fill
confirmation. It does not place broker orders. It produces actionable advice,
keeps a local shadow account, and reconciles that account against a broker
snapshot supplied by the user.

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
option board through AKShare, writes immutable live snapshots, and also updates
the canonical product parquet files so the existing signal pipeline can include
the newest date.

Live signal generation is read-only. It enriches the latest option chain with
IV/Greeks and builds the latest signal row without mutating the shadow account
or the live feature-history store. Account state changes are driven by broker
holding/trade imports and explicit account rebuild/amend tools.

## Roll State

Roll cooldown still lives in the shadow account because it is created by
confirmed close/roll fills. Signal generation reads that cooldown but does not
advance or persist it.

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

The broker holding export is not a per-fill execution report. It has no unique
execution id, execution time, exact commission, or exact cash movement.
Therefore generated fills use open average price, configured option fee, and
occupied margin to estimate `cash_delta`. Account reconciliation now checks
Greeks PnL explainability against fee-adjusted NAV changes instead of comparing
to a separate broker account snapshot. Close/roll confirmations cannot be
inferred safely from a holding snapshot alone when the position is absent or
changed; those still need explicit fill JSON or a real transaction export.

## ETF Hedge Snapshot Import

`import_hedge.py` reads the newest `证券持仓查询*.csv` and
`证券委托查询_实时成交*.csv` under `live_hold/` by default and can auto-confirm
the ETF delta hedge state into the shadow account.

```bash
python scripts/live/import_hedge.py --product kc50etf --dry-run
python scripts/live/import_hedge.py --product kc50etf
```

The holding export supplies the final ETF hedge quantity, cost price, market
value, and ETF code. The trade export supplies the report-date ETF executions
used to estimate `cash_delta`. The generated fill is a `delta_hedge` fill: it
sets the local hedge to the broker-reported target quantity while recording the
matched ETF execution rows for audit. If no matching trade rows are found,
`cash_delta` falls back to an estimate from holding cost and the import result
emits a warning.

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
- `交易记录`: cumulative trade rows from `live_hold/成交明细*.csv` and supported summary exports.

The default report mode shows the operator-facing account, daily PnL, Greeks,
position, and trade fields. `--mode diagnose` adds internal PnL decomposition,
broker reconciliation, and Greeks-explanation fields. Current option positions
are revalued against current mid prices, with IV and Greeks calculated through
the same project valuation functions used by live signals.

## Planned Hedge Advice

Live signals only mutate cash and positions after manual fill confirmation.
When open, close, or roll advice is generated, live mode treats those option
actions as an execution plan first. It then simulates the plan in order and
emits one final `FINAL_DELTA_HEDGE` only if the post-plan account delta would
remain outside tolerance. This final hedge is an execution hint for the
operator; it is not written into the shadow account until the hedge fill is
confirmed.

## First Version Limitations

- Online quote support currently covers SSE ETF options only.
- Intraday quotes are treated as reference data. The optimized strategies are
  daily/EOD strategies, so live reports mark advice as EOD-style unless a
  product-specific intraday source is later added.
