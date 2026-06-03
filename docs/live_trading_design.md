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
- Signal generation must not mutate cash, positions, hedge, or fills. It may
  update the persisted strategy-state memory used by later live signals.
- Only manual fill confirmation updates the shadow account.
- Reconciliation reports differences first; it does not auto-adjust state.

## Quote Sources

`source=local` snapshots the latest existing parquet files and is useful for
repeatable tests.

`source=akshare` currently supports SSE ETF options: `50etf`, `300etf`,
`500etf`, and `kc50etf`. It pulls the latest ETF daily bar and the current SSE
option board through AKShare, writes immutable live snapshots, and also updates
the canonical product parquet files so the existing signal pipeline can include
the newest date.

Live signal generation is incremental. It only enriches the latest option chain
with IV/Greeks, then merges that row into
`state/live/<product>/feature_history.parquet` for rolling IV percentiles. If
the feature history file is missing, the first run seeds it from the historical
cache; later hourly ticks do not recompute the full historical cache.

## Strategy-State Memory

Live signals persist the cross-day state that the backtest state machine keeps
in memory:

- `last_signal_date`: the latest trading date already processed by live mode.
- `strike_mismatch_days`: consecutive trading days where an existing position's
  strike differs from current ATM, by side.
- `roll_cooldown_left`: remaining trading days before a side may open or roll.
- `cooldown_total_days` and `cooldown_started_date`: deterministic cooldown
  anchors so repeated hourly ticks on the same date do not double-count.

Manual fill confirmation resets the relevant side on open/roll. Confirmed
closes start the configured side cooldown; a long close with `exit_reason` set
to `iv_high` also starts the configured short-side cooldown. Signal generation
advances this memory once per trading date, blocks entries during cooldown, and
only recommends rolls when the backtest roll conditions are met: low DTE or
enough consecutive ATM-strike mismatch days, plus an active entry signal.

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
occupied margin to estimate `cash_delta`. Broker cash should still be checked
through reconciliation. Close/roll confirmations cannot be inferred safely from
a holding snapshot alone when the position is absent or changed; those still
need explicit fill JSON or a real transaction export.

## Live Account Report

`live_account_report.py` produces the operator-facing account status report. It
has three spreadsheet sections: cumulative account summary, cumulative position
snapshots, and report-date broker trade details.

```bash
python scripts/live/live_account_report.py --product kc50etf
python scripts/live/live_account_report.py --product kc50etf --source none
python scripts/live/live_account_report.py --product kc50etf --format csv
```

By default it fetches the latest AKShare quotes before reporting and writes an
Excel workbook with separate sheets:

- `账户总体情况`: one row per report date since the account report history began.
- `持仓记录`: position snapshots accumulated by report date. Contract names are
  read from the broker holding export when available.
- `当日交易记录`: report-date rows from `live_hold/成交明细*.csv`.

`--format csv` writes one CSV per section, and `--format both` writes both Excel
and CSV outputs. Current option positions are revalued against current mid
prices, with IV and Greeks calculated through the same project valuation
functions used by live signals.

## Projected Hedge Advice

Live signals only mutate cash and positions after manual fill confirmation.
When an open, close, or roll advice is generated, live mode also emits
`PROJECTED_DELTA_HEDGE` if executing that advice would leave projected account
delta outside tolerance. This projected hedge is an execution hint for the
operator; it is not written into the shadow account until the hedge fill is
confirmed.

## First Version Limitations

- Online quote support currently covers SSE ETF options only.
- Intraday quotes are treated as reference data. The optimized strategies are
  daily/EOD strategies, so live reports mark advice as EOD-style unless a
  product-specific intraday source is later added.
