# Scripts directory

Only reusable project entry points belong under `scripts/`.

- `download/`: reproducible historical-data builders. Use the shared
  `build_data.py --product {300etf,500etf,kc50etf}` entry for SSE ETF options;
  `zz1000/build_data.py` remains separate because it uses a different exchange
  and source workflow.
- `live/`: supported live-account, quote-capture, reconciliation, reporting, and
  execution entry points. `IntradayCaptureApp.spec` is the single desktop build
  definition.
- `research/`: reusable research and backtest tools with a documented input and
  reproducible output. Hard-coded one-off scans should be moved to `策略留档/`
  after their results are recorded.
- `research/archive/`: completed research pipelines retained only for historical
  reproduction; they are not current strategy entry points.
- `ops/`: operational project-maintenance helpers.

Generated directories are not source code:

- `build/` and `dist/` are disposable PyInstaller working/output directories.
- `release/` may contain the latest deployable desktop bundle, but is ignored by
  Git because it is reproducible from `scripts/live/IntradayCaptureApp.spec`.
- Do not create variant release roots such as `release_all_contracts/`; rebuild
  the single `release/` bundle after source changes.

The deprecated `dynamic_position_straddle` plugin and its parameter-scan helper
remain only for reproducing old experiments. New dynamic-position work must use
`dynamic_atm_iv_straddle`.
