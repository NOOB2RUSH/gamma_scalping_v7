# Adding a Live Product

This document is the checklist for adding a new product to the live system. The live system currently assumes SSE ETF options whose hedge instrument is the underlying ETF. Do not add index options or non-SSE options through this flow until their quote, broker import, and hedge workflows are explicitly supported.

## Current Live Product Map

| product key | display name | report code | ETF symbol | ETF order book id | AKShare option prefix | option export marker | config module |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `50etf` | `上证50ETF华夏` | `sh510050` | `510050` | `510050.XSHG` | `510050.XSHG` | `50ETF` | `core.configs.config_50etf` |
| `300etf` | `沪深300ETF华泰柏瑞` | `sh510300` | `510300` | `510300.XSHG` | `300ETF_OPTION` | `300ETF` | `core.configs.config_300etf` |
| `500etf` | `中证500ETF南方` | `sh510500` | `510500` | `510500.XSHG` | `500ETF_OPTION` | `500ETF` | `core.configs.config_500etf` |
| `kc50etf` | `科创50ETF华夏` | `sh588000` | `588000` | `588000.XSHG` | `KC50ETF_OPTION` | `科创50` | `core.configs.config_kc50etf` |

When adding a product, add one row to every equivalent mapping in code. The `product key` is the stable internal id. The `display name` and `report code` are report-only fields.

## Required Code Changes

1. Add the product config.

   Create `core/configs/config_<product>.py`, normally by copying the closest existing ETF option config.

   Required fields:

   - `data.product`
   - `data.etf_dir`
   - `data.opt_dir`
   - `data.hedge_etf_dir`
   - `backtest.initial_cash`
   - `backtest.long_qty = 10`
   - `backtest.short_qty = 10`
   - `backtest.etf_fee_rate`
   - `backtest.option_fee_per_contract`
   - strategy open/close thresholds
   - roll thresholds
   - delta hedge settings
   - `vol.contract_multiplier`
   - ATM selection settings

2. Register the config.

   Update `core/config.py`:

   - Add `<product>: core.configs.config_<product>` to `PRODUCT_CONFIG_MODULES`.
   - Add `<product>` to the `available_live_products()` allowlist.

3. Register the live quote spec.

   Update `core/live/market_data.py`:

   - Add an entry to `SSE_ETF_OPTION_SPECS`.
   - Set `etf_symbol` to the six-digit ETF security code.
   - Set `etf_file_prefix` to `<symbol>.XSHG`.
   - Set `option_file_prefix` to the AKShare option board prefix verified for this ETF.

   The product will automatically enter `LIVE_PRODUCTS` after this mapping is added.

4. Register option broker export markers.

   Update both files:

   - `core/live/holding_importer.py`
   - `core/live/account_report.py`

   Add the product to `PRODUCT_CONTRACT_NAME_MARKERS`. This marker must match the option contract names in `实时持仓(信息导出)*.csv` and `成交明细(信息导出)*.csv`.

   Example: `300ETF购7月4900` is matched by `300ETF`.

5. Register portfolio report labels.

   Update `core/live/portfolio_report.py`:

   - Add the product to `STRATEGY_DISPLAY_NAMES`.
   - Ensure the product is covered by `_position_product_markers`.

   The `合约代码` column in `账户总体情况` is derived automatically from `SSE_ETF_OPTION_SPECS` as `sh<etf_symbol>`.

6. Add historical or live quote download support if needed.

   If the product needs canonical historical research data, extend the product
   specifications in `scripts/download/build_data.py` when it follows the SSE ETF
   option workflow. Add a product-specific downloader only when the exchange or
   source protocol is materially different, as with `zz1000`.

   Minimum requirement for live import/reporting:

   - Local historical data should be available when possible.
   - Missing broker contract metadata must be recoverable through `market_data.fetch_historical_option_metadata`.
   - The AKShare risk indicator `CONTRACT_ID` prefix must match the ETF symbol.

7. Confirm broker ETF import compatibility.

   No product-specific ETF importer change is needed if `SSE_ETF_OPTION_SPECS` is correct and broker files keep the current format:

   - `证券持仓查询(信息导出)*.csv`
   - `证券委托查询_实时成交(信息导出)*.csv`

   ETF rows are matched by `SSE_ETF_OPTION_SPECS[product].etf_symbol`.

8. Update tests.

   At minimum, update or add coverage in:

   - `tests/test_live_multi_product_support.py`
   - `tests/test_live_portfolio_report.py`
   - `tests/test_live_etf_importer.py`

   Required checks:

   - `set(core.config.available_live_products()) == set(market_data.LIVE_PRODUCTS)`
   - `market_data.option_underlying_order_book_id(product)` returns the ETF order book id.
   - Product config has `long_qty == short_qty == 10`.
   - Broker option holding rows are filtered by the new marker.
   - Portfolio report shows the display name and `sh<etf_symbol>` report code.
   - ETF import matches the new ETF symbol.

## Validation Commands

Run these after adding the mapping and config:

```bash
python -m pytest tests/test_live_multi_product_support.py
python -m pytest tests/test_live_etf_importer.py
python -m pytest tests/test_live_portfolio_report.py
python -m pytest
```

Then run smoke tests against the live scripts:

```bash
python scripts/live/fetch_quotes.py --product <product> --source akshare
python scripts/live/generate_signal.py --product <product>
python scripts/live/import_holdings.py --product <product> --include-existing --dry-run
python scripts/live/import_etf.py --product <product> --dry-run
python scripts/live/live_account_report.py --product <product> --source akshare --no-write --no-history
python scripts/live/live_portfolio_report.py --source akshare --no-write --no-history
```

If the product is included in the portfolio console, also start:

```bash
python scripts/live/live_console.py
```

## Broker Export Acceptance Criteria

Before enabling a new live product for real operation, save sample broker exports under `live_hold/` and verify:

- Option holding rows contain a stable product marker in `合约名称`.
- Only normal options are selected for new entries; adjusted contracts with decimal strike/suffix are excluded from fresh selection.
- Existing adjusted positions, if any, can still be resolved by exact contract code for reporting/closing.
- Option trade detail rows use the current `成交明细(信息导出)` format.
- ETF holding and ETF trade rows use the current two standard ETF files.
- ETF symbol matches the intended hedge instrument exactly.

## Common Failure Modes

- Product appears in `core/config.py` but not `SSE_ETF_OPTION_SPECS`: live commands reject it.
- Product appears in `SSE_ETF_OPTION_SPECS` but not `available_live_products()`: portfolio/console commands omit it.
- Option marker is too broad: broker imports may mix two products.
- Option marker is too narrow: broker imports silently find no rows for the product.
- AKShare option prefix is wrong: live quote fetch returns empty chains or unrelated contracts.
- The report display name is changed without legacy matching support: same-day report rows may duplicate under old and new names.
- Config quantities are not 10/10: live open advice violates current fixed-entry requirement.

## Implementation Map

| Purpose | File | Object/function |
| --- | --- | --- |
| Config registration | `core/config.py` | `PRODUCT_CONFIG_MODULES`, `available_live_products()` |
| Live ETF/option quote spec | `core/live/market_data.py` | `SSE_ETF_OPTION_SPECS` |
| Option broker row filtering | `core/live/holding_importer.py` | `PRODUCT_CONTRACT_NAME_MARKERS` |
| Option trade/report row filtering | `core/live/account_report.py` | `PRODUCT_CONTRACT_NAME_MARKERS` |
| Portfolio display name/code | `core/live/portfolio_report.py` | `STRATEGY_DISPLAY_NAMES`, `_position_product_markers()` |
| ETF import matching | `core/live/etf_importer.py` | uses `SSE_ETF_OPTION_SPECS` |
| Console product universe | `scripts/live/live_console.py` | uses `core.config.available_live_products()` |
| Portfolio product universe | `core/live/portfolio_report.py` | uses `core.config.available_live_products()` |
