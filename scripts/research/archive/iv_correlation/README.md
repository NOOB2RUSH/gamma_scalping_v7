# IV correlation research archive

This directory contains the completed cross-product ATM-IV correlation and
pair-strategy experiments from July 2026. They are retained for historical
reproduction and are not active strategy entry points.

The original pipeline was:

1. `etf_atm_iv_ratios.py` builds the aligned ATM-IV panel and ratios.
2. `etf_atm_iv_change_corr.py` calculates IV-change correlations.
3. `iv_corr_regime_shifts.py` detects persistent correlation shifts.
4. `plot_pair_atm_iv.py` plots pair-level IV and correlation diagnostics.
5. `pair_iv_ratio_backtest.py` runs the experimental pair strategy.

Run archived scripts from the project root. Existing outputs remain under
`output/research/` and `output/research/pair_backtest/`.
