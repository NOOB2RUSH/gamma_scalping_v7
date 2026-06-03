"""Simulated live trading helpers.

The live package is deliberately independent from the backtest state machine.
It reuses shared valuation and strategy helpers, while account mutation is
limited to explicit manual fill confirmation.
"""

