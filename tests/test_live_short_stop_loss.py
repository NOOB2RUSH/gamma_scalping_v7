from pathlib import Path
from unittest import mock

import core
import pandas as pd

from core.live import signal_engine


def test_short_daily_loss_aum_stop_uses_strict_negative_threshold():
    config = core.config.load_config("300etf")
    with mock.patch.object(core.strategy, "CONFIG", config):
        assert core.strategy.is_short_daily_loss_aum_stop(-7_500.0, 500_000.0) is False
        assert core.strategy.is_short_daily_loss_aum_stop(-7_501.0, 500_000.0) is True
        assert core.strategy.is_short_daily_loss_aum_stop(1_000.0, 500_000.0) is False


def test_disabled_short_stop_ignores_daily_aum_loss():
    config = core.config.load_config("kc50etf")
    with mock.patch.object(core.strategy, "CONFIG", config):
        assert core.strategy.is_short_daily_loss_aum_stop(-50_000.0, 500_000.0) is False


def test_intraday_previous_snapshot_is_not_treated_as_previous_close():
    snapshot = {
        "snapshot_stamp": "20260618_143000",
        "quote_date": "2026-06-18",
        "option_snapshot": str(Path("unused.parquet")),
    }
    with mock.patch.object(
        signal_engine.market_data,
        "load_previous_quote_snapshot",
        return_value=snapshot,
    ), mock.patch.object(pd, "read_parquet") as read_parquet:
        date, chain = signal_engine._load_previous_close_chain(
            "300etf",
            "2026-06-22",
        )

    assert date is None
    assert chain is None
    read_parquet.assert_not_called()
