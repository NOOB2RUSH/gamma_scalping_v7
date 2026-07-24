import math
import unittest

import pandas as pd

from scripts.research.dynamic_position.iv_level_next_day_realized_vol_curve import (
    build_next_day_realized_volatility,
    kernel_smoothed_realized_volatility,
)


class NextDayRealizedVolatilityCurveTest(unittest.TestCase):
    def test_pairs_today_iv_with_next_trading_day_absolute_log_return(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        atm_iv = pd.Series([0.20, 0.25, 0.30], index=dates)
        close = pd.Series([100.0, 110.0, 99.0], index=dates)

        result = build_next_day_realized_volatility(atm_iv, close)

        self.assertEqual(result.index.tolist(), dates[:2].tolist())
        self.assertEqual(result.loc[dates[0], "next_trading_day"], dates[1])
        self.assertAlmostEqual(
            result.loc[dates[0], "next_day_log_return"], math.log(1.1)
        )
        self.assertAlmostEqual(
            result.loc[dates[1], "next_day_realized_volatility"],
            abs(math.log(99.0 / 110.0)) * math.sqrt(252),
        )
        self.assertAlmostEqual(
            result.loc[dates[0], "next_day_realized_to_atm_iv_ratio"],
            math.log(1.1) * math.sqrt(252) / 0.20,
        )
        self.assertAlmostEqual(
            result.loc[dates[0], "today_atm_iv_minus_next_day_realized_volatility"],
            0.20 - math.log(1.1) * math.sqrt(252),
        )

    def test_kernel_curve_returns_expected_conditional_mean_columns(self):
        dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
        samples = build_next_day_realized_volatility(
            pd.Series([0.20, 0.25, 0.30], index=dates),
            pd.Series([100.0, 110.0, 99.0], index=dates),
        )

        curves = kernel_smoothed_realized_volatility(
            samples,
            bandwidth=0.03,
            grid_points=3,
            lower_percentile=0.0,
            upper_percentile=1.0,
        )

        self.assertEqual(len(curves), 3)
        self.assertTrue(curves["expected_next_day_realized_volatility"].gt(0).all())
        self.assertTrue(
            curves["expected_next_day_realized_to_atm_iv_ratio"].gt(0).all()
        )
        self.assertTrue(
            curves[
                "expected_today_atm_iv_minus_next_day_realized_volatility"
            ].notna().all()
        )
        self.assertTrue(curves["effective_sample_size"].gt(0).all())


if __name__ == "__main__":
    unittest.main()
