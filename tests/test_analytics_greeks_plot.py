import unittest

import pandas as pd

from core import analytics


class AnalyticsGreeksPlotTest(unittest.TestCase):
    def test_vol_features_plots_cumulative_return_from_initial_cash(self):
        initial_cash = float(analytics.CONFIG.backtest.initial_cash)
        index = pd.to_datetime(["2026-01-05", "2026-01-06"])
        features = pd.DataFrame(
            {
                "close": [3.0, 3.1],
                "atm_iv": [0.20, 0.21],
            },
            index=index,
        )
        backtest = pd.DataFrame(
            {
                "nav": [initial_cash, initial_cash * 1.10],
            },
            index=index,
        )

        figure = analytics.plot_vol_features(
            features,
            backtest_df=backtest,
            show=False,
        )

        return_axis = next(
            axis
            for axis in figure.axes
            if axis.get_ylabel() == "Cumulative Return"
        )
        plotted = {
            line.get_label(): list(line.get_ydata())
            for line in return_axis.lines
        }
        self.assertAlmostEqual(plotted["Cumulative Return"][0], 0.0)
        self.assertAlmostEqual(plotted["Cumulative Return"][1], 0.1)
        self.assertEqual(plotted["Initial Capital (0%)"], [0.0, 0.0])

    def test_cumulative_plot_combines_theta_and_gamma(self):
        backtest = pd.DataFrame(
            {
                "delta_pnl": [1.0, 2.0],
                "gamma_pnl": [-1.0, -2.0],
                "vega_pnl": [0.5, 1.0],
                "theta_pnl": [4.0, 6.0],
                "greeks_pnl": [4.5, 7.0],
            },
            index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
        )

        figure = analytics.plot_cumulative_greeks_pnl(backtest, show=False)
        plotted = {
            line.get_label(): list(line.get_ydata())
            for line in figure.axes[0].lines
        }

        self.assertEqual(plotted["Theta + Gamma PnL"], [3.0, 7.0])
        self.assertNotIn("Theta PnL", plotted)
        self.assertNotIn("Gamma PnL", plotted)
        self.assertEqual(plotted["Total Greeks PnL"], [4.5, 11.5])


if __name__ == "__main__":
    unittest.main()
