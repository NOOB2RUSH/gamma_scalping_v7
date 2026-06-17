import unittest
from types import SimpleNamespace
from unittest import mock

from core.live import portfolio_account


class LivePortfolioAccountTest(unittest.TestCase):
    def test_shared_cash_counts_initial_cash_once(self):
        states = {
            "300etf": SimpleNamespace(cash=900_000.0),
            "500etf": SimpleNamespace(cash=800_000.0),
            "kc50etf": SimpleNamespace(cash=9_500_000.0),
        }
        initial = {
            "300etf": 1_000_000.0,
            "500etf": 1_000_000.0,
            "kc50etf": 10_000_000.0,
        }
        with (
            mock.patch.object(
                portfolio_account.account_store,
                "load_account",
                side_effect=lambda product, account_id: states[product],
            ),
            mock.patch.object(
                portfolio_account,
                "product_initial_cash",
                side_effect=lambda product: initial[product],
            ),
        ):
            cash = portfolio_account.shared_cash(
                products=("300etf", "500etf", "kc50etf"),
            )

        self.assertEqual(cash, 9_200_000.0)

    def test_shared_nav_counts_initial_cash_once(self):
        payloads = {
            "300etf": {"summary": {"初始资金": 1_000_000.0, "估算权益": 999_000.0}},
            "500etf": {"summary": {"初始资金": 1_000_000.0, "估算权益": 998_000.0}},
            "kc50etf": {
                "summary": {"初始资金": 10_000_000.0, "估算权益": 10_011_915.34}
            },
        }

        nav = portfolio_account.shared_nav_from_subaccounts(payloads)

        self.assertAlmostEqual(nav, 10_008_915.34)


if __name__ == "__main__":
    unittest.main()
