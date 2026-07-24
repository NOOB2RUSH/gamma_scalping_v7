import unittest
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pandas as pd

from core.live import market_data


class LiveMarketDataBatchTest(unittest.TestCase):
    def tearDown(self):
        market_data._CURRENT_SSE_OPTION_CONTRACTS = None
        market_data._CURRENT_SSE_OPTION_CONTRACTS_FETCHED_AT = 0.0

    def test_parse_sina_option_quote_batch_preserves_live_quote_fields(self):
        values = [""] * 43
        values[1] = "0.1010"
        values[2] = "0.1020"
        values[3] = "0.1030"
        values[5] = "1234"
        values[7] = "3.1000"
        values[32] = "2026-07-14 14:56:03"
        values[36] = "510050"
        values[37] = "50ETF-CALL"
        values[41] = "567"
        text = f'var hq_str_CON_OP_10000001="{",".join(values)}";'

        rows = market_data._parse_sina_option_quote_batch(text)

        self.assertEqual(set(rows), {"10000001"})
        row = rows["10000001"]
        self.assertEqual(row["bid"], 0.101)
        self.assertEqual(row["close"], 0.102)
        self.assertEqual(row["ask"], 0.103)
        self.assertEqual(row["strike_price"], 3.1)
        self.assertEqual(row["volume"], 567.0)
        self.assertEqual(row["open_interest"], 1234.0)
        self.assertEqual(row["quote_time"], "2026-07-14 14:56:03")

    def test_current_day_contracts_are_cached_across_products(self):
        columns = {
            "\u5408\u7ea6\u7f16\u7801": ["10000001", "10000002"],
            "\u5408\u7ea6\u4ea4\u6613\u4ee3\u7801": [
                "510050C2607M03100",
                "510300P2607M04800",
            ],
            "\u5230\u671f\u65e5": ["20260722", "20260722"],
        }
        fake_ak = mock.Mock()
        fake_ak.option_current_day_sse.return_value = pd.DataFrame(columns)

        first = market_data._sse_option_tasks_from_current_day_sse(
            fake_ak,
            market_data.SSE_ETF_OPTION_SPECS["50etf"],
        )
        second = market_data._sse_option_tasks_from_current_day_sse(
            fake_ak,
            market_data.SSE_ETF_OPTION_SPECS["300etf"],
        )

        self.assertEqual(first[0][0], "10000001")
        self.assertEqual(first[0][2], "C")
        self.assertEqual(second[0][0], "10000002")
        self.assertEqual(second[0][2], "P")
        fake_ak.option_current_day_sse.assert_called_once_with()

    def test_current_day_contract_cache_is_shared_by_concurrent_products(self):
        contracts = pd.DataFrame(
            {
                "\u5408\u7ea6\u7f16\u7801": ["10000001"],
                "\u5408\u7ea6\u4ea4\u6613\u4ee3\u7801": ["510050C2607M03100"],
                "\u5230\u671f\u65e5": ["20260722"],
            }
        )
        fake_ak = mock.Mock()

        def fetch_contracts():
            time.sleep(0.05)
            return contracts

        fake_ak.option_current_day_sse.side_effect = fetch_contracts
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda _: market_data._current_sse_option_contracts(fake_ak),
                    range(4),
                )
            )

        self.assertTrue(all(result is contracts for result in results))
        fake_ak.option_current_day_sse.assert_called_once_with()

    def test_chain_batches_quotes_and_falls_back_only_for_missing_codes(self):
        tasks = [
            ("10000001", pd.Timestamp("2026-07-22"), "C"),
            ("10000002", pd.Timestamp("2026-07-22"), "P"),
        ]
        batch_row = {
            "order_book_id": "10000001",
            "strike_price": 3.1,
            "bid": 0.1,
            "ask": 0.11,
            "raw_sina_volume": 10.0,
            "volume": 10.0,
            "open_interest": 20.0,
            "contract_multiplier": 10000,
            "close": 0.105,
            "source": "batch",
        }
        fallback_row = {
            **batch_row,
            "order_book_id": "10000002",
            "option_type": "P",
            "maturity_date": pd.Timestamp("2026-07-22"),
            "source": "fallback",
        }

        with (
            mock.patch.object(
                market_data,
                "_sse_option_tasks_from_current_day_sse",
                return_value=tasks,
            ),
            mock.patch.object(
                market_data,
                "_fetch_sina_option_quote_batch",
                return_value={"10000001": batch_row},
            ) as batch,
            mock.patch.object(
                market_data,
                "_fetch_sse_option_rows_individually",
                return_value=[fallback_row],
            ) as fallback,
        ):
            chain = market_data._fetch_sse_option_chain(
                mock.Mock(),
                market_data.SSE_ETF_OPTION_SPECS["50etf"],
                pd.DatetimeIndex([]),
                pd.Timestamp("2026-07-14"),
            )

        self.assertEqual(set(chain["order_book_id"]), {"10000001", "10000002"})
        batch.assert_called_once_with(["10000001", "10000002"])
        fallback.assert_called_once_with(
            mock.ANY,
            [("10000002", pd.Timestamp("2026-07-22"), "P")],
        )
        first = chain.loc[chain["order_book_id"].eq("10000001")].iloc[0]
        self.assertEqual(first["option_type"], "C")
        self.assertEqual(first["maturity_date"], pd.Timestamp("2026-07-22"))

    def test_chain_fetches_multiple_quote_batches(self):
        tasks = [
            (f"{code:08d}", pd.Timestamp("2026-07-22"), "C")
            for code in range(1, 5)
        ]

        def batch_rows(codes):
            return {
                code: {
                    "order_book_id": code,
                    "strike_price": 3.1,
                    "bid": 0.1,
                    "ask": 0.11,
                    "close": 0.105,
                    "volume": 10.0,
                    "open_interest": 20.0,
                    "contract_multiplier": 10000,
                    "source": "batch",
                }
                for code in codes
            }

        with (
            mock.patch.object(
                market_data,
                "AKSHARE_OPTION_BATCH_SIZE",
                2,
            ),
            mock.patch.object(
                market_data,
                "_sse_option_tasks_from_current_day_sse",
                return_value=tasks,
            ),
            mock.patch.object(
                market_data,
                "_fetch_sina_option_quote_batch",
                side_effect=batch_rows,
            ) as batch,
            mock.patch.object(
                market_data,
                "_fetch_sse_option_rows_individually",
                return_value=[],
            ) as fallback,
        ):
            chain = market_data._fetch_sse_option_chain(
                mock.Mock(),
                market_data.SSE_ETF_OPTION_SPECS["50etf"],
                pd.DatetimeIndex([]),
                pd.Timestamp("2026-07-14"),
            )

        self.assertEqual(batch.call_count, 2)
        self.assertEqual(set(chain["order_book_id"]), {task[0] for task in tasks})
        fallback.assert_called_once_with(mock.ANY, [])

    def test_current_contract_terms_override_batch_defaults_after_dividend(self):
        tasks = [
            ("10011720", pd.Timestamp("2026-07-22"), "C"),
            ("10011729", pd.Timestamp("2026-07-22"), "P"),
        ]
        batch_rows = {
            code: {
                "order_book_id": code,
                "strike_price": 8.25,
                "bid": 0.1,
                "ask": 0.11,
                "close": 0.105,
                "volume": 10.0,
                "open_interest": 20.0,
                "contract_multiplier": 10000,
                "contract_symbol": "500ETF",
                "source": "batch",
            }
            for code, _, _ in tasks
        }
        current_metadata = {
            "10011720": {
                "strike": 8.104,
                "expiry": "2026-07-22",
                "option_type": "C",
                "contract_multiplier": 10180,
                "contract_symbol": "XD500ETF\u8d2d7\u67088104A",
            },
            "10011729": {
                "strike": 8.104,
                "expiry": "2026-07-22",
                "option_type": "P",
                "contract_multiplier": 10180,
                "contract_symbol": "XD500ETF\u6cbd7\u67088104A",
            },
        }

        with (
            mock.patch.object(
                market_data,
                "_sse_option_tasks_from_current_day_sse",
                return_value=tasks,
            ),
            mock.patch.object(
                market_data,
                "_current_sse_option_contract_metadata",
                return_value=current_metadata,
            ),
            mock.patch.object(
                market_data,
                "_fetch_sina_option_quote_batch",
                return_value=batch_rows,
            ),
            mock.patch.object(
                market_data,
                "_fetch_sse_option_rows_individually",
                return_value=[],
            ),
        ):
            chain = market_data._fetch_sse_option_chain(
                mock.Mock(),
                market_data.SSE_ETF_OPTION_SPECS["500etf"],
                pd.DatetimeIndex([]),
                pd.Timestamp("2026-07-15"),
            )

        self.assertEqual(set(chain["strike_price"]), {8.104})
        self.assertEqual(set(chain["contract_multiplier"]), {10180})
        self.assertTrue(chain["contract_symbol"].str.endswith("A").all())


if __name__ == "__main__":
    unittest.main()
