import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import pandas as pd

import core
from core.live import account, holding_importer, market_data, signal_engine


class LiveMultiProductSupportTest(unittest.TestCase):
    def test_product_configs_do_not_include_reference_curves(self):
        for product in core.config.available_products():
            with self.subTest(product=product):
                self.assertFalse(hasattr(core.config.load_config(product), "reference"))

    def test_new_contract_selection_excludes_adjusted_options(self):
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": f"{kind}_{suffix}",
                    "contract_symbol": symbol,
                    "strike_price": strike,
                    "maturity_date": pd.Timestamp("2026-07-22"),
                    "dte": 27,
                    "option_type": kind,
                    "contract_multiplier": 10000,
                    "iv": 0.2,
                    "volume": volume,
                }
                for suffix, symbol_base, strike, volume in [
                    ("ADJ", "300ETF{kind}7月4873A", 4.873, 10000),
                    ("NORMAL", "300ETF{kind}7月4900", 4.9, 100),
                ]
                for kind, symbol in [
                    ("c", symbol_base.format(kind="购")),
                    ("p", symbol_base.format(kind="沽")),
                ]
            ]
        )
        vol_config = SimpleNamespace(
            contract_multiplier=10000,
            atm_target_dte=27,
            atm_target_dte_min=5,
            atm_target_dte_max=60,
            atm_moneyness_tol=0.1,
            atm_selection_mode="target_dte",
            atm_min_total_volume=0,
            atm_low_volume_search_near_month=False,
        )

        with mock.patch.object(core.vol_engine, "CONFIG", SimpleNamespace(vol=vol_config)):
            atm = core.vol_engine.select_atm_from_chain(chain, spot=4.88)
            pool = core.vol_engine.calc_atm_pool_volume(chain, spot=4.88)

        self.assertEqual(atm["strike"], 4.9)
        self.assertFalse(atm["call"]["contract_symbol"].endswith("A"))
        self.assertEqual(pool["atm_pool_total_volume"], 200)

        adjusted_position = {"call_code": "c_ADJ", "put_code": "p_ADJ"}
        call_row, put_row = core.vol_engine.resolve_position_pair(
            adjusted_position,
            chain,
        )
        self.assertTrue(call_row["contract_symbol"].endswith("A"))
        self.assertTrue(put_row["contract_symbol"].endswith("A"))

    def test_open_plan_includes_final_delta_hedge_and_new_position_capacity(self):
        config = SimpleNamespace(
            strategy=SimpleNamespace(
                enable_delta_hedge=True,
                delta_hedge_tolerance_ratio=0.05,
                allow_etf_short_hedge=True,
            ),
            backtest=SimpleNamespace(
                min_cash_reserve=0.0,
                etf_fee_rate=0.0,
                option_fee_per_contract=2.0,
                dynamic_position_control_enabled=False,
                max_margin_to_nav_ratio=0.8,
            ),
            vol=SimpleNamespace(contract_multiplier=10000),
        )
        live_account = account.AccountState(product="300etf", cash=1_000_000)
        chain = pd.DataFrame(
            [
                {
                    "order_book_id": "CALL",
                    "option_type": "c",
                    "contract_multiplier": 10000,
                    "delta": 0.45,
                    "gamma": 0.1,
                    "vega": 0.2,
                    "theta": -0.3,
                    "iv": 0.2,
                    "underlying_order_book_id": "510300.XSHG",
                },
                {
                    "order_book_id": "PUT",
                    "option_type": "p",
                    "contract_multiplier": 10000,
                    "delta": -0.39,
                    "gamma": 0.1,
                    "vega": 0.2,
                    "theta": -0.3,
                    "iv": 0.2,
                    "underlying_order_book_id": "510300.XSHG",
                },
            ]
        )
        open_item = {
            "action": "OPEN_SHORT_STRADDLE",
            "priority": "action",
            "side": "short",
            "call_code": "CALL",
            "put_code": "PUT",
            "call_qty": 10,
            "put_qty": 10,
            "estimated_cash_effect": 0.0,
        }

        plan, planned_greeks = signal_engine._build_execution_plan(
            config,
            live_account,
            chain,
            4.88,
            {"underlying_order_book_id": "510300.XSHG"},
            [open_item],
            core.backtester.empty_greeks(),
            {},
        )

        self.assertAlmostEqual(planned_greeks["delta"], -6000.0)
        self.assertEqual([item["action"] for item in plan], [
            "OPEN_SHORT_STRADDLE",
            "FINAL_DELTA_HEDGE",
        ])
        self.assertEqual(plan[-1]["trade_etf_qty"], 6000.0)
        self.assertEqual(plan[-1]["delta_hedge_capacity"], 100000.0)
        self.assertAlmostEqual(plan[-1]["normalized_account_delta"], -0.06)

    def test_live_supports_four_sse_etf_products_with_expected_underlyings(self):
        expected = {
            "50etf": "510050.XSHG",
            "300etf": "510300.XSHG",
            "500etf": "510500.XSHG",
            "kc50etf": "588000.XSHG",
        }

        self.assertEqual(set(market_data.LIVE_PRODUCTS), set(expected))
        for product, underlying in expected.items():
            with self.subTest(product=product):
                self.assertEqual(
                    market_data.option_underlying_order_book_id(product),
                    underlying,
                )
                chain = market_data.attach_live_underlying_id(
                    product,
                    pd.DataFrame([{"order_book_id": "OPTION"}]),
                )
                self.assertEqual(chain.iloc[0]["underlying_order_book_id"], underlying)

        with self.assertRaisesRegex(ValueError, "Live trading currently supports"):
            market_data.require_live_product("zz1000")

    def test_all_live_product_configs_open_ten_contracts_per_leg(self):
        self.assertEqual(
            set(core.config.available_live_products()),
            set(market_data.LIVE_PRODUCTS),
        )
        for product in market_data.LIVE_PRODUCTS:
            with self.subTest(product=product):
                config = core.config.load_config(product)
                self.assertEqual(config.backtest.long_qty, 10)
                self.assertEqual(config.backtest.short_qty, 10)

    def test_live_entry_advice_uses_ten_contracts_per_leg(self):
        config = SimpleNamespace(backtest=SimpleNamespace(long_qty=99, short_qty=88))
        feature = {
            "long_open_signal": True,
            "short_open_signal": True,
            "short_open_regime": "absolute",
        }
        strategy_state = SimpleNamespace(
            short_entry_cooldown_left=0,
        )

        with mock.patch.object(
            signal_engine,
            "_open_advice",
            side_effect=lambda action, side, qty, atm, spot: {
                "action": action,
                "side": side,
                "call_qty": qty,
                "put_qty": qty,
            },
        ):
            advice = signal_engine._entry_advice(
                config,
                feature,
                {"strike": 1.0},
                1.0,
                strategy_state,
            )

        self.assertEqual(
            [(item["call_qty"], item["put_qty"]) for item in advice],
            [(10, 10), (10, 10)],
        )

    def test_historical_option_metadata_uses_exact_date_and_cache(self):
        risk = pd.DataFrame(
            {
                "SECURITY_ID": ["10000001", "10000002"],
                "CONTRACT_ID": ["510300C2606M04900", "510300P2606M04900"],
                "CONTRACT_SYMBOL": ["300ETF购6月4900", "300ETF沽6月4900"],
            }
        )
        daily = pd.DataFrame(
            {
                "日期": ["2026-06-09"],
                "收盘": [0.123],
                "成交量": [456],
            }
        )
        ak = SimpleNamespace(
            option_risk_indicator_sse=mock.Mock(return_value=risk),
            option_sse_daily_sina=mock.Mock(return_value=daily),
        )

        with TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "historical_option_metadata.csv"
            with (
                mock.patch.dict("sys.modules", {"akshare": ak}),
                mock.patch.object(
                    market_data.storage,
                    "historical_option_metadata_cache_path",
                    return_value=cache_path,
                ),
            ):
                first = market_data.fetch_historical_option_metadata(
                    "300etf",
                    "2026-06-09",
                    codes=["10000001"],
                )
                second = market_data.fetch_historical_option_metadata(
                    "300etf",
                    "2026-06-09",
                    codes=["10000001"],
                )

        self.assertEqual(first["10000001"]["strike"], 4.9)
        self.assertEqual(
            first["10000001"]["underlying_order_book_id"],
            "510300.XSHG",
        )
        self.assertEqual(first["10000001"]["close"], 0.123)
        self.assertEqual(first["10000001"]["volume"], 456)
        self.assertEqual(second, first)
        ak.option_risk_indicator_sse.assert_called_once_with(date="20260609")
        ak.option_sse_daily_sina.assert_called_once_with(symbol="10000001")

    def test_sse_option_chain_falls_back_when_month_list_breaks(self):
        risk = pd.DataFrame(
            {
                "SECURITY_ID": ["10011703", "10011712", "99999999"],
                "CONTRACT_ID": [
                    "510300C2607M04900",
                    "510300P2607M04900",
                    "510500C2607M08500",
                ],
            }
        )
        ak = SimpleNamespace(
            option_sse_list_sina=mock.Mock(side_effect=AttributeError("split")),
            option_risk_indicator_sse=mock.Mock(return_value=risk),
        )

        tasks = market_data._sse_option_tasks_from_risk_indicator(
            ak,
            market_data.SSE_ETF_OPTION_SPECS["300etf"],
            pd.DatetimeIndex([]),
            "2026-06-15",
        )

        self.assertEqual(
            [(code, option_type) for code, _, option_type in tasks],
            [("10011703", "C"), ("10011712", "P")],
        )
        ak.option_risk_indicator_sse.assert_called_once_with(date="20260615")

    def test_holding_metadata_falls_back_to_akshare_when_local_history_is_missing(self):
        with TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(
                data=SimpleNamespace(product="500etf", opt_dir=temp_dir),
                vol=SimpleNamespace(contract_multiplier=10000),
            )
            fallback = {
                "10000003": {
                    "strike": 8.5,
                    "expiry": "2026-06-24",
                    "option_type": "C",
                    "contract_multiplier": 10000,
                    "contract_symbol": "500ETF购6月8500",
                    "underlying_order_book_id": "510500.XSHG",
                    "metadata_source": "akshare_option_risk_indicator_sse",
                }
            }
            with mock.patch.object(
                holding_importer.market_data,
                "fetch_historical_option_metadata",
                return_value=fallback,
            ) as fetch:
                metadata = holding_importer._load_contract_metadata(
                    config,
                    ["10000003"],
                    trade_date="2026-06-09",
                )

        self.assertEqual(metadata, fallback)
        fetch.assert_called_once_with(
            "500etf",
            "2026-06-09",
            codes={"10000003"},
        )

    def test_holding_metadata_uses_local_history_before_akshare(self):
        with TemporaryDirectory() as temp_dir:
            opt_dir = Path(temp_dir)
            pd.DataFrame(
                [
                    {
                        "order_book_id": "10000004",
                        "strike_price": 3.0,
                        "maturity_date": "2026-06-24",
                        "option_type": "P",
                        "contract_multiplier": 10000,
                        "contract_symbol": "50ETF沽6月3000",
                    }
                ]
            ).to_parquet(opt_dir / "510050.XSHG_2026-06-09_chain.parquet")
            config = SimpleNamespace(
                data=SimpleNamespace(product="50etf", opt_dir=str(opt_dir)),
                vol=SimpleNamespace(contract_multiplier=10000),
            )
            with mock.patch.object(
                holding_importer.market_data,
                "fetch_historical_option_metadata",
            ) as fetch:
                metadata = holding_importer._load_contract_metadata(
                    config,
                    ["10000004"],
                    trade_date="2026-06-09",
                )

        self.assertEqual(metadata["10000004"]["metadata_source"], "local_option_chain")
        self.assertEqual(
            metadata["10000004"]["underlying_order_book_id"],
            "510050.XSHG",
        )
        fetch.assert_not_called()

    def test_shared_holding_rows_are_filtered_by_product(self):
        rows = [
            {"contract_name": "300ETF购7月4900"},
            {"contract_name": "500ETF购7月8500"},
            {"contract_name": "科创50沽6月1750"},
        ]

        self.assertEqual(
            holding_importer._rows_for_product(rows, "500etf"),
            [{"contract_name": "500ETF购7月8500"}],
        )
        self.assertEqual(
            holding_importer._rows_for_product(rows, "kc50etf"),
            [{"contract_name": "科创50沽6月1750"}],
        )

    def test_holding_metadata_uses_live_snapshot_before_akshare(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            opt_dir = root / "history"
            live_dir = root / "data" / "live" / "300etf" / "quotes" / "20260615"
            opt_dir.mkdir()
            live_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "order_book_id": "10011703",
                        "strike_price": 4.9,
                        "maturity_date": "2026-07-22",
                        "option_type": "C",
                        "contract_multiplier": 10000,
                        "contract_symbol": "300ETF购7月4900",
                    }
                ]
            ).to_parquet(live_dir / "145000_option_chain.parquet")
            config = SimpleNamespace(
                data=SimpleNamespace(product="300etf", opt_dir=str(opt_dir)),
                vol=SimpleNamespace(contract_multiplier=10000),
            )

            def project_path(path):
                path = Path(path)
                if path.is_absolute():
                    return path
                return root / path

            with (
                mock.patch.object(holding_importer, "project_path", side_effect=project_path),
                mock.patch.object(
                    holding_importer.market_data,
                    "fetch_historical_option_metadata",
                ) as fetch,
            ):
                metadata = holding_importer._load_contract_metadata(
                    config,
                    ["10011703"],
                    trade_date="2026-06-15",
                )

        self.assertEqual(metadata["10011703"]["strike"], 4.9)
        self.assertEqual(metadata["10011703"]["metadata_source"], "local_option_chain")
        fetch.assert_not_called()

    def test_option_trade_detail_is_aggregated_by_contract(self):
        detail = pd.DataFrame(
            [
                {
                    "合约代码": "10011721",
                    "开平": "开仓",
                    "买卖": "卖",
                    "成交数量": 10,
                    "成交价格": 0.24,
                },
                {
                    "合约代码": "10011721",
                    "开平": "开仓",
                    "买卖": "卖",
                    "成交数量": 4,
                    "成交价格": 0.25,
                },
                {
                    "合约代码": "10011721",
                    "开平": "平仓",
                    "买卖": "买",
                    "成交数量": 4,
                    "成交价格": 0.26,
                },
            ]
        )

        summary = holding_importer._trade_detail_by_code(detail)["10011721"]

        self.assertEqual(summary["卖开"], 14)
        self.assertAlmostEqual(summary["卖开均价"], (10 * 0.24 + 4 * 0.25) / 14)
        self.assertEqual(summary["买平"], 4)
        self.assertEqual(summary["买平均价"], 0.26)

    def test_trade_detail_resolver_ignores_legacy_summary_exports(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live_hold = root / "live_hold"
            live_hold.mkdir()
            summary = live_hold / "成交汇总(信息导出)_2026_06_15-15_10_00.csv"
            detail = live_hold / "成交明细(信息导出)_2026_06_15-15_09_00.csv"
            summary.touch()
            detail.touch()

            with mock.patch("pathlib.Path.glob") as glob:
                glob.return_value = [detail]
                resolved = holding_importer._resolve_trade_detail_file("2026-06-15")

        self.assertEqual(resolved, detail)
        glob.assert_called_once_with("成交明细*.csv")

    def test_holding_import_roll_closes_old_position_before_opening_snapshot(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            db_path = root / "account.sqlite"
            holding_path = root / "实时持仓(信息导出)_2026_06_17-14_48_14.csv"
            detail_path = root / "成交明细(信息导出)_2026_06_17-14_48_28.csv"
            pd.DataFrame(
                [
                    {
                        "合约代码": "10011740",
                        "合约名称": "科创50购7月1950",
                        "买卖": "卖",
                        "持仓类型": "义务仓",
                        "总持仓": 10,
                        "今开仓": 10,
                        "开仓均价": 0.1011,
                        "最新价": 0.1009,
                        "占用保证金": 18845.0,
                    },
                    {
                        "合约代码": "10011749",
                        "合约名称": "科创50沽7月1950",
                        "买卖": "卖",
                        "持仓类型": "义务仓",
                        "总持仓": 10,
                        "今开仓": 10,
                        "开仓均价": 0.1385,
                        "最新价": 0.1380,
                        "占用保证金": 40430.0,
                    },
                ]
            ).to_csv(holding_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "合约代码": "10010393",
                        "开平": "平仓",
                        "买卖": "买",
                        "成交数量": 80,
                        "成交价格": 0.1853,
                    },
                    {
                        "合约代码": "10010394",
                        "开平": "平仓",
                        "买卖": "买",
                        "成交数量": 80,
                        "成交价格": 0.0045,
                    },
                ]
            ).to_csv(detail_path, index=False, encoding="utf-8-sig")

            old_fill = {
                "action": "open_short_straddle",
                "side": "short",
                "date": "2026-06-09",
                "call_code": "10010393",
                "put_code": "10010394",
                "strike": 1.75,
                "expiry": "2026-06-24",
                "call_qty": 80,
                "put_qty": 80,
                "entry_call_price": 0.0517,
                "entry_put_price": 0.0717,
                "contract_multiplier": 10000,
                "short_entry_regime": "absolute",
                "entry_option_value": 98720.0,
                "option_margin": 376608.0,
                "cash_delta": 277568.0,
            }
            metadata = {
                "10011740": {
                    "strike": 1.95,
                    "expiry": "2026-07-22",
                    "option_type": "C",
                    "contract_multiplier": 10000,
                },
                "10011749": {
                    "strike": 1.95,
                    "expiry": "2026-07-22",
                    "option_type": "P",
                    "contract_multiplier": 10000,
                },
            }

            with (
                mock.patch.object(account.storage, "account_db_path", return_value=db_path),
                mock.patch.object(holding_importer, "_resolve_trade_detail_file", return_value=detail_path),
                mock.patch.object(holding_importer, "_load_contract_metadata", return_value=metadata),
            ):
                account.record_fill("kc50etf", old_fill)
                result = holding_importer.import_holding_file(
                    "kc50etf",
                    file_path=holding_path,
                    date="2026-06-17",
                )
                local = account.load_account("kc50etf")

        self.assertEqual(
            [item["fill"]["action"] for item in result["applied"]],
            ["close_short_straddle", "open_short_straddle"],
        )
        self.assertEqual(result["warnings"], [])
        self.assertEqual(local.positions["short"]["call_code"], "10011740")
        self.assertEqual(local.positions["short"]["put_code"], "10011749")
        self.assertEqual(local.positions["short"]["call_qty"], 10)
        self.assertEqual(local.positions["short"]["put_qty"], 10)

    def test_holding_import_uses_trade_execution_time_not_export_time_for_snapshot_filter(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            root = Path(temp_dir)
            db_path = root / "account.sqlite"
            holding_path = root / "实时持仓(信息导出)_2026_06_29-15_07_20.csv"
            detail_path = root / "成交明细(信息导出)_2026_06_29-15_07_28.csv"
            pd.DataFrame(
                [
                    {
                        "合约代码": "10011721",
                        "合约名称": "500ETF购7月8500",
                        "买卖": "卖",
                        "持仓类型": "义务仓",
                        "总持仓": 5,
                        "今开仓": 5,
                        "开仓均价": 0.4551,
                        "最新价": 0.4630,
                        "占用保证金": 73471.0,
                    },
                    {
                        "合约代码": "10011723",
                        "合约名称": "500ETF购7月9000",
                        "买卖": "卖",
                        "持仓类型": "义务仓",
                        "总持仓": 1,
                        "今开仓": 0,
                        "开仓均价": 0.2499,
                        "最新价": 0.1724,
                        "占用保证金": 10692.2,
                    },
                    {
                        "合约代码": "10011732",
                        "合约名称": "500ETF沽7月9000",
                        "买卖": "卖",
                        "持仓类型": "义务仓",
                        "总持仓": 10,
                        "今开仓": 0,
                        "开仓均价": 0.2488,
                        "最新价": 0.3020,
                        "占用保证金": 143542.0,
                    },
                ]
            ).to_csv(holding_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [
                    {
                        "合约代码": "10011721",
                        "开平": "平仓",
                        "买卖": "买",
                        "成交数量": 5,
                        "成交价格": 0.4554,
                        "日期": "20260629",
                        "成交时间": "14:47:41",
                        "成交时间(日)": "20260629 14:47:41",
                    },
                    {
                        "合约代码": "10011723",
                        "开平": "平仓",
                        "买卖": "买",
                        "成交数量": 3,
                        "成交价格": 0.1660,
                        "日期": "20260629",
                        "成交时间": "14:47:56",
                        "成交时间(日)": "20260629 14:47:56",
                    },
                    {
                        "合约代码": "10011721",
                        "开平": "开仓",
                        "买卖": "卖",
                        "成交数量": 5,
                        "成交价格": 0.4551,
                        "日期": "20260629",
                        "成交时间": "14:48:15",
                        "成交时间(日)": "20260629 14:48:15",
                    },
                ]
            ).to_csv(detail_path, index=False, encoding="utf-8-sig")
            old_straddle = {
                "action": "open_short_straddle",
                "side": "short",
                "date": "2026-06-25",
                "call_code": "10011723",
                "put_code": "10011732",
                "strike": 9.0,
                "expiry": "2026-07-22",
                "call_qty": 4,
                "put_qty": 10,
                "entry_call_price": 0.2499,
                "entry_put_price": 0.2488,
                "contract_multiplier": 10000,
                "option_margin": 184884.0,
                "cash_delta": 0.0,
            }
            metadata = {
                "10011721": {
                    "strike": 8.5,
                    "expiry": "2026-07-22",
                    "option_type": "C",
                    "contract_multiplier": 10000,
                    "contract_symbol": "500ETF购7月8500",
                },
                "10011723": {
                    "strike": 9.0,
                    "expiry": "2026-07-22",
                    "option_type": "C",
                    "contract_multiplier": 10000,
                    "contract_symbol": "500ETF购7月9000",
                },
                "10011732": {
                    "strike": 9.0,
                    "expiry": "2026-07-22",
                    "option_type": "P",
                    "contract_multiplier": 10000,
                    "contract_symbol": "500ETF沽7月9000",
                },
            }

            with (
                mock.patch.object(account.storage, "account_db_path", return_value=db_path),
                mock.patch.object(holding_importer, "_resolve_trade_detail_file", return_value=detail_path),
                mock.patch.object(holding_importer, "_load_contract_metadata", return_value=metadata),
            ):
                account.record_fill("500etf", old_straddle)
                result = holding_importer.import_holding_file(
                    "500etf",
                    file_path=holding_path,
                    date="2026-06-29",
                )
                local = account.load_account("500etf")

        self.assertEqual(
            [item["fill"]["action"] for item in result["applied"]],
            ["rebalance_straddle_legs"],
        )
        self.assertEqual(local.positions["short"]["call_qty"], 1)
        self.assertEqual(local.positions["short"]["put_qty"], 10)
        self.assertTrue(
            any("unpaired option holding" in item["reason"] for item in result["warnings"])
        )

    def test_roll_fill_does_not_start_a_cooldown(self):
        fill = {
            "action": "roll_short_straddle",
            "side": "short",
            "date": "2026-06-24",
            "call_code": "CALL",
            "put_code": "PUT",
            "strike": 2.5,
            "expiry": "2026-07-22",
            "call_qty": 10,
            "put_qty": 10,
            "entry_call_price": 0.01,
            "entry_put_price": 0.01,
            "contract_multiplier": 10000,
        }

        state = account.AccountState(product="kc50etf")
        account._apply_fill(state, "kc50etf", fill)

        self.assertEqual(
            state.strategy_state.to_dict(),
            {
                "short_entry_cooldown_left": 0,
                "short_entry_cooldown_total_days": 0,
                "short_entry_cooldown_started_date": None,
            },
        )

    def test_legacy_roll_cooldown_state_is_ignored(self):
        state = account._strategy_state_from_payload(
            {
                "roll_cooldown_left": {"long": 5, "short": 5},
                "cooldown_total_days": {"long": 5, "short": 5},
                "cooldown_started_date": {
                    "long": "2026-07-08",
                    "short": "2026-07-08",
                },
            }
        )

        self.assertEqual(state, account.StrategyState())


if __name__ == "__main__":
    unittest.main()
