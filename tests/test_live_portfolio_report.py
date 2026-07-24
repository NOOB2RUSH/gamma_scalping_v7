import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from core.live import account_report, portfolio_report


def _payload(product, date, nav):
    summary = {column: 0.0 for column in account_report.SUMMARY_COLUMNS}
    summary.update(
        {
            "日期": date,
            "账户ID": "default",
            "初始资金": 1_000_000.0,
            "估算权益": nav,
            "账户Delta": 10.0,
        }
    )
    return {
        "product": product,
        "date": date,
        "summary": summary,
        "summary_history": pd.DataFrame([summary], columns=account_report.SUMMARY_COLUMNS),
    }


def _daily_frames(payload):
    product = payload["product"]
    date = payload["date"]
    summary = {
        column: 0.0 for column in account_report.DEFAULT_SUMMARY_REPORT_COLUMNS
    }
    summary.update({"日期": date, "估算权益": 1_000_000.0, "账户Delta": 10.0})
    return {
        "账户总体情况": pd.DataFrame([summary]),
        "持仓记录": pd.DataFrame(
            [{"日期": date, "合约代码": f"{product}-position"}],
            columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS,
        ),
        "交易记录": pd.DataFrame(
            [{"日期": date, "合约代码": f"{product}-trade"}],
            columns=account_report.TRADE_COLUMNS,
        ),
    }


def _report_frames(payload):
    frames = _daily_frames(payload)
    history = payload.get("summary_history")
    if isinstance(history, pd.DataFrame) and not history.empty:
        frames["账户总体情况"] = account_report._summary_report_frame(
            history,
            report_date=payload.get("date"),
        )
    return frames


def _portfolio_frames(payload):
    frames = _daily_frames(payload)
    summary = frames.pop("账户总体情况")
    summary["策略名称"] = portfolio_report._strategy_display_name(payload["product"])
    summary["合约代码"] = portfolio_report._strategy_contract_code(payload["product"])
    summary["AUM"] = None
    summary["备注"] = "new"
    summary = summary.reindex(columns=portfolio_report.SUMMARY_REPORT_COLUMNS)
    frames["持仓记录"] = portfolio_report._product_position_frame(
        payload["product"],
        frames["持仓记录"],
    )
    return {"账户总体情况": summary, **frames}


class LivePortfolioReportTest(unittest.TestCase):
    def test_terminal_summary_starts_with_used_snapshot_times(self):
        payload = {
            "date": "2026-06-22",
            "products": ["300etf", "kc50etf"],
            "subaccounts": {
                "300etf": {
                    "quote_snapshot": {"snapshot_stamp": "20260622_151219"}
                },
                "kc50etf": {
                    "quote_snapshot": {"snapshot_stamp": "20260622_151240"}
                },
            },
            "shared_cash": 1_000_000.0,
            "frames": {portfolio_report.SUMMARY_TEMPLATE_SHEET: pd.DataFrame()},
            "errors": {},
        }

        lines = portfolio_report.format_terminal_summary(payload)

        self.assertEqual(
            lines[0],
            "报告快照时间: 300etf=2026-06-22 15:12:19 "
            "kc50etf=2026-06-22 15:12:40",
        )

    def test_build_rejects_incomplete_product_report(self):
        with mock.patch.object(
            portfolio_report.account_report,
            "build_live_account_report",
            side_effect=[
                _payload("300etf", "2026-06-15", 1.0),
                ValueError("missing market"),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "complete unified account"):
                portfolio_report.build_portfolio_report(
                    products=("300etf", "500etf"),
                    source="none",
                    persist_history=False,
                )

    def test_build_rejects_mixed_valuation_dates(self):
        with mock.patch.object(
            portfolio_report.account_report,
            "build_live_account_report",
            side_effect=[
                _payload("300etf", "2026-06-15", 1.0),
                _payload("500etf", "2026-06-12", 1.0),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "same valuation date"):
                portfolio_report.build_portfolio_report(
                    products=("300etf", "500etf"),
                    source="none",
                    persist_history=False,
                )

    def test_combined_report_exactly_matches_account_report_columns(self):
        payloads = {
            "50etf": _payload("50etf", "2026-06-15", 1_000_000.0),
            "300etf": _payload("300etf", "2026-06-15", 1_000_000.0),
        }
        with (
            mock.patch.object(
                portfolio_report.account_report,
                "_report_frames",
                side_effect=_report_frames,
            ),
            mock.patch.object(
                portfolio_report.account_report,
                "_daily_report_frames",
                side_effect=_daily_frames,
            ),
        ):
            frames = portfolio_report._combined_daily_frames(payloads, {})

        self.assertEqual(list(frames), ["账户总体情况", "持仓记录", "交易记录"])
        self.assertEqual(
            list(frames["账户总体情况"].columns),
            [
                account_report.DEFAULT_SUMMARY_REPORT_COLUMNS[0],
                "策略名称",
                "合约代码",
                *portfolio_report.PORTFOLIO_SUMMARY_VALUE_COLUMNS[1:],
                "备注",
            ],
        )
        self.assertEqual(
            list(frames["持仓记录"].columns),
            [
                account_report.DEFAULT_POSITION_REPORT_COLUMNS[0],
                "策略名称",
                *account_report.DEFAULT_POSITION_REPORT_COLUMNS[1:],
            ],
        )
        self.assertEqual(list(frames["交易记录"].columns), account_report.TRADE_COLUMNS)
        summary = frames["账户总体情况"]
        self.assertEqual(
            set(summary["策略名称"]),
            {"上证50ETF华夏", "沪深300ETF华泰柏瑞"},
        )
        self.assertEqual(set(summary["合约代码"]), {"sh510050", "sh510300"})
        self.assertIn("AUM", summary.columns)
        self.assertNotIn("估算权益", summary.columns)
        self.assertEqual(list(summary["账户Delta"]), [10.0, 10.0])
        self.assertTrue(summary["备注"].isna().all())
        self.assertNotIn("50etf", frames)
        self.assertNotIn("300etf", frames)
        self.assertNotIn("品种", frames["持仓记录"].columns)

    def test_product_summary_keeps_greeks_history_separate(self):
        def payload(product, greeks_pnl):
            rows = []
            for date, value in [
                ("2026-06-12", 0.0),
                ("2026-06-15", greeks_pnl),
            ]:
                row = {column: 0.0 for column in account_report.SUMMARY_COLUMNS}
                row.update(
                    {
                        "日期": date,
                        "账户ID": "default",
                        "初始资金": 1_000_000.0,
                        "估算权益": 1_000_000.0,
                        "对冲单日盈亏": 11.0 if date == "2026-06-15" else 0.0,
                        "ETF单日盈亏": 11.0 if date == "2026-06-15" else 0.0,
                        "总单日盈亏": 100.0 if date == "2026-06-15" else 0.0,
                        "当日手续费": 2.0 if date == "2026-06-15" else 0.0,
                        "单日GreeksPnL": value,
                    }
                )
                rows.append(row)
            return {
                "product": product,
                "date": "2026-06-15",
                "summary": rows[-1],
                "summary_history": pd.DataFrame(
                    rows,
                    columns=account_report.SUMMARY_COLUMNS,
                ),
            }

        payloads = {
            "300etf": payload("300etf", 0.0),
            "kc50etf": payload("kc50etf", -23676.21),
        }
        with (
            mock.patch.object(
                portfolio_report.account_report,
                "_report_frames",
                side_effect=_report_frames,
            ),
            mock.patch.object(
                portfolio_report.account_report,
                "_daily_report_frames",
                side_effect=_daily_frames,
            ),
        ):
            frames = portfolio_report._combined_daily_frames(payloads, {})

        summary = frames["账户总体情况"]
        kc50_summary = summary[summary["策略名称"].eq("科创50ETF华夏")]
        etf300_summary = summary[summary["策略名称"].eq("沪深300ETF华泰柏瑞")]
        self.assertEqual(list(kc50_summary["日期"]), ["2026-06-12", "2026-06-15"])
        self.assertEqual(list(etf300_summary["日期"]), ["2026-06-12", "2026-06-15"])
        self.assertAlmostEqual(kc50_summary.iloc[-1]["单日GreeksPnL"], -23676.21)
        self.assertAlmostEqual(etf300_summary.iloc[-1]["单日GreeksPnL"], 0.0)
        self.assertAlmostEqual(kc50_summary.iloc[-1]["ETF单日盈亏"], 11.0)
        self.assertAlmostEqual(kc50_summary.iloc[-1]["净单日盈亏"], 98.0)

    def test_combined_current_summary_uses_same_position_pnl_as_workbook(self):
        payload = _payload("500etf", "2026-07-21", 779_200.0)
        summary = pd.DataFrame(
            [
                {
                    "日期": "2026-07-21",
                    "AUM": 779_200.0,
                    "当日手续费": 43.19964,
                    "期权单日盈亏": 170.0,
                    "ETF单日盈亏": -21_212.4,
                    "总单日盈亏(手续费前)": -21_042.4,
                    "净单日盈亏": -21_085.59964,
                    "单日GreeksPnL": 71.8003,
                }
            ],
            columns=account_report.DEFAULT_SUMMARY_REPORT_COLUMNS,
        )
        positions = pd.DataFrame(
            [
                {
                    "日期": "2026-07-21",
                    "合约代码": "10012055",
                    "持仓盈亏": 0.0,
                    "交易盈亏": -60.0,
                    "到期日": "2026-08-26",
                },
                {
                    "日期": "2026-07-21",
                    "合约代码": "10012064",
                    "持仓盈亏": 0.0,
                    "交易盈亏": 230.0,
                    "到期日": "2026-08-26",
                },
                {
                    "日期": "2026-07-21",
                    "合约代码": "510500",
                    "持仓盈亏": 0.0,
                    "交易盈亏": -98.4,
                    "到期日": None,
                },
            ],
            columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS,
        )
        daily = {
            "账户总体情况": summary,
            "持仓记录": positions,
            "交易记录": pd.DataFrame(columns=account_report.TRADE_COLUMNS),
        }
        with (
            mock.patch.object(
                portfolio_report.account_report,
                "_report_frames",
                return_value=daily,
            ),
            mock.patch.object(
                portfolio_report.account_report,
                "_daily_report_frames",
                return_value=daily,
            ),
        ):
            frames = portfolio_report._combined_daily_frames(
                {"500etf": payload},
                {},
            )

        row = frames["账户总体情况"].iloc[-1]
        self.assertAlmostEqual(row["期权单日盈亏"], 170.0)
        self.assertAlmostEqual(row["ETF单日盈亏"], -98.4)
        self.assertAlmostEqual(row["总单日盈亏(手续费前)"], 71.6)
        self.assertAlmostEqual(row["净单日盈亏"], 28.40036)
        self.assertAlmostEqual(row["单日GreeksPnL"], 71.8003)

    def test_combined_summary_aum_uses_final_open_position_after_roll(self):
        payload = _payload("300etf", "2026-06-22", 1_000_000.0)
        summary = pd.DataFrame(
            [
                {
                    "日期": "2026-06-22",
                    "估算权益": 1_000_000.0,
                    "账户Delta": 10.0,
                }
            ],
            columns=account_report.DEFAULT_SUMMARY_REPORT_COLUMNS,
        )
        position = pd.DataFrame(
            [
                {
                    "日期": "2026-06-22",
                    "合约代码": "old-call",
                    "合约名称": "300ETF购7月4900",
                    "交易方向": "空",
                    "总持仓张数": 0,
                    "AUM": 900_000.0,
                    "到期日": "2026-07-22",
                },
                {
                    "日期": "2026-06-22",
                    "合约代码": "new-call",
                    "合约名称": "300ETF购7月5000",
                    "交易方向": "空",
                    "总持仓张数": 12,
                    "AUM": 596_880.0,
                    "到期日": "2026-07-22",
                },
                {
                    "日期": "2026-06-22",
                    "合约代码": "new-put",
                    "合约名称": "300ETF沽7月5000",
                    "交易方向": "空",
                    "总持仓张数": 12,
                    "AUM": 596_880.0,
                    "到期日": "2026-07-22",
                },
            ],
            columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS,
        )
        trade = pd.DataFrame(columns=account_report.TRADE_COLUMNS)
        product_frames = {
            "账户总体情况": summary,
            "持仓记录": position,
            "交易记录": trade,
        }
        with (
            mock.patch.object(
                portfolio_report.account_report,
                "_report_frames",
                return_value=product_frames,
            ),
            mock.patch.object(
                portfolio_report.account_report,
                "_daily_report_frames",
                return_value=product_frames,
            ),
        ):
            frames = portfolio_report._combined_daily_frames({"300etf": payload}, {})

        summary_row = frames["账户总体情况"].iloc[0]
        self.assertAlmostEqual(summary_row["AUM"], 596_880.0)
        self.assertNotIn("估算权益", frames["账户总体情况"].columns)

    def test_combined_report_filters_rows_before_product_reset(self):
        payload = _payload("300etf", "2026-06-15", 1_000_000.0)
        old_summary = payload["summary"].copy()
        old_summary["日期"] = "2026-06-12"
        payload["summary_history"] = pd.DataFrame(
            [old_summary, payload["summary"]],
            columns=account_report.SUMMARY_COLUMNS,
        )
        payload["_portfolio_reset_date"] = pd.Timestamp("2026-06-15")

        with (
            mock.patch.object(
                portfolio_report.account_report,
                "_report_frames",
                side_effect=_report_frames,
            ),
            mock.patch.object(
                portfolio_report.account_report,
                "_daily_report_frames",
                side_effect=_daily_frames,
            ),
        ):
            frames = portfolio_report._combined_daily_frames({"300etf": payload}, {})

        summary = frames["账户总体情况"]
        self.assertEqual(list(summary["日期"]), ["2026-06-15"])
        self.assertEqual(
            set(summary["策略名称"]),
            {"沪深300ETF华泰柏瑞"},
        )

    def test_backfills_aum_for_existing_historical_position_rows(self):
        cols = account_report.DEFAULT_POSITION_REPORT_COLUMNS
        date_col, code_col, name_col, direction_col = cols[:4]
        qty_col = cols[4]
        aum_col = cols[5]
        expiry_col = cols[11]
        frame = pd.DataFrame(
            [
                {
                    date_col: "2026-06-15",
                    code_col: "10011721",
                    name_col: "500ETF购7月8500",
                    direction_col: "空",
                    qty_col: 7,
                    aum_col: None,
                    expiry_col: "2026-07-22",
                },
                {
                    date_col: "2026-06-15",
                    code_col: "10011730",
                    name_col: "500ETF沽7月8500",
                    direction_col: "空",
                    qty_col: 10,
                    aum_col: None,
                    expiry_col: "2026-07-22",
                },
            ],
            columns=cols,
        )
        summary_cols = account_report.SUMMARY_COLUMNS
        summary = {column: 0.0 for column in summary_cols}
        summary[summary_cols[0]] = "2026-06-15"
        summary["标的价格"] = 8.544
        payloads = {
            "500etf": {
                "date": "2026-06-15",
                "summary_history": pd.DataFrame([summary], columns=summary_cols),
            }
        }

        result = portfolio_report._backfill_position_aum(frame, payloads)

        for value in result[aum_col]:
            self.assertAlmostEqual(value, 854_400.0)

    def test_backfills_historical_position_holding_pnl(self):
        cols = account_report.DEFAULT_POSITION_REPORT_COLUMNS
        date_col, code_col, name_col, direction_col = cols[:4]
        qty_col = cols[4]
        change_col = cols[6]
        latest_col = cols[7]
        cost_col = cols[8]
        holding_pnl_col = cols[9]
        expiry_col = cols[11]
        frame = pd.DataFrame(
            [
                {
                    date_col: "2026-06-12",
                    code_col: "10010393",
                    name_col: "科创50购6月1750",
                    direction_col: "空",
                    qty_col: 80,
                    change_col: 80,
                    latest_col: 0.04915,
                    cost_col: 0.0517,
                    holding_pnl_col: 0.0,
                    expiry_col: "2026-06-24",
                },
                {
                    date_col: "2026-06-15",
                    code_col: "10010393",
                    name_col: "科创50购6月1750",
                    direction_col: "空",
                    qty_col: 80,
                    change_col: 80,
                    latest_col: 0.1035,
                    cost_col: 0.0517,
                    holding_pnl_col: 0.0,
                    expiry_col: "2026-06-24",
                },
                {
                    date_col: "2026-06-12",
                    code_col: "588000",
                    name_col: "588000.XSHG",
                    direction_col: "多",
                    qty_col: 53100,
                    change_col: 53100,
                    latest_col: 1.756,
                    cost_col: 1.756,
                    holding_pnl_col: 0.0,
                    expiry_col: None,
                },
                {
                    date_col: "2026-06-15",
                    code_col: "588000",
                    name_col: "588000.XSHG",
                    direction_col: "多",
                    qty_col: 53100,
                    change_col: 53100,
                    latest_col: 1.844,
                    cost_col: 1.756,
                    holding_pnl_col: 0.0,
                    expiry_col: None,
                },
            ],
            columns=cols,
        )

        result = portfolio_report._backfill_position_holding_pnl(
            frame,
            {"kc50etf": {"date": "2026-06-15"}},
        )

        self.assertAlmostEqual(result.iloc[0][holding_pnl_col], 0.0)
        self.assertAlmostEqual(result.iloc[1][holding_pnl_col], -43_480.0)
        self.assertEqual(result.iloc[1][change_col], 0)
        self.assertAlmostEqual(result.iloc[2][holding_pnl_col], 0.0)
        self.assertAlmostEqual(result.iloc[3][holding_pnl_col], 4_672.8)
        self.assertEqual(result.iloc[3][change_col], 0)

    def test_backfill_closed_old_position_uses_trade_price(self):
        cols = account_report.DEFAULT_POSITION_REPORT_COLUMNS
        frame = pd.DataFrame(
            [
                {
                    "日期": "2026-06-18",
                    "合约代码": "10011741",
                    "合约名称": "科创50购7月2000",
                    "交易方向": "空",
                    "总持仓张数": 10,
                    "今日变化": 10,
                    "最新价": 0.1170,
                    "持仓均价": 0.1189,
                    "持仓盈亏": 0.0,
                    "到期日": "2026-07-22",
                },
                {
                    "日期": "2026-06-22",
                    "合约代码": "10011741",
                    "合约名称": "科创50购7月2000",
                    "交易方向": "空",
                    "总持仓张数": 0,
                    "今日变化": -10,
                    "最新价": 0.1354,
                    "持仓均价": 0.1189,
                    "持仓盈亏": -1840.0,
                    "到期日": "2026-07-22",
                },
            ],
            columns=cols,
        )
        payloads = {
            "kc50etf": {
                "date": "2026-06-22",
                "trade_rows": [
                    {
                        "日期": "2026-06-22",
                        "合约代码": "10011741",
                        "买卖": "买",
                        "成交数量": 10,
                        "成交价格": 0.1349,
                    }
                ],
            }
        }

        result = portfolio_report._backfill_position_holding_pnl(frame, payloads)

        self.assertAlmostEqual(result.iloc[1]["持仓盈亏"], -1840.0)
        self.assertAlmostEqual(result.iloc[1]["交易盈亏"], 50.0)

    def test_write_uses_template_then_appends_new_date(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template.xlsx"
            with pd.ExcelWriter(template, engine="openpyxl") as writer:
                pd.DataFrame(
                    [{"日期": "2026-06-12", "估算权益": 1.0, "备注": "old"}],
                    columns=[*account_report.DEFAULT_SUMMARY_REPORT_COLUMNS, "备注"],
                ).to_excel(writer, sheet_name="账户总体情况", index=False)
                pd.DataFrame(
                    [{"日期": "2026-06-12", "合约代码": "old-position"}],
                    columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS,
                ).to_excel(writer, sheet_name="持仓记录", index=False)
                pd.DataFrame(
                    [{"日期": "2026-06-12", "合约代码": "old-trade"}],
                    columns=account_report.TRADE_COLUMNS,
                ).to_excel(writer, sheet_name="交易记录", index=False)

            out_dir = root / "out"
            out_dir.mkdir()
            existing = out_dir / "20260614_150000_account_report.xlsx"
            with pd.ExcelWriter(existing, engine="openpyxl") as writer:
                pd.DataFrame(
                    [{"日期": "2026-06-12", "估算权益": 2.0, "备注": "old-product"}],
                    columns=[*account_report.DEFAULT_SUMMARY_REPORT_COLUMNS, "备注"],
                ).to_excel(writer, sheet_name="300etf", index=False)
                pd.DataFrame(
                    [{"日期": "2026-06-12", "合约代码": "old-position"}],
                    columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS,
                ).to_excel(writer, sheet_name="持仓记录", index=False)
                pd.DataFrame(
                    [{"日期": "2026-06-12", "合约代码": "old-trade"}],
                    columns=account_report.TRADE_COLUMNS,
                ).to_excel(writer, sheet_name="交易记录", index=False)

            frames = _portfolio_frames(_payload("300etf", "2026-06-15", 1.0))
            payload = {
                "account_id": "default",
                "date": "2026-06-15",
                "dates": ["2026-06-15"],
                "products": ["300etf"],
                "errors": {},
                "frames": frames,
            }
            with (
                mock.patch.object(portfolio_report, "TEMPLATE_REPORT_PATH", template),
                mock.patch.object(
                    portfolio_report.storage,
                    "portfolio_output_dir",
                    return_value=out_dir,
                ),
                mock.patch.object(
                    portfolio_report.storage,
                    "local_now_stamp",
                    return_value="20260615_160000",
                ),
            ):
                paths = portfolio_report.write_portfolio_report(payload)
            workbook = account_report._read_report_workbook(paths["total_excel"])

        self.assertEqual(list(workbook), ["账户总体情况", "持仓记录", "交易记录"])
        self.assertNotIn("300etf", workbook)
        self.assertEqual(len(workbook["账户总体情况"]), 2)
        self.assertEqual(set(workbook["账户总体情况"]["策略名称"]), {"沪深300ETF华泰柏瑞"})
        self.assertEqual(
            set(workbook["持仓记录"]["合约代码"]),
            {"old-position", "300etf-position"},
        )
        self.assertEqual(list(workbook["交易记录"].columns), account_report.TRADE_COLUMNS)

    def test_merge_deduplicates_trades_by_trade_id(self):
        frames = _portfolio_frames(_payload("300etf", "2026-06-15", 1.0))
        trade = frames["交易记录"].iloc[0].copy()
        trade[account_report.TRADE_COLUMNS[5]] = "trade-1"
        frames["交易记录"] = pd.DataFrame(
            [trade, trade],
            columns=account_report.TRADE_COLUMNS,
        )

        with mock.patch.object(
            portfolio_report,
            "TEMPLATE_REPORT_PATH",
            Path("missing-template.xlsx"),
        ):
            combined = portfolio_report._merge_with_existing(frames, None)

        self.assertEqual(len(combined["交易记录"]), 1)
        self.assertEqual(
            combined["交易记录"].iloc[0][account_report.TRADE_COLUMNS[5]],
            "trade-1",
        )

    def test_merge_ignores_legacy_account_summary_sheet(self):
        with TemporaryDirectory() as temp_dir:
            template = Path(temp_dir) / "template.xlsx"
            columns = [*account_report.DEFAULT_SUMMARY_REPORT_COLUMNS, "备注"]
            with pd.ExcelWriter(template, engine="openpyxl") as writer:
                pd.DataFrame(
                    [{"日期": "2026-06-12", "估算权益": 10_011_915.34}],
                    columns=columns,
                ).to_excel(writer, sheet_name="账户总体情况", index=False)
                pd.DataFrame(columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS).to_excel(
                    writer, sheet_name="持仓记录", index=False
                )
                pd.DataFrame(columns=account_report.TRADE_COLUMNS).to_excel(
                    writer, sheet_name="交易记录", index=False
                )
            current = _portfolio_frames(_payload("300etf", "2026-06-15", 1.0))
            existing = Path(temp_dir) / "existing.xlsx"
            with pd.ExcelWriter(existing, engine="openpyxl") as writer:
                pd.DataFrame(
                    [{"日期": "2026-06-12", "估算权益": 13_011_915.34}],
                    columns=columns,
                ).to_excel(writer, sheet_name="账户总体情况", index=False)
                pd.DataFrame(columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS).to_excel(
                    writer, sheet_name="持仓记录", index=False
                )
                pd.DataFrame(columns=account_report.TRADE_COLUMNS).to_excel(
                    writer, sheet_name="交易记录", index=False
                )

            with mock.patch.object(portfolio_report, "TEMPLATE_REPORT_PATH", template):
                combined = portfolio_report._merge_with_existing(current, existing)

        self.assertIn("账户总体情况", combined)
        self.assertNotIn("300etf", combined)
        self.assertIn("AUM", combined["账户总体情况"].columns)
        self.assertNotIn("估算权益", combined["账户总体情况"].columns)
        self.assertEqual(combined["账户总体情况"].iloc[0]["策略名称"], "沪深300ETF华泰柏瑞")

    def test_merge_replaces_legacy_strategy_name_with_display_name(self):
        columns = [
            account_report.DEFAULT_SUMMARY_REPORT_COLUMNS[0],
            "策略名称",
            "合约代码",
            *portfolio_report.PORTFOLIO_SUMMARY_VALUE_COLUMNS[1:],
            "备注",
        ]
        current = pd.DataFrame(
            [
                {
                    "日期": "2026-06-16",
                    "策略名称": "沪深300ETF华泰柏瑞",
                    "合约代码": "sh510300",
                    "AUM": 2.0,
                }
            ],
            columns=columns,
        )
        existing = pd.DataFrame(
            [
                {
                    "日期": "2026-06-16",
                    "策略名称": "300etf",
                    "合约代码": "sh510300",
                    "AUM": 1.0,
                }
            ],
            columns=columns,
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "existing.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                existing.to_excel(writer, sheet_name="账户总体情况", index=False)
                pd.DataFrame(columns=account_report.DEFAULT_POSITION_REPORT_COLUMNS).to_excel(
                    writer, sheet_name="持仓记录", index=False
                )
                pd.DataFrame(columns=account_report.TRADE_COLUMNS).to_excel(
                    writer, sheet_name="交易记录", index=False
                )
            combined = portfolio_report._merge_with_existing(
                {
                    "账户总体情况": current,
                    "持仓记录": pd.DataFrame(
                        columns=portfolio_report.POSITION_REPORT_COLUMNS
                    ),
                    "交易记录": pd.DataFrame(columns=account_report.TRADE_COLUMNS),
                },
                path,
                products=["300etf"],
            )

        summary = combined["账户总体情况"]
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["策略名称"], "沪深300ETF华泰柏瑞")
        self.assertEqual(summary.iloc[0]["AUM"], 2.0)

    def test_merge_filters_existing_rows_before_product_reset(self):
        current = _portfolio_frames(_payload("300etf", "2026-06-15", 1.0))
        payloads = {
            "300etf": {
                "_portfolio_reset_date": pd.Timestamp("2026-06-15"),
            }
        }
        columns = current["账户总体情况"].columns
        old_summary = pd.DataFrame(
            [
                {
                    "日期": "2026-06-12",
                    "策略名称": "沪深300ETF华泰柏瑞",
                    "合约代码": "sh510300",
                    "AUM": 9.0,
                }
            ],
            columns=columns,
        )
        old_position = pd.DataFrame(
            [
                {
                    "日期": "2026-06-12",
                    "策略名称": "沪深300ETF华泰柏瑞",
                    "合约代码": "510300",
                    "合约名称": "510300.XSHG",
                }
            ],
            columns=portfolio_report.POSITION_REPORT_COLUMNS,
        )
        old_trade = pd.DataFrame(
            [
                {
                    "日期": "2026-06-12",
                    "合约代码": "510300",
                    "合约名称": "510300.XSHG",
                }
            ],
            columns=account_report.TRADE_COLUMNS,
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "existing.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                old_summary.to_excel(writer, sheet_name="账户总体情况", index=False)
                old_position.to_excel(writer, sheet_name="持仓记录", index=False)
                old_trade.to_excel(writer, sheet_name="交易记录", index=False)
            with mock.patch.object(
                portfolio_report,
                "TEMPLATE_REPORT_PATH",
                Path(temp_dir) / "missing-template.xlsx",
            ):
                combined = portfolio_report._merge_with_existing(
                    current,
                    path,
                    payloads=payloads,
                    products=["300etf"],
                )

        self.assertEqual(list(combined["账户总体情况"]["日期"]), ["2026-06-15"])
        self.assertEqual(
            set(combined["持仓记录"]["合约代码"]),
            {"300etf-position"},
        )
        self.assertEqual(
            set(combined["交易记录"]["合约代码"]),
            {"300etf-trade"},
        )

    def test_merge_backfills_historical_etf_trade_pnl_from_merged_trades(self):
        current = {
            "账户总体情况": pd.DataFrame(columns=portfolio_report.SUMMARY_REPORT_COLUMNS),
            "持仓记录": pd.DataFrame(columns=portfolio_report.POSITION_REPORT_COLUMNS),
            "交易记录": pd.DataFrame(columns=account_report.TRADE_COLUMNS),
        }
        position_rows = pd.DataFrame(
            [
                {
                    "日期": "2026-06-26",
                    "策略名称": "科创50ETF华夏",
                    "合约代码": "588000",
                    "合约名称": "588000.XSHG",
                    "交易方向": "多",
                    "总持仓张数": 5800,
                    "今日变化": 5200,
                    "最新价": 2.133,
                    "持仓均价": 2.144258,
                    "持仓盈亏": -27.0,
                    "交易盈亏": 0.0,
                },
                {
                    "日期": "2026-06-29",
                    "策略名称": "科创50ETF华夏",
                    "合约代码": "588000",
                    "合约名称": "588000.XSHG",
                    "交易方向": "多",
                    "总持仓张数": 6900,
                    "今日变化": 1100,
                    "最新价": 2.248,
                    "持仓均价": 2.157927,
                    "持仓盈亏": 667.0,
                    "交易盈亏": 0.0,
                },
            ],
            columns=portfolio_report.POSITION_REPORT_COLUMNS,
        )
        trade_rows = pd.DataFrame(
            [
                {
                    "日期": "2026-06-29",
                    "合约代码": "588000",
                    "合约名称": "科创50ETF华夏",
                    "买卖": "买",
                    "成交价格": 2.23,
                    "成交数量": 1100,
                    "成交时间": "14:46:31",
                    "类型": "ETF对冲",
                }
            ],
            columns=account_report.TRADE_COLUMNS,
        )
        summary_rows = pd.DataFrame(
            [
                {
                    "日期": "2026-06-29",
                    "策略名称": "科创50ETF华夏",
                    "合约代码": "sh588000",
                    "AUM": 1_000_000.0,
                    "当日手续费": 0.2,
                    "期权单日盈亏": 0.0,
                    "ETF单日盈亏": 667.0,
                    "总单日盈亏(手续费前)": 667.0,
                    "净单日盈亏": 666.8,
                    "单日盈亏/AUM": 0.000667,
                }
            ],
            columns=portfolio_report.SUMMARY_REPORT_COLUMNS,
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "existing.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                summary_rows.to_excel(writer, sheet_name="账户总体情况", index=False)
                position_rows.to_excel(writer, sheet_name="持仓记录", index=False)
                trade_rows.to_excel(writer, sheet_name="交易记录", index=False)
            with mock.patch.object(
                portfolio_report,
                "TEMPLATE_REPORT_PATH",
                Path(temp_dir) / "missing-template.xlsx",
            ):
                combined = portfolio_report._merge_with_existing(
                    current,
                    path,
                    payloads={"kc50etf": {"trade_rows": []}},
                    products=["kc50etf"],
                )

        positions = combined["持仓记录"]
        row = positions.loc[
            positions["日期"].astype(str).eq("2026-06-29")
            & positions["合约代码"].astype(str).eq("588000")
        ].iloc[0]
        self.assertAlmostEqual(row["持仓盈亏"], 667.0)
        self.assertAlmostEqual(row["交易盈亏"], 19.8)
        summary = combined["账户总体情况"].iloc[0]
        self.assertAlmostEqual(summary["期权单日盈亏"], 0.0)
        self.assertAlmostEqual(summary["ETF单日盈亏"], 686.8)
        self.assertAlmostEqual(summary["总单日盈亏(手续费前)"], 686.8)
        self.assertAlmostEqual(summary["净单日盈亏"], 686.6)
        self.assertAlmostEqual(summary["单日盈亏/AUM"], 0.0006868)

    def test_backfill_position_pnl_includes_ex_dividend_cash_distribution(self):
        positions = pd.DataFrame(
            [
                {
                    "日期": "2026-07-14",
                    "策略名称": "中证500ETF南方",
                    "合约代码": "510500",
                    "合约名称": "中证500ETF南方",
                    "交易方向": "多",
                    "总持仓张数": 34000,
                    "今日变化": 0,
                    "最新价": 8.413,
                    "持仓均价": 8.413,
                    "持仓盈亏": 0.0,
                    "交易盈亏": 0.0,
                    "到期日": None,
                },
                {
                    "日期": "2026-07-15",
                    "策略名称": "中证500ETF南方",
                    "合约代码": "510500",
                    "合约名称": "中证500ETF南方",
                    "交易方向": "多",
                    "总持仓张数": 0,
                    "今日变化": -34000,
                    "最新价": 8.151,
                    "持仓均价": 8.413,
                    "持仓盈亏": -3842.0,
                    "交易盈亏": -238.0,
                    "到期日": None,
                },
            ],
            columns=portfolio_report.POSITION_REPORT_COLUMNS,
        )
        trades = pd.DataFrame(
            [
                {
                    "日期": "2026-07-15",
                    "合约代码": "510500",
                    "合约名称": "中证500ETF南方",
                    "买卖": "卖",
                    "成交价格": 8.144,
                    "成交数量": 34000,
                    "类型": "ETF对冲",
                }
            ],
            columns=account_report.TRADE_COLUMNS,
        )

        result = portfolio_report._backfill_position_holding_pnl(
            positions,
            payloads={"500etf": {"trade_rows": []}},
            trade_frame=trades,
        )
        row = result.loc[result["日期"].astype(str).eq("2026-07-15")].iloc[0]

        self.assertAlmostEqual(row["持仓盈亏"], -3842.0)
        self.assertAlmostEqual(row["交易盈亏"], -238.0)

    def test_product_summary_copies_dividend_note_to_existing_remark_column(self):
        summary = pd.DataFrame(
            [{"日期": "2026-07-15"}],
            columns=account_report.DEFAULT_SUMMARY_REPORT_COLUMNS,
        )
        payload = {
            "product": "500etf",
            "account_id": "default",
            "date": "2026-07-15",
            "position_history": pd.DataFrame(
                [
                    {
                        "日期": "2026-07-14",
                        "账户ID": "default",
                        "方向": "hedge",
                        "总持仓": 34000,
                    }
                ]
            ),
            "summary_history": summary,
        }

        result = portfolio_report._product_summary_frame(
            "500etf",
            payload,
            {
                "账户总体情况": summary,
                "持仓记录": pd.DataFrame(),
            },
        )

        self.assertIn("应收股息5,066.00元", result.iloc[0]["备注"])


if __name__ == "__main__":
    unittest.main()
