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
    summary["备注"] = "new"
    frames["持仓记录"] = portfolio_report._product_position_frame(
        payload["product"],
        frames["持仓记录"],
    )
    return {"账户总体情况": summary, **frames}


class LivePortfolioReportTest(unittest.TestCase):
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
                *account_report.DEFAULT_SUMMARY_REPORT_COLUMNS[1:],
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
        self.assertEqual(list(summary["估算权益"]), [1_000_000.0, 1_000_000.0])
        self.assertEqual(list(summary["账户Delta"]), [10.0, 10.0])
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
        summary[summary_cols[3]] = 8.544
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
        self.assertEqual(combined["账户总体情况"].iloc[0]["估算权益"], 1_000_000.0)
        self.assertEqual(combined["账户总体情况"].iloc[0]["策略名称"], "沪深300ETF华泰柏瑞")

    def test_merge_replaces_legacy_strategy_name_with_display_name(self):
        columns = [
            account_report.DEFAULT_SUMMARY_REPORT_COLUMNS[0],
            "策略名称",
            "合约代码",
            *account_report.DEFAULT_SUMMARY_REPORT_COLUMNS[1:],
            "备注",
        ]
        current = pd.DataFrame(
            [
                {
                    "日期": "2026-06-16",
                    "策略名称": "沪深300ETF华泰柏瑞",
                    "合约代码": "sh510300",
                    "估算权益": 2.0,
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
                    "估算权益": 1.0,
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
                        columns=portfolio_report._position_report_columns()
                    ),
                    "交易记录": pd.DataFrame(columns=account_report.TRADE_COLUMNS),
                },
                path,
                products=["300etf"],
            )

        summary = combined["账户总体情况"]
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0]["策略名称"], "沪深300ETF华泰柏瑞")
        self.assertEqual(summary.iloc[0]["估算权益"], 2.0)


if __name__ == "__main__":
    unittest.main()
