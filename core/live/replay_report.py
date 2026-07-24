from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pandas as pd


ACTUAL_SUMMARY_COLUMNS = {
    "日期": "date",
    "当日手续费": "actual_fee",
    "期权单日盈亏": "actual_option_daily_pnl",
    "ETF单日盈亏": "actual_etf_daily_pnl",
    "总单日盈亏": "actual_gross_daily_pnl",
    "净单日盈亏": "actual_net_daily_pnl",
    "账户Delta": "actual_account_delta",
}
UNDERLYING_BY_PRODUCT = {
    "50etf": "510050.XSHG",
    "300etf": "510300.XSHG",
    "500etf": "510500.XSHG",
    "kc50etf": "588000.XSHG",
}


def build_live_comparison(
    product: str,
    theoretical_daily: pd.DataFrame,
    theoretical_trades: pd.DataFrame,
    *,
    account_id="default",
    state_root="state/live",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare snapshot replay with broker-derived live account history.

    Monetary attribution is limited to quantities that can be matched exactly
    by date, instrument and direction. Unmatched/delayed actions and ending
    position differences are reported explicitly and left unvalued; they remain
    in the unexplained residual instead of being assigned a fabricated PnL.
    """
    state_dir = Path(state_root) / product
    actual_daily = _load_actual_daily(
        state_dir / f"{account_id}_account_summary_history.csv"
    )
    start = pd.to_datetime(theoretical_daily["date"]).min()
    end = pd.to_datetime(theoretical_daily["date"]).max()
    actual_fills = _load_actual_fills_read_only(
        state_dir / "account.sqlite",
        account_id=account_id,
        start=start,
        end=end,
    )
    theoretical_legs = _expand_theoretical_trades(theoretical_trades)
    actual_legs = _expand_actual_fills(actual_fills, product=product)
    matches = _match_execution_legs(theoretical_legs, actual_legs)
    match_daily = _summarize_matches(matches)
    theoretical_fee = _theoretical_fee_by_date(theoretical_trades)
    actual_positions = _load_actual_position_fingerprints(
        state_dir / f"{account_id}_position_history.csv"
    )

    comparison = theoretical_daily.copy()
    comparison["date"] = pd.to_datetime(comparison["date"]).dt.normalize()
    comparison = comparison.merge(actual_daily, on="date", how="left")
    comparison = comparison.merge(theoretical_fee, on="date", how="left")
    comparison = comparison.merge(match_daily, on="date", how="left")
    comparison = comparison.merge(actual_positions, on="date", how="left")
    numeric_zero = [
        "theoretical_fee",
        "execution_slippage_pnl",
        "matched_leg_count",
        "delayed_leg_count",
        "unexecuted_leg_count",
        "unexpected_actual_leg_count",
        "unmatched_theoretical_leg_count",
        "unmatched_actual_leg_count",
        "unexecuted_notional",
        "delayed_notional",
        "unexecuted_or_delayed_notional",
    ]
    for column in numeric_zero:
        if column not in comparison:
            comparison[column] = 0.0
        comparison[column] = pd.to_numeric(
            comparison[column], errors="coerce"
        ).fillna(0.0)
    comparison["fee_difference_pnl"] = (
        comparison["theoretical_fee"]
        - pd.to_numeric(comparison["actual_fee"], errors="coerce").fillna(0.0)
    )
    comparison["actual_minus_theoretical_pnl"] = (
        pd.to_numeric(comparison["actual_net_daily_pnl"], errors="coerce")
        - pd.to_numeric(comparison["theoretical_daily_pnl"], errors="coerce")
    )
    first_replay_date = comparison["date"].min()
    comparison["comparison_scope_status"] = comparison["date"].map(
        lambda value: (
            "interval_start_partial_theoretical_day"
            if value == first_replay_date
            else "comparable_snapshot_to_snapshot_day"
        )
    )
    comparison["pnl_comparable"] = comparison["date"] != first_replay_date
    comparison["position_difference"] = comparison.apply(
        _position_difference,
        axis=1,
    )
    comparison["unexecuted_or_delayed_status"] = comparison.apply(
        lambda row: (
            "present_not_monetarily_valued"
            if row["unmatched_theoretical_leg_count"] > 0
            or row["unmatched_actual_leg_count"] > 0
            else "none"
        ),
        axis=1,
    )
    comparison["unexecuted_status"] = comparison["unexecuted_leg_count"].map(
        lambda value: "present_not_monetarily_valued" if value > 0 else "none"
    )
    comparison["delayed_execution_status"] = comparison["delayed_leg_count"].map(
        lambda value: "present_not_monetarily_valued" if value > 0 else "none"
    )
    comparison["position_difference_status"] = comparison[
        "position_difference"
    ].map(
        lambda value: (
            "present_not_monetarily_valued" if bool(value) else "none"
        )
    )
    comparison["unexplained_residual_pnl"] = (
        comparison["actual_minus_theoretical_pnl"]
        - comparison["execution_slippage_pnl"]
        - comparison["fee_difference_pnl"]
    )
    comparison["attribution_note"] = comparison.apply(
        _attribution_note,
        axis=1,
    )
    return comparison, matches, actual_fills


def write_live_comparison_report(
    product: str,
    comparison: pd.DataFrame,
    matches: pd.DataFrame,
    theoretical_trades: pd.DataFrame,
    actual_fills: pd.DataFrame,
    *,
    output_dir,
    metadata: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "live_straddle_comparison.csv"
    xlsx_path = output_dir / "live_straddle_comparison.xlsx"
    json_path = output_dir / "live_straddle_comparison.json"
    md_path = output_dir / "live_straddle_comparison.md"
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path) as writer:
        comparison.to_excel(writer, sheet_name="理论实际对比", index=False)
        matches.to_excel(writer, sheet_name="成交匹配", index=False)
        theoretical_trades.to_excel(writer, sheet_name="理论成交", index=False)
        actual_fills.to_excel(writer, sheet_name="实际导入成交", index=False)
    payload = {
        "metadata": metadata,
        "rows": _json_records(comparison),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    md_path.write_text(
        _comparison_markdown(product, comparison, metadata),
        encoding="utf-8",
    )
    return {
        "csv": str(csv_path),
        "xlsx": str(xlsx_path),
        "json": str(json_path),
        "markdown": str(md_path),
    }


def _load_actual_daily(path):
    if not path.exists():
        return pd.DataFrame(columns=list(ACTUAL_SUMMARY_COLUMNS.values()))
    frame = pd.read_csv(path, encoding="utf-8-sig")
    available = {
        source: target
        for source, target in ACTUAL_SUMMARY_COLUMNS.items()
        if source in frame.columns
    }
    result = frame[list(available)].rename(columns=available)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    return result.dropna(subset=["date"]).drop_duplicates("date", keep="last")


def _load_actual_fills_read_only(db_path, *, account_id, start, end):
    if not db_path.exists():
        return pd.DataFrame()
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            select id, action, payload, created_at
            from fills
            where account_id = ? and voided_at is null
            order by id
            """,
            (account_id,),
        ).fetchall()
    records = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        date = pd.to_datetime(payload.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        date = pd.Timestamp(date).normalize()
        if date < start or date > end:
            continue
        records.append(
            {
                "fill_id": row["id"],
                "action": row["action"],
                "created_at": row["created_at"],
                "date": date,
                "payload": payload,
            }
        )
    return pd.DataFrame(records)


def _expand_theoretical_trades(frame):
    rows = []
    if frame is None or frame.empty:
        return pd.DataFrame(rows)
    for _, trade in frame.iterrows():
        date = pd.Timestamp(trade["date"]).normalize()
        if trade.get("asset") == "ETF":
            qty = float(trade.get("trade_qty", 0.0) or 0.0)
            rows.append(
                _execution_leg(
                    date,
                    "ETF",
                    str(trade.get("underlying_order_book_id")),
                    qty,
                    float(trade.get("price", 0.0) or 0.0),
                    1.0,
                    "theoretical",
                    "etf",
                )
            )
            continue
        side = str(trade.get("side") or "short")
        multiplier = float(trade.get("contract_multiplier", 10000) or 10000)
        for leg, code_key, qty_key, price_key in (
            ("call", "call_code", "trade_call_qty", "call_price"),
            ("put", "put_code", "trade_put_qty", "put_price"),
        ):
            qty = float(trade.get(qty_key, 0.0) or 0.0)
            if abs(qty) <= 1e-9:
                continue
            rows.append(
                _execution_leg(
                    date,
                    "OPTION",
                    str(trade.get(code_key)),
                    qty,
                    float(trade.get(price_key, 0.0) or 0.0),
                    multiplier,
                    "theoretical",
                    side,
                    leg=leg,
                )
            )
    return pd.DataFrame(rows)


def _expand_actual_fills(frame, *, product):
    rows = []
    if frame is None or frame.empty:
        return pd.DataFrame(rows)
    for _, fill_row in frame.iterrows():
        payload = fill_row["payload"]
        date = pd.Timestamp(fill_row["date"]).normalize()
        action = str(fill_row["action"])
        if action in {"delta_hedge", "rebalance_hedge", "close_hedge"}:
            price = float(payload.get("price", 0.0) or 0.0)
            cash_delta = float(payload.get("cash_delta", 0.0) or 0.0)
            qty = -cash_delta / price if price > 0 else 0.0
            qty = round(qty / 100.0) * 100.0
            if abs(qty) > 1e-9:
                rows.append(
                    _execution_leg(
                        date,
                        "ETF",
                        str(
                            payload.get("underlying_order_book_id")
                            or UNDERLYING_BY_PRODUCT.get(product)
                        ),
                        qty,
                        price,
                        1.0,
                        "actual",
                        "etf",
                        fill_id=fill_row["fill_id"],
                    )
                )
            continue

        adjustments = payload.get("leg_adjustments") or []
        if adjustments:
            side = str(payload.get("side") or "short")
            multiplier = float(payload.get("contract_multiplier", 10000) or 10000)
            for adjustment in adjustments:
                rows.append(
                    _execution_leg(
                        date,
                        "OPTION",
                        str(adjustment.get("order_book_id")),
                        float(adjustment.get("qty_change", 0.0) or 0.0),
                        float(adjustment.get("price", 0.0) or 0.0),
                        multiplier,
                        "actual",
                        side,
                        leg=str(adjustment.get("leg")),
                        fill_id=fill_row["fill_id"],
                    )
                )
            continue

        if action in {"open_option_hedge", "close_option_hedge"}:
            side = str(payload.get("side") or "short")
            sign = -1.0 if action.startswith("close") else 1.0
            multiplier = float(payload.get("contract_multiplier", 10000) or 10000)
            option_type = str(payload.get("option_type") or "").lower()
            leg = "call" if option_type.startswith("c") else "put"
            code = payload.get("order_book_id") or payload.get(f"{leg}_code")
            qty = sign * float(
                payload.get("qty", payload.get(f"{leg}_qty", 0.0)) or 0.0
            )
            price = (
                payload.get("price")
                if action.startswith("close")
                else payload.get("entry_price")
            )
            if price is None:
                price = payload.get("close_price") or payload.get(f"entry_{leg}_price")
            if code is not None and price is not None and abs(qty) > 1e-9:
                rows.append(
                    _execution_leg(
                        date,
                        "OPTION",
                        str(code),
                        qty,
                        float(price),
                        multiplier,
                        "actual",
                        side,
                        leg=leg,
                        fill_id=fill_row["fill_id"],
                    )
                )
            continue

        if "straddle" not in action:
            continue
        side = str(payload.get("side") or ("short" if "short" in action else "long"))
        sign = -1.0 if action.startswith("close") else 1.0
        multiplier = float(payload.get("contract_multiplier", 10000) or 10000)
        for leg in ("call", "put"):
            code = payload.get(f"{leg}_code")
            qty = sign * float(payload.get(f"{leg}_qty", 0.0) or 0.0)
            price = payload.get(f"entry_{leg}_price")
            if price is None:
                price = payload.get(f"{leg}_price")
            if price is None:
                price = payload.get(f"last_{leg}_price")
            if code is None or price is None or abs(qty) <= 1e-9:
                continue
            rows.append(
                _execution_leg(
                    date,
                    "OPTION",
                    str(code),
                    qty,
                    float(price),
                    multiplier,
                    "actual",
                    side,
                    leg=leg,
                    fill_id=fill_row["fill_id"],
                )
            )
    return pd.DataFrame(rows)


def _execution_leg(
    date,
    asset,
    code,
    qty,
    price,
    multiplier,
    source,
    side,
    *,
    leg=None,
    fill_id=None,
):
    if asset == "ETF":
        cash_flow = -float(qty) * float(price)
    else:
        direction = 1.0 if side == "short" else -1.0
        cash_flow = direction * float(qty) * float(price) * float(multiplier)
    return {
        "date": pd.Timestamp(date).normalize(),
        "asset": asset,
        "code": code,
        "leg": leg,
        "qty": float(qty),
        "price": float(price),
        "multiplier": float(multiplier),
        "source": source,
        "side": side,
        "cash_flow": cash_flow,
        "fill_id": fill_id,
    }


def _match_execution_legs(theoretical, actual):
    if theoretical is None or theoretical.empty:
        theoretical = pd.DataFrame()
    if actual is None or actual.empty:
        actual = pd.DataFrame()
    theoretical = theoretical.copy().reset_index(drop=True)
    actual = actual.copy().reset_index(drop=True)
    if theoretical.empty and actual.empty:
        return pd.DataFrame()

    theoretical["remaining_qty"] = theoretical["qty"].abs()
    actual["remaining_qty"] = actual["qty"].abs()
    rows = []

    # First allocate same-day fills. A plan leg may be split over several broker
    # fills (and vice versa), so matching must consume quantities rather than
    # marking a whole row matched after the first overlap.
    for theoretical_index in theoretical.index:
        expected = theoretical.loc[theoretical_index]
        candidates = _candidate_actual_indices(actual, expected, same_day=True)
        for actual_index in candidates:
            if theoretical.at[theoretical_index, "remaining_qty"] <= 1e-9:
                break
            allocated = min(
                theoretical.at[theoretical_index, "remaining_qty"],
                actual.at[actual_index, "remaining_qty"],
            )
            if allocated <= 1e-9:
                continue
            rows.append(
                _allocation_row(
                    expected,
                    actual.loc[actual_index],
                    allocated,
                    status="matched",
                )
            )
            theoretical.at[theoretical_index, "remaining_qty"] -= allocated
            actual.at[actual_index, "remaining_qty"] -= allocated

    # A later fill in the same instrument/direction is classified separately as
    # delayed. Its timing PnL is deliberately not fabricated: only the evidence
    # and reference notional are reported, leaving the timing effect in residual.
    for theoretical_index in theoretical.index:
        expected = theoretical.loc[theoretical_index]
        candidates = _candidate_actual_indices(actual, expected, later_only=True)
        for actual_index in candidates:
            if theoretical.at[theoretical_index, "remaining_qty"] <= 1e-9:
                break
            allocated = min(
                theoretical.at[theoretical_index, "remaining_qty"],
                actual.at[actual_index, "remaining_qty"],
            )
            if allocated <= 1e-9:
                continue
            rows.append(
                _allocation_row(
                    expected,
                    actual.loc[actual_index],
                    allocated,
                    status="delayed",
                )
            )
            theoretical.at[theoretical_index, "remaining_qty"] -= allocated
            actual.at[actual_index, "remaining_qty"] -= allocated

    for _, expected in theoretical.iterrows():
        if expected["remaining_qty"] > 1e-9:
            rows.append(
                _unmatched_row(
                    expected,
                    "theoretical_only",
                    qty=expected["remaining_qty"],
                )
            )
    for _, observed in actual.iterrows():
        if observed["remaining_qty"] > 1e-9:
            rows.append(
                _unmatched_row(
                    observed,
                    "actual_only",
                    qty=observed["remaining_qty"],
                )
            )
    return pd.DataFrame(rows)


def _candidate_actual_indices(actual, expected, *, same_day=False, later_only=False):
    if actual.empty:
        return []
    mask = (
        (actual["remaining_qty"] > 1e-9)
        & (actual["asset"] == expected["asset"])
        & (actual["code"] == expected["code"])
        & (actual["qty"].apply(_sign) == _sign(expected["qty"]))
    )
    if same_day:
        mask &= actual["date"] == expected["date"]
    if later_only:
        mask &= actual["date"] > expected["date"]
    candidates = actual[mask].copy()
    if candidates.empty:
        return []
    candidates["quantity_distance"] = (
        candidates["remaining_qty"] - float(expected["remaining_qty"])
    ).abs()
    sort_columns = ["date", "quantity_distance"] if later_only else ["quantity_distance"]
    return list(candidates.sort_values(sort_columns).index)


def _allocation_row(expected, observed, matched_qty, *, status):
    expected_sign = _sign(expected["qty"])
    observed_sign = _sign(observed["qty"])
    expected_cf_per_qty = expected["cash_flow"] / abs(expected["qty"])
    observed_cf_per_qty = observed["cash_flow"] / abs(observed["qty"])
    delayed = status == "delayed"
    return {
        "date": expected["date"],
        "actual_date": observed["date"],
        "delay_days": int((observed["date"] - expected["date"]).days),
        "match_status": status,
        "asset": expected["asset"],
        "code": expected["code"],
        "side": expected["side"],
        "theoretical_qty": expected_sign * matched_qty,
        "actual_qty": observed_sign * matched_qty,
        "matched_qty": matched_qty,
        "theoretical_price": expected["price"],
        "actual_price": observed["price"],
        "multiplier": expected["multiplier"],
        "reference_notional": (
            matched_qty * float(expected["price"]) * float(expected["multiplier"])
        ),
        "execution_slippage_pnl": 0.0 if delayed else (
            observed_cf_per_qty - expected_cf_per_qty
        ) * matched_qty,
        "fill_id": observed.get("fill_id"),
    }


def _unmatched_row(leg, status, *, qty=None):
    theoretical = status == "theoretical_only"
    qty = abs(float(leg["qty"])) if qty is None else abs(float(qty))
    signed_qty = _sign(leg["qty"]) * qty
    return {
        "date": leg["date"],
        "actual_date": None if theoretical else leg["date"],
        "delay_days": None,
        "match_status": status,
        "asset": leg["asset"],
        "code": leg["code"],
        "side": leg["side"],
        "theoretical_qty": signed_qty if theoretical else None,
        "actual_qty": None if theoretical else signed_qty,
        "matched_qty": 0.0,
        "theoretical_price": leg["price"] if theoretical else None,
        "actual_price": None if theoretical else leg["price"],
        "multiplier": leg["multiplier"],
        "reference_notional": (
            qty * float(leg["price"]) * float(leg["multiplier"])
        ),
        "execution_slippage_pnl": 0.0,
        "fill_id": None if theoretical else leg.get("fill_id"),
    }


def _summarize_matches(matches):
    if matches.empty:
        return pd.DataFrame(columns=["date"])
    records = []
    for date, frame in matches.groupby("date"):
        unmatched_theoretical = frame[frame["match_status"] == "theoretical_only"]
        unmatched_actual = frame[frame["match_status"] == "actual_only"]
        delayed = frame[frame["match_status"] == "delayed"]
        unexecuted_notional = float(unmatched_theoretical["reference_notional"].sum())
        delayed_notional = float(delayed["reference_notional"].sum())
        records.append(
            {
                "date": pd.Timestamp(date),
                "execution_slippage_pnl": float(
                    frame["execution_slippage_pnl"].sum()
                ),
                "matched_leg_count": int((frame["match_status"] == "matched").sum()),
                "delayed_leg_count": len(delayed),
                "unexecuted_leg_count": len(unmatched_theoretical),
                "unexpected_actual_leg_count": len(unmatched_actual),
                "unmatched_theoretical_leg_count": len(unmatched_theoretical) + len(delayed),
                "unmatched_actual_leg_count": len(unmatched_actual) + len(delayed),
                "unexecuted_notional": unexecuted_notional,
                "delayed_notional": delayed_notional,
                "unexecuted_or_delayed_notional": unexecuted_notional + delayed_notional,
            }
        )
    return pd.DataFrame(records)


def _theoretical_fee_by_date(trades):
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["date", "theoretical_fee"])
    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return (
        frame.groupby("date", as_index=False)["fee"]
        .sum()
        .rename(columns={"fee": "theoretical_fee"})
    )


def _load_actual_position_fingerprints(path):
    columns = ["date", "actual_position_fingerprint", "actual_hedge_qty"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"日期", "方向", "合约代码", "合约名称", "总持仓"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    frame["日期"] = pd.to_datetime(frame["日期"], errors="coerce").dt.normalize()
    records = []
    for date, daily in frame.dropna(subset=["日期"]).groupby("日期"):
        fingerprint = {"long": None, "short": None}
        hedge_qty = 0.0
        for _, row in daily.iterrows():
            row_direction = str(row.get("方向") or "").lower()
            if row_direction == "hedge":
                hedge_qty = float(row.get("总持仓", 0.0) or 0.0)
                continue
            contract_name = str(row.get("合约名称") or "")
            if "购" in contract_name:
                leg = "call"
            elif "沽" in contract_name:
                leg = "put"
            else:
                continue
            side = row_direction if row_direction in {"long", "short"} else (
                "short" if "卖" in str(row.get("买卖") or "") else "long"
            )
            target = fingerprint[side] or {
                "call_code": None,
                "put_code": None,
                "call_qty": 0,
                "put_qty": 0,
            }
            target[f"{leg}_code"] = _contract_code(row.get("合约代码"))
            target[f"{leg}_qty"] = int(float(row.get("总持仓", 0.0) or 0.0))
            fingerprint[side] = target
        records.append(
            {
                "date": pd.Timestamp(date),
                "actual_position_fingerprint": json.dumps(
                    fingerprint, ensure_ascii=False, sort_keys=True
                ),
                "actual_hedge_qty": hedge_qty,
            }
        )
    return pd.DataFrame(records, columns=columns)


def _position_difference(row):
    actual = row.get("actual_position_fingerprint")
    theoretical = row.get("ending_position_fingerprint")
    option_diff = (
        pd.notna(actual)
        and pd.notna(theoretical)
        and json.loads(actual) != json.loads(theoretical)
    )
    hedge_diff = (
        pd.notna(row.get("actual_hedge_qty"))
        and abs(
            float(row.get("actual_hedge_qty") or 0.0)
            - float(row.get("ending_hedge_qty") or 0.0)
        )
        > 1e-6
    )
    return bool(option_diff or hedge_diff)


def _attribution_note(row):
    notes = []
    if not bool(row.get("pnl_comparable", True)):
        notes.append("区间首个快照前的当日损益不在理论重放范围内")
    if row["unexecuted_status"] != "none":
        notes.append("存在未执行计划，其PnL未强行估算")
    if row["delayed_execution_status"] != "none":
        notes.append("存在延迟成交，延迟期间PnL未强行估算")
    if row.get("unexpected_actual_leg_count", 0) > 0:
        notes.append("存在计划外实际成交")
    if row["position_difference_status"] != "none":
        notes.append("理论与实际收盘持仓不同，影响保留在未解释残差")
    if not notes:
        notes.append("成交和持仓均可直接核对")
    return "；".join(notes)


def _comparison_markdown(product, frame, metadata):
    theoretical = pd.to_numeric(
        frame.get("theoretical_daily_pnl"), errors="coerce"
    ).sum()
    actual = pd.to_numeric(frame.get("actual_net_daily_pnl"), errors="coerce").sum()
    residual = pd.to_numeric(
        frame.get("unexplained_residual_pnl"), errors="coerce"
    ).sum()
    comparable = frame[frame.get("pnl_comparable", True).astype(bool)]
    comparable_theoretical = pd.to_numeric(
        comparable.get("theoretical_daily_pnl"), errors="coerce"
    ).sum()
    comparable_actual = pd.to_numeric(
        comparable.get("actual_net_daily_pnl"), errors="coerce"
    ).sum()
    comparable_residual = pd.to_numeric(
        comparable.get("unexplained_residual_pnl"), errors="coerce"
    ).sum()
    return "\n".join(
        [
            f"# {product} live_straddle 理论/实际对比",
            "",
            f"- 区间：{metadata.get('start')} 至 {metadata.get('end')}",
            f"- 当前策略快照重放累计盈亏：{theoretical:,.2f}",
            f"- 券商实际净盈亏：{actual:,.2f}",
            f"- 尚未解释残差：{residual:,.2f}",
            f"- 可比整日理论/实际：{comparable_theoretical:,.2f} / {comparable_actual:,.2f}",
            f"- 可比整日未解释残差：{comparable_residual:,.2f}",
            "",
            "区间首日从首个已保存快照开始，券商日盈亏通常包含该快照之前的时段，因此首日保留展示但不计入‘可比整日’汇总。",
            "",
            "未匹配/延迟成交和持仓差异只在能够可靠定价时才计入货币归因；否则明确标记并留在残差中，避免伪造解释。",
            "",
            frame.to_markdown(index=False),
        ]
    )


def _json_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _sign(value):
    value = float(value)
    return 1 if value > 0 else -1 if value < 0 else 0


def _contract_code(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return str(value)


__all__ = [
    "build_live_comparison",
    "write_live_comparison_report",
]
