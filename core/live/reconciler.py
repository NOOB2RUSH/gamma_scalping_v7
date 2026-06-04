from __future__ import annotations

import json

from . import account as account_store
from . import storage


def reconcile(product, broker_snapshot, account_id="default"):
    local = account_store.load_account(product, account_id=account_id)
    diffs = []

    _compare_number(diffs, "cash", local.cash, broker_snapshot.get("cash"))
    _compare_hedge(diffs, local.hedge.to_dict(), broker_snapshot.get("hedge", {}))
    _compare_positions(diffs, local.positions, broker_snapshot.get("positions", []))

    payload = {
        "product": product,
        "account_id": account_id,
        "ok": not diffs,
        "diffs": diffs,
        "local_account": local.to_dict(),
        "broker_snapshot": broker_snapshot,
    }
    account_store.record_broker_snapshot(product, broker_snapshot, account_id=account_id)
    _record_reconciliation(product, payload, account_id)
    return payload


def write_reconcile_report(product, payload):
    stamp = storage.local_now_stamp()
    path = storage.output_dir(product) / f"{stamp}_reconcile.md"
    lines = [
        f"# Reconciliation: {product}",
        "",
        f"- account_id: {payload['account_id']}",
        f"- ok: {payload['ok']}",
        "",
        "## Differences",
        "",
    ]
    if not payload["diffs"]:
        lines.append("No differences.")
    else:
        for diff in payload["diffs"]:
            lines.append(
                f"- {diff['field']}: local={diff.get('local')} broker={diff.get('broker')}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _compare_number(diffs, field, local, broker, tolerance=1e-6):
    if broker is None:
        diffs.append({"field": field, "local": local, "broker": None})
        return
    if abs(float(local) - float(broker)) > tolerance:
        diffs.append({"field": field, "local": local, "broker": broker})


def _compare_hedge(diffs, local, broker):
    for key in ["qty", "entry_price", "margin"]:
        _compare_number(diffs, f"hedge.{key}", local.get(key, 0.0), broker.get(key, 0.0))
    if local.get("underlying_order_book_id") != broker.get("underlying_order_book_id"):
        diffs.append(
            {
                "field": "hedge.underlying_order_book_id",
                "local": local.get("underlying_order_book_id"),
                "broker": broker.get("underlying_order_book_id"),
            }
        )


def _compare_positions(diffs, local_positions, broker_positions):
    broker_by_side = {item.get("side"): item for item in broker_positions}
    for side, local in local_positions.items():
        broker = broker_by_side.get(side)
        if local is None and broker is None:
            continue
        if local is None or broker is None:
            diffs.append({"field": f"position.{side}", "local": local, "broker": broker})
            continue
        for key in ["call_code", "put_code", "call_qty", "put_qty", "strike", "expiry"]:
            if str(local.get(key)) != str(broker.get(key)):
                diffs.append(
                    {
                        "field": f"position.{side}.{key}",
                        "local": local.get(key),
                        "broker": broker.get(key),
                    }
                )


def _record_reconciliation(product, payload, account_id):
    db_path = storage.account_db_path(product)
    with account_store.connect(db_path) as conn:
        conn.execute(
            """
            insert into reconciliations(account_id, payload, created_at)
            values (?, ?, ?)
            """,
            (
                account_id,
                json.dumps(payload, ensure_ascii=False, default=str),
                storage.utc_now_text(),
            ),
        )
        conn.commit()
