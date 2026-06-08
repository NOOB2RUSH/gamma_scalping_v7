from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import storage


POSITION_SIDES = ("long", "short")


@dataclass
class HedgeState:
    qty: float = 0.0
    entry_price: float = 0.0
    margin: float = 0.0
    underlying_order_book_id: str | None = None
    latest_price: float | None = None
    last_market_value: float | None = None
    last_unrealized_pnl: float | None = None
    last_mark_date: str | None = None
    last_mark_source_file: str | None = None
    last_mark_source_timestamp: str | None = None

    def to_dict(self):
        return {
            "qty": self.qty,
            "entry_price": self.entry_price,
            "margin": self.margin,
            "underlying_order_book_id": self.underlying_order_book_id,
            "latest_price": self.latest_price,
            "last_market_value": self.last_market_value,
            "last_unrealized_pnl": self.last_unrealized_pnl,
            "last_mark_date": self.last_mark_date,
            "last_mark_source_file": self.last_mark_source_file,
            "last_mark_source_timestamp": self.last_mark_source_timestamp,
        }


@dataclass
class AccountState:
    product: str
    account_id: str = "default"
    cash: float = 1_000_000.0
    positions: dict = field(default_factory=lambda: {"long": None, "short": None})
    option_hedges: list = field(default_factory=list)
    hedge: HedgeState = field(default_factory=HedgeState)
    strategy_state: "StrategyState" = field(default_factory=lambda: StrategyState())
    updated_at: str | None = None

    def to_dict(self):
        return {
            "product": self.product,
            "account_id": self.account_id,
            "cash": self.cash,
            "positions": self.positions,
            "option_hedges": self.option_hedges,
            "hedge": self.hedge.to_dict(),
            "strategy_state": self.strategy_state.to_dict(),
            "updated_at": self.updated_at,
        }


@dataclass
class StrategyState:
    roll_cooldown_left: dict = field(
        default_factory=lambda: {"long": 0, "short": 0}
    )
    cooldown_total_days: dict = field(
        default_factory=lambda: {"long": 0, "short": 0}
    )
    cooldown_started_date: dict = field(
        default_factory=lambda: {"long": None, "short": None}
    )

    def to_dict(self):
        return {
            "roll_cooldown_left": {
                side: int(self.roll_cooldown_left.get(side, 0) or 0)
                for side in POSITION_SIDES
            },
            "cooldown_total_days": {
                side: int(self.cooldown_total_days.get(side, 0) or 0)
                for side in POSITION_SIDES
            },
            "cooldown_started_date": {
                side: self.cooldown_started_date.get(side)
                for side in POSITION_SIDES
            },
        }


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn):
    conn.executescript(
        """
        create table if not exists account_state (
            account_id text primary key,
            product text not null,
            cash real not null,
            updated_at text not null
        );
        create table if not exists option_positions (
            account_id text not null,
            side text not null,
            payload text not null,
            updated_at text not null,
            primary key (account_id, side)
        );
        create table if not exists option_hedges (
            account_id text not null,
            hedge_id text not null,
            payload text not null,
            updated_at text not null,
            primary key (account_id, hedge_id)
        );
        create table if not exists hedge_position (
            account_id text primary key,
            payload text not null,
            updated_at text not null
        );
        create table if not exists strategy_state (
            account_id text primary key,
            payload text not null,
            updated_at text not null
        );
        create table if not exists fills (
            id integer primary key autoincrement,
            account_id text not null,
            action text not null,
            payload text not null,
            created_at text not null,
            voided_at text,
            void_reason text,
            replaces_fill_id integer
        );
        create table if not exists broker_snapshots (
            id integer primary key autoincrement,
            account_id text not null,
            payload text not null,
            created_at text not null
        );
        create table if not exists reconciliations (
            id integer primary key autoincrement,
            account_id text not null,
            payload text not null,
            created_at text not null
        );
        """
    )
    _ensure_column(conn, "fills", "voided_at", "text")
    _ensure_column(conn, "fills", "void_reason", "text")
    _ensure_column(conn, "fills", "replaces_fill_id", "integer")
    conn.commit()


def _ensure_column(conn, table, column, column_type):
    existing = {
        row["name"]
        for row in conn.execute(f"pragma table_info({table})")
    }
    if column not in existing:
        conn.execute(f"alter table {table} add column {column} {column_type}")


def initialize_account(product, initial_cash, db_path=None, account_id="default", reset=False):
    db_path = db_path or storage.account_db_path(product)
    with connect(db_path) as conn:
        existing = conn.execute(
            "select 1 from account_state where account_id = ?",
            (account_id,),
        ).fetchone()
        if existing is not None and not reset:
            raise ValueError(
                f"Live account already exists: {product}/{account_id}. "
                "Use --reset to reinitialize and clear positions/fills, or use a "
                "cash_adjustment fill to change cash."
            )
        if reset:
            conn.execute("delete from account_state where account_id = ?", (account_id,))
            conn.execute("delete from option_positions where account_id = ?", (account_id,))
            conn.execute("delete from option_hedges where account_id = ?", (account_id,))
            conn.execute("delete from hedge_position where account_id = ?", (account_id,))
            conn.execute("delete from strategy_state where account_id = ?", (account_id,))
            conn.execute("delete from fills where account_id = ?", (account_id,))
            conn.execute("delete from broker_snapshots where account_id = ?", (account_id,))
            conn.execute("delete from reconciliations where account_id = ?", (account_id,))
        now = storage.utc_now_text()
        conn.execute(
            """
            insert into account_state(account_id, product, cash, updated_at)
            values (?, ?, ?, ?)
            on conflict(account_id) do update set
                product=excluded.product,
                cash=excluded.cash,
                updated_at=excluded.updated_at
            """,
            (account_id, product, float(initial_cash), now),
        )
        conn.execute(
            """
            insert into hedge_position(account_id, payload, updated_at)
            values (?, ?, ?)
            on conflict(account_id) do nothing
            """,
            (
                account_id,
                json.dumps(HedgeState().to_dict(), ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            """
            insert into strategy_state(account_id, payload, updated_at)
            values (?, ?, ?)
            on conflict(account_id) do nothing
            """,
            (
                account_id,
                json.dumps(StrategyState().to_dict(), ensure_ascii=False),
                now,
            ),
        )
        conn.commit()
    return load_account(product, db_path, account_id)


def load_account(product, db_path=None, account_id="default"):
    db_path = db_path or storage.account_db_path(product)
    with connect(db_path) as conn:
        account = _load_account_from_conn(conn, product, account_id)
    if account is None:
        return initialize_account(product, _default_initial_cash(product), db_path, account_id)
    return account


def save_account(account, db_path=None):
    db_path = db_path or storage.account_db_path(account.product)
    now = storage.utc_now_text()
    with connect(db_path) as conn:
        _save_account_to_conn(conn, account, now)
        conn.commit()
    account.updated_at = now
    return account


def record_fill(product, fill, db_path=None, account_id="default"):
    """Apply a manually confirmed fill to the shadow account.

    Fill payloads are intentionally explicit. This keeps broker execution truth
    separate from model advice.
    """
    db_path = db_path or storage.account_db_path(product)
    fill = normalize_fill(fill)
    with connect(db_path) as conn:
        account = _load_account_from_conn(conn, product, account_id)
        if account is None:
            account = AccountState(
                product=product,
                account_id=account_id,
                cash=float(_default_initial_cash(product)),
                positions={side: None for side in POSITION_SIDES},
                option_hedges=[],
                hedge=HedgeState(),
                strategy_state=StrategyState(),
            )
        _apply_fill(account, product, fill)
        now = storage.utc_now_text()
        _save_account_to_conn(conn, account, now)
        _insert_fill_to_conn(conn, fill, account_id, now)
        conn.commit()
        account.updated_at = now
    return account


def normalize_fill(fill):
    result = dict(fill)
    action = result.get("action")
    if action is None:
        raise ValueError("Fill missing action.")

    action_text = str(action)
    action_key = action_text.lower()
    action_map = {
        "open_short_straddle": "open_short_straddle",
        "open_long_straddle": "open_long_straddle",
        "open_straddle": "open_straddle",
        "roll_short_straddle": "roll_short_straddle",
        "roll_long_straddle": "roll_long_straddle",
        "roll_straddle": "roll_straddle",
        "close_short_straddle": "close_short_straddle",
        "close_long_straddle": "close_long_straddle",
        "close_straddle": "close_straddle",
        "delta_hedge": "delta_hedge",
        "projected_delta_hedge": "delta_hedge",
        "final_delta_hedge": "delta_hedge",
        "option_delta_hedge_short_call": "open_option_hedge",
        "final_option_delta_hedge_short_call": "open_option_hedge",
        "open_option_hedge": "open_option_hedge",
        "option_hedge_mark_update": "option_hedge_mark_update",
        "close_option_hedge": "close_option_hedge",
        "rebalance_hedge": "rebalance_hedge",
        "close_hedge": "close_hedge",
        "option_mark_update": "option_mark_update",
        "hedge_mark_update": "hedge_mark_update",
        "cash_adjustment": "cash_adjustment",
    }
    if action_key not in action_map:
        raise ValueError(f"Unsupported fill action: {action}")
    result["action"] = action_map[action_key]
    if result["action"] != action_text:
        result.setdefault("source_action", action_text)

    _copy_if_missing(result, "estimated_call_price", "entry_call_price")
    _copy_if_missing(result, "estimated_put_price", "entry_put_price")
    _copy_if_missing(result, "estimated_trade_value", "entry_option_value")
    _copy_if_missing(result, "estimated_option_margin", "option_margin")
    _copy_if_missing(result, "estimated_cash_effect", "cash_delta")
    _copy_if_missing(result, "estimated_price", "price")
    _copy_if_missing(result, "target_hedge_qty", "new_etf_qty")
    _copy_if_missing(result, "target_hedge_qty", "qty")
    return result


def _copy_if_missing(payload, source_key, target_key):
    if target_key not in payload and source_key in payload:
        payload[target_key] = payload[source_key]


def _option_hedge_id(payload, index=0):
    code = payload.get("order_book_id") or payload.get("call_code") or payload.get("put_code")
    side = payload.get("side", "short")
    if code is None:
        return f"{side}:{index}"
    return f"{side}:{code}"


def _load_account_from_conn(conn, product, account_id):
    state_row = conn.execute(
        "select * from account_state where account_id = ?",
        (account_id,),
    ).fetchone()
    if state_row is None:
        return None

    positions = {side: None for side in POSITION_SIDES}
    for row in conn.execute(
        "select side, payload from option_positions where account_id = ?",
        (account_id,),
    ):
        positions[row["side"]] = json.loads(row["payload"])

    option_hedges = []
    for row in conn.execute(
        "select payload from option_hedges where account_id = ? order by hedge_id",
        (account_id,),
    ):
        option_hedges.append(json.loads(row["payload"]))

    hedge_row = conn.execute(
        "select payload from hedge_position where account_id = ?",
        (account_id,),
    ).fetchone()
    hedge_payload = json.loads(hedge_row["payload"]) if hedge_row else {}
    strategy_row = conn.execute(
        "select payload from strategy_state where account_id = ?",
        (account_id,),
    ).fetchone()
    strategy_payload = json.loads(strategy_row["payload"]) if strategy_row else {}

    return AccountState(
        product=state_row["product"] or product,
        account_id=account_id,
        cash=float(state_row["cash"]),
        positions=positions,
        option_hedges=option_hedges,
        hedge=HedgeState(
            qty=float(hedge_payload.get("qty", 0.0) or 0.0),
            entry_price=float(hedge_payload.get("entry_price", 0.0) or 0.0),
            margin=float(hedge_payload.get("margin", 0.0) or 0.0),
            underlying_order_book_id=hedge_payload.get("underlying_order_book_id"),
            latest_price=_optional_float(hedge_payload.get("latest_price")),
            last_market_value=_optional_float(hedge_payload.get("last_market_value")),
            last_unrealized_pnl=_optional_float(hedge_payload.get("last_unrealized_pnl")),
            last_mark_date=hedge_payload.get("last_mark_date"),
            last_mark_source_file=hedge_payload.get("last_mark_source_file"),
            last_mark_source_timestamp=hedge_payload.get("last_mark_source_timestamp"),
        ),
        strategy_state=_strategy_state_from_payload(strategy_payload),
        updated_at=state_row["updated_at"],
    )


def _save_account_to_conn(conn, account, now):
    conn.execute(
        """
        insert into account_state(account_id, product, cash, updated_at)
        values (?, ?, ?, ?)
        on conflict(account_id) do update set
            product=excluded.product,
            cash=excluded.cash,
            updated_at=excluded.updated_at
        """,
        (account.account_id, account.product, float(account.cash), now),
    )
    conn.execute(
        "delete from option_positions where account_id = ?",
        (account.account_id,),
    )
    for side, payload in account.positions.items():
        if payload is None:
            continue
        conn.execute(
            """
            insert into option_positions(account_id, side, payload, updated_at)
            values (?, ?, ?, ?)
            """,
            (
                account.account_id,
                side,
                json.dumps(payload, ensure_ascii=False, default=str),
                now,
            ),
        )
    conn.execute(
        "delete from option_hedges where account_id = ?",
        (account.account_id,),
    )
    for index, payload in enumerate(account.option_hedges or []):
        hedge_id = _option_hedge_id(payload, index)
        conn.execute(
            """
            insert into option_hedges(account_id, hedge_id, payload, updated_at)
            values (?, ?, ?, ?)
            """,
            (
                account.account_id,
                hedge_id,
                json.dumps(payload, ensure_ascii=False, default=str),
                now,
            ),
        )
    conn.execute(
        """
        insert into hedge_position(account_id, payload, updated_at)
        values (?, ?, ?)
        on conflict(account_id) do update set
            payload=excluded.payload,
            updated_at=excluded.updated_at
        """,
        (
            account.account_id,
            json.dumps(account.hedge.to_dict(), ensure_ascii=False, default=str),
            now,
        ),
    )
    conn.execute(
        """
        insert into strategy_state(account_id, payload, updated_at)
        values (?, ?, ?)
        on conflict(account_id) do update set
            payload=excluded.payload,
            updated_at=excluded.updated_at
        """,
        (
            account.account_id,
            json.dumps(
                account.strategy_state.to_dict(),
                ensure_ascii=False,
                default=str,
            ),
            now,
        ),
    )


def _apply_fill(account, product, fill):
    fill = normalize_fill(fill)
    action = fill["action"]
    side = fill.get("side")
    cash_delta = float(fill.get("cash_delta", 0.0) or 0.0)

    if action in {"open_straddle", "open_short_straddle", "open_long_straddle"}:
        side = side or ("short" if "short" in action else "long")
        account.positions[side] = _position_from_fill(fill, side)
        _reset_strategy_side(account.strategy_state, side)
        account.cash += cash_delta
    elif action in {"roll_straddle", "roll_short_straddle", "roll_long_straddle"}:
        side = side or ("short" if "short" in action else "long")
        account.positions[side] = _position_from_fill(fill, side)
        _reset_strategy_side(account.strategy_state, side)
        account.cash += cash_delta
    elif action in {"close_straddle", "close_short_straddle", "close_long_straddle"}:
        side = side or ("short" if "short" in action else "long")
        account.positions[side] = None
        _start_strategy_cooldown(
            account.strategy_state,
            side,
            _roll_cooldown_days(product),
            fill.get("date"),
        )
        if side == "long" and fill.get("exit_reason") == "iv_high":
            _start_strategy_cooldown(
                account.strategy_state,
                "short",
                _short_cooldown_after_long_iv_high_exit_days(product),
                fill.get("date"),
            )
        account.cash += cash_delta
    elif action in {"delta_hedge", "rebalance_hedge", "close_hedge"}:
        hedge = fill.get("hedge", fill)
        account.hedge = HedgeState(
            qty=float(hedge.get("qty", hedge.get("new_etf_qty", hedge.get("target_hedge_qty", 0.0))) or 0.0),
            entry_price=float(hedge.get("entry_price", hedge.get("price", 0.0)) or 0.0),
            margin=float(hedge.get("margin", 0.0) or 0.0),
            underlying_order_book_id=hedge.get("underlying_order_book_id"),
            latest_price=_optional_float(hedge.get("latest_price")),
            last_market_value=_optional_float(hedge.get("market_value")),
            last_unrealized_pnl=_optional_float(hedge.get("unrealized_pnl")),
            last_mark_date=hedge.get("date"),
            last_mark_source_file=hedge.get("holding_source_file"),
            last_mark_source_timestamp=hedge.get("source_timestamp"),
        )
        account.cash += cash_delta
    elif action == "open_option_hedge":
        _upsert_option_hedge(account, fill)
        account.cash += cash_delta
    elif action == "option_hedge_mark_update":
        _apply_option_hedge_mark_update(account, fill)
    elif action == "close_option_hedge":
        _remove_option_hedge(account, fill)
        account.cash += cash_delta
    elif action == "option_mark_update":
        _apply_option_mark_update(account, fill)
    elif action == "hedge_mark_update":
        _apply_hedge_mark_update(account, fill)
    elif action == "cash_adjustment":
        account.cash += cash_delta
    else:
        raise ValueError(f"Unsupported fill action: {action}")

    return account


def amend_fill(
    product,
    fill_id,
    replacement_fill=None,
    reason=None,
    db_path=None,
    account_id="default",
    initial_cash=None,
):
    db_path = db_path or storage.account_db_path(product)
    now = storage.utc_now_text()
    with connect(db_path) as conn:
        row = conn.execute(
            """
            select id, voided_at from fills
            where id = ? and account_id = ?
            """,
            (int(fill_id), account_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"Fill not found: id={fill_id} account_id={account_id}")
        if row["voided_at"] is not None:
            raise ValueError(f"Fill is already voided: id={fill_id}")

        conn.execute(
            """
            update fills
            set voided_at = ?, void_reason = ?
            where id = ? and account_id = ?
            """,
            (now, reason or "amended", int(fill_id), account_id),
        )
        replacement_fill_id = None
        if replacement_fill is not None:
            replacement_fill = normalize_fill(replacement_fill)
            cursor = conn.execute(
                """
                insert into fills(
                    account_id, action, payload, created_at, replaces_fill_id
                )
                values (?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    replacement_fill["action"],
                    json.dumps(replacement_fill, ensure_ascii=False, default=str),
                    now,
                    int(fill_id),
                ),
            )
            replacement_fill_id = cursor.lastrowid
        conn.commit()

    account = rebuild_account(
        product,
        db_path=db_path,
        account_id=account_id,
        initial_cash=initial_cash,
    )
    return {
        "voided_fill_id": int(fill_id),
        "replacement_fill_id": replacement_fill_id,
        "account": account,
    }


def rebuild_account(product, db_path=None, account_id="default", initial_cash=None):
    db_path = db_path or storage.account_db_path(product)
    if initial_cash is None:
        initial_cash = _default_initial_cash(product)

    account = AccountState(
        product=product,
        account_id=account_id,
        cash=float(initial_cash),
        positions={side: None for side in POSITION_SIDES},
        option_hedges=[],
        hedge=HedgeState(),
        strategy_state=StrategyState(),
    )
    for row in list_fills(
        product,
        db_path=db_path,
        account_id=account_id,
        include_voided=False,
    ):
        _apply_fill(account, product, row["payload"])
    return save_account(account, db_path)


def list_fills(
    product,
    db_path=None,
    account_id="default",
    include_voided=True,
    limit=None,
    order="asc",
):
    db_path = db_path or storage.account_db_path(product)
    if order not in {"asc", "desc"}:
        raise ValueError("order must be 'asc' or 'desc'")
    where = ["account_id = ?"]
    params = [account_id]
    if not include_voided:
        where.append("voided_at is null")
    query = (
        "select * from fills where "
        + " and ".join(where)
        + f" order by id {order}"
    )
    if limit is not None:
        query += " limit ?"
        params.append(int(limit))

    rows = []
    with connect(db_path) as conn:
        for row in conn.execute(query, params):
            payload = json.loads(row["payload"])
            rows.append(
                {
                    "id": row["id"],
                    "account_id": row["account_id"],
                    "action": row["action"],
                    "payload": payload,
                    "created_at": row["created_at"],
                    "voided_at": row["voided_at"],
                    "void_reason": row["void_reason"],
                    "replaces_fill_id": row["replaces_fill_id"],
                }
            )
    return rows


def list_fill_table(
    product,
    db_path=None,
    account_id="default",
    include_voided=True,
    limit=None,
    order="asc",
    expand_security_trades=False,
):
    rows = []
    for row in list_fills(
        product,
        db_path=db_path,
        account_id=account_id,
        include_voided=include_voided,
        limit=limit,
        order=order,
    ):
        rows.append(_fill_table_row(row))
        if expand_security_trades:
            payload = row["payload"]
            for index, trade in enumerate(payload.get("security_trades") or [], start=1):
                rows.append(_security_trade_table_row(row, trade, index))
    return rows


def _fill_table_row(row):
    payload = row["payload"]
    action = payload.get("action", row["action"])
    action_lower = str(action).lower()
    return {
        "id": row["id"],
        "status": "VOID" if row["voided_at"] else "ACTIVE",
        "created_at": row["created_at"],
        "trade_date": payload.get("date"),
        "row_type": "fill",
        "parent_fill_id": None,
        "action": action,
        "side": payload.get("side"),
        "cash_delta": payload.get("cash_delta", 0.0),
        "call_code": payload.get("call_code"),
        "put_code": payload.get("put_code"),
        "call_qty": payload.get("call_qty", payload.get("qty")),
        "put_qty": payload.get("put_qty", payload.get("qty")),
        "strike": payload.get("strike"),
        "expiry": payload.get("expiry"),
        "call_price": payload.get("entry_call_price", payload.get("call_price")),
        "put_price": payload.get("entry_put_price", payload.get("put_price")),
        "last_call_price": payload.get("last_call_price"),
        "last_put_price": payload.get("last_put_price"),
        "option_value": payload.get("entry_option_value"),
        "last_option_value": payload.get("last_option_value"),
        "option_margin": payload.get("option_margin"),
        "exit_reason": payload.get("exit_reason"),
        "hedge_qty": payload.get("qty", payload.get("new_etf_qty", payload.get("target_hedge_qty")))
        if "hedge" in action_lower
        else None,
        "trade_etf_qty": payload.get("trade_etf_qty") if "hedge" in action_lower else None,
        "hedge_price": payload.get("entry_price", payload.get("price"))
        if "hedge" in action_lower
        else None,
        "latest_price": payload.get("latest_price") if "hedge" in action_lower else None,
        "last_market_value": payload.get("market_value") if "hedge" in action_lower else None,
        "last_unrealized_pnl": payload.get("unrealized_pnl") if "hedge" in action_lower else None,
        "source_timestamp": payload.get("source_timestamp"),
        "security_trade_count": len(payload.get("security_trades") or []),
        "security_trade_id": None,
        "security_code": payload.get("security_code"),
        "security_direction": None,
        "security_price": None,
        "security_qty": None,
        "security_signed_qty": None,
        "security_trade_time": None,
        "trade_source_file": payload.get("trade_source_file"),
        "underlying_order_book_id": payload.get("underlying_order_book_id"),
        "short_entry_regime": payload.get("short_entry_regime"),
        "replaces_fill_id": row["replaces_fill_id"],
        "voided_at": row["voided_at"],
        "void_reason": row["void_reason"],
    }


def _security_trade_table_row(row, trade, index):
    payload = row["payload"]
    return {
        "id": f"{row['id']}.{index}",
        "status": "VOID" if row["voided_at"] else "ACTIVE",
        "created_at": row["created_at"],
        "trade_date": payload.get("date"),
        "row_type": "security_trade",
        "parent_fill_id": row["id"],
        "action": payload.get("action", row["action"]),
        "side": None,
        "cash_delta": trade.get("cash_delta", 0.0),
        "call_code": None,
        "put_code": None,
        "call_qty": None,
        "put_qty": None,
        "strike": None,
        "expiry": None,
        "call_price": None,
        "put_price": None,
        "last_call_price": None,
        "last_put_price": None,
        "option_value": None,
        "last_option_value": None,
        "option_margin": None,
        "exit_reason": None,
        "hedge_qty": payload.get("qty", payload.get("new_etf_qty")),
        "trade_etf_qty": trade.get("signed_qty"),
        "hedge_price": trade.get("price"),
        "latest_price": None,
        "last_market_value": None,
        "last_unrealized_pnl": None,
        "source_timestamp": payload.get("source_timestamp"),
        "security_trade_count": None,
        "security_trade_id": trade.get("trade_id"),
        "security_code": trade.get("security_code"),
        "security_direction": trade.get("direction"),
        "security_price": trade.get("price"),
        "security_qty": trade.get("qty"),
        "security_signed_qty": trade.get("signed_qty"),
        "security_trade_time": trade.get("trade_time"),
        "trade_source_file": payload.get("trade_source_file"),
        "underlying_order_book_id": payload.get("underlying_order_book_id"),
        "short_entry_regime": None,
        "replaces_fill_id": row["replaces_fill_id"],
        "voided_at": row["voided_at"],
        "void_reason": row["void_reason"],
    }


def list_reconciliations(product, db_path=None, account_id="default", limit=None):
    db_path = db_path or storage.account_db_path(product)
    query = """
        select * from reconciliations
        where account_id = ?
        order by id asc
    """
    params = [account_id]
    if limit is not None:
        query += " limit ?"
        params.append(int(limit))

    rows = []
    with connect(db_path) as conn:
        for row in conn.execute(query, params):
            rows.append(
                {
                    "id": row["id"],
                    "account_id": row["account_id"],
                    "payload": json.loads(row["payload"]),
                    "created_at": row["created_at"],
                }
            )
    return rows


def _position_from_fill(fill, side):
    multiplier = fill.get("contract_multiplier", 10000)
    call_qty = int(fill.get("call_qty", fill.get("qty", 0)) or 0)
    put_qty = int(fill.get("put_qty", fill.get("qty", 0)) or 0)
    return {
        "entry_date": fill.get("date"),
        "call_code": fill["call_code"],
        "put_code": fill["put_code"],
        "strike": float(fill["strike"]),
        "expiry": fill["expiry"],
        "call_qty": call_qty,
        "put_qty": put_qty,
        "entry_call_price": float(fill.get("entry_call_price", fill.get("call_price", 0.0))),
        "entry_put_price": float(fill.get("entry_put_price", fill.get("put_price", 0.0))),
        "entry_call_volume": fill.get("entry_call_volume"),
        "entry_put_volume": fill.get("entry_put_volume"),
        "entry_total_volume": fill.get("entry_total_volume"),
        "contract_multiplier": multiplier,
        "underlying_order_book_id": fill.get("underlying_order_book_id"),
        "side": side,
        "short_entry_regime": fill.get("short_entry_regime"),
        "entry_option_value": float(fill.get("entry_option_value", 0.0) or 0.0),
        "option_margin": float(fill.get("option_margin", 0.0) or 0.0),
        "last_option_value": float(fill.get("last_option_value", 0.0) or 0.0),
        "last_call_price": _optional_float(fill.get("last_call_price")),
        "last_put_price": _optional_float(fill.get("last_put_price")),
        "last_mark_date": fill.get("date"),
        "last_mark_source_file": fill.get("source_file"),
        "last_mark_source_timestamp": fill.get("source_timestamp"),
    }


def _option_hedge_from_fill(fill):
    option_type = str(fill.get("option_type") or "").lower()
    code = fill.get("order_book_id")
    if not option_type:
        if fill.get("call_code") is not None:
            option_type = "c"
            code = fill.get("call_code")
        elif fill.get("put_code") is not None:
            option_type = "p"
            code = fill.get("put_code")
    qty = int(fill.get("qty", fill.get("call_qty", fill.get("put_qty", 0))) or 0)
    price_key = "entry_call_price" if option_type == "c" else "entry_put_price"
    last_price_key = "last_call_price" if option_type == "c" else "last_put_price"
    return {
        "order_book_id": code,
        "option_type": option_type,
        "side": fill.get("side", "short"),
        "qty": qty,
        "strike": float(fill.get("strike", 0.0) or 0.0),
        "expiry": fill.get("expiry"),
        "entry_price": float(fill.get("entry_price", fill.get(price_key, 0.0)) or 0.0),
        "last_price": _optional_float(fill.get("last_price", fill.get(last_price_key))),
        "contract_multiplier": int(fill.get("contract_multiplier", 10000) or 10000),
        "underlying_order_book_id": fill.get("underlying_order_book_id"),
        "option_margin": float(fill.get("option_margin", 0.0) or 0.0),
        "last_option_value": float(fill.get("last_option_value", 0.0) or 0.0),
        "last_mark_date": fill.get("date"),
        "last_mark_source_file": fill.get("source_file"),
        "last_mark_source_timestamp": fill.get("source_timestamp"),
        "option_hedge_type": fill.get("option_hedge_type"),
        "contract_symbol": fill.get("contract_symbol"),
    }


def _upsert_option_hedge(account, fill):
    hedge = _option_hedge_from_fill(fill)
    hedge_id = _option_hedge_id(hedge)
    for index, existing in enumerate(account.option_hedges or []):
        if _option_hedge_id(existing, index) == hedge_id:
            account.option_hedges[index] = hedge
            return
    account.option_hedges.append(hedge)


def _apply_option_hedge_mark_update(account, fill):
    code = fill.get("order_book_id") or fill.get("call_code") or fill.get("put_code")
    side = fill.get("side", "short")
    for hedge in account.option_hedges or []:
        if str(hedge.get("order_book_id")) != str(code) or str(hedge.get("side")) != str(side):
            continue
        if fill.get("qty") is not None or fill.get("call_qty") is not None or fill.get("put_qty") is not None:
            hedge["qty"] = int(fill.get("qty", fill.get("call_qty", fill.get("put_qty", hedge.get("qty", 0)))) or 0)
        last_price = fill.get("last_price")
        if last_price is None and hedge.get("option_type") == "c":
            last_price = fill.get("last_call_price")
        if last_price is None and hedge.get("option_type") == "p":
            last_price = fill.get("last_put_price")
        hedge["last_price"] = _optional_float(last_price)
        if fill.get("option_margin") is not None:
            hedge["option_margin"] = float(fill.get("option_margin") or 0.0)
        if fill.get("last_option_value") is not None:
            hedge["last_option_value"] = float(fill.get("last_option_value") or 0.0)
        hedge["last_mark_date"] = fill.get("date")
        hedge["last_mark_source_file"] = fill.get("source_file")
        hedge["last_mark_source_timestamp"] = fill.get("source_timestamp")
        return
    raise ValueError(f"Cannot update missing option hedge code={code} side={side}.")


def _remove_option_hedge(account, fill):
    code = fill.get("order_book_id") or fill.get("call_code") or fill.get("put_code")
    side = fill.get("side", "short")
    account.option_hedges = [
        hedge
        for hedge in account.option_hedges or []
        if not (
            str(hedge.get("order_book_id")) == str(code)
            and str(hedge.get("side")) == str(side)
        )
    ]


def _apply_option_mark_update(account, fill):
    side = fill.get("side")
    if side not in account.positions or account.positions.get(side) is None:
        raise ValueError(f"Cannot update option mark for missing position side={side}.")
    position = account.positions[side]
    if (
        str(position.get("call_code")) != str(fill.get("call_code"))
        or str(position.get("put_code")) != str(fill.get("put_code"))
    ):
        raise ValueError(f"Option mark update does not match existing position side={side}.")
    position["last_option_value"] = float(fill.get("last_option_value", 0.0) or 0.0)
    position["last_call_price"] = _optional_float(fill.get("last_call_price"))
    position["last_put_price"] = _optional_float(fill.get("last_put_price"))
    if fill.get("option_margin") is not None:
        position["option_margin"] = float(fill.get("option_margin") or 0.0)
    position["last_mark_date"] = fill.get("date")
    position["last_mark_source_file"] = fill.get("source_file")
    position["last_mark_source_timestamp"] = fill.get("source_timestamp")


def _apply_hedge_mark_update(account, fill):
    qty = float(fill.get("qty", fill.get("new_etf_qty", account.hedge.qty)) or 0.0)
    if abs(float(account.hedge.qty or 0.0) - qty) > 1e-6:
        raise ValueError("Hedge mark update does not match existing hedge qty.")
    account.hedge.latest_price = _optional_float(fill.get("latest_price"))
    account.hedge.last_market_value = _optional_float(fill.get("market_value"))
    account.hedge.last_unrealized_pnl = _optional_float(fill.get("unrealized_pnl"))
    account.hedge.last_mark_date = fill.get("date")
    account.hedge.last_mark_source_file = fill.get("holding_source_file")
    account.hedge.last_mark_source_timestamp = fill.get("source_timestamp")


def _optional_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strategy_state_from_payload(payload):
    state = StrategyState()
    for field_name in [
        "roll_cooldown_left",
        "cooldown_total_days",
        "cooldown_started_date",
    ]:
        values = payload.get(field_name, {}) or {}
        current = getattr(state, field_name)
        for side in POSITION_SIDES:
            if side in values:
                current[side] = values[side]
    state.roll_cooldown_left = {
        side: int(state.roll_cooldown_left.get(side, 0) or 0)
        for side in POSITION_SIDES
    }
    state.cooldown_total_days = {
        side: int(state.cooldown_total_days.get(side, 0) or 0)
        for side in POSITION_SIDES
    }
    return state


def _reset_strategy_side(strategy_state, side):
    strategy_state.roll_cooldown_left[side] = 0
    strategy_state.cooldown_total_days[side] = 0
    strategy_state.cooldown_started_date[side] = None


def _start_strategy_cooldown(strategy_state, side, days, date_text):
    days = int(days or 0)
    if days <= 0:
        strategy_state.roll_cooldown_left[side] = 0
        strategy_state.cooldown_total_days[side] = 0
        strategy_state.cooldown_started_date[side] = None
        return
    strategy_state.roll_cooldown_left[side] = max(
        int(strategy_state.roll_cooldown_left.get(side, 0) or 0),
        days,
    )
    strategy_state.cooldown_total_days[side] = max(
        int(strategy_state.cooldown_total_days.get(side, 0) or 0),
        days,
    )
    strategy_state.cooldown_started_date[side] = date_text


def _roll_cooldown_days(product):
    import core

    return core.config.load_config(product).strategy.roll_cooldown_days


def _short_cooldown_after_long_iv_high_exit_days(product):
    import core

    return (
        core.config.load_config(product)
        .strategy.short_cooldown_after_long_iv_high_exit_days
    )


def _default_initial_cash(product):
    import core

    return core.config.load_config(product).backtest.initial_cash


def _insert_fill(product, fill, db_path=None, account_id="default"):
    db_path = db_path or storage.account_db_path(product)
    fill = normalize_fill(fill)
    with connect(db_path) as conn:
        _insert_fill_to_conn(conn, fill, account_id, storage.utc_now_text())
        conn.commit()


def _insert_fill_to_conn(conn, fill, account_id, now):
    conn.execute(
        """
        insert into fills(account_id, action, payload, created_at)
        values (?, ?, ?, ?)
        """,
        (
            account_id,
            fill["action"],
            json.dumps(fill, ensure_ascii=False, default=str),
            now,
        ),
    )


def record_broker_snapshot(product, snapshot, db_path=None, account_id="default"):
    db_path = db_path or storage.account_db_path(product)
    with connect(db_path) as conn:
        conn.execute(
            """
            insert into broker_snapshots(account_id, payload, created_at)
            values (?, ?, ?)
            """,
            (
                account_id,
                json.dumps(snapshot, ensure_ascii=False, default=str),
                storage.utc_now_text(),
            ),
        )
        conn.commit()
