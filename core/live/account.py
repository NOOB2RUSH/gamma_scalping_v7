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

    def to_dict(self):
        return {
            "qty": self.qty,
            "entry_price": self.entry_price,
            "margin": self.margin,
            "underlying_order_book_id": self.underlying_order_book_id,
        }


@dataclass
class AccountState:
    product: str
    account_id: str = "default"
    cash: float = 1_000_000.0
    positions: dict = field(default_factory=lambda: {"long": None, "short": None})
    hedge: HedgeState = field(default_factory=HedgeState)
    strategy_state: "StrategyState" = field(default_factory=lambda: StrategyState())
    updated_at: str | None = None

    def to_dict(self):
        return {
            "product": self.product,
            "account_id": self.account_id,
            "cash": self.cash,
            "positions": self.positions,
            "hedge": self.hedge.to_dict(),
            "strategy_state": self.strategy_state.to_dict(),
            "updated_at": self.updated_at,
        }


@dataclass
class StrategyState:
    strike_mismatch_days: dict = field(
        default_factory=lambda: {"long": 0, "short": 0}
    )
    roll_cooldown_left: dict = field(
        default_factory=lambda: {"long": 0, "short": 0}
    )
    cooldown_total_days: dict = field(
        default_factory=lambda: {"long": 0, "short": 0}
    )
    cooldown_started_date: dict = field(
        default_factory=lambda: {"long": None, "short": None}
    )
    last_signal_date: str | None = None

    def to_dict(self):
        return {
            "strike_mismatch_days": {
                side: int(self.strike_mismatch_days.get(side, 0) or 0)
                for side in POSITION_SIDES
            },
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
            "last_signal_date": self.last_signal_date,
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
        if reset:
            conn.execute("delete from account_state where account_id = ?", (account_id,))
            conn.execute("delete from option_positions where account_id = ?", (account_id,))
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
        state_row = conn.execute(
            "select * from account_state where account_id = ?",
            (account_id,),
        ).fetchone()
        if state_row is None:
            return initialize_account(product, 1_000_000.0, db_path, account_id)

        positions = {side: None for side in POSITION_SIDES}
        for row in conn.execute(
            "select side, payload from option_positions where account_id = ?",
            (account_id,),
        ):
            positions[row["side"]] = json.loads(row["payload"])

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
        product=state_row["product"],
        account_id=account_id,
        cash=float(state_row["cash"]),
        positions=positions,
        hedge=HedgeState(
            qty=float(hedge_payload.get("qty", 0.0) or 0.0),
            entry_price=float(hedge_payload.get("entry_price", 0.0) or 0.0),
            margin=float(hedge_payload.get("margin", 0.0) or 0.0),
            underlying_order_book_id=hedge_payload.get("underlying_order_book_id"),
        ),
        strategy_state=_strategy_state_from_payload(strategy_payload),
        updated_at=state_row["updated_at"],
    )


def save_account(account, db_path=None):
    db_path = db_path or storage.account_db_path(account.product)
    now = storage.utc_now_text()
    with connect(db_path) as conn:
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
        conn.commit()
    account.updated_at = now
    return account


def record_fill(product, fill, db_path=None, account_id="default"):
    """Apply a manually confirmed fill to the shadow account.

    Fill payloads are intentionally explicit. This keeps broker execution truth
    separate from model advice.
    """
    account = load_account(product, db_path, account_id)
    _apply_fill(account, product, fill)
    save_account(account, db_path)
    _insert_fill(product, fill, db_path, account_id)
    return account


def _apply_fill(account, product, fill):
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
            qty=float(hedge.get("qty", hedge.get("new_etf_qty", 0.0)) or 0.0),
            entry_price=float(hedge.get("entry_price", hedge.get("price", 0.0)) or 0.0),
            margin=float(hedge.get("margin", 0.0) or 0.0),
            underlying_order_book_id=hedge.get("underlying_order_book_id"),
        )
        account.cash += cash_delta
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
):
    db_path = db_path or storage.account_db_path(product)
    where = ["account_id = ?"]
    params = [account_id]
    if not include_voided:
        where.append("voided_at is null")
    query = (
        "select * from fills where "
        + " and ".join(where)
        + " order by id asc"
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
):
    return [
        _fill_table_row(row)
        for row in list_fills(
            product,
            db_path=db_path,
            account_id=account_id,
            include_voided=include_voided,
            limit=limit,
        )
    ]


def _fill_table_row(row):
    payload = row["payload"]
    action = payload.get("action", row["action"])
    return {
        "id": row["id"],
        "status": "VOID" if row["voided_at"] else "ACTIVE",
        "created_at": row["created_at"],
        "trade_date": payload.get("date"),
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
        "option_value": payload.get("entry_option_value"),
        "option_margin": payload.get("option_margin"),
        "exit_reason": payload.get("exit_reason"),
        "hedge_qty": payload.get("qty", payload.get("new_etf_qty"))
        if "hedge" in action
        else None,
        "hedge_price": payload.get("entry_price", payload.get("price"))
        if "hedge" in action
        else None,
        "underlying_order_book_id": payload.get("underlying_order_book_id"),
        "short_entry_regime": payload.get("short_entry_regime"),
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
    }


def _strategy_state_from_payload(payload):
    state = StrategyState()
    for field_name in [
        "strike_mismatch_days",
        "roll_cooldown_left",
        "cooldown_total_days",
        "cooldown_started_date",
    ]:
        values = payload.get(field_name, {}) or {}
        current = getattr(state, field_name)
        for side in POSITION_SIDES:
            if side in values:
                current[side] = values[side]
    state.last_signal_date = payload.get("last_signal_date")
    state.strike_mismatch_days = {
        side: int(state.strike_mismatch_days.get(side, 0) or 0)
        for side in POSITION_SIDES
    }
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
    strategy_state.strike_mismatch_days[side] = 0
    strategy_state.roll_cooldown_left[side] = 0
    strategy_state.cooldown_total_days[side] = 0
    strategy_state.cooldown_started_date[side] = None


def _start_strategy_cooldown(strategy_state, side, days, date_text):
    days = int(days or 0)
    strategy_state.strike_mismatch_days[side] = 0
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
    with connect(db_path) as conn:
        conn.execute(
            """
            insert into fills(account_id, action, payload, created_at)
            values (?, ?, ?, ?)
            """,
            (
                account_id,
                fill["action"],
                json.dumps(fill, ensure_ascii=False, default=str),
                storage.utc_now_text(),
            ),
        )
        conn.commit()


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
