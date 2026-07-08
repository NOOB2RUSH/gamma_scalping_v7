from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

try:
    from pythongo import BaseParams, Field
    from pythongo.base import BaseStrategy
except Exception:  # pragma: no cover - only available inside InfiniTrader.
    BaseParams = object
    BaseStrategy = object

    def Field(default=None, title=None):
        del title
        return default


class Params(BaseParams):
    project_root = Field(
        default=r"C:\Users\交易员\strategy\gamma_scalping_v7",
        title="策略工程目录",
    )
    product = Field(default="300etf", title="产品")
    account_id = Field(default="default", title="本地账户ID")
    quote_source = Field(default="snapshot", title="行情源")
    quote_date = Field(default="latest", title="行情日期")
    command_file = Field(default="", title="本地命令文件")
    investor = Field(default="", title="报单账号")
    dry_run = Field(default=True, title="只生成计划不发单")
    run_once = Field(default=True, title="启动后只执行一次")


class GammaScalpingAuto(BaseStrategy):
    """InfiniTrader PythonGO bridge for the local gamma scalping live strategy."""

    author = "gamma_scalping_v7"
    default_params = Params()

    def __init__(self):
        super().__init__()
        self._submitted = False
        self._orders = []
        self._runtime_dir = None
        self._command = None

    def on_start(self):
        self._run_once_if_needed()

    def on_tick(self, tick):
        if not self.params.run_once:
            self._run_once_if_needed()

    def on_order(self, order):
        self._write_event("order", _to_plain_dict(order))

    def on_trade(self, trade):
        self._write_event("trade", _to_plain_dict(trade))

    def _run_once_if_needed(self):
        if self.params.run_once and self._submitted:
            return
        self._submitted = True
        self._bootstrap_project()
        orders = self._build_orders()
        self._orders = orders
        self._write_plan(orders)
        if self.params.dry_run:
            self.output(f"dry_run=True orders={len(orders)}")
            return
        for order in orders:
            self._submit_order(order)
        self._archive_command()

    def _bootstrap_project(self):
        project_root = Path(str(self.params.project_root)).resolve()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        self._runtime_dir = (
            project_root
            / "output"
            / "live"
            / str(self.params.product)
            / "infinitrader"
        )
        self._runtime_dir.mkdir(parents=True, exist_ok=True)

    def _build_orders(self):
        from core.live import infinitrader, market_data, signal_engine

        command = self._load_command()
        if command is not None:
            self._command = command
            self._write_json("loaded_command", command)
            return command.get("orders", [])

        snapshot = market_data.fetch_quote_snapshot(
            str(self.params.product),
            str(self.params.quote_source),
            str(self.params.quote_date),
        )
        payload = signal_engine.generate_signal(
            str(self.params.product),
            str(self.params.account_id),
            snapshot["quote_date"],
            quote_snapshot=snapshot,
        )
        payload["quote_snapshot"] = snapshot
        self._write_json("signal", payload)
        orders = infinitrader.compile_signal_orders(payload)
        self._command = {
            "run_id": _stamp(),
            "product": str(self.params.product),
            "account_id": str(self.params.account_id),
            "date": payload.get("date"),
            "signal": payload,
            "orders": orders,
        }
        return orders

    def _load_command(self):
        explicit = str(self.params.command_file or "").strip()
        if explicit:
            path = Path(explicit)
        else:
            path = self._runtime_path("pending_command.json")
        if not path.exists():
            return None
        command = json.loads(path.read_text(encoding="utf-8"))
        command["_loaded_from"] = str(path)
        return command

    def _submit_order(self, order):
        if order["offset"] is None:
            order_id = self.send_order(
                exchange=order["exchange"],
                instrument_id=order["instrument_id"],
                volume=order["volume"],
                price=order["price"],
                order_direction=order["order_direction"],
                order_type=order["order_type"],
                investor=str(self.params.investor),
                hedgeflag=order["hedgeflag"],
                market=order["market"],
                memo=self._order_memo(order),
            )
        else:
            order_id = self.make_order_req(
                exchange=order["exchange"],
                instrument_id=order["instrument_id"],
                volume=order["volume"],
                price=order["price"],
                order_direction=order["order_direction_type"],
                offset=order["offset"],
                order_type=order["order_type"],
                investor=str(self.params.investor),
                hedgeflag=order["hedgeflag"],
                market=order["market"],
                memo=self._order_memo(order),
            )
        payload = dict(order)
        payload["order_id"] = order_id
        payload["run_id"] = self._run_id()
        self._write_event("submit", payload)
        self.output(
            "submit seq={sequence} {exchange}.{instrument_id} "
            "{order_direction} offset={offset} qty={volume} price={price}".format(
                **order
            )
        )

    def _write_plan(self, orders):
        self._write_json(
            "plan",
            {
                "product": str(self.params.product),
                "account_id": str(self.params.account_id),
                "dry_run": bool(self.params.dry_run),
                "run_id": self._run_id(),
                "orders": orders,
            },
        )

    def _archive_command(self):
        loaded_from = self._command.get("_loaded_from")
        if not loaded_from:
            return
        source = Path(loaded_from)
        if not source.exists() or source.name != "pending_command.json":
            return
        target = self._runtime_path(f"executed_{self._run_id()}.json")
        source.replace(target)

    def _run_id(self):
        if self._command:
            return str(self._command.get("run_id") or "")
        return ""

    def _order_memo(self, order):
        base = str(order.get("memo") or "")
        run_id = self._run_id()
        return f"{run_id}:{base}" if run_id else base

    def _write_json(self, name, payload):
        path = self._runtime_path(f"{name}_{_stamp()}.json")
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_event(self, event_type, payload):
        path = self._runtime_path(f"events_{pd.Timestamp.now():%Y%m%d}.jsonl")
        row = {
            "ts": pd.Timestamp.now().isoformat(),
            "event": event_type,
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _runtime_path(self, name):
        if self._runtime_dir is None:
            self._bootstrap_project()
        return self._runtime_dir / name


def _stamp():
    return pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def _to_plain_dict(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    result = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if callable(item):
            continue
        result[key] = item
    return result
