from __future__ import annotations

import json
import sys
from datetime import datetime

import _bootstrap  # noqa: F401
import core
from core.live import (
    account,
    etf_importer,
    holding_importer,
    intraday_pnl,
    market_data,
    portfolio_account,
    portfolio_report,
    position_checker,
    reconciler,
    report,
    signal_engine,
    storage,
)
from core.live.runtime import load_product_config


MARK_UPDATE_ACTIONS = {
    "option_mark_update",
    "hedge_mark_update",
}


def main():
    session = {
        "products": tuple(core.config.available_live_products()),
        "account_id": "default",
    }
    while True:
        _print_header(session)
        action = _menu_choice(
            [
                ("1", "拉取并保存 AKShare 最新快照"),
                ("2", "使用本地最新快照生成/预览策略信号"),
                ("3", "导入期权/ETF 持仓成交并自动确认"),
                ("4", "生成账户报告"),
                ("5", "查看当前持仓"),
                ("6", "查看成交记录"),
                ("7", "重建账户状态"),
                ("8", "账户对账"),
                ("9", "初始化/重置账户"),
                ("10", "盘中盈亏统计"),
                ("11", "检查本地/券商持仓一致性"),
                ("0", "退出"),
            ]
        )
        try:
            if action == "1":
                _action_fetch_akshare_snapshots(session)
            elif action == "2":
                _action_quote_signal(session)
            elif action == "3":
                _action_import_holdings(session)
            elif action == "4":
                _action_account_report(session)
            elif action == "5":
                _action_show_positions(session)
            elif action == "6":
                _action_show_fills(session)
            elif action == "7":
                _action_rebuild_account(session)
            elif action == "8":
                _action_reconcile(session)
            elif action == "9":
                _action_init_account(session)
            elif action == "10":
                _action_intraday_pnl(session)
            elif action == "11":
                _action_check_positions(session)
            elif action == "0":
                print("已退出。")
                return
        except Exception as exc:
            print(f"ERROR: {exc}")
        _pause()


def _action_init_account(session):
    reset = _confirm("是否重置四个子账户并清空持仓/fills", False)
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            config = load_product_config(product)
            state = account.initialize_account(
                product,
                config.backtest.initial_cash,
                account_id=session["account_id"],
                reset=reset,
            )
            _print_account_state(state)
        except Exception as exc:
            print(f"FAILED {exc}")


def _action_quote_signal(session):
    source = "snapshot"
    date = _prompt("本地快照日期", "latest")
    snapshots = {}
    snapshot_errors = {}
    for product in session["products"]:
        try:
            snapshots[product] = market_data.fetch_quote_snapshot(
                product,
                source,
                date,
            )
        except Exception as exc:
            snapshot_errors[product] = exc

    snapshot_times = " ".join(
        f"{product}={_snapshot_time_text(snapshots.get(product))}"
        for product in session["products"]
    )
    print(f"\n快照时间: {snapshot_times}")
    for product in session["products"]:
        print(f"\n== {product} ==")
        if product in snapshot_errors:
            print(f"FAILED {snapshot_errors[product]}")
            continue
        try:
            snapshot = snapshots[product]
            signal_date = snapshot["quote_date"]
            payload = signal_engine.generate_signal(
                product,
                session["account_id"],
                signal_date,
                quote_snapshot=snapshot,
            )
            if snapshot is not None:
                payload["quote_snapshot"] = snapshot
            report_path = report.write_signal_report(product, payload)
            json_path = report_path.with_suffix(".json")
            storage.write_json(json_path, payload)
            for line in report.format_signal_summary(payload):
                print(line)
        except Exception as exc:
            print(f"FAILED {exc}")


def _snapshot_time_text(snapshot):
    if not snapshot:
        return "不可用"
    stamp = str(snapshot.get("snapshot_stamp") or "")
    try:
        return datetime.strptime(stamp, "%Y%m%d_%H%M%S").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (TypeError, ValueError):
        return stamp or "未知"


def _action_fetch_akshare_snapshots(session):
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            snapshot = market_data.fetch_quote_snapshot(
                product,
                source="akshare",
                date="latest",
            )
            print(
                f"saved={snapshot['snapshot_stamp']} "
                f"quote_date={snapshot['quote_date']} "
                f"option_rows={snapshot.get('option_rows', 0)}"
            )
        except Exception as exc:
            print(f"FAILED {exc}")


def _action_import_holdings(session):
    date = _prompt_optional("交易日期，留空则自动选择最新实时持仓和成交明细")
    include_existing = _confirm("导入总持仓，而不是仅导入今日开仓", False)
    dry_run = _confirm("先预览，不写入账户", True)
    results_by_product = {}
    for product in session["products"]:
        print(f"\n## {product}")
        results = _run_auto_imports(
            product,
            session["account_id"],
            date=date or None,
            include_existing=include_existing,
            dry_run=dry_run,
        )
        results_by_product[product] = results
        _print_auto_import_results(results)

    has_writes = any(
        _auto_import_has_writes(results)
        for results in results_by_product.values()
    )
    if dry_run and has_writes:
        if not _confirm("是否将上述内容写入四个子账户", False):
            print("PREVIEW_ONLY 未写入任何账户；如需导入，请在写入确认时输入 y。")
            return
        for product, results in results_by_product.items():
            kinds = _auto_import_writable_kinds(results)
            if not kinds:
                continue
            print(f"\n## {product}")
            confirmed = _run_auto_imports(
                product,
                session["account_id"],
                date=date or None,
                include_existing=include_existing,
                dry_run=False,
                only_kinds=kinds,
            )
            _print_auto_import_results(confirmed)
        print("IMPORT_CONFIRMED 已完成账户写入。")
    elif dry_run:
        print("PREVIEW_ONLY 没有识别到需要写入账户的新内容。")


def _run_auto_imports(product, account_id, date, include_existing, dry_run, only_kinds=None):
    kinds = set(only_kinds or ["options", "etf"])
    results = []
    if "options" in kinds:
        results.append(
            _try_auto_import(
                "options",
                lambda: holding_importer.import_holding_file(
                    product,
                    file_path=None,
                    account_id=account_id,
                    date=date,
                    include_existing=include_existing,
                    dry_run=dry_run,
                ),
            )
        )
    if "etf" in kinds:
        results.append(
            _try_auto_import(
                "etf",
                lambda: etf_importer.import_etf_files(
                    product,
                    holding_file=None,
                    trade_file=None,
                    account_id=account_id,
                    date=date,
                    dry_run=dry_run,
                ),
            )
        )
    return results


def _try_auto_import(kind, factory):
    try:
        return {"kind": kind, "result": factory(), "error": None}
    except FileNotFoundError as exc:
        return {"kind": kind, "result": None, "error": str(exc), "skipped": True}
    except Exception as exc:
        return {"kind": kind, "result": None, "error": str(exc), "skipped": False}


def _print_auto_import_results(results):
    by_kind = {item["kind"]: item for item in results}
    for kind, label in [("options", "Option"), ("etf", "ETF")]:
        item = by_kind.get(kind)
        if item is None:
            continue
        result = item.get("result")
        if result is None:
            status = "跳过" if item.get("skipped") else "失败"
            print(f"{label}: {status} - {item.get('error')}")
            continue

        applied = result.get("applied") or []
        position_changes = [
            change
            for change in applied
            if str((change.get("fill") or {}).get("action"))
            not in MARK_UPDATE_ACTIONS
        ]
        if not position_changes:
            print(f"{label}: 无变化")
            continue

        mode = (
            "预览"
            if any(change.get("dry_run") for change in position_changes)
            else "已写入"
        )
        print(f"{label}: 有变化 ({mode})")
        for change in position_changes:
            print(f"  - {_format_auto_import_change(kind, change)}")


def _auto_import_has_writes(results):
    return any(
        item.get("result") is not None and item["result"].get("applied")
        for item in results
    )


def _auto_import_writable_kinds(results):
    return [
        item["kind"]
        for item in results
        if item.get("result") is not None and item["result"].get("applied")
    ]


def _format_auto_import_change(kind, change):
    fill = change["fill"]
    prefix = "DRY_RUN" if change.get("dry_run") else "CONFIRMED"
    if kind == "etf" or _is_hedge_fill(fill):
        return _format_auto_import_etf_change(prefix, fill)
    return _format_auto_import_option_change(prefix, fill)


def _format_auto_import_etf_change(prefix, fill):
    action = fill.get("action")
    new_qty = _number_or_none(
        fill.get("target_hedge_qty", fill.get("new_etf_qty", fill.get("qty")))
    )
    trade_qty = _number_or_none(fill.get("trade_etf_qty"))
    old_qty = (
        new_qty - trade_qty
        if new_qty is not None and trade_qty is not None
        else None
    )
    qty_text = (
        f"qty={_fmt_qty(old_qty)}->{_fmt_qty(new_qty)}"
        if old_qty is not None
        else f"qty={_fmt_qty(new_qty)}"
    )
    parts = [
        prefix,
        str(action),
        qty_text,
        f"trade={_fmt_qty(trade_qty)}",
        f"entry={_fmt_optional(fill.get('entry_price'))}",
        f"latest={_fmt_optional(fill.get('latest_price'))}",
    ]
    if fill.get("cash_delta") is not None:
        parts.append(f"cash_delta={float(fill['cash_delta']):.2f}")
    if fill.get("unrealized_pnl") is not None:
        parts.append(f"unrealized={float(fill['unrealized_pnl']):.2f}")
    return " ".join(parts)


def _format_auto_import_option_change(prefix, fill):
    action = str(fill.get("action"))
    parts = [
        prefix,
        action,
        f"side={fill.get('side')}",
    ]
    code_text = _format_option_fill_codes(fill)
    if code_text:
        parts.append(code_text)
    qty_text = _format_option_fill_qty(fill)
    if qty_text:
        parts.append(qty_text)
    if fill.get("strike") is not None:
        parts.append(f"strike={fill.get('strike')}")
    if fill.get("expiry") is not None:
        parts.append(f"expiry={fill.get('expiry')}")
    price_text = _format_option_fill_prices(fill)
    if price_text:
        parts.append(price_text)
    if fill.get("last_option_value") is not None:
        parts.append(f"value={_fmt_optional(fill.get('last_option_value'))}")
    if fill.get("option_margin") is not None:
        parts.append(f"margin={_fmt_optional(fill.get('option_margin'))}")
    if fill.get("option_margin_release") is not None:
        parts.append(
            f"margin_release={_fmt_optional(fill.get('option_margin_release'))}"
        )
    if fill.get("cash_delta") is not None:
        parts.append(f"cash_delta={float(fill['cash_delta']):.2f}")
    return " ".join(parts)


def _format_option_fill_codes(fill):
    if fill.get("order_book_id") is not None:
        return f"code={fill.get('order_book_id')}"
    call = fill.get("call_code")
    put = fill.get("put_code")
    if call is None and put is None:
        return None
    return f"call={call} put={put}"


def _format_option_fill_qty(fill):
    if fill.get("qty") is not None:
        return f"qty={_fmt_qty(fill.get('qty'))}"
    call_qty = fill.get("call_qty")
    put_qty = fill.get("put_qty")
    if call_qty is None and put_qty is None:
        return None
    return f"qty={_fmt_qty(call_qty)}/{_fmt_qty(put_qty)}"


def _format_option_fill_prices(fill):
    if fill.get("price") is not None or fill.get("close_price") is not None:
        return f"price={_fmt_optional(fill.get('price', fill.get('close_price')))}"
    call_price = fill.get(
        "entry_call_price",
        fill.get("call_price", fill.get("last_call_price")),
    )
    put_price = fill.get(
        "entry_put_price",
        fill.get("put_price", fill.get("last_put_price")),
    )
    if call_price is None and put_price is None:
        return None
    return f"call_px={_fmt_optional(call_price)} put_px={_fmt_optional(put_price)}"


def _number_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_qty(value):
    number = _number_or_none(value)
    if number is None:
        return "nan"
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _action_account_report(session):
    source = "snapshot"
    date = _prompt_optional("日期，留空则最新")
    persist_history = _confirm("更新累计账户/持仓历史", True)
    write_files = _confirm("写出组合报告文件", True)
    payload = portfolio_report.build_portfolio_report(
        account_id=session["account_id"],
        products=session["products"],
        source=source,
        date=date or None,
        persist_history=persist_history,
    )
    if write_files:
        paths = portfolio_report.write_portfolio_report(payload)
        _print_report_paths(paths)
    for line in portfolio_report.format_terminal_summary(payload):
        print(line)


def _action_intraday_pnl(session):
    date = _prompt_optional("本地快照日期，留空则最新")
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            payload = intraday_pnl.calculate_intraday_pnl(
                product,
                account_id=session["account_id"],
                date=date,
            )
            for line in intraday_pnl.format_intraday_pnl(payload):
                print(line)
        except Exception as exc:
            print(f"FAILED {exc}")


def _action_check_positions(session):
    date = _prompt_optional("导出文件日期，留空则最新")
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            payload = position_checker.check_account_positions(
                product,
                account_id=session["account_id"],
                date=date or None,
            )
            for line in position_checker.format_position_check(payload):
                print(line)
        except Exception as exc:
            print(f"FAILED {exc}")


def _action_show_positions(session):
    print(
        f"\nshared_account={session['account_id']} "
        f"cash={portfolio_account.shared_cash(session['account_id'], session['products']):.2f}"
    )
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            state = account.load_account(product, account_id=session["account_id"])
            _print_account_state(state, show_cash=False, show_account=False)
        except Exception as exc:
            print(f"FAILED {exc}")


def _action_show_fills(session):
    limit = _prompt_int_optional("最多显示多少条最新成交，留空则全部")
    active_only = _confirm("只显示有效成交", False)
    table = _confirm("使用统一表格展示成交", True)
    include_voided = not active_only
    rows = []
    for product in session["products"]:
        product_rows = (
            account.list_fill_table(
                product,
                account_id=session["account_id"],
                include_voided=include_voided,
                limit=limit,
                order="desc",
                expand_security_trades=True,
            )
            if table
            else account.list_fills(
                product,
                account_id=session["account_id"],
                include_voided=include_voided,
                limit=limit,
                order="desc",
            )
        )
        rows.extend([{"品种": product, **row} for row in product_rows])
    _print_table(rows) if table else _print_json_rows(rows)


def _action_rebuild_account(session):
    if not _confirm("确认按有效 fills 重建四个子账户状态", False):
        print("已取消。")
        return
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            state = account.rebuild_account(
                product,
                account_id=session["account_id"],
            )
            _print_account_state(state)
        except Exception as exc:
            print(f"FAILED {exc}")


def _action_reconcile(session):
    start_date = _prompt_optional("开始日期，留空则只对最新日期")
    end_date = _prompt_optional("结束日期，留空则只对最新日期")
    abs_tolerance = _prompt_float(
        "绝对残差容忍度",
        reconciler.DEFAULT_ABS_TOLERANCE,
    )
    rel_tolerance = _prompt_float(
        "相对残差容忍度",
        reconciler.DEFAULT_REL_TOLERANCE,
    )
    try:
        fund_rows = reconciler.fund_reconciliation_rows(
            account_id=session["account_id"],
            products=session["products"],
            start_date=start_date,
            end_date=end_date,
        )
        for line in reconciler.format_fund_reconciliation_terminal(fund_rows):
            print(line)
    except Exception as exc:
        print(f"资金对账 FAILED {exc}")
    for product in session["products"]:
        print(f"\n== {product} ==")
        try:
            payload = reconciler.reconcile(
                product,
                account_id=session["account_id"],
                start_date=start_date,
                end_date=end_date,
                abs_tolerance=abs_tolerance,
                rel_tolerance=rel_tolerance,
            )
            report_path = reconciler.write_reconcile_report(product, payload)
            storage.write_json(report_path.with_suffix(".json"), payload)
            print(f"reconcile_report={report_path}")
            for line in reconciler.format_terminal_summary(payload, include_fund=False):
                print(line)
        except Exception as exc:
            print(f"FAILED {exc}")


def _print_header(session):
    print("")
    print("=" * 72)
    print(f"Live 组合管理 | products={','.join(session['products'])}")
    print("=" * 72)


def _menu_choice(items):
    for key, label in items:
        print(f"{key}. {label}")
    valid = {key for key, _ in items}
    while True:
        value = _prompt("选择")
        if value in valid:
            return value
        print("无效选择。")


def _prompt(label, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = _readline(f"{label}{suffix}: ").strip()
    if value == "" and default is not None:
        return str(default)
    return value


def _prompt_optional(label):
    value = _readline(f"{label}: ").strip()
    return value or None


def _prompt_choice(label, choices, default=None):
    choices = list(choices)
    default = default if default in choices else choices[0]
    prompt = f"{label} ({'/'.join(choices)})"
    while True:
        value = _prompt(prompt, default)
        if value in choices:
            return value
        print(f"请输入：{', '.join(choices)}")


def _prompt_int_optional(label):
    value = _prompt_optional(label)
    if value is None:
        return None
    return int(value)


def _prompt_float(label, default=None):
    while True:
        value = _prompt(label, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            print("请输入数字。")


def _confirm(label, default=False):
    default_text = "Y" if default else "N"
    value = _readline(f"{label} [默认: {default_text}] (y/n): ").strip().lower()
    if value == "":
        return default
    return value in {"y", "yes", "是", "1", "true"}


def _pause():
    _readline("\n按 Enter 返回菜单...")


def _readline(prompt):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return input(prompt)
    if sys.platform.startswith("win"):
        return _readline_windows(prompt)
    return _readline_posix(prompt)


def _readline_windows(prompt):
    import msvcrt

    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars = []
    while True:
        char = msvcrt.getwch()
        if char in {"\r", "\n"}:
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            return "".join(chars)
        if char == "\x03":
            raise KeyboardInterrupt
        if char == "\x1a":
            raise EOFError
        if char in {"\b", "\x7f"}:
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if char in {"\x00", "\xe0"}:
            msvcrt.getwch()
            continue
        if char == "\x15":
            while chars:
                chars.pop()
                sys.stdout.write("\b \b")
            sys.stdout.flush()
            continue
        chars.append(char)
        sys.stdout.write(char)
        sys.stdout.flush()


def _readline_posix(prompt):
    import termios
    import tty

    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars = []
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            char = sys.stdin.read(1)
            if char in {"\r", "\n"}:
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(chars)
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x04":
                raise EOFError
            if char in {"\b", "\x7f"}:
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if char == "\x15":
                while chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                sys.stdout.flush()
                continue
            if char == "\x1b":
                _consume_escape_sequence()
                continue
            chars.append(char)
            sys.stdout.write(char)
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _consume_escape_sequence():
    if not sys.stdin.isatty():
        return
    import select

    for _ in range(2):
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if not ready:
            return
        sys.stdin.read(1)


def _print_account_state(state, show_cash=True, show_account=True):
    if show_account:
        print(f"account={state.product}/{state.account_id}")
    if show_cash:
        print(f"cash={state.cash:.2f}")
    print(
        "hedge="
        f"qty={state.hedge.qty} "
        f"entry_price={state.hedge.entry_price} "
        f"margin={state.hedge.margin} "
        f"underlying={state.hedge.underlying_order_book_id}"
    )
    for side, position in state.positions.items():
        if position is None:
            print(f"position.{side}=None")
        else:
            expiry = position.get("expiry", position.get("maturity_date"))
            strike = position.get("strike", position.get("strike_price"))
            print(
                f"position.{side}={position.get('call_code')}/{position.get('put_code')} "
                f"qty={position.get('call_qty')}/{position.get('put_qty')} "
                f"strike={strike} expiry={expiry}"
            )
def _print_import_result(result):
    if "file" in result:
        print(f"file={result['file']}")
    if "holding_file" in result:
        print(f"holding_file={result['holding_file']}")
    if "trade_file" in result:
        print(f"trade_file={result['trade_file']}")
    print(f"trade_date={result['trade_date']}")
    if "input_rows" in result:
        print(f"input_rows={result['input_rows']} usable_rows={result['usable_rows']}")
    if "holding_rows" in result:
        print(
            f"holding_rows={result['holding_rows']} "
            f"trade_rows={result['trade_rows']} "
            f"matched_trade_rows={result['matched_trade_rows']}"
        )
    print(f"dry_run={result['dry_run']}")
    for item in result["applied"]:
        fill = item["fill"]
        prefix = "DRY_RUN" if item["dry_run"] else "CONFIRMED"
        if _is_hedge_fill(fill):
            print(
                f"{prefix} {fill['action']} "
                f"target_qty={fill['target_hedge_qty']:.0f} "
                f"trade_qty={fill.get('trade_etf_qty', 0.0):.0f} "
                f"entry_price={fill['entry_price']:.6f} "
                f"trade_price={_fmt_optional(fill.get('price'))} "
                f"latest_price={_fmt_optional(fill.get('latest_price'))} "
                f"cash_delta={fill['cash_delta']:.2f}"
            )
        elif fill["action"] == "option_mark_update":
            print(
                f"{prefix} {fill['action']} side={fill.get('side')} "
                f"qty={fill.get('call_qty')}/{fill.get('put_qty')} "
                f"call={fill.get('call_code')} put={fill.get('put_code')} "
                f"last_call_px={_fmt_optional(fill.get('last_call_price'))} "
                f"last_put_px={_fmt_optional(fill.get('last_put_price'))} "
                f"last_option_value={_fmt_optional(fill.get('last_option_value'))} "
                f"cash_delta={fill['cash_delta']:.2f}"
            )
        else:
            print(
                f"{prefix} {fill['action']} side={fill['side']} "
                f"qty={fill['call_qty']}/{fill['put_qty']} "
                f"call={fill['call_code']} put={fill['put_code']} "
                f"cash_delta={fill['cash_delta']:.2f}"
            )
    for item in result["skipped"]:
        fill = item["fill"]
        if _is_hedge_fill(fill):
            print(
                f"SKIPPED reason={item['reason']} "
                f"target_qty={fill['target_hedge_qty']:.0f} "
                f"entry_price={fill['entry_price']:.6f}"
            )
        else:
            print(
                f"SKIPPED side={item['side']} reason={item['reason']} "
                f"call={fill['call_code']} put={fill['put_code']}"
            )
    for warning in result["warnings"]:
        print(f"WARNING {warning}")


def _is_hedge_fill(fill):
    return fill.get("action") in {
        "delta_hedge",
        "rebalance_hedge",
        "close_hedge",
        "hedge_mark_update",
    }


def _fmt_optional(value):
    if value is None:
        return "nan"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _print_report_paths(paths):
    if "total_excel" in paths:
        print(f"portfolio_report_total_excel={paths['total_excel']}")
    if "json" in paths:
        print(f"portfolio_report_json={paths['json']}")


def _print_dict(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _print_json_rows(rows):
    if not rows:
        print("(none)")
        return
    for row in rows:
        _print_dict(row)


def _print_table(rows):
    if not rows:
        print("(none)")
        return
    columns = []
    for row in rows:
        for column in row.keys():
            if column not in columns:
                columns.append(column)
    widths = {
        column: min(
            24,
            max(len(str(column)), *[len(_cell(row.get(column))) for row in rows]),
        )
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            " | ".join(
                _cell(row.get(column))[: widths[column]].ljust(widths[column])
                for column in columns
            )
        )


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
