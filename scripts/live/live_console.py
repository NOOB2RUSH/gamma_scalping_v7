from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
import core
from core.live import (
    account,
    account_report,
    hedge_importer,
    holding_importer,
    market_data,
    reconciler,
    report,
    signal_engine,
    storage,
)
from core.live.runtime import load_product_config


def main():
    session = {
        "product": _choose_product(),
        "account_id": _prompt("账户ID", "default"),
    }
    while True:
        _print_header(session)
        action = _menu_choice(
            [
                ("1", "初始化/重置账户"),
                ("2", "拉取行情并生成/预览策略信号"),
                ("3", "导入期权/ETF 持仓成交并自动确认"),
                ("4", "生成账户报告"),
                ("5", "查看账户与成交记录"),
                ("6", "导出成交记录表"),
                ("7", "修改/作废已确认成交"),
                ("8", "重建账户状态"),
                ("9", "账户对账"),
                ("10", "切换品种/账户"),
                ("0", "退出"),
            ]
        )
        try:
            if action == "1":
                _action_init_account(session)
            elif action == "2":
                _action_quote_signal(session)
            elif action == "3":
                _action_import_holdings(session)
            elif action == "4":
                _action_account_report(session)
            elif action == "5":
                _action_show_account(session)
            elif action == "6":
                _action_export_fills(session)
            elif action == "7":
                _action_amend_fill(session)
            elif action == "8":
                _action_rebuild_account(session)
            elif action == "9":
                _action_reconcile(session)
            elif action == "10":
                session["product"] = _choose_product(session["product"])
                session["account_id"] = _prompt("账户ID", session["account_id"])
            elif action == "0":
                print("已退出。")
                return
        except Exception as exc:
            print(f"ERROR: {exc}")
        _pause()


def _action_init_account(session):
    product = session["product"]
    config = load_product_config(product)
    cash = _prompt_float("初始现金", config.backtest.initial_cash)
    reset = _confirm("是否重置已有账户并清空持仓/fills", False)
    state = account.initialize_account(
        product,
        cash,
        account_id=session["account_id"],
        reset=reset,
    )
    _print_account_state(state)


def _action_quote_signal(session):
    source = _prompt_choice("行情源", ["akshare", "local", "none"], "akshare")
    if source == "none":
        date = _prompt_optional("日期，留空则使用已有数据最新日期")
        snapshot = None
        signal_date = date or None
    else:
        date = _prompt("行情日期", "latest")
        snapshot = market_data.fetch_quote_snapshot(session["product"], source, date)
        signal_date = snapshot["quote_date"]
        print("snapshot saved")
        _print_dict(snapshot)

    payload = signal_engine.generate_signal(
        session["product"],
        session["account_id"],
        signal_date,
        quote_snapshot=snapshot,
    )
    if snapshot is not None:
        payload["quote_snapshot"] = snapshot
    report_path = report.write_signal_report(session["product"], payload)
    json_path = report_path.with_suffix(".json")
    storage.write_json(json_path, payload)
    print(f"quote_date={payload['date']}")
    print(f"signal_report={report_path}")
    print(f"signal_json={json_path}")
    print("read_only=True")
    for line in report.format_signal_summary(payload):
        print(line)


def _action_confirm_fill(session):
    payload = _read_fill_or_signal_payload()
    fill = _select_fill_payload(payload)
    fill = _edit_fill_payload(fill)
    print("")
    print("即将确认成交：")
    _print_dict(account.normalize_fill(fill))
    if not _confirm("确认写入账户", False):
        print("已取消。")
        return
    state = account.record_fill(
        session["product"],
        fill,
        account_id=session["account_id"],
    )
    print(f"fill_applied={account.normalize_fill(fill)['action']}")
    _print_account_state(state)


def _action_import_holdings(session):
    date = _prompt_optional("交易日期，留空则分别自动选择最新 option/hedge 导出")
    include_existing = _confirm("导入总持仓，而不是仅导入今日开仓", False)
    dry_run = _confirm("先预览，不写入账户", True)
    results = _run_auto_imports(
        session,
        date=date or None,
        include_existing=include_existing,
        dry_run=dry_run,
    )
    _print_auto_import_results(results)

    if dry_run and _auto_import_has_writes(results) and _confirm("是否将上述内容写入账户", False):
        results = _run_auto_imports(
            session,
            date=date or None,
            include_existing=include_existing,
            dry_run=False,
            only_kinds=_auto_import_writable_kinds(results),
        )
        _print_auto_import_results(results)


def _run_auto_imports(session, date, include_existing, dry_run, only_kinds=None):
    kinds = set(only_kinds or ["option", "hedge"])
    results = []
    if "option" in kinds:
        results.append(
            _try_auto_import(
                "option",
                lambda: holding_importer.import_holding_file(
                    session["product"],
                    file_path=None,
                    account_id=session["account_id"],
                    date=date,
                    include_existing=include_existing,
                    dry_run=dry_run,
                ),
            )
        )
    if "hedge" in kinds:
        results.append(
            _try_auto_import(
                "hedge",
                lambda: hedge_importer.import_hedge_files(
                    session["product"],
                    holding_file=None,
                    trade_file=None,
                    account_id=session["account_id"],
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
    for item in results:
        print("")
        print(f"== {item['kind']} import ==")
        if item.get("result") is not None:
            _print_import_result(item["result"])
            continue
        label = "SKIPPED" if item.get("skipped") else "FAILED"
        print(f"{label} reason={item.get('error')}")


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


def _action_import_option(session):
    file_path = _prompt_optional("实时持仓导出文件，留空则自动选择 live_hold/ 最新实时持仓文件")
    date = _prompt_optional("交易日期，留空则从文件名解析")
    include_existing = _confirm("导入总持仓，而不是仅导入今日开仓", False)
    dry_run = _confirm("先预览，不写入账户", True)
    result = holding_importer.import_holding_file(
        session["product"],
        file_path=file_path or None,
        account_id=session["account_id"],
        date=date or None,
        include_existing=include_existing,
        dry_run=dry_run,
    )
    _print_import_result(result)
    if dry_run and _confirm_import_write(result):
        result = holding_importer.import_holding_file(
            session["product"],
            file_path=file_path or None,
            account_id=session["account_id"],
            date=date or None,
            include_existing=include_existing,
            dry_run=False,
        )
        _print_import_result(result)


def _action_import_hedge(session):
    holding_file = _prompt_optional("证券持仓查询文件，留空则自动选择 live_hold/ 最新证券持仓查询文件")
    trade_file = _prompt_optional("证券委托查询文件，留空则自动选择 live_hold/ 最新证券委托文件")
    date = _prompt_optional("交易日期，留空则从文件名解析")
    dry_run = _confirm("先预览，不写入账户", True)
    result = hedge_importer.import_hedge_files(
        session["product"],
        holding_file=holding_file or None,
        trade_file=trade_file or None,
        account_id=session["account_id"],
        date=date or None,
        dry_run=dry_run,
    )
    _print_import_result(result)
    if dry_run and _confirm_import_write(result):
        result = hedge_importer.import_hedge_files(
            session["product"],
            holding_file=holding_file or None,
            trade_file=trade_file or None,
            account_id=session["account_id"],
            date=date or None,
            dry_run=False,
        )
        _print_import_result(result)


def _action_account_report(session):
    source = _prompt_choice("行情源", ["akshare", "local", "none"], "akshare")
    date = _prompt_optional("日期，留空则最新")
    output_format = _prompt_choice("输出格式", ["excel", "csv", "both"], "excel")
    persist_history = _confirm("更新累计账户/持仓历史", True)
    write_files = _confirm("写出报告文件", True)
    payload = account_report.build_live_account_report(
        session["product"],
        account_id=session["account_id"],
        source=source,
        date=date or None,
        persist_history=persist_history,
    )
    if write_files:
        paths = account_report.write_live_account_report(
            session["product"],
            payload,
            output_format=output_format,
        )
        _print_report_paths(paths)
    for line in account_report.format_terminal_summary(payload):
        print(line)


def _action_show_account(session):
    limit = _prompt_int_optional("最多显示多少条最新成交，留空则全部")
    active_only = _confirm("只显示有效成交", False)
    table = _confirm("使用统一表格展示成交", True)
    state = account.load_account(session["product"], account_id=session["account_id"])
    _print_account_state(state)
    include_voided = not active_only
    if table:
        rows = account.list_fill_table(
            session["product"],
            account_id=session["account_id"],
            include_voided=include_voided,
            limit=limit,
            order="desc",
            expand_security_trades=True,
        )
        _print_table(rows)
    else:
        rows = account.list_fills(
            session["product"],
            account_id=session["account_id"],
            include_voided=include_voided,
            limit=limit,
            order="desc",
        )
        _print_json_rows(rows)


def _action_export_fills(session):
    active_only = _confirm("只导出有效成交", False)
    limit = _prompt_int_optional("最多导出多少条，留空则全部")
    out = _prompt_optional("输出 CSV 路径，留空则写到 output/live/<product>")
    rows = account.list_fill_table(
        session["product"],
        account_id=session["account_id"],
        include_voided=not active_only,
        limit=limit,
    )
    df = pd.DataFrame(rows)
    path = Path(out) if out else storage.output_dir(session["product"]) / f"{storage.local_now_stamp()}_fill_table.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"fill_table_csv={path}")
    print(f"rows={len(df)}")


def _action_amend_fill(session):
    fill_id = _prompt_int("要作废/修改的 fill id")
    reason = _prompt("原因", "amended_by_user")
    replacement = None
    if _confirm("是否插入替换成交", False):
        replacement = _select_fill_payload(_read_fill_or_signal_payload())
        replacement = _edit_fill_payload(replacement)
    initial_cash = _prompt_float_optional("重建账户使用的初始现金，留空则使用产品配置")
    result = account.amend_fill(
        session["product"],
        fill_id,
        replacement_fill=replacement,
        reason=reason,
        account_id=session["account_id"],
        initial_cash=initial_cash,
    )
    print(f"voided_fill_id={result['voided_fill_id']}")
    if result["replacement_fill_id"] is not None:
        print(f"replacement_fill_id={result['replacement_fill_id']}")
    _print_account_state(result["account"])


def _action_rebuild_account(session):
    initial_cash = _prompt_float_optional("重建账户使用的初始现金，留空则使用产品配置")
    if not _confirm("确认按有效 fills 重建账户状态", False):
        print("已取消。")
        return
    state = account.rebuild_account(
        session["product"],
        account_id=session["account_id"],
        initial_cash=initial_cash,
    )
    _print_account_state(state)


def _action_reconcile(session):
    start_date = _prompt_optional("开始日期，留空则从第二条历史开始")
    end_date = _prompt_optional("结束日期，留空则到最新历史")
    abs_tolerance = _prompt_float(
        "绝对残差容忍度",
        reconciler.DEFAULT_ABS_TOLERANCE,
    )
    rel_tolerance = _prompt_float(
        "相对残差容忍度",
        reconciler.DEFAULT_REL_TOLERANCE,
    )
    payload = reconciler.reconcile(
        session["product"],
        account_id=session["account_id"],
        start_date=start_date,
        end_date=end_date,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    report_path = reconciler.write_reconcile_report(session["product"], payload)
    storage.write_json(report_path.with_suffix(".json"), payload)
    print(f"reconcile_report={report_path}")
    for line in reconciler.format_terminal_summary(payload):
        print(line)


def _read_fill_or_signal_payload():
    path = _prompt_optional("成交/信号 JSON 文件路径，留空则直接粘贴 JSON")
    if path:
        return storage.read_json(path)
    text = _prompt("粘贴一行 JSON")
    return json.loads(text)


def _select_fill_payload(payload):
    if isinstance(payload, dict) and "advice" in payload:
        advice = [
            item for item in payload["advice"]
            if item.get("priority") == "action"
        ]
        if not advice:
            raise ValueError("Signal JSON does not contain actionable advice.")
        print("可确认的策略建议：")
        for idx, item in enumerate(advice, start=1):
            side = f" side={item.get('side')}" if item.get("side") else ""
            reason = item.get("reason", "")
            print(f"{idx}. {item.get('action')}{side} reason={reason}")
        index = _prompt_int("选择建议序号", 1)
        if index < 1 or index > len(advice):
            raise ValueError("Invalid advice index.")
        fill = dict(advice[index - 1])
        fill.setdefault("date", payload.get("date"))
        return fill
    if isinstance(payload, dict):
        return payload
    raise ValueError("Payload must be a JSON object.")


def _edit_fill_payload(fill):
    result = dict(fill)
    action = str(result.get("action", "")).lower()
    result["date"] = _prompt("成交日期", result.get("date") or pd.Timestamp.today().date())

    if "straddle" in action:
        result["side"] = _prompt_choice("方向", ["short", "long"], result.get("side") or "short")
        for key, label in [
            ("call_code", "Call 合约代码"),
            ("put_code", "Put 合约代码"),
            ("strike", "行权价"),
            ("expiry", "到期日"),
        ]:
            default = result.get(key) or result.get(f"target_{key}")
            result[key] = _prompt(label, default)
        result["call_qty"] = _prompt_int("Call 张数", result.get("call_qty") or result.get("target_call_qty") or result.get("qty") or 0)
        result["put_qty"] = _prompt_int("Put 张数", result.get("put_qty") or result.get("target_put_qty") or result.get("qty") or 0)
        result["entry_call_price"] = _prompt_float(
            "实际 Call 成交价",
            result.get("entry_call_price", result.get("estimated_call_price", result.get("call_price", 0))),
        )
        result["entry_put_price"] = _prompt_float(
            "实际 Put 成交价",
            result.get("entry_put_price", result.get("estimated_put_price", result.get("put_price", 0))),
        )
        result["entry_option_value"] = _prompt_float(
            "成交权利金合计/期权价值",
            result.get("entry_option_value", result.get("estimated_trade_value", 0)),
        )
        result["option_margin"] = _prompt_float(
            "占用保证金",
            result.get("option_margin", result.get("estimated_option_margin", 0)),
        )
        result["cash_delta"] = _prompt_float(
            "现金变动 cash_delta",
            result.get("cash_delta", result.get("estimated_cash_effect", 0)),
        )
    elif "hedge" in action:
        result["new_etf_qty"] = _prompt_float(
            "对冲后目标 ETF 持仓数量",
            result.get("new_etf_qty", result.get("target_hedge_qty", result.get("qty", 0))),
        )
        result["qty"] = result["new_etf_qty"]
        result["price"] = _prompt_float(
            "实际 ETF 成交价",
            result.get("price", result.get("entry_price", result.get("estimated_price", 0))),
        )
        result["entry_price"] = result["price"]
        result["margin"] = _prompt_float("对冲保证金", result.get("margin", 0))
        result["cash_delta"] = _prompt_float("现金变动 cash_delta", result.get("cash_delta", 0))
        underlying = result.get("underlying_order_book_id")
        result["underlying_order_book_id"] = _prompt("标的代码", underlying or "")
    elif action == "cash_adjustment":
        result["cash_delta"] = _prompt_float("现金变动 cash_delta", result.get("cash_delta", 0))
    else:
        if _confirm("是否逐字段确认/编辑 JSON", False):
            text = _prompt("粘贴修改后的完整 JSON")
            result = json.loads(text)
    return result


def _choose_product(default=None):
    products = list(core.config.available_products())
    default = default or ("kc50etf" if "kc50etf" in products else products[0])
    return _prompt_choice("品种", products, default)


def _print_header(session):
    print("")
    print("=" * 72)
    print(f"Live 账户管理 | product={session['product']} account={session['account_id']}")
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


def _prompt_int(label, default=None):
    while True:
        value = _prompt(label, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            print("请输入整数。")


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


def _prompt_float_optional(label):
    value = _prompt_optional(label)
    if value is None:
        return None
    return float(value)


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


def _print_account_state(state):
    print(f"account={state.product}/{state.account_id}")
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
            print(
                f"position.{side}={position['call_code']}/{position['put_code']} "
                f"qty={position['call_qty']}/{position['put_qty']} "
                f"strike={position['strike']} expiry={position['expiry']}"
            )


def _confirm_import_write(result):
    if not result.get("applied"):
        print("没有可写入账户的导入项。")
        return False
    if all(item.get("dry_run") is False for item in result["applied"]):
        return False
    return _confirm("是否将上述内容写入账户", False)


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
        elif fill["action"] in {"option_mark_update", "option_hedge_mark_update"}:
            print(
                f"{prefix} {fill['action']} side={fill.get('side')} "
                f"qty={fill.get('call_qty')}/{fill.get('put_qty')} "
                f"call={fill.get('call_code')} put={fill.get('put_code')} "
                f"last_call_px={_fmt_optional(fill.get('last_call_price'))} "
                f"last_put_px={_fmt_optional(fill.get('last_put_price'))} "
                f"last_option_value={_fmt_optional(fill.get('last_option_value'))} "
                f"cash_delta={fill['cash_delta']:.2f}"
            )
        elif fill["action"] in {"open_option_hedge", "close_option_hedge"}:
            print(
                f"{prefix} {fill['action']} side={fill.get('side')} "
                f"qty={fill.get('call_qty')}/{fill.get('put_qty')} "
                f"call={fill.get('call_code')} put={fill.get('put_code')} "
                f"price={_fmt_optional(fill.get('price', fill.get('entry_price')))} "
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
    if "excel" in paths:
        print(f"account_report_excel={paths['excel']}")
    if "csv" in paths:
        for sheet_name, path in paths["csv"].items():
            print(f"account_report_csv[{sheet_name}]={path}")
    if "json" in paths:
        print(f"account_report_json={paths['json']}")
    if "diagnostics" in paths:
        print(f"account_report_diagnostics={paths['diagnostics']}")


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
