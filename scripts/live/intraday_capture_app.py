from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, Button, Checkbutton, Entry, Frame, IntVar, Label, Listbox, Scrollbar, StringVar, Text, Tk, Toplevel, messagebox
from tkinter.ttk import Progressbar

import _bootstrap  # noqa: F401
import core
from core.live import runtime, storage


APP_TITLE = "Intraday Option Capture"
DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_ACCOUNT_ID = "default"
CAPTURE_WORKER_ARG = "--capture-worker"


def configure_project_root():
    root = _resolve_project_root()
    runtime.PROJECT_ROOT = root
    storage.PROJECT_ROOT = root
    return root


def _resolve_project_root() -> Path:
    env_root = os.environ.get("GAMMA_SCALPING_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.extend([Path.cwd().resolve(), exe_dir, *exe_dir.parents])
    else:
        script_root = Path(__file__).resolve().parents[2]
        candidates.extend([script_root, Path.cwd().resolve(), *script_root.parents])

    for candidate in candidates:
        if (candidate / "run.py").exists() and (candidate / "core").exists():
            return candidate
    return candidates[0]


PROJECT_ROOT = configure_project_root()


@dataclass
class CaptureStatus:
    product: str
    pid: int | None
    running: bool
    latest_file: Path | None
    latest_file_mtime: str | None
    log_path: Path
    pid_path: Path


def pid_path(product: str) -> Path:
    return storage.output_dir(product) / "intraday_capture.pid"


def log_path(product: str) -> Path:
    return storage.output_dir(product) / "intraday_capture.log"


def read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    return pid if pid > 0 else None


def process_running(pid: int | None) -> bool:
    if pid is None:
        return False
    if os.name == "nt":
        return _windows_process_running(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _windows_process_running(pid: int) -> bool:
    try:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=_creation_flags(),
        )
    except Exception:
        return False
    return str(pid) in completed.stdout


def stop_process(pid: int | None) -> bool:
    if pid is None:
        return False
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            creationflags=_creation_flags(),
        )
        return completed.returncode == 0
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    return True


def latest_intraday_file(product: str) -> tuple[Path | None, str | None]:
    root = Path(storage.PROJECT_ROOT) / "data" / "live" / product / "intraday"
    if not root.exists():
        return None, None
    candidates = [
        path
        for path in root.rglob("option_*")
        if path.is_file()
        and path.name.endswith(("_1m.csv", "_spot_snapshots.csv", "_greeks_snapshots.csv"))
    ]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    stamp = _format_mtime(latest.stat().st_mtime)
    return latest, stamp


def capture_status(product: str) -> CaptureStatus:
    p_path = pid_path(product)
    pid = read_pid(p_path)
    latest, mtime = latest_intraday_file(product)
    return CaptureStatus(
        product=product,
        pid=pid,
        running=process_running(pid),
        latest_file=latest,
        latest_file_mtime=mtime,
        log_path=log_path(product),
        pid_path=p_path,
    )


def intraday_status_summary(product: str) -> str:
    path = Path(storage.PROJECT_ROOT) / "data" / "live" / product / "intraday" / "status.json"
    if not path.exists():
        return "dates=-"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "dates=invalid"
    rows = payload.get("dates", {})
    total = len(rows)
    complete = sum(1 for row in rows.values() if row.get("complete"))
    return f"dates={complete}/{total}"


def start_capture(
    product: str,
    account_id: str,
    interval_seconds: int,
    extra_option_codes: list[str] | None = None,
    once: bool = False,
    save_option_greeks_snapshot: bool = False,
    all_option_contracts: bool = False,
) -> subprocess.Popen:
    status = capture_status(product)
    if status.running:
        raise RuntimeError(f"{product} capture is already running with pid={status.pid}")

    status.pid_path.parent.mkdir(parents=True, exist_ok=True)
    status.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = capture_command(
        product,
        account_id,
        interval_seconds,
        status.pid_path,
        extra_option_codes=extra_option_codes,
        once=once,
        save_option_greeks_snapshot=save_option_greeks_snapshot,
        all_option_contracts=all_option_contracts,
    )

    log = status.log_path.open("a", encoding="utf-8")
    mode = "once" if once else "continuous"
    log.write(f"\n[capture_mode] product={product} mode={mode}\n")
    log.write(f"[start] command={' '.join(command)}\n")
    log.flush()
    return subprocess.Popen(
        command,
        cwd=str(storage.PROJECT_ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=_creation_flags(),
    )


def capture_command(
    product: str,
    account_id: str,
    interval_seconds: int,
    worker_pid_path: Path,
    extra_option_codes: list[str] | None = None,
    once: bool = False,
    save_option_greeks_snapshot: bool = False,
    all_option_contracts: bool = False,
) -> list[str]:
    if getattr(sys, "frozen", False):
        command = [sys.executable, CAPTURE_WORKER_ARG]
    else:
        command = [
            sys.executable,
            str(Path(__file__).with_name("capture_intraday_quotes.py")),
        ]
    command.extend(
        [
            "--product",
            product,
            "--account-id",
            account_id,
            "--interval-seconds",
            str(interval_seconds),
            "--pid-file",
            str(worker_pid_path),
        ]
    )
    if once:
        command.append("--once")
    if save_option_greeks_snapshot:
        command.append("--save-option-greeks-snapshot")
    if all_option_contracts:
        command.append("--all-option-contracts")
    for code in extra_option_codes or []:
        clean = str(code).strip()
        if clean:
            command.extend(["--option-code", clean])
    return command


def run_capture_worker(argv: list[str]):
    import capture_intraday_quotes

    sys.argv = ["capture_intraday_quotes.py", *argv]
    capture_intraday_quotes.main()


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )


def _format_mtime(value: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Cannot read log: {exc}"
    return "\n".join(lines[-max_lines:])


def capture_progress_summary(product: str) -> dict[str, object]:
    """Read the most recent worker progress event from a product's log."""
    mode = "unknown"
    progress: dict[str, object] = {
        "mode": mode,
        "phase": "idle",
        "completed": 0,
        "total": 0,
        "detail": "",
    }
    for line in tail_text(log_path(product), max_lines=400).splitlines():
        if line.startswith("[capture_mode] "):
            for token in line.split()[1:]:
                key, separator, value = token.partition("=")
                if separator and key == "mode":
                    progress["mode"] = value
        if not line.startswith("capture_progress "):
            continue
        fields = {}
        for token in line.split()[1:]:
            key, separator, value = token.partition("=")
            if separator:
                fields[key] = value
        try:
            progress["completed"] = int(fields.get("completed", 0))
            progress["total"] = int(fields.get("total", 0))
        except ValueError:
            continue
        progress["phase"] = fields.get("phase", "unknown")
        progress["detail"] = " ".join(
            token for token in line.split()[1:] if not token.startswith(("product=", "phase=", "completed=", "total="))
        )
    return progress


class IntradayCaptureApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.products = list(core.config.available_live_products())
        self.product_vars = {product: IntVar(value=1) for product in self.products}
        self.account_id = StringVar(value=DEFAULT_ACCOUNT_ID)
        self.interval = StringVar(value=str(DEFAULT_INTERVAL_SECONDS))
        self.extra_codes = StringVar(value="")
        self.backfill_days = StringVar(value="5")
        self.browse_date = StringVar(value="")
        self.browse_symbol = StringVar(value="")
        self.save_greeks_snapshot = IntVar(value=0)
        self.all_option_contracts = IntVar(value=0)
        self.status_list: Listbox | None = None
        self.log_text: Listbox | None = None
        self.progress_vars: dict[str, StringVar] = {}
        self.progress_bars: dict[str, Progressbar] = {}
        self._build()
        self._schedule_refresh()

    def _build(self):
        top = Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)
        Label(top, text="Products").pack(side=LEFT)
        for product in self.products:
            Checkbutton(top, text=product, variable=self.product_vars[product]).pack(
                side=LEFT
            )

        settings = Frame(self.root)
        settings.pack(fill="x", padx=8, pady=4)
        Label(settings, text="Account").pack(side=LEFT)
        Entry(settings, textvariable=self.account_id, width=12).pack(side=LEFT, padx=4)
        Label(settings, text="Interval seconds").pack(side=LEFT)
        Entry(settings, textvariable=self.interval, width=8).pack(side=LEFT, padx=4)
        Label(settings, text="Extra option codes comma-separated").pack(side=LEFT)
        Entry(settings, textvariable=self.extra_codes, width=34).pack(side=LEFT, padx=4)
        Checkbutton(
            settings,
            text="Save Greeks snapshot",
            variable=self.save_greeks_snapshot,
        ).pack(side=LEFT, padx=4)
        Checkbutton(
            settings,
            text="All available contracts → daily Parquet (Run Once)",
            variable=self.all_option_contracts,
        ).pack(side=LEFT, padx=4)

        data_tools = Frame(self.root)
        data_tools.pack(fill="x", padx=8, pady=4)
        Label(data_tools, text="Backfill days").pack(side=LEFT)
        Entry(data_tools, textvariable=self.backfill_days, width=6).pack(side=LEFT, padx=4)
        Label(data_tools, text="Browse date YYYY-MM-DD").pack(side=LEFT)
        Entry(data_tools, textvariable=self.browse_date, width=12).pack(side=LEFT, padx=4)
        Label(data_tools, text="Symbol/code").pack(side=LEFT)
        Entry(data_tools, textvariable=self.browse_symbol, width=16).pack(side=LEFT, padx=4)

        controls = Frame(self.root)
        controls.pack(fill="x", padx=8, pady=4)
        Button(controls, text="Start Selected", command=self.start_selected).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Run Once Selected", command=self.run_once_selected).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Stop Selected", command=self.stop_selected).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Refresh", command=self.refresh).pack(side=LEFT, padx=4)
        Button(controls, text="Show Selected Log", command=self.show_selected_log).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Backfill Missing", command=self.backfill_missing).pack(
            side=LEFT, padx=4
        )
        Button(controls, text="Browse Data", command=self.browse_data).pack(
            side=LEFT, padx=4
        )

        progress_panel = Frame(self.root)
        progress_panel.pack(fill="x", padx=8, pady=4)
        Label(progress_panel, text="Capture progress").grid(row=0, column=0, sticky="w")
        for row, product in enumerate(self.products, start=1):
            Label(progress_panel, text=product, width=9, anchor="w").grid(
                row=row, column=0, sticky="w"
            )
            bar = Progressbar(progress_panel, length=310, mode="determinate", maximum=1)
            bar.grid(row=row, column=1, padx=4, sticky="w")
            variable = StringVar(value="Stopped")
            Label(progress_panel, textvariable=variable, width=95, anchor="w").grid(
                row=row, column=2, sticky="w"
            )
            self.progress_bars[product] = bar
            self.progress_vars[product] = variable

        body = Frame(self.root)
        body.pack(fill=BOTH, expand=True, padx=8, pady=8)

        left = Frame(body)
        left.pack(side=LEFT, fill=BOTH, expand=True)
        Label(left, text="Status").pack(anchor="w")
        status_scroll = Scrollbar(left, orient=VERTICAL)
        self.status_list = Listbox(
            left, height=10, width=120, yscrollcommand=status_scroll.set
        )
        status_scroll.config(command=self.status_list.yview)
        self.status_list.pack(side=LEFT, fill=BOTH, expand=True)
        status_scroll.pack(side=RIGHT, fill="y")

        bottom = Frame(self.root)
        bottom.pack(fill=BOTH, expand=True, padx=8, pady=8)
        Label(bottom, text="Log tail").pack(anchor="w")
        log_scroll = Scrollbar(bottom, orient=VERTICAL)
        self.log_text = Listbox(
            bottom, height=16, width=140, yscrollcommand=log_scroll.set
        )
        log_scroll.config(command=self.log_text.yview)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)
        log_scroll.pack(side=RIGHT, fill="y")

    def selected_products(self) -> list[str]:
        return [
            product
            for product in self.products
            if int(self.product_vars[product].get() or 0) == 1
        ]

    def extra_option_codes(self) -> list[str]:
        return [
            item.strip()
            for item in self.extra_codes.get().split(",")
            if item.strip()
        ]

    def start_selected(self):
        if self.all_option_contracts.get():
            messagebox.showinfo(
                "Use Run Once",
                "Full-chain minute capture uses many source requests. Use Run Once Selected.",
            )
            return
        try:
            interval = int(self.interval.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid interval", "Interval seconds must be positive.")
            return
        codes = self.extra_option_codes()
        errors = []
        for product in self.selected_products():
            try:
                start_capture(
                    product,
                    self.account_id.get().strip() or DEFAULT_ACCOUNT_ID,
                    interval,
                    codes,
                    save_option_greeks_snapshot=bool(self.save_greeks_snapshot.get()),
                    all_option_contracts=bool(self.all_option_contracts.get()),
                )
            except Exception as exc:
                errors.append(f"{product}: {exc}")
        self.refresh()
        if errors:
            messagebox.showerror("Start failed", "\n".join(errors))

    def run_once_selected(self):
        try:
            interval = int(self.interval.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid interval", "Interval seconds must be positive.")
            return
        codes = self.extra_option_codes()
        errors = []
        for product in self.selected_products():
            try:
                start_capture(
                    product,
                    self.account_id.get().strip() or DEFAULT_ACCOUNT_ID,
                    interval,
                    codes,
                    once=True,
                    save_option_greeks_snapshot=bool(self.save_greeks_snapshot.get()),
                    all_option_contracts=bool(self.all_option_contracts.get()),
                )
            except Exception as exc:
                errors.append(f"{product}: {exc}")
        self.refresh()
        if errors:
            messagebox.showerror("Run once failed", "\n".join(errors))

    def stop_selected(self):
        errors = []
        for product in self.selected_products():
            status = capture_status(product)
            if not status.running:
                continue
            if not stop_process(status.pid):
                errors.append(f"{product}: failed to stop pid={status.pid}")
        self.refresh()
        if errors:
            messagebox.showerror("Stop failed", "\n".join(errors))

    def refresh(self):
        assert self.status_list is not None
        self.status_list.delete(0, END)
        for product in self.products:
            status = capture_status(product)
            running = "RUNNING" if status.running else "STOPPED"
            latest = str(status.latest_file) if status.latest_file else "-"
            mtime = status.latest_file_mtime or "-"
            pid = status.pid if status.pid is not None else "-"
            date_status = intraday_status_summary(product)
            self._refresh_progress(product, status)
            self.status_list.insert(
                END,
                f"{product:8s} {running:8s} pid={pid} {date_status} latest={mtime} file={latest}",
            )

    def _schedule_refresh(self):
        self.refresh()
        self.root.after(1000, self._schedule_refresh)

    def _refresh_progress(self, product: str, status: CaptureStatus):
        progress = capture_progress_summary(product)
        completed = int(progress["completed"])
        total = int(progress["total"])
        bar = self.progress_bars[product]
        bar.configure(maximum=max(total, 1), value=min(completed, max(total, 1)))

        mode = str(progress["mode"])
        phase = str(progress["phase"])
        detail = str(progress["detail"])
        if status.running and mode == "continuous":
            text = f"正在持续运行 · 最近一轮 {completed}/{total} · {phase} {detail}".strip()
        elif status.running:
            text = f"单次抓取中 · {completed}/{total} · {phase} {detail}".strip()
        elif phase == "completed":
            text = f"单次抓取完成 · {completed}/{total} · {detail}".strip()
        else:
            text = "Stopped"
        self.progress_vars[product].set(text)

    def show_selected_log(self):
        assert self.status_list is not None
        assert self.log_text is not None
        selection = self.status_list.curselection()
        if selection:
            product = self.products[int(selection[0])]
        else:
            products = self.selected_products()
            product = products[0] if products else self.products[0]
        text = tail_text(log_path(product))
        self.log_text.delete(0, END)
        self.log_text.insert(END, f"== {product} {log_path(product)} ==")
        if not text:
            self.log_text.insert(END, "(no log yet)")
            return
        for line in text.splitlines():
            self.log_text.insert(END, line)

    def backfill_missing(self):
        try:
            days = int(self.backfill_days.get())
            if days <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid days", "Backfill days must be positive.")
            return

        import argparse
        import capture_intraday_quotes

        account_id = self.account_id.get().strip() or DEFAULT_ACCOUNT_ID
        codes = self.extra_option_codes()
        products = self.selected_products()
        if not products:
            messagebox.showinfo("No product", "Select at least one product.")
            return

        plans = []
        errors = []
        for product in products:
            try:
                discovery = capture_intraday_quotes.discover_backfill_dates(
                    product,
                    account_id=account_id,
                    option_codes=codes,
                    max_days=days,
                )
                status = capture_intraday_quotes.refresh_intraday_status(
                    product,
                    option_codes=discovery["option_codes"],
                    etf_symbol=discovery["etf_symbol"],
                )
                status_by_date = status.get("dates", {})
                missing = [
                    row["date"]
                    for row in discovery["dates"]
                    if row["complete"]
                    and not status_by_date.get(row["date"], {}).get("complete")
                ]
                plans.append((product, discovery, missing))
            except Exception as exc:
                errors.append(f"{product}: {exc}")

        if errors:
            messagebox.showerror("Backfill discovery failed", "\n".join(errors))
            return

        summary_lines = []
        for product, discovery, missing in plans:
            available = [row["date"] for row in discovery["dates"] if row["complete"]]
            summary_lines.append(
                f"{product}: source_available={len(available)} "
                f"missing={len(missing)} dates={','.join(missing) or '-'}"
            )
        if not any(missing for _, _, missing in plans):
            messagebox.showinfo("Backfill", "No missing available dates.")
            self.refresh()
            return
        if not messagebox.askyesno(
            "Confirm backfill",
            "Backfill missing minute data?\n\n" + "\n".join(summary_lines),
        ):
            return

        run_errors = []
        results = []
        for product, _, _ in plans:
            args = argparse.Namespace(
                product=product,
                account_id=account_id,
                interval_seconds=None,
                daily_schedule="",
                option_code=codes,
                no_account_positions=False,
                once=True,
                save_option_greeks_snapshot=False,
                max_runs=None,
                output_dir=None,
                pid_file=None,
                target_date=None,
                backfill_missing=True,
                backfill_days=days,
            )
            try:
                results.append(capture_intraday_quotes.backfill_missing_dates(args))
            except Exception as exc:
                run_errors.append(f"{product}: {exc}")

        self.refresh()
        self._show_lines(
            "Backfill result",
            [
                f"{item['product']}: filled={len(item['results'])} "
                f"missing={','.join(item['missing_dates']) or '-'} "
                f"status={item['status_path']}"
                for item in results
            ]
            + run_errors,
        )
        if run_errors:
            messagebox.showerror("Backfill failed", "\n".join(run_errors))

    def browse_data(self):
        import pandas as pd
        import capture_intraday_quotes

        product = self._selected_product_for_detail()
        date_text = self.browse_date.get().strip()
        symbol = self.browse_symbol.get().strip()
        if not date_text:
            status = capture_intraday_quotes.load_intraday_status(product)
            dates = sorted(status.get("dates", {}))
            date_text = dates[-1] if dates else pd.Timestamp.now().strftime("%Y-%m-%d")
        try:
            date_part = pd.Timestamp(date_text).strftime("%Y%m%d")
        except Exception:
            messagebox.showerror("Invalid date", "Browse date must be YYYY-MM-DD.")
            return

        root = Path(storage.PROJECT_ROOT) / "data" / "live" / product / "intraday" / date_part
        if not root.exists():
            messagebox.showinfo("No data", f"No intraday directory: {root}")
            return
        if not symbol:
            files = sorted(path.name for path in root.glob("*.csv"))
            self._show_lines(
                f"{product} {date_text} files",
                files or ["(no csv files)"],
            )
            return

        candidates = [
            root / f"option_{symbol}_1m.csv",
            root / f"etf_{symbol}_1m.csv",
            root / symbol,
        ]
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            files = sorted(item.name for item in root.glob("*.csv"))
            self._show_lines(
                "Data not found",
                [f"Not found for symbol/code: {symbol}", f"Directory: {root}", *files],
            )
            return
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Read failed", str(exc))
            return
        preview = frame.head(300).to_string(index=False)
        self._show_text(
            f"{product} {date_text} {path.name}",
            f"path={path}\nrows={len(frame)} columns={list(frame.columns)}\n\n{preview}",
        )

    def _selected_product_for_detail(self) -> str:
        assert self.status_list is not None
        selection = self.status_list.curselection()
        if selection:
            return self.products[int(selection[0])]
        products = self.selected_products()
        return products[0] if products else self.products[0]

    def _show_lines(self, title: str, lines: list[str]):
        self._show_text(title, "\n".join(lines))

    def _show_text(self, title: str, text: str):
        window = Toplevel(self.root)
        window.title(title)
        window.geometry("1100x640")
        box = Text(window, wrap="none")
        box.pack(fill=BOTH, expand=True)
        box.insert("1.0", text)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == CAPTURE_WORKER_ARG:
        run_capture_worker(sys.argv[2:])
        return
    root = Tk()
    root.geometry("1180x720")
    IntradayCaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
