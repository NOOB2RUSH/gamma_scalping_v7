from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, Button, Checkbutton, Entry, Frame, IntVar, Label, Listbox, Scrollbar, StringVar, Tk, messagebox

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


def start_capture(
    product: str,
    account_id: str,
    interval_seconds: int,
    extra_option_codes: list[str] | None = None,
    once: bool = False,
    save_option_greeks_snapshot: bool = False,
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
    )

    log = status.log_path.open("a", encoding="utf-8")
    log.write(f"\n[start] command={' '.join(command)}\n")
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


class IntradayCaptureApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.products = list(core.config.available_live_products())
        self.product_vars = {product: IntVar(value=1) for product in self.products}
        self.account_id = StringVar(value=DEFAULT_ACCOUNT_ID)
        self.interval = StringVar(value=str(DEFAULT_INTERVAL_SECONDS))
        self.extra_codes = StringVar(value="")
        self.save_greeks_snapshot = IntVar(value=0)
        self.status_list: Listbox | None = None
        self.log_text: Listbox | None = None
        self._build()
        self.refresh()

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

    def start_selected(self):
        try:
            interval = int(self.interval.get())
            if interval <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid interval", "Interval seconds must be positive.")
            return
        codes = [
            item.strip()
            for item in self.extra_codes.get().split(",")
            if item.strip()
        ]
        errors = []
        for product in self.selected_products():
            try:
                start_capture(
                    product,
                    self.account_id.get().strip() or DEFAULT_ACCOUNT_ID,
                    interval,
                    codes,
                    save_option_greeks_snapshot=bool(self.save_greeks_snapshot.get()),
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
        codes = [
            item.strip()
            for item in self.extra_codes.get().split(",")
            if item.strip()
        ]
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
            self.status_list.insert(
                END,
                f"{product:8s} {running:8s} pid={pid} latest={mtime} file={latest}",
            )

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
