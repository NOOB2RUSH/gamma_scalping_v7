import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "live"))
import intraday_capture_app


class IntradayCaptureAppHelpersTest(TestCase):
    def test_latest_intraday_file_finds_newest_option_file(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            product = "50etf"
            intraday = root / "data" / "live" / product / "intraday" / "20260707"
            intraday.mkdir(parents=True)
            older = intraday / "option_10000001_1m.csv"
            newer = intraday / "option_10000001_greeks_snapshots.csv"
            ignored = intraday / "etf_510050_1m.csv"
            older.write_text("older", encoding="utf-8")
            newer.write_text("newer", encoding="utf-8")
            ignored.write_text("ignored", encoding="utf-8")
            os.utime(older, (1_000_000, 1_000_000))
            os.utime(newer, (2_000_000, 2_000_000))
            os.utime(ignored, (3_000_000, 3_000_000))

            with mock.patch.object(intraday_capture_app.storage, "PROJECT_ROOT", root):
                path, stamp = intraday_capture_app.latest_intraday_file(product)

        self.assertEqual(path, newer)
        self.assertEqual(stamp, "1970-01-24 11:33:20")

    def test_read_pid_and_tail_text(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pid_path = root / "capture.pid"
            pid_path.write_text("12345\n", encoding="utf-8")
            log_path = root / "capture.log"
            log_path.write_text("a\nb\nc\n", encoding="utf-8")

            self.assertEqual(intraday_capture_app.read_pid(pid_path), 12345)
            self.assertEqual(intraday_capture_app.tail_text(log_path, max_lines=2), "b\nc")

    def test_capture_command_uses_python_script_when_not_frozen(self):
        command = intraday_capture_app.capture_command(
            "50etf",
            "default",
            60,
            Path("capture.pid"),
            extra_option_codes=["10000001"],
        )

        self.assertEqual(command[0], sys.executable)
        self.assertTrue(command[1].endswith("capture_intraday_quotes.py"))
        self.assertIn("--interval-seconds", command)
        self.assertIn("10000001", command)
        self.assertNotIn("--save-option-greeks-snapshot", command)

    def test_capture_command_uses_exe_worker_when_frozen(self):
        with (
            mock.patch.object(sys, "executable", "IntradayCaptureApp.exe"),
            mock.patch.object(sys, "frozen", True, create=True),
        ):
            command = intraday_capture_app.capture_command(
                "50etf",
                "default",
                60,
                Path("capture.pid"),
            )

        self.assertEqual(command[:2], ["IntradayCaptureApp.exe", "--capture-worker"])
        self.assertIn("--product", command)

    def test_capture_command_supports_run_once_mode(self):
        command = intraday_capture_app.capture_command(
            "50etf",
            "default",
            60,
            Path("capture.pid"),
            once=True,
        )

        self.assertIn("--once", command)

    def test_capture_command_supports_optional_greeks_snapshot(self):
        command = intraday_capture_app.capture_command(
            "50etf",
            "default",
            60,
            Path("capture.pid"),
            save_option_greeks_snapshot=True,
        )

        self.assertIn("--save-option-greeks-snapshot", command)
