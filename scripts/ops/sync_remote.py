from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_REMOTE = "yangziqi@172.16.128.67"
DEFAULT_REMOTE_DIR = "/home/yangziqi/strategy/gamma_scalping_v7"


CODE_EXCLUDES = [
    ".git/",
    ".venv/",
    ".cache/",
    "__pycache__/",
    "*.pyc",
    "output/",
    "data/",
    "state/",
]

ALL_EXCLUDES = [
    ".git/",
    ".venv/",
    ".cache/",
    "__pycache__/",
    "*.pyc",
    "output/",
    "state/",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="用 rsync 同步本地项目到远程服务器，或拉回远程输出结果。"
    )
    parser.add_argument(
        "--mode",
        choices=["code", "data", "all", "pull-output"],
        default="code",
        help=(
            "code=只同步代码和配置；data=只同步 data；"
            "all=同步代码和 data；pull-output=从远程拉回 output。"
        ),
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="远程登录地址。")
    parser.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help="远程项目目录。",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="让远程目录删除本地不存在的文件。谨慎使用。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要同步的内容，不实际修改远程文件。",
    )
    return parser.parse_args()


def require_rsync():
    if shutil.which("rsync") is None:
        raise SystemExit(
            "未找到 rsync。请在 Git Bash/WSL/Linux/macOS 中运行，"
            "或先安装 rsync 后再执行本脚本。"
        )


def run_command(command):
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def rsync_base_args(args):
    command = ["rsync", "-avz", "--progress"]
    if args.delete:
        command.append("--delete")
    if args.dry_run:
        command.append("--dry-run")
    return command


def add_excludes(command, excludes):
    for pattern in excludes:
        command.extend(["--exclude", pattern])


def remote_path(args, subdir=""):
    base = args.remote_dir.rstrip("/")
    if subdir:
        return f"{args.remote}:{base}/{subdir.strip('/')}/"
    return f"{args.remote}:{base}/"


def sync_code(args, project_root):
    command = rsync_base_args(args)
    add_excludes(command, CODE_EXCLUDES)
    command.extend([f"{project_root}/", remote_path(args)])
    run_command(command)


def sync_data(args, project_root):
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise SystemExit(f"本地 data 目录不存在: {data_dir}")
    command = rsync_base_args(args)
    command.extend([f"{data_dir}/", remote_path(args, "data")])
    run_command(command)


def sync_all(args, project_root):
    command = rsync_base_args(args)
    add_excludes(command, ALL_EXCLUDES)
    command.extend([f"{project_root}/", remote_path(args)])
    run_command(command)


def pull_output(args, project_root):
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    command = rsync_base_args(args)
    command.extend([remote_path(args, "output"), f"{output_dir}/"])
    run_command(command)


def main():
    args = parse_args()
    require_rsync()
    project_root = Path(__file__).resolve().parents[2]

    if args.mode == "code":
        sync_code(args, project_root)
    elif args.mode == "data":
        sync_data(args, project_root)
    elif args.mode == "all":
        sync_all(args, project_root)
    elif args.mode == "pull-output":
        pull_output(args, project_root)


if __name__ == "__main__":
    main()
