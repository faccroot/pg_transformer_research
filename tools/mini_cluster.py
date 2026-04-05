#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


MINI_ALIAS_RE = re.compile(r"^mini\d{2}$")


def parse_ssh_config_aliases(config_path: Path) -> list[str]:
    aliases: list[str] = []
    if not config_path.is_file():
        raise FileNotFoundError(f"SSH config not found: {config_path}")
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.lower().startswith("host "):
            continue
        for token in line.split()[1:]:
            if MINI_ALIAS_RE.fullmatch(token):
                aliases.append(token)
    return sorted(dict.fromkeys(aliases))


def ssh_g(alias: str) -> dict[str, str]:
    proc = subprocess.run(
        ["ssh", "-G", alias],
        check=True,
        capture_output=True,
        text=True,
    )
    info: dict[str, str] = {"alias": alias}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        if key in {"user", "hostname", "port"}:
            info[key] = value
    return info


def discover_hosts(config_path: Path, selected: list[str] | None = None) -> list[dict[str, str]]:
    aliases = parse_ssh_config_aliases(config_path)
    if selected:
        wanted = set(selected)
        aliases = [alias for alias in aliases if alias in wanted]
    return [ssh_g(alias) for alias in aliases]


def check_host(alias: str, timeout: int) -> dict[str, str]:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", alias, "hostname"],
        capture_output=True,
        text=True,
    )
    stderr = proc.stderr.strip()
    stdout = proc.stdout.strip()
    if proc.returncode == 0:
        status = "ok"
    elif "Permission denied" in stderr:
        status = "auth_failed"
    elif "Could not resolve hostname" in stderr or "No route to host" in stderr or "Operation timed out" in stderr:
        status = "unreachable"
    else:
        status = "error"
    return {
        "alias": alias,
        "status": status,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": str(proc.returncode),
    }


def run_remote(alias: str, command: str, timeout: int, batch_mode: bool) -> dict[str, str]:
    ssh_cmd = ["ssh"]
    if batch_mode:
        ssh_cmd.extend(["-o", "BatchMode=yes"])
    ssh_cmd.extend(["-o", f"ConnectTimeout={timeout}", alias, command])
    proc = subprocess.run(ssh_cmd, capture_output=True, text=True)
    return {
        "alias": alias,
        "returncode": str(proc.returncode),
        "stdout": proc.stdout.rstrip(),
        "stderr": proc.stderr.rstrip(),
    }


def print_rows(rows: list[dict[str, str]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    for row in rows:
        parts = [f"{key}={value}" for key, value in row.items()]
        print(" ".join(parts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and use Mac mini SSH aliases from ~/.ssh/config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".ssh" / "config",
        help="SSH config to inspect",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List configured mini hosts")
    list_cmd.add_argument("--host", action="append", help="Restrict to one or more aliases")
    list_cmd.add_argument("--json", action="store_true")

    check_cmd = sub.add_parser("check", help="Run a non-interactive auth/reachability check")
    check_cmd.add_argument("--host", action="append", help="Restrict to one or more aliases")
    check_cmd.add_argument("--timeout", type=int, default=5)
    check_cmd.add_argument("--json", action="store_true")

    exec_cmd = sub.add_parser("exec", help="Run one command sequentially across mini hosts")
    exec_cmd.add_argument("remote_command", help="Command string to execute remotely")
    exec_cmd.add_argument("--host", action="append", help="Restrict to one or more aliases")
    exec_cmd.add_argument("--timeout", type=int, default=5)
    exec_cmd.add_argument("--batch-mode", action="store_true", help="Disable interactive auth prompts")
    exec_cmd.add_argument("--json", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "list":
        rows = discover_hosts(args.config, args.host)
        print_rows(rows, args.json)
        return

    hosts = discover_hosts(args.config, args.host)
    aliases = [host["alias"] for host in hosts]
    if args.command == "check":
        rows = [check_host(alias, args.timeout) for alias in aliases]
        print_rows(rows, args.json)
        return
    if args.command == "exec":
        rows = [run_remote(alias, args.remote_command, args.timeout, args.batch_mode) for alias in aliases]
        print_rows(rows, args.json)
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
