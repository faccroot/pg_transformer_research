#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DAEMON_TOOL = ROOT / "tools" / "branch_memoryd.py"
BRANCH_MEMORY_TOOL = ROOT / "tools" / "branch_memory.py"
DEFAULT_LOG_PATH = ROOT / "research" / "branch_memory" / "daemon.log"
DEFAULT_SOCKET = Path(
    os.environ.get("BRANCH_MEMORY_SOCKET")
    or os.path.join(
        os.environ.get("XDG_RUNTIME_DIR") or "/tmp",
        f"parameter-golf-branch-memory-{os.getuid()}.sock",
    )
)


def send_request(socket_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(2.0)
    client.connect(str(socket_path))
    client.sendall((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
    chunks: list[bytes] = []
    while True:
        part = client.recv(65536)
        if not part:
            break
        chunks.append(part)
        if b"\n" in part:
            break
    client.close()
    raw = b"".join(chunks).decode("utf-8").strip()
    return json.loads(raw) if raw else {}


def ping_daemon(socket_path: Path) -> dict[str, Any] | None:
    try:
        response = send_request(socket_path, {"command": "ping"})
    except (FileNotFoundError, ConnectionError, OSError, json.JSONDecodeError):
        return None
    return response if response.get("ok") else None


def wait_for_ping(socket_path: Path, timeout_sec: float) -> dict[str, Any] | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = ping_daemon(socket_path)
        if response:
            return response
        time.sleep(0.1)
    return None


def send_request_with_retries(socket_path: Path, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any] | None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            return send_request(socket_path, payload)
        except (FileNotFoundError, ConnectionError, OSError, json.JSONDecodeError, TimeoutError):
            time.sleep(0.1)
    return None


def autostart_daemon(socket_path: Path, log_path: Path, start_timeout_sec: float) -> bool:
    if socket_path.exists():
        if wait_for_ping(socket_path, min(start_timeout_sec, 1.5)):
            return True
        try:
            socket_path.unlink()
        except OSError:
            pass
    if socket_path.exists() and ping_daemon(socket_path):
        return True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(os.devnull, "wb") as devnull:
        subprocess.Popen(
            [
                sys.executable,
                str(DAEMON_TOOL),
                "--socket-path",
                str(socket_path),
                "--log-path",
                str(log_path),
            ],
            cwd=ROOT,
            stdout=devnull,
            stderr=devnull,
            stdin=devnull,
            start_new_session=True,
            close_fds=True,
        )
    return wait_for_ping(socket_path, start_timeout_sec) is not None


def run_direct(argv: list[str]) -> int:
    proc = subprocess.run([sys.executable, str(BRANCH_MEMORY_TOOL), *argv], cwd=ROOT, check=False)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Client wrapper for branch_memory. Talks to a local daemon when available."
    )
    parser.add_argument(
        "--socket-path",
        default=str(DEFAULT_SOCKET),
        help="Unix socket path for the daemon. Defaults to $BRANCH_MEMORY_SOCKET, $XDG_RUNTIME_DIR, or /tmp.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Bypass the daemon and invoke branch_memory.py directly.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        help="Optional daemon-side timeout for the requested command.",
    )
    parser.add_argument(
        "--daemon-log-path",
        default=str(DEFAULT_LOG_PATH),
        help="Daemon log path used for autostart. Defaults to research/branch_memory/daemon.log.",
    )
    parser.add_argument(
        "--start-timeout-sec",
        type=float,
        default=5.0,
        help="How long to wait for an autostarted daemon socket to become ready.",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Do not start the daemon automatically if the socket is missing or stale.",
    )
    parser.add_argument(
        "--ping",
        action="store_true",
        help="Ping the daemon instead of running a branch_memory.py subcommand.",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Ask the daemon to shut down cleanly.",
    )
    parser.add_argument(
        "branch_memory_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to branch_memory.py, for example: list-frontier --lane hardmax",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    passthrough = list(args.branch_memory_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    if args.ping and args.shutdown:
        print("choose at most one of --ping or --shutdown", file=sys.stderr)
        raise SystemExit(2)
    if not args.ping and not args.shutdown and not passthrough:
        print("branch_memoryctl.py requires branch_memory arguments", file=sys.stderr)
        raise SystemExit(2)

    if args.direct:
        raise SystemExit(run_direct(passthrough))

    socket_path = Path(args.socket_path).expanduser()
    daemon_log_path = Path(args.daemon_log_path).expanduser()
    if args.shutdown:
        if not socket_path.exists():
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "daemon_unavailable",
                        "message": f"socket not found at {socket_path}",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise SystemExit(1)
        response = send_request_with_retries(socket_path, {"command": "shutdown"}, args.start_timeout_sec)
        if response is None:
            print(json.dumps({"ok": False, "error": "shutdown_failed", "socket_path": str(socket_path)}, indent=2, sort_keys=True))
            raise SystemExit(1)
        print(json.dumps(response, indent=2, sort_keys=True))
        raise SystemExit(0 if response.get("ok") else 1)

    should_autostart = not args.no_autostart and not args.shutdown
    ready = wait_for_ping(socket_path, min(args.start_timeout_sec, 1.0)) if socket_path.exists() else None
    if not socket_path.exists() or not ready:
        if not should_autostart:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "daemon_unavailable",
                        "message": f"socket not ready at {socket_path}",
                        "hint": f"start: {sys.executable} {DAEMON_TOOL} --socket-path {socket_path}",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not autostart_daemon(socket_path, daemon_log_path, args.start_timeout_sec):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": "daemon_start_failed",
                        "message": f"could not autostart daemon at {socket_path}",
                        "log_path": str(daemon_log_path),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            raise SystemExit(1)

    if not socket_path.exists():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "daemon_unavailable",
                    "message": f"socket not found at {socket_path}",
                    "hint": f"start: {sys.executable} {DAEMON_TOOL}",
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)

    if args.ping:
        response = send_request_with_retries(socket_path, {"command": "ping"}, args.start_timeout_sec)
        if response is None:
            print(json.dumps({"ok": False, "error": "ping_failed", "socket_path": str(socket_path)}, indent=2, sort_keys=True))
            raise SystemExit(1)
        print(json.dumps(response, indent=2, sort_keys=True))
        raise SystemExit(0 if response.get("ok") else 1)

    response = send_request_with_retries(
        socket_path,
        {"command": "run", "argv": passthrough, "timeout_sec": args.timeout_sec},
        args.start_timeout_sec,
    )
    if response is None:
        print(
            json.dumps(
                {"ok": False, "error": "request_failed", "socket_path": str(socket_path), "argv": passthrough},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
    stdout = response.get("stdout", "")
    stderr = response.get("stderr", "")
    if stdout:
        sys.stdout.write(stdout)
        if not stdout.endswith("\n"):
            sys.stdout.write("\n")
    if stderr:
        sys.stderr.write(stderr)
        if not stderr.endswith("\n"):
            sys.stderr.write("\n")
    raise SystemExit(int(response.get("returncode", 1)))


if __name__ == "__main__":
    main()
