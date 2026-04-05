#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BRANCH_MEMORY_TOOL = ROOT / "tools" / "branch_memory.py"
DEFAULT_SOCKET = Path(
    os.environ.get("BRANCH_MEMORY_SOCKET")
    or os.path.join(
        os.environ.get("XDG_RUNTIME_DIR") or "/tmp",
        f"parameter-golf-branch-memory-{os.getuid()}.sock",
    )
)
DEFAULT_LOG = ROOT / "research" / "branch_memory" / "daemon.log"


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log_line(log_path: Path | None, payload: dict[str, Any]) -> None:
    if not log_path:
        return
    ensure_parent(log_path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def send_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    conn.sendall((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))


def recv_json(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        part = conn.recv(65536)
        if not part:
            break
        chunks.append(part)
        if b"\n" in part:
            break
    raw = b"".join(chunks).decode("utf-8").strip()
    if not raw:
        return {}
    return json.loads(raw)


def run_branch_memory(argv: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(BRANCH_MEMORY_TOOL), *argv]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def handle_request(request: dict[str, Any], *, log_path: Path | None, socket_path: Path, start_time: str) -> dict[str, Any]:
    command = request.get("command")
    if command == "ping":
        return {
            "ok": True,
            "command": "ping",
            "socket_path": str(socket_path),
            "started_at_utc": start_time,
            "time_utc": now_utc(),
        }
    if command == "shutdown":
        return {
            "ok": True,
            "command": "shutdown",
            "shutdown": True,
            "time_utc": now_utc(),
        }
    if command != "run":
        return {
            "ok": False,
            "error": "unsupported_command",
            "message": "expected command=run|ping|shutdown",
            "time_utc": now_utc(),
        }
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return {
            "ok": False,
            "error": "invalid_argv",
            "message": "argv must be a list of strings",
            "time_utc": now_utc(),
        }
    timeout = request.get("timeout_sec")
    timeout_value = int(timeout) if isinstance(timeout, (int, float)) else None
    proc = run_branch_memory(argv, timeout=timeout_value)
    response = {
        "ok": proc.returncode == 0,
        "command": "run",
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "time_utc": now_utc(),
    }
    log_line(
        log_path,
        {
            "event": "run",
            "argv": argv,
            "ok": response["ok"],
            "returncode": proc.returncode,
            "time_utc": response["time_utc"],
        },
    )
    return response


def serve_forever(socket_path: Path, log_path: Path | None) -> None:
    ensure_parent(socket_path)
    if socket_path.exists():
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(64)
    start_time = now_utc()
    stop = {"value": False}

    def mark_stop(_: int, __: Any) -> None:
        stop["value"] = True

    signal.signal(signal.SIGTERM, mark_stop)
    signal.signal(signal.SIGINT, mark_stop)

    log_line(log_path, {"event": "start", "socket_path": str(socket_path), "time_utc": start_time})
    try:
        while not stop["value"]:
            conn, _ = server.accept()
            with conn:
                try:
                    request = recv_json(conn)
                    response = handle_request(
                        request,
                        log_path=log_path,
                        socket_path=socket_path,
                        start_time=start_time,
                    )
                    send_json(conn, response)
                    if response.get("shutdown"):
                        stop["value"] = True
                except Exception as exc:  # pragma: no cover - defensive control-plane path
                    error_payload = {
                        "ok": False,
                        "error": "server_exception",
                        "message": str(exc),
                        "time_utc": now_utc(),
                    }
                    send_json(conn, error_payload)
                    log_line(log_path, {"event": "exception", "message": str(exc), "time_utc": error_payload["time_utc"]})
    finally:
        server.close()
        if socket_path.exists():
            socket_path.unlink()
        log_line(log_path, {"event": "stop", "socket_path": str(socket_path), "time_utc": now_utc()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resident Unix-socket daemon for branch_memory operations.")
    parser.add_argument(
        "--socket-path",
        default=str(DEFAULT_SOCKET),
        help="Unix socket path. Defaults to $BRANCH_MEMORY_SOCKET, $XDG_RUNTIME_DIR, or /tmp.",
    )
    parser.add_argument(
        "--log-path",
        default=str(DEFAULT_LOG),
        help="JSONL daemon log path. Set to empty string to disable.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    socket_path = Path(args.socket_path).expanduser()
    log_path = Path(args.log_path).expanduser() if args.log_path else None
    serve_forever(socket_path, log_path)


if __name__ == "__main__":
    main()
