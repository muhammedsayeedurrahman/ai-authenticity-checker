"""
Single-command dev launcher: starts the FastAPI backend (main.py) and the
Vite frontend dev server together, both streaming to this terminal.
Ctrl+C stops both.

Usage:
    python run_dev.py
"""

from __future__ import annotations

import os
import platform
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(ROOT, "frontend")
IS_WINDOWS = platform.system() == "Windows"


def _venv_python() -> str:
    """Prefer this repo's own .venv interpreter over whatever `python`
    happens to be on PATH, so backend deps resolve the same way they do
    when running `python main.py` directly."""
    candidate = (
        os.path.join(ROOT, ".venv", "Scripts", "python.exe")
        if IS_WINDOWS
        else os.path.join(ROOT, ".venv", "bin", "python")
    )
    return candidate if os.path.exists(candidate) else sys.executable


def _popen_frontend() -> subprocess.Popen:
    """`npm run dev` needs shell=True to resolve npm.cmd/npm on PATH,
    which makes the Popen object track a shell wrapper, not the actual
    vite/node process underneath it - terminating just the wrapper
    leaves vite running as an orphan (observed: it kept the dev port
    bound after this script had already exited). Put it in its own
    process group/session so _kill_tree can take out the whole tree."""
    if IS_WINDOWS:
        return subprocess.Popen(
            "npm run dev", cwd=FRONTEND, shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        "npm run dev", cwd=FRONTEND, shell=True, preexec_fn=os.setsid,
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass


def _wait_for_backend(host: str = "127.0.0.1", port: int = 7861, timeout: int = 120) -> bool:
    """Poll until the backend accepts TCP connections or timeout expires."""
    print(f"[dev] waiting for backend on {host}:{port} (ML models may take a moment)...", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"[dev] backend is ready on port {port} ✓", flush=True)
                return True
        except OSError:
            time.sleep(0.5)
    print(f"[dev] WARNING: backend did not become ready within {timeout}s — starting frontend anyway")
    return False


def main() -> int:
    backend_cmd = [_venv_python(), "main.py"]
    print(f"[dev] backend:  {' '.join(backend_cmd)}  (cwd={ROOT})")
    backend = subprocess.Popen(backend_cmd, cwd=ROOT)

    _wait_for_backend()

    print(f"[dev] frontend: npm run dev  (cwd={FRONTEND})")
    frontend = _popen_frontend()

    exit_code = 0
    try:
        while True:
            if backend.poll() is not None:
                exit_code = backend.returncode
                print(f"[dev] backend exited ({exit_code}) - stopping frontend...")
                _kill_tree(frontend)
                break
            if frontend.poll() is not None:
                exit_code = frontend.returncode
                print(f"[dev] frontend exited ({exit_code}) - stopping backend...")
                _kill_tree(backend)
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[dev] stopping backend and frontend...")
        _kill_tree(backend)
        _kill_tree(frontend)
    finally:
        backend.wait()
        frontend.wait()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
