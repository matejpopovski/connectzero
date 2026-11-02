"""Runs training under a progress watchdog.

Training touches a heartbeat file between phases. If the heartbeat goes stale or the trainer
dies, the watchdog stops it and starts it again, and it resumes from its last atomic checkpoint.
Waiting on a process alone cannot tell a slow generation from a hung one.

Usage: python -m connectzero.supervise --stall 420 -- --run runs/main --minutes 42 --device mps
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_dir_from(arguments: list[str]) -> Path:
    if "--run" in arguments:
        return Path(arguments[arguments.index("--run") + 1])
    return Path("runs/main")


def stop(child: subprocess.Popen, grace: float = 20.0) -> None:
    child.terminate()
    try:
        child.wait(grace)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def supervise(arguments: list[str], stall: float, poll: float, max_restarts: int) -> int:
    heartbeat = run_dir_from(arguments) / "heartbeat"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "connectzero.train", *arguments]
    for attempt in range(max_restarts + 1):
        heartbeat.touch()
        print(f"[supervise] attempt {attempt} starting", flush=True)
        child = subprocess.Popen(command)
        while True:
            code = child.poll()
            if code == 0:
                return 0
            if code is not None:
                print(f"[supervise] trainer exited with {code}", flush=True)
                break
            quiet = time.time() - heartbeat.stat().st_mtime
            if quiet > stall:
                print(f"[supervise] no progress for {quiet:.0f}s, restarting", flush=True)
                stop(child)
                break
            time.sleep(poll)
    print("[supervise] giving up", flush=True)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stall", type=float, default=420.0, help="seconds without a heartbeat")
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = [item for item in args.train_args if item != "--"]
    sys.exit(supervise(forwarded, args.stall, args.poll, args.max_restarts))


if __name__ == "__main__":
    main()
