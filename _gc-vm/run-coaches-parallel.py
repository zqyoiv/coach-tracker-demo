"""
Run multiple coach-date-camera-id-runner.py jobs in parallel (one OS process per camera).

Each camera writes to its own coach-store CSV-state / person-ID paths — safe to parallelize.
On one GPU, many concurrent YOLO runs may OOM or slow down; use --max-parallel 1 or 2,
or assign GPUs with --cuda-devices 0,1,2,3.

Examples (from repo root):
  python _gc-vm/run-coaches-parallel.py --date 3-13 --cameras coach-1 coach-2 coach-3 coach-4 --no-viewer

  python _gc-vm/run-coaches-parallel.py --date 3-13 --video-root "%USERPROFILE%\\coach-raw-video" ^
    --cameras coach-1 coach-2 coach-3 coach-4 coach-5 --no-viewer --max-parallel 2

Linux / VM:
  python _gc-vm/run-coaches-parallel.py --date 3-13 --video-root ~/coach-raw-video --no-viewer
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# coach-tracker-demo root (parent of _gc-vm)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "video-script" / "coach-date-camera-id-runner.py"

DEFAULT_VIDEO_ROOT = Path.home() / "coach-raw-video"


def _build_runner_cmd(
    *,
    video_root: str,
    drive_mount_root: str,
    date: str,
    camera: str,
    no_viewer: bool,
    stop_on_error: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(RUNNER),
        "--date",
        date,
        "--camera",
        camera,
        "--video-root",
        video_root,
        "--drive-mount-root",
        drive_mount_root,
    ]
    if no_viewer:
        cmd.append("--no-viewer")
    if stop_on_error:
        cmd.append("--stop-on-error")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run coach-date-camera-id-runner.py for several cameras in parallel."
    )
    parser.add_argument("--date", required=True, help="Date folder (e.g. 3-13).")
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="Cameras to run in parallel, e.g. coach-1 coach-2 coach-3 coach-4",
    )
    parser.add_argument(
        "--video-root",
        default=str(DEFAULT_VIDEO_ROOT),
        help=f"Root with date/Coach-x folders (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument(
        "--drive-mount-root",
        default="/content/drive",
        help="Colab Drive mount (only if using Drive URL from runner; default /content/drive).",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Pass --no-viewer to each runner.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Pass --stop-on-error to each runner.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Max concurrent jobs (default: number of cameras). Lower on a single GPU.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Comma-separated CUDA device ids per camera in order, e.g. 0,1,2,3. Empty slot = no env change.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Per-camera logs (default: coach-store/parallel-run-logs/<timestamp>).",
    )
    args = parser.parse_args()

    video_root = str(Path(args.video_root).expanduser().resolve())

    if not RUNNER.exists():
        print(f"Error: runner not found: {RUNNER}", file=sys.stderr)
        return 2

    cameras: list[str] = list(args.cameras)
    cuda_list: list[str | None] | None = None
    if args.cuda_devices:
        parts = [p.strip() for p in args.cuda_devices.split(",")]
        if len(parts) != len(cameras):
            print(
                f"Error: --cuda-devices must have {len(cameras)} comma-separated values "
                f"(got {len(parts)}) to match --cameras.",
                file=sys.stderr,
            )
            return 2
        cuda_list = [p if p != "" else None for p in parts]

    max_parallel = args.max_parallel if args.max_parallel is not None else len(cameras)
    if max_parallel < 1:
        print("Error: --max-parallel must be >= 1", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else PROJECT_ROOT / "coach-store" / "parallel-run-logs" / stamp
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(enumerate(cameras))
    running: list[tuple[int, str, subprocess.Popen, object]] = []
    results: dict[int, tuple[str, int | None]] = {}

    def start_next() -> bool:
        nonlocal pending
        if not pending:
            return False
        idx, camera = pending.pop(0)
        env = os.environ.copy()
        if cuda_list is not None and cuda_list[idx] is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_list[idx]

        cmd = _build_runner_cmd(
            video_root=video_root,
            drive_mount_root=args.drive_mount_root,
            date=args.date,
            camera=camera,
            no_viewer=args.no_viewer,
            stop_on_error=args.stop_on_error,
        )

        log_path = log_dir / f"{camera.replace(' ', '_')}.log"
        log_f = open(log_path, "w", encoding="utf-8", newline="\n")
        log_f.write(f"# cmd: {' '.join(cmd)}\n# cwd: {PROJECT_ROOT}\n\n")
        log_f.flush()

        print(f"[launch] {camera} -> log: {log_path}")
        p = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        running.append((idx, camera, p, log_f))
        return True

    def refill() -> None:
        while len(running) < max_parallel and pending:
            start_next()

    refill()

    while running or pending:
        if not running and pending:
            refill()
        if not running and not pending:
            break
        time.sleep(0.25)
        still: list[tuple[int, str, subprocess.Popen, object]] = []
        for idx, camera, p, log_f in running:
            rc = p.poll()
            if rc is None:
                still.append((idx, camera, p, log_f))
                continue
            try:
                log_f.close()
            except Exception:
                pass
            results[idx] = (camera, rc)
            print(f"[done] {camera} exit={rc}")
        running = still
        refill()

    print("\n--- Summary ---")
    failed = 0
    for i in range(len(cameras)):
        cam, rc = results[i]
        ok = rc == 0
        if not ok:
            failed += 1
        status = "OK" if ok else "FAIL"
        print(f"  {cam}: {status} (exit {rc})")
    print(f"Logs: {log_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
