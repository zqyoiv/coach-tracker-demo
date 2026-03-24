"""
Run multiple coach-date-camera-id-runner.py jobs in parallel (one OS process per camera).

Each camera writes to its own coach-store CSV-state / person-ID paths — safe to parallelize.
On one GPU, many concurrent YOLO runs may OOM or slow down; use --max-parallel 1 or 2,
or assign GPUs with --cuda-devices 0,1,2,3.

Use --dates to process several day folders in order (e.g. 3-14, 3-15, 3-16); each day
runs the same camera pool before moving to the next day.

Examples (from repo root):

  Windows:
    python _gc-vm/run-coaches-parallel.py --date 3-13 --video-root "%USERPROFILE%\\coach-raw-video" ...

  Linux / VM:
    python _gc-vm/run-coaches-parallel.py --date 3-13 --no-viewer
    python _gc-vm/run-coaches-parallel.py --dates 3-14 3-15 3-16 --no-viewer --max-parallel 2
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


def run_parallel_for_date(
    date_str: str,
    *,
    video_root: str,
    cameras: list[str],
    drive_mount_root: str,
    no_viewer: bool,
    stop_on_error: bool,
    max_parallel: int,
    cuda_list: list[str | None] | None,
    log_dir: Path,
) -> int:
    """Run all cameras for one date. Returns 0 if all OK, 1 if any failed."""
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
            drive_mount_root=drive_mount_root,
            date=date_str,
            camera=camera,
            no_viewer=no_viewer,
            stop_on_error=stop_on_error,
        )

        log_path = log_dir / f"{camera.replace(' ', '_')}.log"
        log_f = open(log_path, "w", encoding="utf-8", newline="\n")
        log_f.write(f"# date={date_str}\n# cmd: {' '.join(cmd)}\n# cwd: {PROJECT_ROOT}\n\n")
        log_f.flush()

        print(f"[launch] {date_str} {camera} -> log: {log_path}")
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
            print(f"[done] {date_str} {camera} exit={rc}")
        running = still
        refill()

    print(f"\n--- Summary for {date_str} ---")
    failed = 0
    for i in range(len(cameras)):
        cam, rc = results[i]
        ok = rc == 0
        if not ok:
            failed += 1
        status = "OK" if ok else "FAIL"
        print(f"  {cam}: {status} (exit {rc})")
    print(f"Logs: {log_dir}\n")

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run coach-date-camera-id-runner.py for several cameras in parallel."
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Single date folder (e.g. 3-13). Ignored if --dates is set.",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=None,
        metavar="M-D",
        help="Multiple date folders, processed in order (e.g. 3-14 3-15 3-16). Overrides --date.",
    )
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
        help="Base log directory (default: coach-store/parallel-run-logs/<timestamp>/<date>/).",
    )
    args = parser.parse_args()

    if args.dates:
        date_list = list(args.dates)
    elif args.date:
        date_list = [args.date]
    else:
        parser.error("Pass --date once or --dates with one or more folders (e.g. --dates 3-14 3-15 3-16).")

    raw_root = args.video_root.strip()
    if "%" in raw_root and any(
        x in raw_root.upper() for x in ("USERPROFILE", "HOMEPATH", "LOCALAPPDATA")
    ):
        print(
            "ERROR: You used a Windows-style path (e.g. %USERPROFILE%...).\n"
            "On Linux that string is NOT expanded and points to the wrong place.\n"
            "Use one of:\n"
            f"  --video-root ~/coach-raw-video\n"
            f"  --video-root {Path.home() / 'coach-raw-video'}\n"
            "Or omit --video-root (default is ~/coach-raw-video).",
            file=sys.stderr,
        )
        return 2

    video_root = str(Path(raw_root).expanduser().resolve())
    if not Path(video_root).is_dir():
        print(
            f"ERROR: --video-root is not a directory:\n  {video_root}\n"
            "Fix the path, or create the folder.",
            file=sys.stderr,
        )
        return 2

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
    log_base = (
        Path(args.log_dir)
        if args.log_dir
        else PROJECT_ROOT / "coach-store" / "parallel-run-logs" / stamp
    )

    any_fail = False
    for d in date_list:
        # Safe subfolder name (e.g. 3/13 -> 3-13 if user passed wrong; keep as-is for 3-14)
        sub = d.replace("/", "-").strip()
        log_dir = log_base / sub
        print(f"\n{'=' * 60}\n  DATE {sub}  ({len(date_list)} day(s) total)\n{'=' * 60}")
        rc = run_parallel_for_date(
            sub,
            video_root=video_root,
            cameras=cameras,
            drive_mount_root=args.drive_mount_root,
            no_viewer=args.no_viewer,
            stop_on_error=args.stop_on_error,
            max_parallel=max_parallel,
            cuda_list=cuda_list,
            log_dir=log_dir,
        )
        if rc != 0:
            any_fail = True

    if len(date_list) > 1:
        print(f"\nAll dates finished. Log base: {log_base}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
