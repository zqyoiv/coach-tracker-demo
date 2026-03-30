"""
Run one camera across multiple dates in parallel.

This script launches multiple `video-script/coach-date-camera-id-runner.py` jobs,
one per date, for a single camera.

Example:
  python _gc-vm/run-camera-parallel.py --camera coach-1 --dates 3-13 3-14 3-15 --no-viewer
"""

from __future__ import annotations

import argparse
import os
import re
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


def _zone_file_for_camera(camera: str) -> Path:
    m = re.search(r"coach[-_]?([1-5])$", camera.strip().lower())
    if not m:
        raise ValueError(
            f"Unsupported camera '{camera}'. Use coach-1..coach-5 (or coach1..coach5)."
        )
    slug = f"coach{m.group(1)}"
    return PROJECT_ROOT / "coach-store" / "zone" / f"{slug}.json"


def run_parallel_dates_for_camera(
    *,
    dates: list[str],
    camera: str,
    video_root: str,
    drive_mount_root: str,
    no_viewer: bool,
    stop_on_error: bool,
    max_parallel: int,
    cuda_devices: list[str] | None,
    log_base: Path,
) -> int:
    """Run one camera across dates. Returns 0 if all succeed, 1 otherwise."""
    log_base.mkdir(parents=True, exist_ok=True)

    pending = list(enumerate(dates))
    running: list[tuple[int, str, subprocess.Popen, object]] = []
    results: dict[int, tuple[str, int | None]] = {}

    def start_next() -> bool:
        nonlocal pending
        if not pending:
            return False

        idx, date_str = pending.pop(0)
        env = os.environ.copy()
        if cuda_devices is not None and idx < len(cuda_devices):
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices[idx]

        cmd = _build_runner_cmd(
            video_root=video_root,
            drive_mount_root=drive_mount_root,
            date=date_str,
            camera=camera,
            no_viewer=no_viewer,
            stop_on_error=stop_on_error,
        )

        safe_date = date_str.replace("/", "-").strip()
        log_path = log_base / f"{safe_date}.log"
        log_f = open(log_path, "w", encoding="utf-8", newline="\n")
        log_f.write(f"# camera={camera}\n# date={date_str}\n# cmd: {' '.join(cmd)}\n# cwd: {PROJECT_ROOT}\n\n")
        log_f.flush()

        print(f"[launch] {camera} {date_str} -> log: {log_path}")
        p = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        running.append((idx, date_str, p, log_f))
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
        for idx, date_str, p, log_f in running:
            rc = p.poll()
            if rc is None:
                still.append((idx, date_str, p, log_f))
                continue
            try:
                log_f.close()
            except Exception:
                pass
            results[idx] = (date_str, rc)
            print(f"[done] {camera} {date_str} exit={rc}")
        running = still
        refill()

    print(f"\n--- Summary for camera {camera} ---")
    failed = 0
    for i in range(len(dates)):
        d, rc = results[i]
        ok = rc == 0
        if not ok:
            failed += 1
        status = "OK" if ok else "FAIL"
        print(f"  {d}: {status} (exit {rc})")
    print(f"Logs: {log_base}\n")

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one camera across multiple dates in parallel."
    )
    parser.add_argument(
        "--camera",
        required=True,
        help="Single camera to run, e.g. coach-1",
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
        help="Multiple date folders, processed in parallel (e.g. 3-13 3-14 3-15).",
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
        help="Max concurrent jobs (default: number of dates). Lower on single GPU.",
    )
    parser.add_argument(
        "--cuda-devices",
        default=None,
        help="Comma-separated CUDA ids by date order, e.g. 0,1,2. If one value, it is reused for all dates.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Base log directory (default: coach-store/parallel-run-logs/<timestamp>/camera-<id>/).",
    )
    args = parser.parse_args()

    if args.dates:
        date_list = [d.replace("/", "-").strip() for d in args.dates]
    elif args.date:
        date_list = [args.date.replace("/", "-").strip()]
    else:
        parser.error("Pass --date once or --dates with one or more folders.")

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

    try:
        zone_file = _zone_file_for_camera(args.camera)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if not zone_file.exists():
        print(
            f"Error: zone file not found for camera '{args.camera}':\n  {zone_file}",
            file=sys.stderr,
        )
        return 2
    print(f"Using zone config: {zone_file}")

    if args.max_parallel is None:
        max_parallel = len(date_list)
    else:
        max_parallel = args.max_parallel
    if max_parallel < 1:
        print("Error: --max-parallel must be >= 1", file=sys.stderr)
        return 2

    cuda_list: list[str] | None = None
    if args.cuda_devices:
        parts = [p.strip() for p in args.cuda_devices.split(",") if p.strip() != ""]
        if len(parts) == 1:
            cuda_list = [parts[0]] * len(date_list)
        elif len(parts) == len(date_list):
            cuda_list = parts
        else:
            print(
                "Error: --cuda-devices must have 1 value (reused) or exactly one value per date.",
                file=sys.stderr,
            )
            return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_camera = args.camera.replace("/", "-").replace(" ", "_")
    log_base = (
        Path(args.log_dir)
        if args.log_dir
        else PROJECT_ROOT / "coach-store" / "parallel-run-logs" / stamp / safe_camera
    )

    return run_parallel_dates_for_camera(
        dates=date_list,
        camera=args.camera,
        video_root=video_root,
        drive_mount_root=args.drive_mount_root,
        no_viewer=args.no_viewer,
        stop_on_error=args.stop_on_error,
        max_parallel=max_parallel,
        cuda_devices=cuda_list,
        log_base=log_base,
    )


if __name__ == "__main__":
    raise SystemExit(main())
