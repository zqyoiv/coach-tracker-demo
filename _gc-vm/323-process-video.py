#!/usr/bin/env python3
"""
Process only date folder 3-23 under the Coach video root — same steps as process-video.sh:

  1) Run coach-raw-video-concat-hours.py (--root, --dates 3-23, optional --coaches).
  2) For each Coach-N: if hourly/ has .mp4 files, delete top-level *.mp4 in that folder,
     move hourly/*.mp4 up, remove hourly/.

Default root matches this machine:
  C:\\Users\\vioyq\\Desktop\\Coach_Tracker\\_Coach_Video

Override with env COACH_VIDEO_ROOT or --root.

Examples:
  python _gc-vm/323-process-video.py
  python _gc-vm/323-process-video.py --coaches 1 2
  python _gc-vm/323-process-video.py --root "D:\\Videos\\_Coach_Video"

Note: coach-raw-video-concat-hours.py only merges chunks whose names match
  coach-N-<stream>-<start14>-<end14>.mp4
Other filenames (e.g. RecS0A_...) are skipped by that script; cleanup still runs if hourly/ exists.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CONCAT_SCRIPT = SCRIPT_DIR / "coach-raw-video-concat-hours.py"

DEFAULT_VIDEO_ROOT = Path(r"C:\Users\vioyq\Desktop\Coach_Tracker\_Coach_Video")
TARGET_DATE = "3-23"

COACH_DIR_RE = re.compile(r"^Coach-(\d+)$", re.IGNORECASE)


def _video_root_from_env_or_default(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser().resolve()
    env = os.environ.get("COACH_VIDEO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_VIDEO_ROOT.expanduser().resolve()


def cleanup_one_coach(coach_dir: Path) -> None:
    name = coach_dir.name
    if not coach_dir.is_dir():
        print(f"⚠️ Warning: No folder {name}. Skipping.")
        return

    print(f"Processing {name}...")
    hourly = coach_dir / "hourly"
    if not hourly.is_dir():
        print(
            f"⚠️ Warning: No hourly folder or no merged videos path in {name}. Skipping cleanup."
        )
        return

    hour_mp4 = sorted(hourly.glob("*.mp4"))
    if not hour_mp4:
        print(
            f"⚠️ Warning: No merged videos (.mp4) in {name}/hourly. Skipping cleanup."
        )
        return

    for f in coach_dir.glob("*.mp4"):
        try:
            f.unlink()
        except OSError as e:
            print(f"⚠️ Warning: could not remove {f}: {e}")

    for f in hour_mp4:
        dest = coach_dir / f.name
        shutil.move(str(f), str(dest))

    try:
        hourly.rmdir()
    except OSError:
        # e.g. non-mp4 leftovers
        shutil.rmtree(hourly, ignore_errors=False)

    print(f"✅ {name} done.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Merge + cleanup for date {TARGET_DATE} only (same logic as process-video.sh)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=f"Video root containing date folders (default: COACH_VIDEO_ROOT or {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument(
        "--coaches",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Only Coach-N folders (1-5). Default: all Coach-* under the date folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to coach-raw-video-concat-hours.py; skip cleanup step.",
    )
    args = parser.parse_args()

    if args.coaches is not None:
        bad = [n for n in args.coaches if n < 1 or n > 5]
        if bad:
            print(f"ERROR: --coaches must be 1..5 (got {bad})", file=sys.stderr)
            return 2

    video_root = _video_root_from_env_or_default(args.root)
    base_path = video_root / TARGET_DATE

    print(f"🚀 Starting processing for Date: {TARGET_DATE}")
    print(f"   VIDEO_ROOT={video_root}")
    if args.coaches:
        print(f"   COACHES only: {' '.join(str(n) for n in args.coaches)}")

    if not CONCAT_SCRIPT.is_file():
        print(f"❌ Error: concat script not found: {CONCAT_SCRIPT}", file=sys.stderr)
        return 2

    if not base_path.is_dir():
        print(f"❌ Error: Directory does not exist: {base_path}", file=sys.stderr)
        return 1

    print("🎬 Step 1: Merging videos into 1-hour chunks...")
    cmd = [
        sys.executable,
        str(CONCAT_SCRIPT),
        "--root",
        str(video_root),
        "--dates",
        TARGET_DATE,
    ]
    if args.coaches:
        cmd.append("--coaches")
        cmd.extend(str(n) for n in args.coaches)
    if args.dry_run:
        cmd.append("--dry-run")

    r = subprocess.run(cmd, cwd=str(SCRIPT_DIR.parent))
    if r.returncode != 0:
        print("❌ Error: Python merge script failed. Stopping.", file=sys.stderr)
        return r.returncode

    if args.dry_run:
        print("✨ Dry-run: skipped cleanup step.")
        return 0

    print("🧹 Step 2: Cleaning up and moving files...")
    if args.coaches:
        for n in args.coaches:
            cleanup_one_coach(base_path / f"Coach-{n}")
    else:
        for sub in sorted(base_path.iterdir(), key=lambda p: p.name.lower()):
            if not sub.is_dir():
                continue
            if COACH_DIR_RE.match(sub.name.strip()):
                cleanup_one_coach(sub)

    print(f"✨ All tasks for {TARGET_DATE} are finished!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
