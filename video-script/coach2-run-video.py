"""
Run coach videos chronologically for a given date + camera.

Examples:
  python video-script/coach2-run-video.py --date 3-13 --camera coach-2
  python video-script/coach2-run-video.py --date 03-13 --camera coach2 --no-viewer
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_ROOT = Path(r"C:\Users\vioyq\Desktop\Coach_Tracker\_Coach_Video")
TRACKER_SCRIPT = PROJECT_ROOT / "coach-store" / "coach-play-tracker.py"


def normalize_camera(camera: str) -> str:
    m = re.search(r"coach[-_]?([1-5])$", camera.strip().lower())
    if not m:
        raise ValueError(f"Unsupported camera '{camera}'. Use coach-1..coach-5 (or coach1..coach5).")
    return f"Coach-{m.group(1)}"


def normalize_date(date_text: str) -> str:
    # Accept "3-13", "03-13", "3/13", "03/13"
    m = re.match(r"^\s*(\d{1,2})[-/](\d{1,2})\s*$", date_text)
    if not m:
        raise ValueError(f"Invalid date '{date_text}'. Use M-D or MM-DD, e.g. 3-13.")
    month = int(m.group(1))
    day = int(m.group(2))
    return f"{month}-{day}"


def extract_start_timestamp(name: str) -> str:
    # Preferred: first 14-digit timestamp in filename
    m = re.search(r"(\d{14})", name)
    return m.group(1) if m else ""


def list_videos_chronological(target_folder: Path) -> list[Path]:
    # Only top-level .mp4 files, ignoring folders and other files.
    videos = [p for p in target_folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    videos.sort(key=lambda p: (extract_start_timestamp(p.name), p.name.lower()))
    return videos


def run_tracker_for_video(video_path: Path, no_viewer: bool) -> int:
    cmd = [sys.executable, str(TRACKER_SCRIPT), str(video_path)]
    if no_viewer:
        cmd.append("--no-viewer")
    print(f"\n=== Running: {video_path.name} ===")
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all .mp4 snippets in chronological order for one date/camera.")
    parser.add_argument("--date", required=True, help="Date folder under _Coach_Video (e.g. 3-13).")
    parser.add_argument("--camera", required=True, help="Camera name (e.g. coach-2 or coach2).")
    parser.add_argument(
        "--video-root",
        default=str(DEFAULT_VIDEO_ROOT),
        help=f"Root containing date folders (default: {DEFAULT_VIDEO_ROOT})",
    )
    parser.add_argument("--no-viewer", action="store_true", help="Pass --no-viewer to coach-play-tracker.py")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if one video run fails. By default, continue to next file.",
    )
    args = parser.parse_args()

    if not TRACKER_SCRIPT.exists():
        print(f"Error: tracker script not found: {TRACKER_SCRIPT}")
        return 1

    try:
        date_folder = normalize_date(args.date)
        camera_folder = normalize_camera(args.camera)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    target_folder = Path(args.video_root) / date_folder / camera_folder
    if not target_folder.exists() or not target_folder.is_dir():
        print(f"Error: target folder not found: {target_folder}")
        return 1

    videos = list_videos_chronological(target_folder)
    if not videos:
        print(f"No .mp4 files found in: {target_folder}")
        return 0

    print(f"Found {len(videos)} video(s) in {target_folder}")
    for idx, v in enumerate(videos, start=1):
        print(f"{idx:03d}. {v.name}")

    failures = 0
    for v in videos:
        rc = run_tracker_for_video(v, no_viewer=args.no_viewer)
        if rc != 0:
            failures += 1
            print(f"Warning: run failed ({rc}) for {v.name}")
            if args.stop_on_error:
                break

    if failures:
        print(f"\nDone with {failures} failed run(s).")
        return 1
    print("\nDone. All videos processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

