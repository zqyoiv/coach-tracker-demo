"""
Run coach videos chronologically for a given date + camera.

Examples:
  python video-script/coach-date-camera-is-runner.py --date 3-13 --camera coach-2
  python video-script/coach-date-camera-is-runner.py --date 03-13 --camera coach2 --no-viewer
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


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


def _canon(s: str) -> str:
    """Canonical string for case-insensitive, punctuation-insensitive matching."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_child_dir(parent: Path, candidates: list[str]) -> Path | None:
    """
    Find a subdirectory under `parent` matching any candidate name, using
    canonical comparison (case-insensitive; ignores separators like -/_/space).
    """
    if not parent.exists() or not parent.is_dir():
        return None

    wanted = {_canon(c) for c in candidates}
    for p in parent.iterdir():
        if p.is_dir() and _canon(p.name) in wanted:
            return p
    return None


def is_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


def extract_drive_folder_id(url: str) -> str:
    # Typical forms:
    # - https://drive.google.com/drive/folders/<FOLDER_ID>
    # - https://drive.google.com/drive/u/0/folders/<FOLDER_ID>?usp=sharing
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    if not m:
        raise ValueError(
            f"Could not parse Drive folder ID from URL: {url}\n"
            "Use a URL like: https://drive.google.com/drive/folders/<FOLDER_ID>"
        )
    return m.group(1)


def resolve_video_root(video_root_arg: str, drive_folder_url: str | None, drive_mount_root: str) -> Path:
    """
    Resolve the base folder that contains date/camera folders.
    Priority:
      1) --drive-folder-url (Google Drive URL)
      2) --video-root
    """
    if drive_folder_url:
        folder_id = extract_drive_folder_id(drive_folder_url)
        # In Colab, shortcut/shared folders are usually available here:
        # /content/drive/.shortcut-targets-by-id/<FOLDER_ID>
        candidate = Path("/content/drive/.shortcut-targets-by-id") / folder_id
        if candidate.exists() and candidate.is_dir():
            return candidate

        # Fallback: allow custom mount root (if user mounted elsewhere).
        candidate2 = Path(drive_mount_root) / ".shortcut-targets-by-id" / folder_id
        if candidate2.exists() and candidate2.is_dir():
            return candidate2

        raise FileNotFoundError(
            "Drive folder URL was provided but the mounted path could not be resolved.\n"
            f"Tried:\n  - {candidate}\n  - {candidate2}\n"
            "In Colab, mount Drive first:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')\n"
            "Then either use a shortcut/shared folder URL, or pass --video-root with explicit mounted path."
        )

    # Allow passing a URL through --video-root too, for convenience.
    if is_url(video_root_arg):
        root_id = extract_drive_folder_id(video_root_arg)
        candidate = Path("/content/drive/.shortcut-targets-by-id") / root_id
        if candidate.exists() and candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"--video-root looks like a URL but folder is not resolvable in mount: {video_root_arg}\n"
            "Use --drive-folder-url (preferred) or pass --video-root as local mounted path."
        )

    return Path(video_root_arg)


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
    parser.add_argument(
        "--drive-folder-url",
        default=None,
        help=(
            "Google Drive folder URL that contains date/camera subfolders "
            "(e.g., .../drive/folders/<id>). Useful for Colab."
        ),
    )
    parser.add_argument(
        "--drive-mount-root",
        default="/content/drive",
        help="Drive mount root in Colab (default: /content/drive).",
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

    try:
        root_folder = resolve_video_root(args.video_root, args.drive_folder_url, args.drive_mount_root)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1

    # Robust folder resolution for Drive/Colab where names may vary in case/format.
    month, day = [int(x) for x in date_folder.split("-")]
    date_candidates = [
        f"{month}-{day}",
        f"{month:02d}-{day:02d}",
        f"{month}/{day}",
        f"{month:02d}/{day:02d}",
    ]
    date_dir = find_child_dir(root_folder, date_candidates)
    if date_dir is None:
        print(f"Error: date folder not found under {root_folder}")
        print(f"Tried variants: {date_candidates}")
        return 1

    cam_num_match = re.search(r"([1-5])$", args.camera.strip())
    cam_num = cam_num_match.group(1) if cam_num_match else ""
    camera_candidates = [
        camera_folder,           # Coach-1
        camera_folder.lower(),   # coach-1
        f"coach-{cam_num}",
        f"Coach-{cam_num}",
        f"coach{cam_num}",
        f"Coach{cam_num}",
    ]
    camera_dir = find_child_dir(date_dir, camera_candidates)
    if camera_dir is None:
        print(f"Error: camera folder not found under {date_dir}")
        print(f"Tried variants: {camera_candidates}")
        return 1

    target_folder = camera_dir

    videos = list_videos_chronological(target_folder)
    if not videos:
        print(f"No .mp4 files found in: {target_folder}")
        return 0

    print(f"Found {len(videos)} video(s) in {target_folder}")
    for idx, v in enumerate(videos, start=1):
        print(f"{idx:03d}. {v.name}")

    failures = 0
    run_start = time.perf_counter()
    total = len(videos)
    for i, v in enumerate(videos, start=1):
        elapsed = time.perf_counter() - run_start
        progress = i / total if total else 1.0
        eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
        print(
            f"\n[Progress] {i}/{total} ({progress * 100:5.1f}%) "
            f"| elapsed {elapsed:7.1f}s | ETA {eta:7.1f}s"
        )
        rc = run_tracker_for_video(v, no_viewer=args.no_viewer)
        if rc != 0:
            failures += 1
            print(f"Warning: run failed ({rc}) for {v.name}")
            if args.stop_on_error:
                break

    total_elapsed = time.perf_counter() - run_start
    if failures:
        print(f"\nDone with {failures} failed run(s). Total time: {total_elapsed:.1f}s")
        return 1
    print(f"\nDone. All videos processed. Total time: {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

