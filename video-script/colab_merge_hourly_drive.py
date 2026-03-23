#!/usr/bin/env python3
"""
Merge sequential video clips from a Google Drive folder tree into ~1-hour files.

Intended for Google Colab: each date folder (e.g. 3-13) contains Coach-1 … Coach-5
subfolders with many short .mp4 clips. This script sorts clips chronologically,
groups them into batches whose total duration is at most TARGET_SECONDS (default 3600),
and concatenates each batch with ffmpeg (stream copy).

Example Colab setup (run in cells):

    # 1) Mount Drive
    from google.colab import drive
    drive.mount("/content/drive")

    # 2) Ensure ffmpeg + ffprobe
    !apt-get update -qq && apt-get install -qq -y ffmpeg

    # 3) Clone or upload this repo, then:
    %cd /content/coach-tracker-demo
    !python video-script/colab_merge_hourly_drive.py \\
        --input-root "/content/drive/.shortcut-targets-by-id/<FOLDER_ID>/Coach-AV-store-Test" \\
        --output-root "/content/drive/.shortcut-targets-by-id/<FOLDER_ID>/Coach-AV-store-Test-hourly"

Adjust --input-root / --output-root to your paths. You can keep output inside the same
Drive tree or next to it (recommended: separate folder so inputs stay untouched).

Output layout (default):

    <output-root>/
      3-13/
        Coach-1/
          Coach-1_part-001_of-011_20260313110000-20260313120000.mp4
          ...
        Coach-2/
        ...
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
# Match folders like 3-13, 12-25 (month-day)
DATE_DIR_RE = re.compile(r"^\d{1,2}-\d{1,2}$")
# Coach-N (case-insensitive)
COACH_DIR_RE = re.compile(r"^coach-(\d+)$", re.IGNORECASE)
# 14-digit timestamps in filenames (e.g. Reolink-style)
TS14_RE = re.compile(r"\d{14}")


@dataclass(frozen=True)
class VideoEntry:
    path: Path
    duration_sec: float


def _ffprobe_duration(path: Path) -> float:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"ffprobe failed for {path}: {e}") from e


def _sort_key(path: Path) -> tuple:
    """
    Chronological sort: prefer embedded 14-digit timestamps in the filename;
    fall back to mtime then name.
    """
    found = TS14_RE.findall(path.name)
    if found:
        return (0, found[0], path.name.lower())
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (1, f"{mtime:.6f}", path.name.lower())


def _list_videos(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            out.append(p)
    out.sort(key=_sort_key)
    return out


def _batch_by_duration(
    entries: list[VideoEntry], target_seconds: float
) -> list[list[VideoEntry]]:
    """Split ordered entries into batches with total duration <= target_seconds.
    A single clip longer than target becomes its own batch.
    """
    batches: list[list[VideoEntry]] = []
    current: list[VideoEntry] = []
    acc = 0.0

    for e in entries:
        dur = e.duration_sec
        if not current:
            current.append(e)
            acc = dur
            continue
        if acc + dur <= target_seconds + 1e-6:
            current.append(e)
            acc += dur
        else:
            batches.append(current)
            current = [e]
            acc = dur

    if current:
        batches.append(current)
    return batches


def _escape_concat_path(p: Path) -> str:
    # ffmpeg concat demuxer: use single quotes, escape embedded quotes
    s = str(p.resolve())
    return s.replace("'", r"'\''")


def _write_concat_list(paths: list[Path], list_path: Path) -> None:
    lines = []
    for p in paths:
        lines.append(f"file '{_escape_concat_path(p)}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_ffmpeg_concat(list_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def _parse_ts14(s: str) -> str:
    """Format YYYYMMDDHHmmss as YYYYMMDD_HHmmss for filenames."""
    if len(s) < 14:
        return s
    return f"{s[0:8]}_{s[8:14]}"


def _build_output_name(
    coach_label: str,
    part_idx: int,
    total_parts: int,
    first_clip: Path,
    last_clip: Path,
) -> str:
    f = TS14_RE.search(first_clip.name)
    l = TS14_RE.search(last_clip.name)
    start = _parse_ts14(f.group(0)) if f else "start"
    end = _parse_ts14(l.group(0)) if l else "end"
    return f"{coach_label}_part-{part_idx:03d}_of-{total_parts:03d}_{start}-{end}.mp4"


def process_camera_folder(
    date_name: str,
    coach_folder: Path,
    coach_label: str,
    out_dir: Path,
    target_seconds: float,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Returns (number of output files written, number of source clips processed).
    """
    videos = _list_videos(coach_folder)
    if not videos:
        return 0, 0

    entries: list[VideoEntry] = []
    for p in videos:
        d = _ffprobe_duration(p)
        if d <= 0:
            print(f"  skip zero-duration: {p.name}", file=sys.stderr)
            continue
        entries.append(VideoEntry(path=p, duration_sec=d))

    if not entries:
        return 0, 0

    batches = _batch_by_duration(entries, target_seconds)
    total_parts = len(batches)
    written = 0

    for i, batch in enumerate(batches, start=1):
        paths = [e.path for e in batch]
        out_name = _build_output_name(
            coach_label, i, total_parts, paths[0], paths[-1]
        )
        dest = out_dir / out_name
        total_dur = sum(e.duration_sec for e in batch)
        print(
            f"    [{coach_label}] part {i}/{total_parts}: "
            f"{len(batch)} clips, ~{total_dur / 60:.1f} min -> {dest.name}"
        )
        if dry_run:
            written += 1
            continue

        tmp_list = out_dir / f".concat_list_{coach_label}_part{i:03d}.txt"
        try:
            _write_concat_list(paths, tmp_list)
            _run_ffmpeg_concat(tmp_list, dest)
            written += 1
        finally:
            if tmp_list.exists():
                try:
                    tmp_list.unlink()
                except OSError:
                    pass

    return written, len(entries)


def _find_child_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def _match_date_dir(name: str) -> bool:
    return bool(DATE_DIR_RE.match(name.strip()))


def _parse_coach_label(folder_name: str) -> str | None:
    m = COACH_DIR_RE.match(folder_name.strip())
    if not m:
        return None
    return f"Coach-{int(m.group(1))}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Drive video clips into ~1-hour concatenated files (Colab-friendly)."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root folder containing date folders (e.g. .../Coach-AV-store-Test).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root folder for merged output (created if missing).",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=3600.0,
        help="Maximum total duration per merged file (default: 3600).",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Optional list of date folder names to process (e.g. 3-13 3-14). Default: all date-like folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned outputs without running ffmpeg.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser()

    if not input_root.is_dir():
        print(f"ERROR: input root does not exist or is not a directory:\n  {input_root}", file=sys.stderr)
        return 2

    date_filter = set(args.dates) if args.dates else None
    total_outputs = 0
    total_clips = 0

    date_dirs = [d for d in _find_child_dirs(input_root) if _match_date_dir(d.name)]
    if date_filter is not None:
        date_dirs = [d for d in date_dirs if d.name in date_filter]

    if not date_dirs:
        print(f"No date folders found under {input_root}", file=sys.stderr)
        return 1

    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Target: {args.target_seconds}s per merged file (~{args.target_seconds/3600:.2f} h)")
    print(f"Dry run: {args.dry_run}")
    print("---")

    for ddir in date_dirs:
        print(f"Date: {ddir.name}")
        subdirs = _find_child_dirs(ddir)
        for sub in subdirs:
            label = _parse_coach_label(sub.name)
            if not label:
                continue
            out_dir = output_root / ddir.name / label
            n_out, n_clip = process_camera_folder(
                ddir.name,
                sub,
                label,
                out_dir,
                target_seconds=args.target_seconds,
                dry_run=args.dry_run,
            )
            total_outputs += n_out
            total_clips += n_clip
            if n_clip:
                print(f"  -> {label}: {n_out} merged file(s) from {n_clip} clip(s)")

    print("---")
    print(f"Done. Merged files: {total_outputs}, source clips counted: {total_clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
