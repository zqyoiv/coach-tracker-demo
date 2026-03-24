#!/usr/bin/env python3
"""
Merge 5-minute MP4 chunks into ~1-hour videos (12 × 5 min), mirroring
concat-to-hours.ps1 on Windows.

Layout (e.g. on a Linux VM):
  ~/coach-raw-video/
    3-14/
      Coach-1/*.mp4   -> 3-14/Coach-1/hourly/Coach-1-hour-1-20260313105712.mp4
                        (suffix = first chunk's start timestamp in that hour)
      Coach-2/
      ...

Filename pattern (per camera N):
  coach-N-<stream>-<start14>-<end14>[-<dup>].mp4

Same (start,end) window may appear more than once; the largest file is kept.

Requires: ffmpeg in PATH

Examples:
  python video-script/coach-raw-video-concat-hours.py
  python video-script/coach-raw-video-concat-hours.py --root ~/coach-raw-video
  python video-script/coach-raw-video-concat-hours.py --dates 3-14 3-15 --dry-run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# Date folder like 3-14, 12-25 (skip 00 DownloadsInProgress, etc.)
DATE_DIR_RE = re.compile(r"^\d{1,2}-\d{1,2}$")
COACH_DIR_RE = re.compile(r"^Coach-(\d+)$", re.IGNORECASE)

# coach-2-1-20260313120000-20260313120500-2.mp4  -> groups: cam, stream?, start, end, dup
CHUNK_NAME_RE = re.compile(
    r"^coach-(\d+)-(\d+)-(\d{14})-(\d{14})(?:-(\d+))?\.mp4$",
    re.IGNORECASE,
)

CHUNKS_PER_HOUR = 12


def _chunk_start_ts(p: Path) -> str | None:
    """First 14-digit start time from a chunk filename, e.g. coach-1-1-20260313105712-....mp4"""
    m = CHUNK_NAME_RE.match(p.name)
    if m:
        return m.group(3)
    return None


def _escape_concat_path(p: Path) -> str:
    s = p.resolve().as_posix()
    return s.replace("'", r"'\''")


def _write_concat_list(paths: list[Path], list_path: Path) -> None:
    lines = [f"file '{_escape_concat_path(p)}'" for p in paths]
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _run_ffmpeg_concat(list_path: Path, out_path: Path) -> int:
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
    r = subprocess.run(cmd, check=False)
    return int(r.returncode)


def process_coach_folder(
    coach_dir: Path,
    *,
    coach_label: str,
    coach_num: int,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Returns (hours_written, chunks_used).
    Only reads .mp4 directly under coach_dir (not in subfolders like hourly/).
    """
    # Files only at top level
    mp4s = [p for p in coach_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    if not mp4s:
        return 0, 0

    # slot_key -> list of (path, size)
    slots: dict[str, list[tuple[Path, int]]] = defaultdict(list)

    for p in mp4s:
        m = CHUNK_NAME_RE.match(p.name)
        if not m:
            continue
        cam_in_name = int(m.group(1))
        if cam_in_name != coach_num:
            continue
        start_ts, end_ts = m.group(3), m.group(4)
        slot_key = f"{start_ts}-{end_ts}"
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        slots[slot_key].append((p, sz))

    if not slots:
        print(f"  [{coach_label}] no matching coach-{coach_num}-*.mp4 chunks in {coach_dir}")
        return 0, 0

    # Sort by start time; per slot pick largest file
    def slot_start(key: str) -> str:
        return key.split("-")[0]

    ordered: list[Path] = []
    for slot_key in sorted(slots.keys(), key=slot_start):
        candidates = slots[slot_key]
        best = max(candidates, key=lambda x: x[1])[0]
        ordered.append(best)

    n_chunks = len(ordered)
    hourly_dir = coach_dir / "hourly"
    hours_written = 0

    print(f"  [{coach_label}] {n_chunks} chunk(s) -> up to {(n_chunks + CHUNKS_PER_HOUR - 1) // CHUNKS_PER_HOUR} hour file(s)")

    hour_num = 1
    for i in range(0, len(ordered), CHUNKS_PER_HOUR):
        chunk_paths = ordered[i : i + CHUNKS_PER_HOUR]
        start_ts = _chunk_start_ts(chunk_paths[0]) or "unknown"
        out_name = f"{coach_label}-hour-{hour_num}-{start_ts}.mp4"
        out_path = hourly_dir / out_name
        list_path = hourly_dir / f".concat-hour-{hour_num}-{start_ts}.txt"

        if dry_run:
            print(f"    [dry-run] hour {hour_num}: {len(chunk_paths)} chunks -> {out_path}")
            hours_written += 1
            hour_num += 1
            continue

        hourly_dir.mkdir(parents=True, exist_ok=True)
        _write_concat_list(chunk_paths, list_path)
        print(f"    hour {hour_num} ({len(chunk_paths)} chunks) -> {out_path.name}")
        rc = _run_ffmpeg_concat(list_path, out_path)
        try:
            list_path.unlink(missing_ok=True)
        except OSError:
            pass
        if rc != 0:
            print(f"    ERROR: ffmpeg failed (exit {rc}) for {out_path}", file=sys.stderr)
        else:
            hours_written += 1
        hour_num += 1

    return hours_written, n_chunks


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concat 5-min chunks into hourly videos under each Coach-x/hourly/."
    )
    ap.add_argument(
        "--root",
        default="~/coach-raw-video",
        help="Root folder containing date dirs (default: ~/coach-raw-video)",
    )
    ap.add_argument(
        "--dates",
        nargs="*",
        help="Only these date folder names (e.g. 3-14 3-15). Default: all M-D dirs under root.",
    )
    ap.add_argument(
        "--skip-prefix",
        default="00",
        help="Skip top-level dirs whose name starts with this (default: 00 e.g. 00 DownloadsInProgress)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; no ffmpeg")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    if not args.dry_run:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: ffmpeg not found in PATH.", file=sys.stderr)
            return 2

    date_filter = set(args.dates) if args.dates else None
    date_dirs: list[Path] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if args.skip_prefix and p.name.startswith(args.skip_prefix):
            continue
        if not DATE_DIR_RE.match(p.name.strip()):
            continue
        if date_filter is not None and p.name not in date_filter:
            continue
        date_dirs.append(p)

    if not date_dirs:
        print(f"No date folders found under {root}")
        return 1

    total_hours = 0
    total_chunks = 0

    for date_dir in date_dirs:
        print(f"\n=== Date: {date_dir.name} ===")
        subs = sorted(date_dir.iterdir(), key=lambda x: x.name.lower())
        for sub in subs:
            if not sub.is_dir():
                continue
            m = COACH_DIR_RE.match(sub.name.strip())
            if not m:
                continue
            num = int(m.group(1))
            if num < 1 or num > 5:
                continue
            # Normalize label Coach-1 style for output filenames
            label = f"Coach-{num}"
            print(f" {label} ({sub})")
            h, c = process_coach_folder(
                sub,
                coach_label=label,
                coach_num=num,
                dry_run=bool(args.dry_run),
            )
            total_hours += h
            total_chunks += c

    print(f"\nDone. Hour files written: {total_hours} (chunks counted: {total_chunks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
