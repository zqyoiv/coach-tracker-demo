#!/usr/bin/env python3
"""
Merge 5-minute NVR chunks (3-23 / RecS0A-style names) into ~1-hour videos.

Input naming (e.g. under .../3-23/Coach-1/):
  RecS0A_DST20260323_095957_100457_0_55148100000000_D0CA7B.mp4
    -> date 20260323, clip start 09:59:57, clip end 10:04:57 (same day)
  RecS0A_DST20260323_204500_000000_0_55148100000000_0.mp4
    -> crosses midnight: end becomes next calendar day 00:00:00

Dedup: same (start14, end14) window -> keep largest file.

Output (same as coach-raw-video-concat-hours.py):
  Coach-N/hourly/Coach-N-hour-<k>-<first_chunk_start14>.mp4

Examples:
  python _gc-vm/3-23-video-concat-hours.py
  python _gc-vm/3-23-video-concat-hours.py --root "C:\\...\\_Coach_Video" --dates 3-23
  python _gc-vm/3-23-video-concat-hours.py --coaches 1 2 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


DATE_DIR_RE = re.compile(r"^\d{1,2}-\d{1,2}$")
COACH_DIR_RE = re.compile(r"^Coach-(\d+)$", re.IGNORECASE)

# RecS0A_DST20260323_095957_100457_0_55148100000000_D0CA7B.mp4
REC_CHUNK_RE = re.compile(
    r"^RecS0A_DST(\d{8})_(\d{6})_(\d{6})_.+\.mp4$",
    re.IGNORECASE,
)

CHUNKS_PER_HOUR = 12

DEFAULT_VIDEO_ROOT = Path(r"C:\Users\vioyq\Desktop\Coach_Tracker\_Coach_Video")
DEFAULT_DATES = ("3-23",)


def _video_root(cli_root: Path | None) -> Path:
    if cli_root is not None:
        return cli_root.expanduser().resolve()
    env = os.environ.get("COACH_VIDEO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_VIDEO_ROOT.expanduser().resolve()


def _clip_start_end_14(ymd: str, start_hms: str, end_hms: str) -> tuple[str, str]:
    """Return (start14, end14) with correct day for overnight clips."""
    start14 = f"{ymd}{start_hms}"
    s = datetime.strptime(start14, "%Y%m%d%H%M%S")
    e_same = datetime.strptime(f"{ymd}{end_hms}", "%Y%m%d%H%M%S")
    if e_same >= s:
        end14 = e_same.strftime("%Y%m%d%H%M%S")
    else:
        end14 = (e_same + timedelta(days=1)).strftime("%Y%m%d%H%M%S")
    return start14, end14


def _parse_rec_chunk(p: Path) -> tuple[str, str] | None:
    m = REC_CHUNK_RE.match(p.name)
    if not m:
        return None
    ymd, h1, h2 = m.group(1), m.group(2), m.group(3)
    return _clip_start_end_14(ymd, h1, h2)


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


def process_coach_folder_rec(
    coach_dir: Path,
    *,
    coach_label: str,
    dry_run: bool,
) -> tuple[int, int]:
    """
    Returns (hours_written, chunks_used).
    Only reads .mp4 directly under coach_dir (not hourly/).
    """
    mp4s = [p for p in coach_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    if not mp4s:
        return 0, 0

    slots: dict[str, list[tuple[Path, int, str]]] = defaultdict(list)

    for p in mp4s:
        parsed = _parse_rec_chunk(p)
        if not parsed:
            continue
        start14, end14 = parsed
        slot_key = f"{start14}-{end14}"
        try:
            sz = p.stat().st_size
        except OSError:
            sz = 0
        slots[slot_key].append((p, sz, start14))

    if not slots:
        print(
            f"  [{coach_label}] no matching RecS0A_DST*_*_*_*.mp4 chunks in {coach_dir}"
        )
        return 0, 0

    def slot_start(key: str) -> str:
        return key.split("-", 1)[0]

    ordered: list[tuple[Path, str]] = []
    for slot_key in sorted(slots.keys(), key=slot_start):
        candidates = slots[slot_key]
        best_path, _sz, start14 = max(candidates, key=lambda x: x[1])
        ordered.append((best_path, start14))

    n_chunks = len(ordered)
    hourly_dir = coach_dir / "hourly"
    hours_written = 0

    print(
        f"  [{coach_label}] {n_chunks} chunk(s) -> up to "
        f"{(n_chunks + CHUNKS_PER_HOUR - 1) // CHUNKS_PER_HOUR} hour file(s)"
    )

    hour_num = 1
    for i in range(0, len(ordered), CHUNKS_PER_HOUR):
        chunk_batch = ordered[i : i + CHUNKS_PER_HOUR]
        chunk_paths = [t[0] for t in chunk_batch]
        start_ts = chunk_batch[0][1]
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
        description=(
            "Concat RecS0A_DST* 5-min chunks into hourly videos "
            "(same output names as coach-raw-video-concat-hours.py)."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Root folder containing date dirs (default: COACH_VIDEO_ROOT env or "
            f"{DEFAULT_VIDEO_ROOT})"
        ),
    )
    ap.add_argument(
        "--dates",
        nargs="+",
        default=list(DEFAULT_DATES),
        metavar="M-D",
        help=f"Date folder names under root (default: {' '.join(DEFAULT_DATES)}).",
    )
    ap.add_argument(
        "--skip-prefix",
        default="00",
        help="Skip top-level dirs whose name starts with this (default: 00).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan only; no ffmpeg")
    ap.add_argument(
        "--coaches",
        nargs="+",
        type=int,
        metavar="N",
        default=None,
        help="Only these Coach-N folders (1-5). Default: all Coach-* under each date.",
    )
    args = ap.parse_args()

    if args.coaches is not None:
        bad = [n for n in args.coaches if n < 1 or n > 5]
        if bad:
            print(f"ERROR: --coaches must be 1..5 (got {bad})", file=sys.stderr)
            return 2

    root = _video_root(args.root)
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    if not args.dry_run:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: ffmpeg not found in PATH.", file=sys.stderr)
            return 2

    date_filter = set(args.dates)
    date_dirs: list[Path] = []
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if args.skip_prefix and p.name.startswith(args.skip_prefix):
            continue
        if not DATE_DIR_RE.match(p.name.strip()):
            continue
        if p.name not in date_filter:
            continue
        date_dirs.append(p)

    if not date_dirs:
        print(f"No matching date folders under {root} for: {sorted(date_filter)}")
        return 1

    total_hours = 0
    total_chunks = 0

    for date_dir in date_dirs:
        print(f"\n=== Date: {date_dir.name} (RecS0A chunk names) ===")
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
            if args.coaches is not None and num not in set(args.coaches):
                continue
            label = f"Coach-{num}"
            print(f" {label} ({sub})")
            h, c = process_coach_folder_rec(
                sub,
                coach_label=label,
                dry_run=bool(args.dry_run),
            )
            total_hours += h
            total_chunks += c

    print(f"\nDone. Hour files written: {total_hours} (chunks counted: {total_chunks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
