"""
Dashboard: read-only analytics on CSV data.

Never delete, move, or overwrite user CSV files on disk.

Clear dashboard: clears charts and collapses the CSV picker (session only).
Show CSV list: expands it again. Short labels in the list; full path on hover.
Never deletes files on disk.
"""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import IO, Any, Iterable, Optional

from flask import Flask, render_template, request, session


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-dashboard-session-not-for-production")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


BIN_DEFS = [
    (0.0, 10.0, "< 10"),
    (10.0, 20.0, "10 - 20"),
    (20.0, 30.0, "20 - 30"),
    (30.0, 60.0, "30 - 60"),
    (60.0, None, ">= 60"),
]

BIN_LABELS = [b[2] for b in BIN_DEFS]

CAMERA_COLORS = ["#8B5CF6", "#22C55E", "#60A5FA", "#F59E0B", "#EF4444", "#14B8A6", "#A3E635"]
ZONE_COLORS = ["#8B5CF6", "#22C55E", "#60A5FA", "#F59E0B", "#EF4444", "#14B8A6", "#A3E635"]


def _parse_timestamp(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    ts = ts.strip()
    # Expected formats from our CSV: "M/D/YYYY HH:MM" or "M/D/YYYY"
    fmts = [
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            pass

    # Last resort: try "3/13/2026" without time
    # (some files may store leading zeros inconsistently)
    try:
        if len(ts.split(" ")) == 1:
            mo, d, y = ts.split("/")
            return datetime(int(y), int(mo), int(d))
    except Exception:
        pass

    return None


def _dwell_bin_label(dwell_sec: float) -> str:
    for low, high, label in BIN_DEFS:
        if high is None:
            if dwell_sec >= low:
                return label
        else:
            if low <= dwell_sec < high:
                return label
    return BIN_LABELS[0]


def _iter_events_from_csv_stream(stream: IO[str]) -> Iterable[dict[str, Any]]:
    reader = csv.DictReader(stream)
    for row in reader:
        ts = row.get("timestamp", "")
        dt = _parse_timestamp(ts)
        if dt is None:
            continue

        dwell_raw = row.get("dwell_sec", "").strip()
        try:
            dwell_sec = float(dwell_raw)
        except Exception:
            continue

        person_id_raw = row.get("person_id", "").strip()
        try:
            person_id = int(person_id_raw) if person_id_raw != "" else None
        except Exception:
            person_id = None

        zone_id = row.get("zone_id", "").strip() or "unknown_zone"
        camera_id = row.get("camera_id", "").strip() or "unknown_camera"

        yield {
            "dt": dt,
            "dwell_sec": dwell_sec,
            "person_id": person_id,
            "zone_id": zone_id,
            "camera_id": camera_id,
        }


def _events_from_any_csv(csv_source: str | Path | IO[bytes] | IO[str]) -> list[dict[str, Any]]:
    if hasattr(csv_source, "read"):
        # file-like
        if isinstance(csv_source, io.BufferedIOBase) or "b" in getattr(csv_source, "mode", ""):
            text_stream: IO[str] = io.TextIOWrapper(csv_source, encoding="utf-8", errors="ignore")
            return list(_iter_events_from_csv_stream(text_stream))
        return list(_iter_events_from_csv_stream(csv_source))  # type: ignore[arg-type]

    # path-like
    p = Path(csv_source)
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(_iter_events_from_csv_stream(f))


def _format_day_label(d: date) -> str:
    # Avoid %-d issues on Windows.
    return d.strftime("%b %d")


@dataclass
class DashboardData:
    total_events: int
    dwell_bins: list[str]
    dwell_all_counts: list[int]
    dwell_by_zone: dict[str, list[int]]
    daily_labels: list[str]
    daily_counts: list[int]
    camera_labels: list[str]
    camera_counts: list[int]
    avg_dwell_camera_labels: list[str]
    avg_dwell_camera_values: list[float]


def _compute_dashboard(events: list[dict[str, Any]]) -> DashboardData:
    total_events = len(events)

    # Dwell bins
    dwell_all_counts = [0] * len(BIN_LABELS)
    dwell_by_zone: dict[str, list[int]] = {}
    for e in events:
        label = _dwell_bin_label(float(e["dwell_sec"]))
        idx = BIN_LABELS.index(label)
        dwell_all_counts[idx] += 1

        z = str(e["zone_id"])
        if z not in dwell_by_zone:
            dwell_by_zone[z] = [0] * len(BIN_LABELS)
        dwell_by_zone[z][idx] += 1

    # Daily distribution: last 7 days relative to max event date
    if events:
        last_day = max(e["dt"].date() for e in events)
    else:
        last_day = datetime.now().date()

    start_day = last_day - timedelta(days=6)
    days = [start_day + timedelta(days=i) for i in range(7)]
    daily_labels = [_format_day_label(d) for d in days]
    daily_counts = [0] * len(days)

    day_to_idx = {days[i]: i for i in range(len(days))}
    for e in events:
        d = e["dt"].date()
        if d in day_to_idx:
            daily_counts[day_to_idx[d]] += 1

    # Camera ID distribution
    cam_counts: dict[str, int] = {}
    cam_sum_dwell: dict[str, float] = {}
    for e in events:
        cam = str(e["camera_id"])
        cam_counts[cam] = cam_counts.get(cam, 0) + 1
        cam_sum_dwell[cam] = cam_sum_dwell.get(cam, 0.0) + float(e["dwell_sec"])
    # Sort by count desc, then label
    cam_items = sorted(cam_counts.items(), key=lambda x: (-x[1], x[0]))[:12]
    camera_labels = [k for k, _v in cam_items]
    camera_counts = [v for _k, v in cam_items]

    # Average dwell time per camera (seconds), sorted by avg desc
    cam_avg_items = []
    for cam, cnt in cam_counts.items():
        avg = (cam_sum_dwell.get(cam, 0.0) / cnt) if cnt > 0 else 0.0
        cam_avg_items.append((cam, avg))
    cam_avg_items = sorted(cam_avg_items, key=lambda x: (-x[1], x[0]))[:12]
    avg_dwell_camera_labels = [k for k, _v in cam_avg_items]
    avg_dwell_camera_values = [round(v, 2) for _k, v in cam_avg_items]

    return DashboardData(
        total_events=total_events,
        dwell_bins=BIN_LABELS,
        dwell_all_counts=dwell_all_counts,
        dwell_by_zone=dwell_by_zone,
        daily_labels=daily_labels,
        daily_counts=daily_counts,
        camera_labels=camera_labels,
        camera_counts=camera_counts,
        avg_dwell_camera_labels=avg_dwell_camera_labels,
        avg_dwell_camera_values=avg_dwell_camera_values,
    )


def _list_available_csvs() -> list[str]:
    """CSV paths relative to project root (for the picker). Does not delete or move files."""
    out: set[str] = set()
    for pattern in (
        "coach-*/**/*.csv",
        "coach-store/CSV-state/**/*.csv",
        "dashboard/csv/**/*.csv",
    ):
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_file() and p.suffix.lower() == ".csv":
                out.add(str(p.relative_to(PROJECT_ROOT)))
    return sorted(out)


def _short_csv_label(rel: str) -> str:
    """Short text for the dropdown; full path stays in <option title>."""
    parts = rel.replace("\\", "/").split("/")
    name = parts[-1] if parts else rel
    try:
        if "CSV-state" in parts:
            i = parts.index("CSV-state")
            cam = parts[i + 1] if i + 1 < len(parts) else ""
            return f"{cam} · {name}"
    except (ValueError, IndexError):
        pass
    if len(parts) >= 3 and parts[0] == "dashboard" and parts[1] == "csv":
        return f"{parts[2]} · {name}"
    if len(parts) >= 2:
        return f"{parts[0]} · {name}"
    return name


def _csv_rows_for_template(paths: list[str]) -> list[dict[str, str]]:
    return [{"rel": r, "label": _short_csv_label(r)} for r in paths]


def _parse_md_date(text: str) -> tuple[int, int] | None:
    """Parse '3-13' or '03-13' -> (month, day)."""
    text = text.strip()
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})$", text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_coach_num(camera: str) -> int | None:
    m = re.search(r"coach[-_]?([1-5])", camera.strip().lower())
    return int(m.group(1)) if m else None


def _normalize_coach_select(camera: str) -> str:
    m = re.search(r"coach[-_]?([1-5])", camera.strip().lower())
    return f"coach-{m.group(1)}" if m else "coach-1"


def _normalize_camera_key(camera_id: str) -> str:
    m = re.search(r"coach[-_]?([1-5])", str(camera_id).lower())
    return f"coach{m.group(1)}" if m else str(camera_id).strip().lower()


def _load_events_from_dashboard_csv_for_coach(
    coach_num: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """All dashboard/csv/**/*.csv whose filename identifies this coach (e.g. *-coach3-*)."""
    root = PROJECT_ROOT / "dashboard" / "csv"
    if not root.is_dir():
        return [], []
    out: list[dict[str, Any]] = []
    scanned: list[str] = []
    for p in sorted(root.rglob("*.csv")):
        if not p.is_file():
            continue
        n = _parse_coach_num(p.name)
        if n != coach_num:
            continue
        rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        scanned.append(rel)
        for e in _events_from_any_csv(p):
            e["source_csv"] = rel
            out.append(e)
    return out, scanned


def _filter_events_date_camera(
    events: list[dict[str, Any]],
    month: int,
    day: int,
    coach_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        dt = e["dt"]
        if dt.month != month or dt.day != day:
            continue
        if _normalize_camera_key(str(e.get("camera_id", ""))) != coach_key:
            continue
        out.append(e)
    return out


def _split_am_pm(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    am = [e for e in events if e["dt"].hour < 12]
    pm = [e for e in events if e["dt"].hour >= 12]
    return am, pm


def _metrics_window(events: list[dict[str, Any]]) -> dict[str, float]:
    if not events:
        return {"avg_dwell": 0.0, "total_dwell_sec": 0.0, "unique_customers": 0.0}
    dwells = [float(e["dwell_sec"]) for e in events]
    total = sum(dwells)
    avg = total / len(events)
    persons = {e["person_id"] for e in events if e.get("person_id") is not None}
    return {
        "avg_dwell": round(avg, 2),
        "total_dwell_sec": round(total, 2),
        "unique_customers": float(len(persons)),
    }


def _zone_label_from_events(events: list[dict[str, Any]]) -> str:
    if not events:
        return "Zone"
    # most common zone_id
    counts: dict[str, int] = {}
    for e in events:
        z = str(e.get("zone_id") or "").strip() or "unknown"
        counts[z] = counts.get(z, 0) + 1
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best


@app.get("/compare")
def compare_get():
    d1 = request.args.get("d1", "3-13").strip()
    d2 = request.args.get("d2", "3-20").strip()

    err: str | None = None
    pd1 = _parse_md_date(d1)
    pd2 = _parse_md_date(d2)

    compare_payload: dict[str, Any] = {
        "primary_label": d1,
        "baseline_label": d2,
        "by_camera": {},
        "sources_unique_csvs": [],
        "sources_unique_csv_count": 0,
    }

    if not pd1 or not pd2:
        err = "Invalid date format. Use M-D (e.g. 3-13)."
    else:
        primary_month, primary_day = pd1
        baseline_month, baseline_day = pd2

        def pack(m_primary: dict[str, float], m_baseline: dict[str, float]) -> dict[str, Any]:
            return {
                "primary": {
                    "avg_dwell": m_primary["avg_dwell"],
                    "total_dwell_hr": round(m_primary["total_dwell_sec"] / 3600.0, 2),
                    "customers": int(m_primary["unique_customers"]),
                },
                "baseline": {
                    "avg_dwell": m_baseline["avg_dwell"],
                    "total_dwell_hr": round(m_baseline["total_dwell_sec"] / 3600.0, 2),
                    "customers": int(m_baseline["unique_customers"]),
                },
            }

        sources_set: set[str] = set()
        for coach_num in range(1, 6):
            coach_key = f"coach{coach_num}"
            all_events, scanned_csvs = _load_events_from_dashboard_csv_for_coach(coach_num)

            ev_primary = _filter_events_date_camera(all_events, primary_month, primary_day, coach_key)
            ev_baseline = _filter_events_date_camera(
                all_events, baseline_month, baseline_day, coach_key
            )

            ev_primary_am, ev_primary_pm = _split_am_pm(ev_primary)
            ev_baseline_am, ev_baseline_pm = _split_am_pm(ev_baseline)

            primary_csvs = sorted({str(e.get("source_csv")) for e in ev_primary if e.get("source_csv")})
            baseline_csvs = sorted({str(e.get("source_csv")) for e in ev_baseline if e.get("source_csv")})

            compare_payload["by_camera"][coach_num] = {
                "label": f"coach-{coach_num}",
                "zone_title": _zone_label_from_events(ev_primary + ev_baseline),
                "primary_events": len(ev_primary),
                "baseline_events": len(ev_baseline),
                "scanned_csvs": scanned_csvs,
                "primary_csvs": primary_csvs,
                "baseline_csvs": baseline_csvs,
                "am": pack(_metrics_window(ev_primary_am), _metrics_window(ev_baseline_am)),
                "pm": pack(_metrics_window(ev_primary_pm), _metrics_window(ev_baseline_pm)),
            }

            sources_set.update(scanned_csvs)

        compare_payload["sources_unique_csvs"] = sorted(sources_set)
        compare_payload["sources_unique_csv_count"] = len(sources_set)

    return render_template(
        "compare.html",
        active_tab="compare",
        d1=d1,
        d2=d2,
        err=err,
        compare=compare_payload,
    )


@app.get("/")
def index_get():
    csv_files = _list_available_csvs()
    hide_csv_list = bool(session.get("hide_csv_list", False))

    if request.args.get("show_csv_list"):
        session["hide_csv_list"] = False
        hide_csv_list = False

    # Support multi-select: ?files=a.csv&files=b.csv
    selected_files = request.args.getlist("files")
    if not selected_files:
        single = request.args.get("file")
        if single:
            selected_files = [single]

    # ?clear=1 — clear charts, clear selection, collapse long file list (session only; no disk delete)
    if request.args.get("clear"):
        session["hide_csv_list"] = True
        hide_csv_list = True
        selected_files = []
    elif not selected_files and csv_files and not hide_csv_list:
        selected_files = [csv_files[0]]

    # When list is collapsed, do not render long paths in the <select>
    csv_rows = [] if hide_csv_list else _csv_rows_for_template(csv_files)

    events: list[dict[str, Any]] = []
    for rel in selected_files:
        selected_path = (PROJECT_ROOT / rel).resolve()
        if selected_path.exists():
            events.extend(_events_from_any_csv(selected_path))

    data = _compute_dashboard(events)
    return render_template(
        "index.html",
        active_tab="overview",
        csv_rows=csv_rows,
        hide_csv_list=hide_csv_list,
        selected_files=selected_files,
        data=data.__dict__,
    )


@app.post("/upload")
def upload_post():
    files = request.files.getlist("csv_upload")
    if not files:
        return index_get()
    events: list[dict[str, Any]] = []
    any_loaded = False
    for f in files:
        if not f or not f.filename:
            continue
        any_loaded = True
        events.extend(_events_from_any_csv(f.stream))

    if not any_loaded:
        return index_get()

    session["hide_csv_list"] = False
    data = _compute_dashboard(events)
    all_csv = _list_available_csvs()
    csv_rows = _csv_rows_for_template(all_csv)
    return render_template(
        "index.html",
        active_tab="overview",
        csv_rows=csv_rows,
        hide_csv_list=False,
        selected_files=[],
        data=data.__dict__,
    )


if __name__ == "__main__":
    # Run from: python app.py
    app.run(host="127.0.0.1", port=5055, debug=True)

