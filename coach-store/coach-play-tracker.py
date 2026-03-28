"""
Unified coach tracker for coach1..coach5 cameras.

Usage:
  python coach-store/coach-play-tracker.py <video_path> [--no-viewer]

Behavior:
  - Detect camera from video filename prefix (coach1/coach-1 ... coach5/coach-5)
  - Load per-camera tracker YAML (coach1.yaml..coach5.yaml)
  - Load per-camera zone file (zone/coach1.json..zone/coach5.json).
    If zone_norm is set (4 numbers), it wins; otherwise zone_polygon_norm is used.
  - Write CSV into coach-store/CSV-state/<coachN>/ with timestamped filename
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# Add project root to path when running from coach-store/ subfolder
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.person_id_cache import PersonFeatureCache, extract_feature
from utils.utils import load_env, env_bool, env_float
from utils import onsite_video_path

load_env()

BASE_DIR = Path(__file__).resolve().parent
ZONE_DIR = BASE_DIR / "zone"
CSV_STATE_DIR = BASE_DIR / "CSV-state"
PERSON_ID_DIR = BASE_DIR / "person-ID"
CSV_HEADER = ["timestamp", "person_id", "dwell_sec", "zone_id", "camera_id", "source_video"]
# Optional local default; on cloud/batch runs a positional video_path is provided.
VIDEO_PATH = getattr(onsite_video_path, "TAPO_EYELEVEL_0", "")

USE_ENSEMBLE = False
USE_PERSON_CACHE = env_bool("USE_PERSON_CACHE", True)
USE_SUPERVISION = env_bool("USE_SUPERVISION", True)
USE_CPU = False
DRAW_MODEL_YELLOW_BOX = True
MODEL_SOURCE = "yolo11s.pt"
TRACK_CLASSES = [0, 1] if ("visdrone" in str(MODEL_SOURCE).lower()) else [0]
IMG_SIZE = 640
CONF_THRESHOLD = 0.08
IOU_THRESHOLD = 0.5
MIN_BOX_WIDTH_PX = 40
MIN_BOX_HEIGHT_PX = 40
MIN_BOX_AREA_PX = 2500
MAX_ASPECT_RATIO = 3.5
ENSEMBLE_MODEL_SOURCE = ("erbayat/yolov11s-visdrone", "yolo11s-visdrone.pt")
ENSEMBLE_CONF = 0.15
ENSEMBLE_IOU_OVERLAP = 0.4
CACHE_MATCH_THRESH = 0.95
DWELL_LEAVE_BUFFER_SEC = env_float("DWELL_LEAVE_BUFFER_SEC", 5.0)
WINDOW_NAME = "Unified Coach Play Tracker"

if USE_SUPERVISION:
    from utils.supervision_helpers import SupervisionZoneTracker
    import supervision as sv


def _detect_camera_slug(video_path: str) -> str:
    """
    Detect camera slug from filename.
    Accepts patterns like coach1-..., coach-1-..., coach_1-...
    Returns: coach1..coach5
    """
    stem = Path(video_path).stem.lower()
    m = re.search(r"coach[-_]?([1-5])", stem)
    if not m:
        raise ValueError(
            f"Could not detect camera id from video filename '{Path(video_path).name}'. "
            "Expected prefix containing coach1..coach5 (or coach-1..coach-5)."
        )
    return f"coach{m.group(1)}"


def _load_zone_config(camera_slug: str) -> dict:
    zone_path = ZONE_DIR / f"{camera_slug}.json"
    if not zone_path.exists():
        raise FileNotFoundError(f"Zone file missing: {zone_path}")
    with open(zone_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rect_raw = data.get("zone_norm")
    poly_raw = data.get("zone_polygon_norm")
    if rect_raw is not None:
        if (
            not isinstance(rect_raw, list)
            or len(rect_raw) != 4
            or not all(isinstance(v, (int, float)) for v in rect_raw)
        ):
            raise ValueError(f"Invalid zone_norm in {zone_path}")
        data["zone_norm"] = tuple(float(v) for v in rect_raw)
        data["zone_polygon_norm"] = None
    elif poly_raw is not None:
        if not isinstance(poly_raw, list) or len(poly_raw) < 3:
            raise ValueError(f"zone_polygon_norm must be a list of at least 3 [x,y] points in {zone_path}")
        pts = []
        for i, p in enumerate(poly_raw):
            if (
                not isinstance(p, (list, tuple))
                or len(p) != 2
                or not all(isinstance(v, (int, float)) for v in p)
            ):
                raise ValueError(f"Invalid zone_polygon_norm point {i} in {zone_path}")
            pts.append((float(p[0]), float(p[1])))
        data["zone_polygon_norm"] = tuple(pts)
        data["zone_norm"] = None
    else:
        raise ValueError(f"Missing zone_norm or zone_polygon_norm in {zone_path}")
    data["zone_label"] = str(data.get("zone_label", "Zone 1"))
    return data


def _timestamp_from_video_path(video_path: str) -> str:
    m = re.search(r"(\d{4})(\d{2})(\d{2})", Path(video_path).stem)
    if m:
        y, mo, d = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{mo}/{d}/{y}"
    now = datetime.now()
    return f"{now.month}/{now.day}/{now.year}"


def _video_start_datetime(video_path: str) -> Optional[datetime]:
    parts = re.findall(r"\d{14}", Path(video_path).stem)
    if not parts:
        return None
    try:
        return datetime.strptime(parts[0], "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _zone_polygons(frame_shape, zone_cfg):
    h, w = frame_shape[:2]
    zone_norm = zone_cfg.get("zone_norm")
    if zone_norm is not None:
        x1 = int(zone_norm[0] * w)
        y1 = int(zone_norm[1] * h)
        x2 = int(zone_norm[2] * w)
        y2 = int(zone_norm[3] * h)
        zone_1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    else:
        poly_norm = zone_cfg["zone_polygon_norm"]
        zone_1 = np.array(
            [[int(x * w), int(y * h)] for x, y in poly_norm],
            dtype=np.int32,
        )
    zone_2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    return zone_1, zone_2


def _box_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _is_valid_person_box(box, min_w, min_h, min_area, max_aspect):
    w = box[2] - box[0]
    h = box[3] - box[1]
    if w < min_w or h < min_h:
        return False
    if w * h < min_area:
        return False
    longer = max(w, h)
    shorter = min(w, h)
    if shorter <= 0:
        return False
    if longer / shorter > max_aspect:
        return False
    return True


def _load_model(source):
    if isinstance(source, (tuple, list)) and len(source) >= 2:
        repo_id, filename = source[0], source[1]
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=repo_id, filename=filename)
        return YOLO(path)
    if isinstance(source, str) and "/" in source and not source.endswith(".pt"):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=source, filename="best.pt")
        return YOLO(path)
    return YOLO(source)


def _get_device():
    if USE_CPU:
        return "cpu", False
    if not torch.cuda.is_available():
        return "cpu", False
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 12:
        try:
            torch.zeros(1, device="cuda")
            return "cuda", True
        except RuntimeError:
            return "cpu", False
    return "cuda", True


model = _load_model(MODEL_SOURCE)
ensemble_model = _load_model(ENSEMBLE_MODEL_SOURCE) if USE_ENSEMBLE else None
DEVICE, HALF = _get_device()


def main():
    parser = argparse.ArgumentParser(description="Unified coach person tracking with per-camera zone")
    parser.add_argument("video_path", nargs="?", default=VIDEO_PATH, help="Path to video file")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (no display window)")
    parser.add_argument(
        "--csv-output",
        default="",
        help="Optional CSV output path. If provided, rows are appended (header written once).",
    )
    args = parser.parse_args()
    show_viewer = not args.no_viewer
    video_path = args.video_path

    if not video_path:
        print("Error: video_path is required (no default sample path available in this environment).")
        sys.exit(1)
    if not Path(video_path).exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    camera_slug = _detect_camera_slug(video_path)  # coach1..coach5
    camera_id = camera_slug
    tracker_cfg = BASE_DIR / "yaml" / f"{camera_slug}.yaml"
    if not tracker_cfg.exists():
        print(f"Error: tracker yaml not found: {tracker_cfg}")
        sys.exit(1)
    zone_cfg = _load_zone_config(camera_slug)
    zone_label = zone_cfg["zone_label"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay_ms = max(1, int(1000.0 / fps))

    first_seen_sec = {}
    last_seen_sec = {}
    person_cache = PersonFeatureCache(match_thresh=CACHE_MATCH_THRESH) if USE_PERSON_CACHE else None
    sv_helper = None

    try:
        from utils.mixpanel_logger import log_dwell
    except ImportError:
        log_dwell = None

    csv_rows = []
    video_date = _timestamp_from_video_path(video_path)
    video_start_dt = _video_start_datetime(video_path)
    mmdd = video_start_dt.strftime("%m%d") if video_start_dt is not None else datetime.now().strftime("%m%d")
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_dir = CSV_STATE_DIR / camera_slug
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{mmdd}-{camera_slug}-{run_stamp}.csv"
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    person_id_state_path = PERSON_ID_DIR / camera_slug / f"{mmdd}.json"

    if person_cache is not None:
        try:
            loaded_count = person_cache.load_from_json(person_id_state_path)
            print(f"Person-ID memory: loaded {loaded_count} IDs from {person_id_state_path}")
        except Exception as e:
            print(f"Warning: could not load Person-ID memory ({person_id_state_path}): {e}")

    def _append_csv(row: dict):
        out = {k: row.get(k, "") for k in CSV_HEADER}
        csv_rows.append(out)

    viewer_msg = "Press 'q' to quit." if show_viewer else "(headless)"
    print(f"[{camera_slug}] {video_path} ({total_frames} frames @ {fps:.1f} fps). {viewer_msg}")
    print(f"Config: yaml={tracker_cfg.name} zone={ZONE_DIR / (camera_slug + '.json')}")
    print(f"Output CSV: {csv_path}")

    if show_viewer:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_idx = 0
    t_start = time.perf_counter()
    real_start_unix = time.time()
    mixpanel_sent = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1

            results = model.track(
                frame,
                persist=True,
                classes=TRACK_CLASSES,
                tracker=str(tracker_cfg),
                imgsz=IMG_SIZE,
                device=DEVICE,
                half=HALF,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )

            t_sec = (frame_idx - 1) * frame_duration
            main_boxes = []
            current_detections = []
            track_id_to_resolved = {}
            zone_display_ids = []
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
            person_ok = None
            track_ids_raw = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else None
            )
            cls_raw = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else None

            if len(boxes) > 0:
                person_ok = (
                    (cls_raw == 0) if cls_raw is not None and len(cls_raw) == len(boxes)
                    else np.ones(len(boxes), dtype=bool)
                )
                track_ids = (
                    track_ids_raw
                    if track_ids_raw is not None and len(track_ids_raw) == len(boxes)
                    else list(range(len(boxes)))
                )
                for i in range(len(boxes)):
                    if not person_ok[i]:
                        continue
                    box = boxes[i]
                    if not _is_valid_person_box(
                        box, MIN_BOX_WIDTH_PX, MIN_BOX_HEIGHT_PX, MIN_BOX_AREA_PX, MAX_ASPECT_RATIO
                    ):
                        continue
                    track_id = track_ids[i]
                    main_boxes.append(box)
                    if person_cache is not None and track_ids_raw is not None:
                        feat = extract_feature(frame, box)
                        resolved_id = person_cache.resolve(track_id, feat)
                    else:
                        resolved_id = track_id
                    track_id_to_resolved[track_id] = resolved_id
                    current_detections.append((box, track_id, resolved_id))
                    if resolved_id not in first_seen_sec:
                        first_seen_sec[resolved_id] = t_sec
                    last_seen_sec[resolved_id] = t_sec + frame_duration

            if USE_SUPERVISION and sv_helper is None and frame is not None:
                z1_poly, z2_poly = _zone_polygons(frame.shape, zone_cfg)
                purple_color = sv.Color.from_hex("#E040FB")
                sv_helper = SupervisionZoneTracker(
                    frame.shape,
                    dwell_leave_buffer_sec=DWELL_LEAVE_BUFFER_SEC,
                    zone_1_polygon=z1_poly,
                    zone_2_polygon=z2_poly,
                    zone_1_color=purple_color,
                    zone_membership_mode="center",
                    enable_heatmap=False,
                    debug_prints=False,
                )

            if USE_SUPERVISION and sv_helper is not None:
                frame, people_in_zone_1, _pz2, in_zone_1_flags, _iz2, zone_display_ids, dwell_events_ready = sv_helper.update(
                    frame, results[0], track_id_to_resolved=track_id_to_resolved, video_t_sec=t_sec
                )
                if log_dwell and dwell_events_ready:
                    for zone_id, person_id, dwell_sec, first_sec, last_sec in dwell_events_ready:
                        if log_dwell(
                            int(person_id), dwell_sec, zone_id,
                            real_start_unix + first_sec, real_start_unix + last_sec,
                            camera_id=camera_id,
                        ):
                            mixpanel_sent += 1
                cv2.putText(
                    frame, f"People in zone: {people_in_zone_1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

            if DRAW_MODEL_YELLOW_BOX and len(boxes) > 0 and person_ok is not None:
                for i in range(len(boxes)):
                    if not person_ok[i]:
                        continue
                    x1, y1, x2, y2 = map(int, boxes[i])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for box, _track_id, resolved_id in current_detections:
                duration = last_seen_sec[resolved_id] - first_seen_sec[resolved_id]
                z1 = sv_helper.get_zone_time(1, resolved_id) if (USE_SUPERVISION and sv_helper is not None) else 0.0
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{resolved_id} {duration:.1f}s Z:{z1:.1f}s"
                cv2.putText(
                    frame, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            if USE_ENSEMBLE and ensemble_model is not None:
                ens = ensemble_model.predict(
                    frame, classes=[0], imgsz=IMG_SIZE, device=DEVICE, half=HALF,
                    conf=ENSEMBLE_CONF, iou=IOU_THRESHOLD, verbose=False,
                )
                if ens and len(ens[0].boxes) > 0:
                    ens_boxes = ens[0].boxes.xyxy.cpu().numpy()
                    for box in ens_boxes:
                        if not _is_valid_person_box(
                            box, MIN_BOX_WIDTH_PX, MIN_BOX_HEIGHT_PX, MIN_BOX_AREA_PX, MAX_ASPECT_RATIO
                        ):
                            continue
                        if any(_box_iou(box, mb) >= ENSEMBLE_IOU_OVERLAP for mb in main_boxes):
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            if show_viewer:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q"):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nCtrl+C received. Saving CSV with dwell data so far...")

    cap.release()
    if show_viewer:
        cv2.destroyAllWindows()

    print("\n--- Report ---")
    print(f"Time spent processing: {time.perf_counter() - t_start:.1f}s")
    print("Time on screen per person:")

    if USE_SUPERVISION and sv_helper is not None:
        ids_in_zone = {tid for tid in first_seen_sec if sv_helper.get_zone_time(1, tid) > 0}
    else:
        ids_in_zone = set()

    for track_id in sorted(first_seen_sec.keys(), key=lambda x: int(x)):
        start_sec = first_seen_sec[track_id]
        end_sec = last_seen_sec.get(track_id, start_sec)
        secs = end_sec - start_sec
        dwell_z1 = sv_helper.get_zone_time(1, track_id) if (USE_SUPERVISION and sv_helper is not None) else 0
        print(f"  Person {track_id} on screen for {secs:.1f}s | zone={dwell_z1:.1f}s")
        if track_id not in ids_in_zone:
            continue

        if USE_SUPERVISION and sv_helper is not None and video_start_dt is not None:
            first_zone_sec, _last_zone_sec = sv_helper.get_zone_first_last(1, track_id)
            person_dt = video_start_dt + timedelta(seconds=float(first_zone_sec))
            person_ts = f"{person_dt.month}/{person_dt.day}/{person_dt.year} {person_dt.strftime('%H:%M')}"
        else:
            person_ts = video_date

        _append_csv({
            "timestamp": person_ts,
            "person_id": int(track_id),
            "dwell_sec": round(dwell_z1, 2),
            "zone_id": zone_label,
            "camera_id": camera_id,
            "source_video": Path(video_path).name,
        })

    if USE_SUPERVISION and sv_helper is not None:
        flush_events = sv_helper.get_dwell_events_to_flush()
        for zone_id, person_id, dwell_sec, first_sec, last_sec in flush_events:
            if zone_id != 1 or dwell_sec <= 0:
                continue
            if log_dwell:
                log_dwell(
                    int(person_id), dwell_sec, zone_id,
                    real_start_unix + first_sec, real_start_unix + last_sec,
                    camera_id=camera_id,
                )

    if person_cache is not None:
        try:
            person_cache.save_to_json(person_id_state_path)
            print(f"Person-ID memory saved: {person_id_state_path}")
        except Exception as e:
            print(f"Warning: could not save Person-ID memory ({person_id_state_path}): {e}")

    append_mode = bool(args.csv_output)
    write_header = True
    if append_mode and csv_path.exists():
        try:
            write_header = csv_path.stat().st_size == 0
        except OSError:
            write_header = True

    with open(csv_path, "a" if append_mode else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows(csv_rows)
    print(f"CSV written: {csv_path}")
    print("--- end report ---")


if __name__ == "__main__":
    main()

