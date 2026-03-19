"""
Coach-1 person tracking: purple/pink zone over handbag shelving area.
Usage: python coach-1-play-tracker.py [video_path] [--no-viewer]

  --no-viewer   Run headless (no display window, faster for batch processing)

Press 'q' to quit early when viewer is on. Speed: uses GPU + FP16 if available.

Run from project root: python coach-1/coach-1-play-tracker.py <video_path>
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path when running from coach-1/ subfolder
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from utils.person_id_cache import PersonFeatureCache, extract_feature
from utils.utils import load_env, env_bool, env_float
from utils import onsite_video_path

load_env()

# Paths and numbers — tracker config and CSV log in same folder as this script
_COACH1_DIR = Path(__file__).resolve().parent
TRACKER_CFG = str(_COACH1_DIR / "coach-1.yaml")
COACH1_CSV = _COACH1_DIR / "coach-1.csv"
CSV_HEADER = [
    "timestamp", "video_path", "event_type", "person_id", "zone_id",
    "dwell_sec", "time_on_screen_sec", "start_sec", "end_sec",
    "total_frames", "fps", "processing_sec", "camera_id",
]
VIDEO_PATH = onsite_video_path.TAPO_EYELEVEL_0
# --- Flags / config (all at top) ---
USE_ENSEMBLE = False
USE_PERSON_CACHE = env_bool("USE_PERSON_CACHE", True)
USE_SUPERVISION = env_bool("USE_SUPERVISION", True)
USE_CPU = False
DRAW_MODEL_YELLOW_BOX = True
MODEL_SOURCE = "yolo11s.pt"
TRACK_CLASSES = [0, 1] if ("visdrone" in (str(MODEL_SOURCE[0]) if isinstance(MODEL_SOURCE, (tuple, list)) else str(MODEL_SOURCE)).lower()) else [0]
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
CAMERA_ID = "coach-1"
DWELL_LEAVE_BUFFER_SEC = env_float("DWELL_LEAVE_BUFFER_SEC", 5.0)
WINDOW_NAME = "Coach-1 Play Tracker"

# Purple/pink zone: normalized (0.24, 0) top-left to (0.68, 0.61) bottom-right
# Covers handbag shelving area (center-left of frame)
ZONE_NORM = (0.24, 0.00, 0.68, 0.61)  # x1, y1, x2, y2

if USE_SUPERVISION:
    from utils.supervision_helpers import SupervisionZoneTracker
    import supervision as sv


def _log_to_csv(row: dict):
    """Append a log row to coach-1.csv. Creates file with header if new."""
    write_header = not COACH1_CSV.exists()
    with open(COACH1_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_HEADER})


def _coach1_zone_polygons(frame_shape):
    """Build zone 1 (purple rectangle) and zone 2 (empty) from frame shape."""
    h, w = frame_shape[:2]
    x1 = int(ZONE_NORM[0] * w)
    y1 = int(ZONE_NORM[1] * h)
    x2 = int(ZONE_NORM[2] * w)
    y2 = int(ZONE_NORM[3] * h)
    zone_1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    # Zone 2: tiny polygon in corner (unused; supervision rejects negative coords)
    zone_2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    return zone_1, zone_2


def _box_iou(box_a, box_b):
    """IoU of two boxes, each (x1, y1, x2, y2)."""
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
    """Filter out small and thin (pole/stand) false positives. box = (x1,y1,x2,y2)."""
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
        try:
            from huggingface_hub import hf_hub_download
            print(f"Downloading/loading VisDrone model from Hugging Face: {repo_id} ({filename}) ...")
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"Using model at: {path}")
            return YOLO(path)
        except Exception as e:
            raise FileNotFoundError(
                f"Hugging Face model '{repo_id}' ({filename}) could not be loaded. Error: {e}"
            ) from e
    if isinstance(source, str) and "/" in source and not source.endswith(".pt"):
        try:
            from huggingface_hub import hf_hub_download
            print(f"Downloading/loading model from Hugging Face: {source} (best.pt) ...")
            path = hf_hub_download(repo_id=source, filename="best.pt")
            print(f"Using model at: {path}")
            return YOLO(path)
        except Exception as e:
            raise FileNotFoundError(
                f"Hugging Face model '{source}' could not be loaded. Error: {e}"
            ) from e
    print(f"Using local/Ultralytics model: {source}")
    return YOLO(source)


model = _load_model(MODEL_SOURCE)
ensemble_model = _load_model(ENSEMBLE_MODEL_SOURCE) if USE_ENSEMBLE else None
if USE_ENSEMBLE:
    print(f"Ensemble enabled: second model {ENSEMBLE_MODEL_SOURCE} (yellow boxes when main misses)")
if USE_SUPERVISION:
    print("Supervision enabled: Coach-1 purple zone (handbag shelving area), heatmap.")


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
        except RuntimeError as e:
            if "no kernel image" in str(e) or "not compatible" in str(e).lower():
                print(
                    "RTX 50 / Blackwell detected but this PyTorch build has no GPU kernels for it."
                )
            return "cpu", False
    return "cuda", True


DEVICE, HALF = _get_device()


def main():
    parser = argparse.ArgumentParser(description="Coach-1 person tracking with purple zone")
    parser.add_argument("video_path", nargs="?", default=VIDEO_PATH, help="Path to video file")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless (no display window)")
    args = parser.parse_args()
    show_viewer = not args.no_viewer
    video_path = args.video_path

    if not Path(video_path).exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

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
    if USE_PERSON_CACHE:
        print("Person feature cache enabled: new tracker IDs will be matched to previously seen people.")

    try:
        from utils.mixpanel_logger import log_dwell
    except ImportError:
        log_dwell = None

    viewer_msg = "Press 'q' to quit." if show_viewer else "(headless)"
    print(f"Playing with tracking: {video_path} ({total_frames} frames @ {fps:.1f} fps). {viewer_msg}")
    _log_to_csv({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_path": video_path,
        "event_type": "run_start",
        "total_frames": total_frames,
        "fps": fps,
        "camera_id": CAMERA_ID,
    })
    if show_viewer:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_idx = 0
    t_start = time.perf_counter()
    real_start_unix = time.time()
    mixpanel_sent = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            classes=TRACK_CLASSES,
            tracker=TRACKER_CFG,
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
        people_in_zone = 0
        in_zone_flags = []
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
            z1_poly, z2_poly = _coach1_zone_polygons(frame.shape)
            purple_color = sv.Color.from_hex("#E040FB")  # purple/pink
            sv_helper = SupervisionZoneTracker(
                frame.shape,
                dwell_leave_buffer_sec=DWELL_LEAVE_BUFFER_SEC,
                zone_1_polygon=z1_poly,
                zone_2_polygon=z2_poly,
                zone_1_color=purple_color,
            )
        if USE_SUPERVISION and sv_helper is not None:
            frame, people_in_zone_1, people_in_zone_2, in_zone_1_flags, in_zone_2_flags, zone_display_ids, dwell_events_ready = sv_helper.update(
                frame, results[0], track_id_to_resolved=track_id_to_resolved, video_t_sec=t_sec
            )
            if log_dwell and dwell_events_ready:
                for zone_id, person_id, dwell_sec, first_sec, last_sec in dwell_events_ready:
                    if log_dwell(
                        int(person_id), dwell_sec, zone_id,
                        real_start_unix + first_sec, real_start_unix + last_sec,
                        camera_id=CAMERA_ID,
                    ):
                        mixpanel_sent += 1
            if dwell_events_ready:
                for zone_id, person_id, dwell_sec, first_sec, last_sec in dwell_events_ready:
                    _log_to_csv({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "video_path": video_path,
                        "event_type": "dwell",
                        "person_id": int(person_id),
                        "zone_id": zone_id,
                        "dwell_sec": round(dwell_sec, 2),
                        "start_sec": round(first_sec, 2),
                        "end_sec": round(last_sec, 2),
                        "camera_id": CAMERA_ID,
                    })
            cv2.putText(
                frame,
                f"People in zone: {people_in_zone_1}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if zone_display_ids:
                y_off = 55
                for i, rid in enumerate(zone_display_ids):
                    if rid is None:
                        continue
                    in_z1 = in_zone_1_flags[i] if i < len(in_zone_1_flags) else False
                    if not in_z1:
                        continue
                    t_z1 = sv_helper.get_zone_time(1, rid)
                    cv2.putText(
                        frame,
                        f"ID:{int(rid)} Z:{t_z1:.1f}s",
                        (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 200),
                        1,
                    )
                    y_off += 22

        if DRAW_MODEL_YELLOW_BOX and len(boxes) > 0 and person_ok is not None:
            for i in range(len(boxes)):
                if not person_ok[i]:
                    continue
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "model",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
        for box, track_id, resolved_id in current_detections:
            duration = last_seen_sec[resolved_id] - first_seen_sec[resolved_id]
            z1 = sv_helper.get_zone_time(1, resolved_id) if (USE_SUPERVISION and sv_helper is not None) else 0.0
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{resolved_id} {duration:.1f}s"
            if USE_SUPERVISION and sv_helper is not None:
                label += f" Z:{z1:.1f}s"
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if USE_ENSEMBLE and ensemble_model is not None:
            ens = ensemble_model.predict(
                frame,
                classes=[0],
                imgsz=IMG_SIZE,
                device=DEVICE,
                half=HALF,
                conf=ENSEMBLE_CONF,
                iou=IOU_THRESHOLD,
                verbose=False,
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
                    cv2.putText(
                        frame,
                        "det",
                        (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

        fc_y = 55
        if USE_SUPERVISION and sv_helper is not None and zone_display_ids:
            n_lines = sum(1 for i in range(len(zone_display_ids)) if zone_display_ids[i] is not None and (in_zone_1_flags[i] if i < len(in_zone_1_flags) else False))
            fc_y = 55 + 22 * n_lines if n_lines else 55
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames}",
            (10, fc_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        if show_viewer:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                print("Quit by user.")
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
        else:
            # Headless: minimal wait to allow interrupt
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit by user (q).")
                break

    cap.release()
    if show_viewer:
        cv2.destroyAllWindows()

    t_end = time.perf_counter()
    processing_secs = t_end - t_start
    video_length_secs = total_frames / fps if total_frames else 0

    print("\n--- Report ---")
    print(f"Video length: {video_length_secs:.1f} s ({total_frames} frames)")
    print(f"Time spent processing: {processing_secs:.1f} s")
    print("Time on screen per person:")
    try:
        from utils.mixpanel_logger import log_dwell, _is_send_enabled, _get_token
    except ImportError:
        log_dwell = None
        _is_send_enabled = lambda: False
        _get_token = lambda: ""
    send_ok = _is_send_enabled()
    has_token = bool(_get_token())
    print(f"Mixpanel: SEND_TO_MIXPANEL={os.environ.get('SEND_TO_MIXPANEL', '(not set)')} -> enabled={send_ok}, has_token={has_token}")
    for track_id in sorted(first_seen_sec.keys(), key=lambda x: int(x)):
        start_sec = first_seen_sec[track_id]
        end_sec = last_seen_sec.get(track_id, start_sec)
        secs = end_sec - start_sec
        dwell_z1 = sv_helper.get_zone_time(1, track_id) if (USE_SUPERVISION and sv_helper is not None) else 0
        print(f"  Person {track_id} on screen for {secs:.1f} s (first→last) | zone={dwell_z1:.1f}s")
        _log_to_csv({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "video_path": video_path,
            "event_type": "person_summary",
            "person_id": int(track_id),
            "zone_id": 1,
            "dwell_sec": round(dwell_z1, 2),
            "time_on_screen_sec": round(secs, 2),
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "total_frames": total_frames,
            "fps": fps,
            "processing_sec": round(processing_secs, 2),
            "camera_id": CAMERA_ID,
        })
    if USE_SUPERVISION and sv_helper is not None:
        flush_events = sv_helper.get_dwell_events_to_flush()
        for zone_id, person_id, dwell_sec, first_sec, last_sec in flush_events:
            if log_dwell and log_dwell(int(person_id), dwell_sec, zone_id, real_start_unix + first_sec, real_start_unix + last_sec, camera_id=CAMERA_ID):
                mixpanel_sent += 1
            _log_to_csv({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "video_path": video_path,
                "event_type": "dwell",
                "person_id": int(person_id),
                "zone_id": zone_id,
                "dwell_sec": round(dwell_sec, 2),
                "start_sec": round(first_sec, 2),
                "end_sec": round(last_sec, 2),
                "camera_id": CAMERA_ID,
            })
    if log_dwell is not None:
        if mixpanel_sent == 0:
            reasons = []
            if not _is_send_enabled():
                reasons.append("SEND_TO_MIXPANEL is false/off in .env")
            elif not _get_token():
                reasons.append("MIXPANEL_TOKEN missing or not loaded from .env")
            elif not USE_SUPERVISION or sv_helper is None:
                reasons.append("Supervision was off or no frames processed")
            else:
                reasons.append("no person had dwell time in zone (check zone position)")
            print(f"Mixpanel: no events sent — {'; '.join(reasons)}")
        else:
            print(f"Mixpanel: sent {mixpanel_sent} dwell event(s)")
    _log_to_csv({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "video_path": video_path,
        "event_type": "run_complete",
        "total_frames": total_frames,
        "fps": fps,
        "processing_sec": round(processing_secs, 2),
        "camera_id": CAMERA_ID,
    })
    print("--- end report ---")


if __name__ == "__main__":
    main()
