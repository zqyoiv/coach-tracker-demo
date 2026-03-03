"""
Run person tracking on a video file and print a report of time on screen per person.
Usage: python video-tracker.py <video_path>

Speed: uses GPU + FP16 if available; reduce IMG_SIZE or use yolo11n.pt for faster runs.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from person_id_cache import PersonFeatureCache, extract_feature

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

def _env_bool(key: str, default: bool = True) -> bool:
    v = os.environ.get(key, str(default)).strip().lower()
    return v in ("1", "true", "yes")

# --- Flags / config (all at top) ---
# Boolean flags (USE_PERSON_CACHE, USE_SUPERVISION read from .env)
USE_CPU = False
USE_PERSON_CACHE = _env_bool("USE_PERSON_CACHE", True)
USE_SUPERVISION = _env_bool("USE_SUPERVISION", True)
# Paths and numbers
VIDEO_PATH = "C:/Users/vioyq/Desktop/Coach_Tracker/lighting-videos/daylight-cloudy.mp4"
IMG_SIZE = 480
TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker.yaml")
ZONE_ID = 1
CACHE_MATCH_THRESH = 0.75

if USE_SUPERVISION:
    from supervision_helpers import SupervisionZoneTracker

model = YOLO("yolo11x.pt")

def _get_device():
    if USE_CPU:
        return "cpu", False
    if not torch.cuda.is_available():
        return "cpu", False
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 12:  # sm_120 (Blackwell): stable PyTorch has no kernels; nightly does
        try:
            torch.zeros(1, device="cuda")
            return "cuda", True
        except RuntimeError as e:
            if "no kernel image" in str(e) or "not compatible" in str(e).lower():
                print(
                    "RTX 50 / Blackwell detected but this PyTorch build has no GPU kernels for it.\n"
                    "To use your GPU, install PyTorch nightly:\n"
                    "  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128\n"
                )
            return "cpu", False
    return "cuda", True

DEVICE, HALF = _get_device()

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
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

    time_on_screen = {}
    first_seen_sec = {}
    last_seen_sec = {}
    person_cache = PersonFeatureCache(match_thresh=CACHE_MATCH_THRESH) if USE_PERSON_CACHE else None
    sv_helper = None
    if USE_PERSON_CACHE:
        print("Person ID cache enabled: new tracker IDs will be matched to previously seen people.")
    if USE_SUPERVISION:
        print("Supervision enabled: zone and time-in-zone will be reported.")

    print(f"Processing: {video_path} ({total_frames} frames @ {fps:.1f} fps)")
    dev_note = " (FP16)" if HALF else " (using CPU; install PyTorch nightly for RTX 50 GPU)" if (DEVICE == "cpu" and torch.cuda.is_available()) else ""
    print(f"Device: {DEVICE}{dev_note}")
    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total_frames}")

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker=TRACKER_CFG,
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=HALF,
            verbose=False,
        )

        if USE_SUPERVISION and sv_helper is None:
            sv_helper = SupervisionZoneTracker(frame.shape)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            t_sec = (frame_idx - 1) * frame_duration
            track_id_to_resolved = {}
            for box, track_id in zip(boxes, track_ids):
                if person_cache is not None:
                    feat = extract_feature(frame, box)
                    resolved_id = person_cache.resolve(track_id, feat)
                else:
                    resolved_id = track_id
                track_id_to_resolved[track_id] = resolved_id
                time_on_screen[resolved_id] = time_on_screen.get(resolved_id, 0.0) + frame_duration
                if resolved_id not in first_seen_sec:
                    first_seen_sec[resolved_id] = t_sec
                last_seen_sec[resolved_id] = t_sec + frame_duration

            if USE_SUPERVISION and sv_helper is not None:
                sv_helper.update(frame, results[0], track_id_to_resolved=track_id_to_resolved)

    cap.release()
    t_end = time.perf_counter()
    processing_secs = t_end - t_start
    video_length_secs = total_frames / fps

    print("\n--- Report ---")
    print(f"Video length: {video_length_secs:.1f} s ({total_frames} frames)")
    print(f"Time spent processing: {processing_secs:.1f} s")
    print("Time on screen per person:")
    try:
        from mixpanel_logger import log_dwell
    except ImportError:
        log_dwell = None
    for track_id in sorted(time_on_screen.keys(), key=lambda x: int(x)):
        secs = time_on_screen[track_id]
        zone_sec = sv_helper.get_zone_time(track_id) if (USE_SUPERVISION and sv_helper is not None) else None
        if zone_sec is not None:
            print(f"  Person {track_id} on screen for {secs:.1f} s, time in zone: {zone_sec:.1f} s")
        else:
            print(f"  Person {track_id} on screen for {secs:.1f} s")
        if log_dwell:
            start_sec = first_seen_sec.get(track_id, 0.0)
            end_sec = last_seen_sec.get(track_id, start_sec + secs)
            log_dwell(int(track_id), secs, ZONE_ID, t_start + start_sec, t_start + end_sec)
    print("--- end report ---")

if __name__ == "__main__":
    main()
