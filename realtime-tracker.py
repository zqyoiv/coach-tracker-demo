import os
import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
from utils.supervision_helpers import SupervisionZoneTracker
from utils.person_id_cache import PersonFeatureCache, extract_feature

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
USE_SUPERVISION = _env_bool("USE_SUPERVISION", True)
USE_REALSENSE = False
USE_PERSON_CACHE = _env_bool("USE_PERSON_CACHE", True)
# Paths and numbers
MODEL_SOURCE = "yolo11x.pt"
TRACKER_CFG = str(Path(__file__).resolve().parent / "yaml" / "basic-tracker-config.yaml")
ZONE_ID = 1
CAMERA_INDEX = 0
CACHE_MATCH_THRESH = 0.75

def _load_model(source: str):
    if "/" in source and not source.endswith(".pt"):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=source, filename="best.pt")
            return YOLO(path)
        except Exception as e:
            raise FileNotFoundError(
                f"Hugging Face model '{source}' could not be loaded. Install: pip install huggingface_hub. Error: {e}"
            ) from e
    return YOLO(source)


model = _load_model(MODEL_SOURCE)

if USE_REALSENSE:
    import pyrealsense2 as rs
    import numpy as np
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(cfg)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# Create a named window that can be resized
cv2.namedWindow("Store Monitor", cv2.WINDOW_NORMAL) 

# Set the window size (Width, Height)
cv2.resizeWindow("Store Monitor", 1920, 1080)

# First-seen time per track ID; not cleared when person leaves so timer continues if same ID returns within track_buffer
start_times = {}
# Track IDs from previous frame, used to detect who "left" the frame
previous_track_ids = set()
# IDs that have left and not yet "returned"; used to log "returned" when they reappear
ids_who_left = set()

# Supervision helper (created lazily on first frame so it can see frame size)
sv_helper = None
person_cache = PersonFeatureCache(match_thresh=CACHE_MATCH_THRESH) if USE_PERSON_CACHE else None
if USE_PERSON_CACHE:
    print("Person ID cache enabled: new tracker IDs will be matched to previously seen people.")

while True:
    if USE_REALSENSE:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asarray(color_frame.get_data())
            success = True
        except Exception:
            break
    else:
        success, frame = cap.read()
        if not success:
            break

    # Mirror the frame horizontally (mirror/selfie view)
    frame = cv2.flip(frame, 1)

    if USE_SUPERVISION and sv_helper is None:
        sv_helper = SupervisionZoneTracker(frame.shape)

    # 3. Run tracking (persist=True keeps same ID across frames; classes=[0] = person only)
    results = model.track(
        frame, 
        persist=True,
        classes=[0], 
        tracker=TRACKER_CFG,
        verbose=False)

    # Resolve tracker IDs to canonical IDs via person cache (one pass per detection)
    current_detections = []  # list of (box, track_id, resolved_id)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            if person_cache is not None:
                feat = extract_feature(frame, box)
                resolved_id = person_cache.resolve(track_id, feat)
            else:
                resolved_id = track_id
            current_detections.append((box, track_id, resolved_id))
    current_track_ids = {r for (_, _, r) in current_detections}
    # Detect who "left" this frame (was in previous frame, not in current) and log
    left_ids = previous_track_ids - current_track_ids
    for resolved_id in left_ids:
        start_t = start_times.get(resolved_id, time.time())
        duration = time.time() - start_t
        print(f"Person ID:{resolved_id} left (was on screen for {duration:.1f}s)")
        ids_who_left.add(resolved_id)
        try:
            from utils.mixpanel_logger import log_dwell
            log_dwell(int(resolved_id), duration, ZONE_ID, start_t, time.time())
        except ImportError:
            pass
    # Who "returned" this frame = was in ids_who_left and is now visible again (same ID, timer continues)
    returned_ids = ids_who_left & current_track_ids
    for resolved_id in returned_ids:
        print(f"Person ID:{resolved_id} returned (timer continues)")
        ids_who_left.discard(resolved_id)
    previous_track_ids = current_track_ids

    # Supervision: update zone + time-in-zone + heatmap (use resolved IDs so same person = one ID)
    people_in_zone = 0
    in_zone_flags = []
    zone_display_ids = []
    if USE_SUPERVISION and sv_helper is not None:
        track_id_to_resolved = (
            {tid: rid for (_, tid, rid) in current_detections} if person_cache else None
        )
        frame, people_in_zone_1, people_in_zone_2, in_zone_1_flags, in_zone_2_flags, zone_display_ids = sv_helper.update(
            frame, results[0], track_id_to_resolved=track_id_to_resolved
        )
        cv2.putText(
            frame,
            f"People in zone 1: {people_in_zone_1}  zone 2: {people_in_zone_2}",
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
                in_z2 = in_zone_2_flags[i] if i < len(in_zone_2_flags) else False
                if not in_z1 and not in_z2:
                    continue
                t_z1 = sv_helper.get_zone_time(1, rid)
                t_z2 = sv_helper.get_zone_time(2, rid)
                cv2.putText(
                    frame,
                    f"ID:{int(rid)} Z1:{t_z1:.1f}s Z2:{t_z2:.1f}s",
                    (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 200),
                    1,
                )
                y_off += 22

    # Draw results if any detections
    for box, track_id, resolved_id in current_detections:
        if resolved_id not in start_times:
            start_times[resolved_id] = time.time()

        duration = time.time() - start_times[resolved_id]
        z1 = sv_helper.get_zone_time(1, resolved_id) if sv_helper is not None else 0.0
        z2 = sv_helper.get_zone_time(2, resolved_id) if sv_helper is not None else 0.0
        zone_duration = z1 + z2

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID:{resolved_id} T:{duration:.1f}s Z:{zone_duration:.1f}s",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Show the window
    cv2.imshow("Store Monitor", frame)

    # Exit on 'q' key or when window X is clicked
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:
        if cv2.getWindowProperty("Store Monitor", cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

if USE_REALSENSE:
    pipeline.stop()
else:
    cap.release()
cv2.destroyAllWindows()