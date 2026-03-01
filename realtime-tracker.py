import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# 1. Load model (n = nano, fastest for real-time; x = larger, more accurate)
# model = YOLO("yolo11n.pt") 
model = YOLO("yolo11x.pt") 

# Use absolute path for tracker config so it loads correctly regardless of cwd
TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker.yaml")

# Use Intel RealSense (pyrealsense2); set False to use OpenCV with a normal webcam
USE_REALSENSE = True
CAMERA_INDEX = 0  # for OpenCV: 0 = default, 1 = second camera

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

# Set the window size (Width, Height) - adjust these numbers to fit your screen
cv2.resizeWindow("Store Monitor", 800, 450)

# First-seen time per track ID; not cleared when person leaves so timer continues if same ID returns within track_buffer
start_times = {}
# Track IDs from previous frame, used to detect who "left" the frame
previous_track_ids = set()
# IDs that have left and not yet "returned"; used to log "returned" when they reappear
ids_who_left = set()

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

    # 3. Run tracking (persist=True keeps same ID across frames; classes=[0] = person only)
    results = model.track(
        frame, 
        persist=True,
        classes=[0], 
        tracker=TRACKER_CFG,
        verbose=False)

    # Track IDs visible in current frame
    current_track_ids = set(
        results[0].boxes.id.int().cpu().tolist()
        if results[0].boxes.id is not None
        else []
    )
    # Detect who "left" this frame (was in previous frame, not in current) and log
    left_ids = previous_track_ids - current_track_ids
    for track_id in left_ids:
        duration = time.time() - start_times.get(track_id, time.time())
        print(f"Person ID:{track_id} left (was on screen for {duration:.1f}s)")
        ids_who_left.add(track_id)
    # Who "returned" this frame = was in ids_who_left and is now visible again (same ID, timer continues)
    returned_ids = ids_who_left & current_track_ids
    for track_id in returned_ids:
        print(f"Person ID:{track_id} returned (timer continues)")
        ids_who_left.discard(track_id)
    previous_track_ids = current_track_ids

    # Draw results if any detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # bbox coordinates
        track_ids = results[0].boxes.id.int().cpu().tolist()  # unique ID per person

        for box, track_id in zip(boxes, track_ids):
            # New ID: record start time; if same ID returns within track_buffer, we don't reset, timer continues
            if track_id not in start_times:
                start_times[track_id] = time.time()

            # Elapsed time on screen
            duration = time.time() - start_times[track_id]

            # Draw box and label on frame
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} Time:{duration:.1f}s", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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