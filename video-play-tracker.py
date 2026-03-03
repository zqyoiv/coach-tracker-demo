"""
Run person tracking on a video file, play the video in a window with detection boxes, then print a report.
Usage: python video-play-tracker.py <video_path>

Press 'q' to quit early. Speed: uses GPU + FP16 if available.
"""
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Video to process (or pass as first command-line argument)
# VIDEO_PATH = "C:/Users/vioyq/Desktop/Coach_Tracker/lighting-videos/daylight-cloudy.mp4"
VIDEO_PATH = "C:/Users/vioyq/Desktop/Coach_Tracker/test-video/Videos_MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset/1_1_crop.mp4"
# --- Detection tuning for top-down / unusual angles (e.g. overhead 90° view) ---
# Larger model (s/m/l/x) and lower conf often help when the person is barely visible or from above.
MODEL_SOURCE = "yolo11x.pt"  # try yolo11s.pt or yolo11m.pt if x is too slow; n often misses top-down
IMG_SIZE = 640              # larger = more detail (helps odd angles), slower; try 480 if too slow
CONF_THRESHOLD = 0.15       # lower = more detections (helps top-down), more false positives; default 0.25
IOU_THRESHOLD = 0.5         # NMS overlap threshold; default 0.7
# MODEL_SOURCE = "mshamrai/yolov8s-visdrone"  # alternative: different classes 

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

TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker.yaml")
ZONE_ID = 1
USE_CPU = False

WINDOW_NAME = "Video Play Tracker"


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
    delay_ms = max(1, int(1000.0 / fps))

    time_on_screen = {}
    first_seen_sec = {}
    last_seen_sec = {}

    print(f"Playing with tracking: {video_path} ({total_frames} frames @ {fps:.1f} fps). Press 'q' to quit.")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker=TRACKER_CFG,
            imgsz=IMG_SIZE,
            device=DEVICE,
            half=HALF,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )

        t_sec = (frame_idx - 1) * frame_duration
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                time_on_screen[track_id] = time_on_screen.get(track_id, 0.0) + frame_duration
                if track_id not in first_seen_sec:
                    first_seen_sec[track_id] = t_sec
                last_seen_sec[track_id] = t_sec + frame_duration

                x1, y1, x2, y2 = map(int, box)
                duration = time_on_screen[track_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} {duration:.1f}s",
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        # Frame counter overlay
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
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

    cap.release()
    cv2.destroyAllWindows()

    t_end = time.perf_counter()
    processing_secs = t_end - t_start
    video_length_secs = total_frames / fps if total_frames else 0

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
        print(f"  Person {track_id} on screen for {secs:.1f} s")
        if log_dwell:
            start_sec = first_seen_sec.get(track_id, 0.0)
            end_sec = last_seen_sec.get(track_id, start_sec + secs)
            log_dwell(int(track_id), secs, ZONE_ID, t_start + start_sec, t_start + end_sec)
    print("--- end report ---")


if __name__ == "__main__":
    main()
