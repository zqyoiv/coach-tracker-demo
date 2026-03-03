"""
Run person tracking on a video file, play the video in a window with detection boxes, then print a report.
Usage: python video-play-tracker.py <video_path>

Press 'q' to quit early. Speed: uses GPU + FP16 if available.
"""
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from ultralytics import YOLO


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

# Video to process (or pass as first command-line argument)
VIDEO_PATH = "C:/Users/vioyq/Desktop/Coach_Tracker/lighting-videos/daylight-cloudy.mp4"

# --- Model: standard COCO vs drone/top-down ---
# Standard (frontal/side view; often misses 90° top-down):
#   MODEL_SOURCE = "yolo11x.pt"   # or yolo11n.pt, yolo11s.pt, yolo11m.pt
#   TRACK_CLASSES = [0]           # COCO class 0 = person
#
# Drone / top-down options (trained on aerial/overhead data; use TRACK_CLASSES below):
#   1) erbayat/yolov11s-visdrone  — YOLOv11s on VisDrone (pedestrian + people)
#   2) mshamrai/yolov8s-visdrone  — YOLOv8s on VisDrone
#   3) Mahadih534/YoloV8-VisDrone — YOLOv8 for small objects from aerial/drone
# For (1) the HF file is not "best.pt", so use the tuple form.
# If nothing works well for true 90° top-down: fine-tune on 50–200 labeled frames from your camera (Ultralytics train custom data).
MODEL_SOURCE = "yolo11s.pt"   # COCO person (main model; tracker assigns IDs)
# MODEL_SOURCE = ("erbayat/yolov11s-visdrone", "yolo11s-visdrone.pt")
# MODEL_SOURCE = "mshamrai/yolov8s-visdrone"
# MODEL_SOURCE = "yolo11x.pt"

# VisDrone classes: 0=pedestrian, 1=people (both count as person). COCO: 0=person only.
def _is_visdrone_model(src):
    if isinstance(src, (tuple, list)) and len(src) >= 1:
        return "visdrone" in str(src[0]).lower()
    return isinstance(src, str) and "visdrone" in src.lower()
TRACK_CLASSES = [0, 1] if _is_visdrone_model(MODEL_SOURCE) else [0]

# --- Detection tuning for top-down / unusual angles ---
IMG_SIZE = 640
CONF_THRESHOLD = 0.08
IOU_THRESHOLD = 0.5

# Filter out non-person false positives (small blobs, thin poles/stands)
MIN_BOX_WIDTH_PX = 40
MIN_BOX_HEIGHT_PX = 40
MIN_BOX_AREA_PX = 2500          # width*height; avoids tiny detections
MAX_ASPECT_RATIO = 3.5          # max(longer/shorter); thin poles/stands have high ratio

# --- Ensemble: optional second model (e.g. VisDrone) when main misses; yellow "det" boxes ---
USE_ENSEMBLE = False
ENSEMBLE_MODEL_SOURCE = ("erbayat/yolov11s-visdrone", "yolo11s-visdrone.pt")
ENSEMBLE_CONF = 0.15            # raise to reduce ensemble false positives (e.g. poles)
ENSEMBLE_IOU_OVERLAP = 0.4


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

# Use video-specific tracker with lower track_high_thresh so low-conf (e.g. top-down) detections get IDs
TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker-video.yaml")
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
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                if not _is_valid_person_box(
                    box, MIN_BOX_WIDTH_PX, MIN_BOX_HEIGHT_PX, MIN_BOX_AREA_PX, MAX_ASPECT_RATIO
                ):
                    continue
                main_boxes.append(box)
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

        # Ensemble: run second model and draw detections main model missed (yellow)
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
