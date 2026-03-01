"""
Run person tracking on a video file and print a report of time on screen per person.
Usage: python video-tracker.py <video_path>

Speed: uses GPU + FP16 if available; reduce IMG_SIZE or use yolo11n.pt for faster runs.
"""
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# Video to process (or pass as first command-line argument)
# VIDEO_PATH = "C:/Users/vioyq/Desktop/Coach_Tracker/lighting-videos/night-warm.mp4"
VIDEO_PATH = "C:/Users/vioyq/Desktop/Coach_Tracker/angle-videos/eyelevel.mp4"


# Speed: smaller = faster (480 or 640); use yolo11n.pt for much faster, less accurate
IMG_SIZE = 480
model = YOLO("yolo11x.pt")
# Same config as realtime-tracker so same person keeps same ID when leaving/re-entering (track_buffer + ReID)
TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker.yaml")

# Force CPU (set True to ignore GPU)
USE_CPU = True

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

    # Total seconds on screen per track ID
    time_on_screen = {}

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

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for track_id in track_ids:
                time_on_screen[track_id] = time_on_screen.get(track_id, 0.0) + frame_duration

    cap.release()
    t_end = time.perf_counter()
    processing_secs = t_end - t_start
    video_length_secs = total_frames / fps

    # Report: video info, processing time, then one line per person
    print("\n--- Report ---")
    print(f"Video length: {video_length_secs:.1f} s ({total_frames} frames)")
    print(f"Time spent processing: {processing_secs:.1f} s")
    print("Time on screen per person:")
    for track_id in sorted(time_on_screen.keys(), key=lambda x: int(x)):
        secs = time_on_screen[track_id]
        print(f"  Person {track_id} on screen for {secs:.1f} s")
    print("--- end report ---")

if __name__ == "__main__":
    main()
