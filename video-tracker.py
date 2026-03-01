"""
Run person tracking on a video file and print a report of time on screen per person.
Usage: python video-tracker.py <video_path>
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Video to process (or pass as first command-line argument)
VIDEO_PATH = "video.mp4"  # default; override with: python video-tracker.py path/to/video.mp4

# Load model and tracker config
model = YOLO("yolo11x.pt")
TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker.yaml")

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
    frame_idx = 0

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
            verbose=False,
        )

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for track_id in track_ids:
                time_on_screen[track_id] = time_on_screen.get(track_id, 0.0) + frame_duration

    cap.release()

    # Report: one line per person
    print("\n--- Report: time on screen ---")
    for track_id in sorted(time_on_screen.keys(), key=lambda x: int(x)):
        secs = time_on_screen[track_id]
        print(f"Person {track_id} on screen for {secs:.1f} s")
    print("--- end report ---")

if __name__ == "__main__":
    main()
