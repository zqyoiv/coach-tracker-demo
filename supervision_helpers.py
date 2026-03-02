import time
from typing import List, Tuple

import cv2
import numpy as np
import supervision as sv


class SupervisionZoneTracker:
    """
    Helper for:
    - defining a polygon zone
    - computing who is in the zone (box overlaps polygon)
    - accumulating per-ID time-in-zone
    - drawing the zone + heatmap on the frame
    """

    def __init__(self, frame_shape: Tuple[int, int, int]):
        h, w = frame_shape[:2]
        polygon = np.array(
            [
                [w // 4, h // 4],
                [3 * w // 4, h // 4],
                [3 * w // 4, 3 * h // 4],
                [w // 4, 3 * h // 4],
            ],
            dtype=np.int32,
        )
        self.zone = sv.PolygonZone(polygon=polygon)
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.zone, color=sv.Color.GREEN, thickness=2
        )
        self.heatmap_annotator = sv.HeatMapAnnotator(
            position=sv.Position.CENTER, opacity=0.4, radius=15, kernel_size=25
        )

        # Cumulative time each ID has spent in the zone (seconds)
        self.time_in_zone_sec = {}  # track_id -> seconds in zone
        # For detecting "left zone" events between frames
        self._prev_in_zone_by_id = {}  # track_id -> bool (was inside last frame)
        self.last_frame_time = time.time()

    def update(
        self, frame: np.ndarray, result
    ) -> Tuple[np.ndarray, int, List[bool], List[int]]:
        """
        Update zone/time/heatmap for a single YOLO result.

        Returns:
            frame: annotated frame
            people_in_zone: count of detections overlapping the zone
            in_zone: list[bool] per detection
            tracker_ids: list[int] per detection (may contain None)
        """
        detections = sv.Detections.from_ultralytics(result)

        polygon = self.zone.polygon.astype(np.float32)
        in_zone: List[bool] = []
        tracker_ids: List[int] = []

        if len(detections) > 0:
            boxes_xyxy = detections.xyxy
            if detections.tracker_id is not None:
                tracker_ids = [int(t) for t in detections.tracker_id]
            else:
                tracker_ids = [None] * len(detections)

            for box in boxes_xyxy:
                x1, y1, x2, y2 = map(float, box)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                test_points = [
                    (x1, y1),
                    (x2, y1),
                    (x2, y2),
                    (x1, y2),
                    (cx, cy),
                ]
                inside = any(
                    cv2.pointPolygonTest(polygon, (px, py), False) >= 0
                    for (px, py) in test_points
                )
                in_zone.append(inside)

        now = time.time()
        delta = now - self.last_frame_time
        self.last_frame_time = now

        if detections.tracker_id is not None and len(detections.tracker_id) == len(in_zone):
            for i, tid in enumerate(detections.tracker_id):
                if in_zone[i]:
                    tid_int = int(tid)
                    self.time_in_zone_sec[tid_int] = (
                        self.time_in_zone_sec.get(tid_int, 0.0) + delta
                    )

        # Log dwell time for IDs that just left the zone this frame
        current_ids = set()
        if detections.tracker_id is not None:
            for inside, tid in zip(in_zone, detections.tracker_id):
                tid_int = int(tid)
                current_ids.add(tid_int)
                was_inside = self._prev_in_zone_by_id.get(tid_int, False)
                # Transition: inside -> not inside
                if was_inside and not inside:
                    dwell = self.time_in_zone_sec.get(tid_int, 0.0)
                    print(f"ID:{tid_int} left zone, dwell in zone so far: {dwell:.1f}s")
                # Update prev state for this ID
                self._prev_in_zone_by_id[tid_int] = inside

        # Also handle IDs that disappeared from frame while they were inside the zone
        for tid_int, was_inside in list(self._prev_in_zone_by_id.items()):
            if was_inside and tid_int not in current_ids:
                dwell = self.time_in_zone_sec.get(tid_int, 0.0)
                print(f"ID:{tid_int} left zone (off screen), dwell in zone so far: {dwell:.1f}s")
                self._prev_in_zone_by_id[tid_int] = False

        frame = self.zone_annotator.annotate(scene=frame)
        out = self.heatmap_annotator.annotate(scene=frame, detections=detections)
        frame = out[0] if isinstance(out, tuple) else out

        people_in_zone = sum(1 for v in in_zone if v)
        return frame, people_in_zone, in_zone, tracker_ids

    def get_zone_time(self, track_id: int) -> float:
        return float(self.time_in_zone_sec.get(int(track_id), 0.0))

