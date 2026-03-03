import time
from typing import Dict, List, Optional, Tuple

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

        # Cumulative time each ID has spent in the zone (seconds); key = resolved_id when mapping provided
        self.time_in_zone_sec = {}
        # For detecting "left zone" events between frames
        self._prev_in_zone_by_id = {}
        self.last_frame_time = time.time()

    def update(
        self,
        frame: np.ndarray,
        result,
        track_id_to_resolved: Optional[Dict[int, int]] = None,
    ) -> Tuple[np.ndarray, int, List[bool], List[Optional[int]]]:
        """
        Update zone/time/heatmap for a single YOLO result.

        When track_id_to_resolved is provided (e.g. from person_id_cache), zone time
        and "left zone" logs use resolved IDs so the same person keeps one ID.

        Returns:
            frame: annotated frame
            people_in_zone: count of detections overlapping the zone
            in_zone: list[bool] per detection
            ids_for_display: list of resolved_id (or tracker_id) per detection for display
        """
        detections = sv.Detections.from_ultralytics(result)
        track_id_to_resolved = track_id_to_resolved or {}

        polygon = self.zone.polygon.astype(np.float32)
        in_zone: List[bool] = []
        ids_for_display: List[Optional[int]] = []

        if len(detections) > 0:
            boxes_xyxy = detections.xyxy
            raw_tracker_ids = (
                [int(t) for t in detections.tracker_id]
                if detections.tracker_id is not None
                else [None] * len(detections)
            )
            for tid in raw_tracker_ids:
                rid = track_id_to_resolved.get(tid, tid) if tid is not None else None
                ids_for_display.append(rid)

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
                    rid = track_id_to_resolved.get(int(tid), int(tid))
                    self.time_in_zone_sec[rid] = (
                        self.time_in_zone_sec.get(rid, 0.0) + delta
                    )

        # Log dwell time for IDs that just left the zone (use resolved ID so same person = one ID)
        current_resolved_ids = set()
        if detections.tracker_id is not None:
            for inside, tid in zip(in_zone, detections.tracker_id):
                tid_int = int(tid)
                rid = track_id_to_resolved.get(tid_int, tid_int)
                current_resolved_ids.add(rid)
                was_inside = self._prev_in_zone_by_id.get(rid, False)
                if was_inside and not inside:
                    dwell = self.time_in_zone_sec.get(rid, 0.0)
                    print(f"ID:{rid} left zone, dwell in zone so far: {dwell:.1f}s")
                self._prev_in_zone_by_id[rid] = inside

        for rid, was_inside in list(self._prev_in_zone_by_id.items()):
            if was_inside and rid not in current_resolved_ids:
                dwell = self.time_in_zone_sec.get(rid, 0.0)
                print(f"ID:{rid} left zone (off screen), dwell in zone so far: {dwell:.1f}s")
                self._prev_in_zone_by_id[rid] = False

        frame = self.zone_annotator.annotate(scene=frame)
        # Run heatmap only when we have valid detections; suppress numpy divide/cast warnings from supervision
        try:
            has_boxes = len(detections) > 0 and np.isfinite(detections.xyxy).all()
        except Exception:
            has_boxes = False
        if has_boxes:
            with np.errstate(invalid="ignore", divide="ignore"):
                out = self.heatmap_annotator.annotate(scene=frame, detections=detections)
            frame = out[0] if isinstance(out, tuple) else out

        people_in_zone = sum(1 for v in in_zone if v)
        return frame, people_in_zone, in_zone, ids_for_display

    def get_zone_time(self, person_id: int) -> float:
        """person_id should be resolved_id when using the cache."""
        return float(self.time_in_zone_sec.get(int(person_id), 0.0))

