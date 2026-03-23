import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv


# Type for dwell event ready to send: (zone_id, person_id, dwell_sec, first_sec, last_sec)
DwellEventReady = Tuple[int, int, float, float, float]


class SupervisionZoneTracker:
    """
    Helper for:
    - defining two polygon zones (zone 1 and zone 2)
    - computing who is in each zone (box overlaps polygon)
    - per-ID per-zone dwell = last time seen in zone minus first time (no accumulation)
    - drawing both zones + heatmap on the frame
    - buffer-based dwell events: send when person has been gone > buffer_sec without returning
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int, int],
        dwell_leave_buffer_sec: float = 5.0,
        zone_1_polygon: Optional[np.ndarray] = None,
        zone_2_polygon: Optional[np.ndarray] = None,
        zone_1_color: Optional[sv.Color] = None,
        zone_2_color: Optional[sv.Color] = None,
        enable_heatmap: bool = True,
        debug_prints: bool = False,
        zone_membership_mode: str = "any-point",
    ):
        h, w = frame_shape[:2]
        # Zone 1: left half of center area; Zone 2: right half (defaults)
        if zone_1_polygon is not None:
            polygon_1 = np.asarray(zone_1_polygon, dtype=np.int32)
        else:
            polygon_1 = np.array(
                [
                    [w // 4, h // 4],
                    [w // 2, h // 4],
                    [w // 2, 3 * h // 4],
                    [w // 4, 3 * h // 4],
                ],
                dtype=np.int32,
            )
        if zone_2_polygon is not None:
            polygon_2 = np.asarray(zone_2_polygon, dtype=np.int32)
        else:
            polygon_2 = np.array(
                [
                    [w // 2, h // 4],
                    [3 * w // 4, h // 4],
                    [3 * w // 4, 3 * h // 4],
                    [w // 2, 3 * h // 4],
                ],
                dtype=np.int32,
            )
        self.zone_1 = sv.PolygonZone(polygon=polygon_1)
        self.zone_2 = sv.PolygonZone(polygon=polygon_2)
        color_1 = zone_1_color if zone_1_color is not None else sv.Color.GREEN
        color_2 = zone_2_color if zone_2_color is not None else sv.Color.BLUE
        self.zone_1_annotator = sv.PolygonZoneAnnotator(
            zone=self.zone_1, color=color_1, thickness=2
        )
        self.zone_2_annotator = sv.PolygonZoneAnnotator(
            zone=self.zone_2, color=color_2, thickness=2
        )
        self._enable_heatmap = enable_heatmap
        self._debug_prints = debug_prints
        self._zone_membership_mode = str(zone_membership_mode).strip().lower()
        self.heatmap_annotator = sv.HeatMapAnnotator(
            position=sv.Position.CENTER, opacity=0.4, radius=15, kernel_size=25
        )

        # Per-zone first/last time: _first_seen[zone_id][rid], _last_seen[zone_id][rid]
        self._first_seen_in_zone_sec: Dict[int, Dict[int, float]] = {1: {}, 2: {}}
        self._last_seen_in_zone_sec: Dict[int, Dict[int, float]] = {1: {}, 2: {}}
        self._prev_in_zone_by_id: Dict[int, Dict[int, bool]] = {1: {}, 2: {}}
        self._cumulative_sec = 0.0
        self._heatmap_min_move_px = 2.0
        self._prev_centroid_by_id: Dict[int, Tuple[float, float]] = {}
        self.last_frame_time = time.time()
        self._dwell_leave_buffer_sec = dwell_leave_buffer_sec
        # Pending: (zone_id, rid, first, last, left_at) — waiting for buffer before sending
        self._pending_dwell: List[Tuple[int, int, float, float, float]] = []

    def update(
        self,
        frame: np.ndarray,
        result,
        track_id_to_resolved: Optional[Dict[int, int]] = None,
        video_t_sec: Optional[float] = None,
    ) -> Tuple[
        np.ndarray,
        int,
        int,
        List[bool],
        List[bool],
        List[Optional[int]],
        List[DwellEventReady],
    ]:
        """
        Update zone/time/heatmap for a single YOLO result.
        Returns:
            frame, people_in_zone_1, people_in_zone_2, in_zone_1_flags, in_zone_2_flags, ids_for_display,
            dwell_events_ready: [(zone_id, person_id, dwell_sec, first_sec, last_sec), ...] when buffer exceeded
        """
        detections = sv.Detections.from_ultralytics(result)
        track_id_to_resolved = track_id_to_resolved or {}

        in_zone_1: List[bool] = []
        in_zone_2: List[bool] = []
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

            poly1 = self.zone_1.polygon.astype(np.float32)
            poly2 = self.zone_2.polygon.astype(np.float32)
            for box in boxes_xyxy:
                x1, y1, x2, y2 = map(float, box)
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if self._zone_membership_mode == "center":
                    inside_1 = cv2.pointPolygonTest(poly1, (cx, cy), False) >= 0
                    inside_2 = cv2.pointPolygonTest(poly2, (cx, cy), False) >= 0
                else:
                    test_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (cx, cy)]
                    inside_1 = any(
                        cv2.pointPolygonTest(poly1, (px, py), False) >= 0 for (px, py) in test_points
                    )
                    inside_2 = any(
                        cv2.pointPolygonTest(poly2, (px, py), False) >= 0 for (px, py) in test_points
                    )
                in_zone_1.append(inside_1)
                in_zone_2.append(inside_2)

        now = time.time()
        delta = now - self.last_frame_time
        self.last_frame_time = now
        self._cumulative_sec += delta
        t_sec = video_t_sec if video_t_sec is not None else self._cumulative_sec

        dwell_events_ready: List[DwellEventReady] = []

        for zone_id in (1, 2):
            in_zone = in_zone_1 if zone_id == 1 else in_zone_2
            if detections.tracker_id is not None and len(detections.tracker_id) == len(in_zone):
                for i, tid in enumerate(detections.tracker_id):
                    if in_zone[i]:
                        rid = track_id_to_resolved.get(int(tid), int(tid))
                        # If returning after leaving (was in pending), remove from pending and start new visit
                        was_pending = any(p[0] == zone_id and p[1] == rid for p in self._pending_dwell)
                        self._pending_dwell = [
                            p for p in self._pending_dwell
                            if not (p[0] == zone_id and p[1] == rid)
                        ]
                        if was_pending or rid not in self._first_seen_in_zone_sec[zone_id]:
                            self._first_seen_in_zone_sec[zone_id][rid] = t_sec
                        self._last_seen_in_zone_sec[zone_id][rid] = t_sec

            current_resolved_ids = set()
            if detections.tracker_id is not None:
                for inside, tid in zip(in_zone, detections.tracker_id):
                    tid_int = int(tid)
                    rid = track_id_to_resolved.get(tid_int, tid_int)
                    current_resolved_ids.add(rid)
                    was_inside = self._prev_in_zone_by_id[zone_id].get(rid, False)
                    if was_inside and not inside:
                        dwell = self.get_zone_time(zone_id, rid)
                        first, last = self.get_zone_first_last(zone_id, rid)
                        self._pending_dwell.append((zone_id, rid, first, last, t_sec))
                        if self._debug_prints:
                            print(
                                f"ID:{rid} left zone {zone_id}, dwell (first→last): {dwell:.1f}s "
                                f"(buffer {self._dwell_leave_buffer_sec}s)"
                            )
                    self._prev_in_zone_by_id[zone_id][rid] = inside

            for rid, was_inside in list(self._prev_in_zone_by_id[zone_id].items()):
                if was_inside and rid not in current_resolved_ids:
                    dwell = self.get_zone_time(zone_id, rid)
                    first, last = self.get_zone_first_last(zone_id, rid)
                    self._pending_dwell.append((zone_id, rid, first, last, t_sec))
                    if self._debug_prints:
                        print(
                            f"ID:{rid} left zone {zone_id} (off screen), dwell (first→last): {dwell:.1f}s "
                            f"(buffer {self._dwell_leave_buffer_sec}s)"
                        )
                    self._prev_in_zone_by_id[zone_id][rid] = False

        # Check pending: buffer exceeded = permanently left, ready to send
        still_pending: List[Tuple[int, int, float, float, float]] = []
        for zone_id_p, rid_p, first_p, last_p, left_at in self._pending_dwell:
            if t_sec - left_at >= self._dwell_leave_buffer_sec:
                dwell_p = max(0.0, last_p - first_p)
                dwell_events_ready.append((zone_id_p, rid_p, dwell_p, first_p, last_p))
                if self._debug_prints:
                    print(
                        f"ID:{rid_p} permanently left zone {zone_id_p} (buffer exceeded), sending dwell {dwell_p:.1f}s"
                    )
            else:
                still_pending.append((zone_id_p, rid_p, first_p, last_p, left_at))
        self._pending_dwell = still_pending

        frame = self.zone_1_annotator.annotate(scene=frame)
        frame = self.zone_2_annotator.annotate(scene=frame)

        if self._enable_heatmap:
            # Heatmap
            detections_for_heatmap = detections
            if len(detections) > 0:
                if detections.class_id is not None:
                    person_mask = np.asarray(detections.class_id == 0)
                else:
                    person_mask = np.ones(len(detections), dtype=bool)
                move_mask = np.ones(len(detections), dtype=bool)
                if detections.tracker_id is not None and len(detections.tracker_id) == len(detections):
                    boxes = detections.xyxy
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = boxes[i]
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        tid = int(detections.tracker_id[i])
                        prev = self._prev_centroid_by_id.get(tid)
                        self._prev_centroid_by_id[tid] = (float(cx), float(cy))
                        if prev is not None:
                            dx, dy = cx - prev[0], cy - prev[1]
                            if dx * dx + dy * dy < self._heatmap_min_move_px * self._heatmap_min_move_px:
                                move_mask[i] = False
                combined = person_mask & move_mask
                if np.any(combined):
                    detections_for_heatmap = detections[combined]
                else:
                    detections_for_heatmap = detections[[]]
            try:
                has_boxes = len(detections_for_heatmap) > 0 and np.isfinite(detections_for_heatmap.xyxy).all()
            except Exception:
                has_boxes = False
            if has_boxes:
                with np.errstate(invalid="ignore", divide="ignore"):
                    out = self.heatmap_annotator.annotate(scene=frame, detections=detections_for_heatmap)
                frame = out[0] if isinstance(out, tuple) else out

        people_in_zone_1 = sum(1 for v in in_zone_1 if v)
        people_in_zone_2 = sum(1 for v in in_zone_2 if v)
        return frame, people_in_zone_1, people_in_zone_2, in_zone_1, in_zone_2, ids_for_display, dwell_events_ready

    def get_zone_time(self, zone_id: int, person_id: int) -> float:
        """Dwell in zone_id = last - first for that person in that zone. zone_id is 1 or 2."""
        pid = int(person_id)
        first = self._first_seen_in_zone_sec.get(zone_id, {}).get(pid, 0.0)
        last = self._last_seen_in_zone_sec.get(zone_id, {}).get(pid, 0.0)
        return float(max(0.0, last - first))

    def get_zone_first_last(self, zone_id: int, person_id: int) -> Tuple[float, float]:
        """(first_sec, last_sec) in zone for this person (for Mixpanel start/end)."""
        pid = int(person_id)
        first = self._first_seen_in_zone_sec.get(zone_id, {}).get(pid, 0.0)
        last = self._last_seen_in_zone_sec.get(zone_id, {}).get(pid, 0.0)
        return (first, last)

    def get_dwell_events_to_flush(self) -> List[DwellEventReady]:
        """
        Call at video end: return all dwell events to send (pending + still in zone).
        Buffer is ignored — we flush everything.
        """
        events: List[DwellEventReady] = []
        for zone_id_p, rid_p, first_p, last_p, _ in self._pending_dwell:
            dwell_p = max(0.0, last_p - first_p)
            events.append((zone_id_p, rid_p, dwell_p, first_p, last_p))
        self._pending_dwell = []
        for zone_id in (1, 2):
            for rid, inside in self._prev_in_zone_by_id[zone_id].items():
                if inside:
                    first, last = self.get_zone_first_last(zone_id, rid)
                    dwell = max(0.0, last - first)
                    if dwell > 0:
                        events.append((zone_id, rid, dwell, first, last))
        return events
