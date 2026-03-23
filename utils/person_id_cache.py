"""
Person ID feature cache for demo/video-tracker, demo/video-play-tracker, and coach-1.
Stores each resolved person ID and their appearance features; when the tracker assigns
a "new" ID, we match against this cache (cosine similarity) and reuse the same ID if
it's the same person. This reduces ID fragmentation when the tracker loses and
re-creates tracks (e.g. after long occlusion).
"""
from collections import deque
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Default crop size for feature extraction (H, W); aspect ~2:1 like person ReID.
_CROP_H, _CROP_W = 64, 32
# Grid for simple spatial descriptor: (grid_h, grid_w) -> (grid_h * grid_w * 3) dims
_GRID_H, _GRID_W = 4, 4


def extract_feature(frame: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Extract a fixed-size appearance feature from the person crop.
    Uses a simple spatial color descriptor (no extra model). For stronger re-id,
    replace with a ReID model (e.g. OSNet) and call it here.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(_GRID_H * _GRID_W * 3, dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(_GRID_H * _GRID_W * 3, dtype=np.float32)

    crop = cv2.resize(crop, (_CROP_W, _CROP_H), interpolation=cv2.INTER_AREA)
    # Grid pooling: mean per cell, per channel
    cell_h, cell_w = _CROP_H // _GRID_H, _CROP_W // _GRID_W
    feats: List[float] = []
    for i in range(_GRID_H):
        for j in range(_GRID_W):
            cell = crop[
                i * cell_h : (i + 1) * cell_h,
                j * cell_w : (j + 1) * cell_w,
                :,
            ]
            for c in range(3):
                feats.append(float(np.mean(cell[:, :, c])))
    vec = np.array(feats, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1] (assumes L2-normalized vectors)."""
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


class PersonFeatureCache:
    """
    Stores resolved person IDs and their appearance features. When the tracker
    assigns a new ID, we match the detection's feature against the cache and
    reuse a previous ID if similarity is above threshold.
    """

    def __init__(
        self,
        match_thresh: float = 0.75,
        max_vectors_per_id: int = 50,
    ):
        self.match_thresh = match_thresh
        self.max_vectors_per_id = max_vectors_per_id
        # resolved_id -> deque of feature vectors
        self._features: Dict[int, deque] = {}
        # tracker_id -> resolved_id (when we merged a new tracker ID to a known person)
        self._tracker_to_resolved: Dict[int, int] = {}

    def find_match(self, feature: np.ndarray) -> Optional[int]:
        """
        Find a cached person ID whose stored features are close enough to
        this feature. Returns resolved_id or None.
        """
        best_id: Optional[int] = None
        best_sim = self.match_thresh
        for resolved_id, vecs in self._features.items():
            if not vecs:
                continue
            # Compare to the most recent vector (or max over all)
            for v in vecs:
                sim = _cosine_similarity(feature, v)
                if sim > best_sim:
                    best_sim = sim
                    best_id = resolved_id
        return best_id

    def add(self, resolved_id: int, feature: np.ndarray) -> None:
        """Register a new person ID with its first feature."""
        self._features[resolved_id] = deque([feature], maxlen=self.max_vectors_per_id)

    def update(self, resolved_id: int, feature: np.ndarray) -> None:
        """Append feature for an existing person (keeps last N)."""
        if resolved_id not in self._features:
            self._features[resolved_id] = deque(maxlen=self.max_vectors_per_id)
        self._features[resolved_id].append(feature.copy())

    def resolve(self, tracker_id: int, feature: np.ndarray) -> int:
        """
        Map tracker_id to a resolved (canonical) person ID. If we've already
        mapped this tracker_id (e.g. from a previous frame), return that.
        If this is a new tracker_id, try to match to a cached person; if
        match found, register the mapping and return that ID. Otherwise
        treat as new person and return tracker_id.
        """
        if tracker_id in self._tracker_to_resolved:
            resolved = self._tracker_to_resolved[tracker_id]
            self.update(resolved, feature)
            return resolved

        match = self.find_match(feature)
        if match is not None:
            self._tracker_to_resolved[tracker_id] = match
            self.update(match, feature)
            return match

        self._tracker_to_resolved[tracker_id] = tracker_id
        self.add(tracker_id, feature)
        return tracker_id

    def reset_tracker_map(self) -> None:
        """Clear transient tracker_id -> resolved_id map (safe between video files)."""
        self._tracker_to_resolved = {}

    def load_from_json(self, path: Path) -> int:
        """
        Load cached appearance features from disk.
        Returns number of person IDs loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", {})
        loaded = 0
        self._features = {}
        self._tracker_to_resolved = {}
        for k, vec_list in features.items():
            try:
                rid = int(k)
            except Exception:
                continue
            if not isinstance(vec_list, list):
                continue
            dq = deque(maxlen=self.max_vectors_per_id)
            for v in vec_list:
                try:
                    arr = np.asarray(v, dtype=np.float32).reshape(-1)
                except Exception:
                    continue
                if arr.size == 0:
                    continue
                norm = float(np.linalg.norm(arr))
                if norm > 1e-6:
                    arr = arr / norm
                dq.append(arr)
            if dq:
                self._features[rid] = dq
                loaded += 1
        return loaded

    def save_to_json(self, path: Path) -> None:
        """Persist cached appearance features to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "match_thresh": float(self.match_thresh),
            "max_vectors_per_id": int(self.max_vectors_per_id),
            "features": {
                str(rid): [v.astype(float).tolist() for v in vecs]
                for rid, vecs in self._features.items()
                if vecs
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True)
