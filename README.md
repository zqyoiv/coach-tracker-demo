Install: `pip install -r requirements.txt`

Run: `python realtime-tracker.py` or `python video-tracker.py <video.mp4>`

**Venv (optional):**
```bash
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
```

**GPU (RTX 50 / Blackwell):**
```bash
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

**GPU (other NVIDIA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(Or `cu124` / `cu118` for other CUDA versions.)

**Mixpanel (dwell logging)**
No web server: scripts send events directly to Mixpanel when a person leaves (realtime) or at end of run (video). In `.env`: `MIXPANEL_TOKEN=your_token`, and `SEND_TO_MIXPANEL=false` to disable (local-only, no requests). Set zone in code: `ZONE_ID = 1` in realtime-tracker.py and video-tracker.py. Events sent: `Dwell` with properties `person_id`, `dwell_time_sec`, `zone`, `start_time`, `end_time`.

**Files**
- **realtime-tracker.py** — Live camera (or RealSense) person tracking; shows boxes and on-screen time, logs when someone leaves or returns.
- **video-tracker.py** — Runs the same tracker on a video file and prints a report of time-on-screen per person (and processing stats).
- **yaml/basic-tracker-config.yaml** — Tracker config shared by both scripts (BoT-SORT + ReID, buffer, thresholds). Tune this so one person keeps one ID across leave/re-enter.

**yaml/basic-tracker-config.yaml parameters**

| Parameter | Type | Example values | Meaning |
|-----------|------|----------------|---------|
| `tracker_type` | str | `botsort`, `bytetrack` | Tracker backend; `botsort` enables ReID. |
| `track_high_thresh` | float | 0.25–0.5 | First-stage match threshold; higher = stricter, fewer noisy tracks. |
| `track_low_thresh` | float | 0.1 | Second-stage threshold for low-score detections; balances recovery vs drift. |
| `new_track_thresh` | float | 0.25–0.95 | Min confidence to start a **new** track; higher = fewer new IDs, prefer re-activate. |
| `track_buffer` | int | 30–20000 | Frames to keep a lost track alive; higher = same person keeps ID longer when leaving frame. |
| `match_thresh` | float | 0.2–0.8 | Association similarity (IoU/cost); lower = easier to match detection to existing track. |
| `fuse_score` | bool | `true`, `false` | Fuse detection score with motion/IoU in matching. |
| `gmc_method` | str | `sparseOptFlow`, `orb`, `none` | Global motion compensation for moving camera. |
| `proximity_thresh` | float | 0.1–0.5 | Min IoU to consider tracks “proximate” for ReID; lower = more re-associations. |
| `appearance_thresh` | float | 0.1–0.8 | Min appearance similarity for ReID; lower = same person re-entering more likely to keep ID. |
| `with_reid` | bool | `true`, `false` | Use ReID (appearance) to re-identify when person re-enters; needs `botsort`. |
| `model` | str | `auto` | ReID model; `auto` uses detector features when available. |


