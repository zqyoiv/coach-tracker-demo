"""
On-site test video paths. Scans reolink/ and tapo/ under 0304cymy-test-video;
excludes raw-footage (and "raw footage") and any video whose filename matches
its parent folder name. Use in video-play-tracker: e.g.
  from utils.onsite_video_path import REOLINK_EYELEVEL_0
  VIDEO_PATH = REOLINK_EYELEVEL_0

Possible variable names (depend on files on disk; below is the typical set):
  REOLINK_EYELEVEL_0, REOLINK_EYELEVEL_1, REOLINK_EYELEVEL_2, REOLINK_EYELEVEL_3
  REOLINK_FOOTLEVEL_0, REOLINK_FOOTLEVEL_1, REOLINK_FOOTLEVEL_2
  REOLINK_TD_HIGH_0, REOLINK_TD_HIGH_1, REOLINK_TD_HIGH_2, REOLINK_TD_HIGH_3, REOLINK_TD_HIGH_4
  REOLINK_TD_MID_0, REOLINK_TD_MID_1, REOLINK_TD_MID_2, REOLINK_TD_MID_3, REOLINK_TD_MID_4, REOLINK_TD_MID_5
  TAPO_ABNORMAL_ABNORMAL2
  TAPO_EYELEVEL_0, TAPO_EYELEVEL_1, TAPO_EYELEVEL_2, TAPO_EYELEVEL_3
  TAPO_FOOTLEVEL_0, TAPO_FOOTLEVEL_1, TAPO_FOOTLEVEL_2, TAPO_FOOTLEVEL_3, TAPO_FOOTLEVEL_4
  (Also: ALL_PATHS, REOLINK_PATHS, TAPO_PATHS)
"""
from pathlib import Path

_BASE = Path(r"C:\Users\vioyq\Desktop\Coach_Tracker\0304cymy-test-video")
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".gif", ".MP4", ".AVI", ".MOV", ".GIF"}


def _is_raw_footage(rel_parts) -> bool:
    return any(
        p.lower() in ("raw-footage", "raw footage") for p in rel_parts
    )


def _to_var(s: str) -> str:
    out = s.replace("-", "_").replace(" ", "_").upper()
    return out or "FILE"


def _scan_folder(root: Path, prefix: str) -> dict:
    out = {}
    if not root.exists():
        return out
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix not in _VIDEO_EXT:
            continue
        try:
            rel = f.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if _is_raw_footage(parts):
            continue
        parent_name = f.parent.name
        stem = f.stem
        if stem.lower() == parent_name.lower():
            continue
        subfolder = _to_var(parent_name)
        var_stem = _to_var(stem)
        var_name = f"{prefix}_{subfolder}_{var_stem}"
        cnt = 0
        while var_name in out:
            cnt += 1
            var_name = f"{prefix}_{subfolder}_{var_stem}_{cnt}"
        out[var_name] = str(f.resolve())
    return out


_reolink = _scan_folder(_BASE / "reolink", "REOLINK")
_tapo = _scan_folder(_BASE / "tapo", "TAPO")
_all = {**_reolink, **_tapo}
globals().update(_all)

# Convenience: list of all paths, and by prefix
ALL_PATHS = list(_all.values())
REOLINK_PATHS = list(_reolink.values())
TAPO_PATHS = list(_tapo.values())
