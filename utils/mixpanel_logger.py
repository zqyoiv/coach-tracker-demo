"""
Send dwell events to Mixpanel (person_id, dwell_time_sec, zone, start_time, end_time).
Scripts POST directly to Mixpanel's API.
.env: MIXPANEL_TOKEN=..., SEND_TO_MIXPANEL=true|false (false = local-only, no requests).
"""
import os
from pathlib import Path
from datetime import datetime, timezone

# Project root (parent of utils/) for loading .env
_BASE_DIR = Path(__file__).resolve().parent.parent

# Set to True to print send status (token missing, sent, or error)
VERBOSE = True

def _get_token():
    # 1) env (already set or from dotenv)
    token = (os.environ.get("MIXPANEL_TOKEN") or os.environ.get("mixpanel_token") or "").strip()
    if token:
        return token
    # 2) dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(_BASE_DIR / ".env")
        load_dotenv()
        token = (os.environ.get("MIXPANEL_TOKEN") or os.environ.get("mixpanel_token") or "").strip()
        if token:
            return token
    except ImportError:
        pass
    # 3) read .env file directly (fallback)
    for env_path in [_BASE_DIR / ".env", Path.cwd() / ".env"]:
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("MIXPANEL_TOKEN="):
                        return line.split("=", 1)[1].strip().strip('"\'')
                    if line.startswith("mixpanel_token="):
                        return line.split("=", 1)[1].strip().strip('"\'')
        except Exception:
            pass
    return ""


def _is_send_enabled():
    """True if SEND_TO_MIXPANEL is not false/0/no/off (default True for local testing off)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(_BASE_DIR / ".env")
        load_dotenv()
    except ImportError:
        pass
    v = (os.environ.get("SEND_TO_MIXPANEL") or os.environ.get("send_to_mixpanel") or "").strip().lower()
    if v in ("false", "0", "no", "off"):
        return False
    for env_path in [_BASE_DIR / ".env", Path.cwd() / ".env"]:
        if not env_path.exists():
            continue
        try:
            with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line.upper().startswith("SEND_TO_MIXPANEL="):
                        val = line.split("=", 1)[1].strip().strip('"\'').lower()
                        if val in ("false", "0", "no", "off"):
                            return False
                        break
        except Exception:
            pass
    return True


def log_dwell(person_id: int, dwell_time_sec: float, zone: int, start_time: float, end_time: float) -> bool:
    """
    Send one "Dwell" event to Mixpanel.
    start_time/end_time are Unix timestamps (e.g. from time.time()).
    Returns True if sent, False if disabled, no token, or request failed.
    """
    if not _is_send_enabled():
        return False
    token = _get_token()
    if not token:
        if VERBOSE:
            print("Mixpanel: no token (add MIXPANEL_TOKEN to .env and run from project dir)")
        return False

    try:
        import urllib.request
        import urllib.parse
        import json
        import base64

        event = {
            "event": "Dwell",
            "properties": {
                "token": token.strip(),
                "distinct_id": f"zone{zone}_person{person_id}",
                "person_id": int(person_id),
                "dwell_time_sec": round(dwell_time_sec, 2),
                "zone": zone,
                "start_time": datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
                "end_time": datetime.fromtimestamp(end_time, tz=timezone.utc).isoformat(),
                "time": int(start_time),
            },
        }
        data = json.dumps([event])
        payload = base64.b64encode(data.encode()).decode()
        body = f"data={urllib.parse.quote(payload)}".encode()
        req = urllib.request.Request(
            "https://api.mixpanel.com/track",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = r.read().decode()
        if VERBOSE:
            print(f"Mixpanel: sent Dwell person_id={person_id} zone={zone} dwell={dwell_time_sec:.1f}s")
        return True
    except Exception as e:
        if VERBOSE:
            print(f"Mixpanel: failed — {e}")
        return False
