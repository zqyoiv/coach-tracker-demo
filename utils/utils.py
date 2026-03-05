"""Shared helpers: env loading, env_bool."""
import os
from pathlib import Path


def load_env(base_dir=None):
    """Load .env from base_dir (default: project root, parent of utils/) and cwd."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    env_path = base_dir / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        load_dotenv()  # also cwd
    except ImportError:
        pass
    # Fallback: read .env manually so USE_* flags work even without dotenv or wrong cwd
    for p in [env_path, Path.cwd() / ".env"]:
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"\'')
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass


def env_bool(key: str, default: bool = True) -> bool:
    """Read a boolean from os.environ; default when key is missing."""
    v = (os.environ.get(key) or str(default)).strip().lower()
    return v in ("1", "true", "yes")
