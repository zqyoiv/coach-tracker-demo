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

