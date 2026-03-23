# Merge ~1-hour videos on Google Colab (Drive)

Use `colab_merge_hourly_drive.py` to walk `date/Coach-N/` folders, sort clips, and concatenate into batches of up to **3600 seconds** (configurable).

## 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Install FFmpeg

Colab images often include `ffmpeg`; if `ffmpeg -version` fails:

```bash
apt-get update -qq && apt-get install -qq -y ffmpeg
```

## 3. Get the script

Clone your repo or upload `video-script/colab_merge_hourly_drive.py`, then:

```bash
cd /content/coach-tracker-demo  # or your path
```

## 4. Set paths

Replace the placeholders with your real Drive paths (shortcut-targets-by-id is fine).

- **Input**: folder that **directly contains** date folders (`3-13`, `3-14`, …), each with `Coach-1` … `Coach-5`.
- **Output**: a **separate** root so originals stay unchanged, e.g. `Coach-AV-store-Test-hourly`.

Example:

```text
INPUT  = "/content/drive/.shortcut-targets-by-id/1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK/Coach-AV-store-Test"
OUTPUT = "/content/drive/.shortcut-targets-by-id/1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK/Coach-AV-store-Test-hourly"
```

## 5. Dry run (recommended first)

```bash
python video-script/colab_merge_hourly_drive.py \
  --input-root "$INPUT" \
  --output-root "$OUTPUT" \
  --dry-run
```

## 6. Run for real

```bash
python video-script/colab_merge_hourly_drive.py \
  --input-root "$INPUT" \
  --output-root "$OUTPUT"
```

### Optional: only some dates

```bash
python video-script/colab_merge_hourly_drive.py \
  --input-root "$INPUT" \
  --output-root "$OUTPUT" \
  --dates 3-13 3-14
```

### Optional: different target length (seconds)

```bash
python video-script/colab_merge_hourly_drive.py \
  --input-root "$INPUT" \
  --output-root "$OUTPUT" \
  --target-seconds 1800
```

## Output layout

```text
<OUTPUT>/
  3-13/
    Coach-1/
      Coach-1_part-001_of-011_20260313_110000-20260313_120000.mp4
      ...
    Coach-2/
    ...
  3-14/
    ...
```

## Notes

- Sorting uses **14-digit timestamps** in filenames when present; otherwise **modification time**, then name.
- Concat uses **stream copy** (`-c copy`). If you see **non-monotonic DTS** warnings, the files usually still play; re-encoding would require a different mode (slower).
- Very long single clips stay in **one** output file even if over 1 hour.
