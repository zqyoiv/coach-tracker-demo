#!/bin/bash
#
# Same behavior as process-video.sh, but fixed to date 3-23 and default video root:
#   Git Bash:  /c/Users/vioyq/Desktop/Coach_Tracker/_Coach_Video
#   WSL:       export COACH_VIDEO_ROOT="/mnt/c/Users/vioyq/Desktop/Coach_Tracker/_Coach_Video"
#              ./3-23-process-video.sh
#
# Usage:
#   ./3-23-process-video.sh
#   ./3-23-process-video.sh "/mnt/c/Users/.../_Coach_Video"
#   ./3-23-process-video.sh "/c/Users/.../_Coach_Video" 1 2 3
#   ./3-23-process-video.sh 3 4 5
#
# If COACH_VIDEO_ROOT is set, it is used unless the first argument is a directory (then that is root).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/coach-raw-video-concat-hours.py"

TARGET_DATE="3-23"

if [ -n "${COACH_VIDEO_ROOT:-}" ]; then
    VIDEO_ROOT="$COACH_VIDEO_ROOT"
else
    VIDEO_ROOT="/c/Users/vioyq/Desktop/Coach_Tracker/_Coach_Video"
fi

COACHES=()

if [ $# -eq 0 ]; then
    :
elif [ -d "$1" ]; then
    VIDEO_ROOT="$1"
    shift
    COACHES=("$@")
else
    COACHES=("$@")
fi

BASE_PATH="$VIDEO_ROOT/$TARGET_DATE"

echo "🚀 Starting processing for Date: $TARGET_DATE (fixed)"
echo "   VIDEO_ROOT=$VIDEO_ROOT"
if [ ${#COACHES[@]} -gt 0 ]; then
    echo "   COACHES only: ${COACHES[*]}"
fi

if [ ! -d "$BASE_PATH" ]; then
    echo "❌ Error: Directory $BASE_PATH does not exist."
    exit 1
fi

echo "🎬 Step 1: Merging videos into 1-hour chunks (date=$TARGET_DATE only)..."
if [ ${#COACHES[@]} -gt 0 ]; then
    python3 "$PYTHON_SCRIPT" --root "$VIDEO_ROOT" --dates "$TARGET_DATE" --coaches "${COACHES[@]}"
else
    python3 "$PYTHON_SCRIPT" --root "$VIDEO_ROOT" --dates "$TARGET_DATE"
fi

if [ $? -ne 0 ]; then
    echo "❌ Error: Python merge script failed. Stopping."
    exit 1
fi

echo "🧹 Step 2: Cleaning up and moving files..."

_cleanup_one_coach() {
    local coach_dir="$1"
    if [ ! -d "$coach_dir" ]; then
        echo "⚠️ Warning: No folder $(basename "$coach_dir"). Skipping."
        return
    fi
    echo "Processing $(basename "$coach_dir")..."
    if [ -d "${coach_dir}hourly" ] && [ "$(ls -A "${coach_dir}hourly")" ]; then
        rm -f "$coach_dir"*.mp4
        mv "${coach_dir}hourly"/*.mp4 "$coach_dir"
        rmdir "${coach_dir}hourly" 2>/dev/null || rm -rf "${coach_dir}hourly"
        echo "✅ $(basename "$coach_dir") done."
    else
        echo "⚠️ Warning: No hourly folder or no merged videos found in $(basename "$coach_dir"). Skipping cleanup."
    fi
}

if [ ${#COACHES[@]} -gt 0 ]; then
    for n in "${COACHES[@]}"; do
        _cleanup_one_coach "$BASE_PATH/Coach-$n/"
    done
else
    for coach_dir in "$BASE_PATH"/Coach-*/; do
        [ -d "$coach_dir" ] || continue
        _cleanup_one_coach "$coach_dir"
    done
fi

echo "✨ All tasks for $TARGET_DATE are finished!"
