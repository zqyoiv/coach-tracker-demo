#!/bin/bash

# Windows PowerShell: use process-video.ps1 (same args, Windows paths like C:\...\Coach_Video).
#   cd ...\coach-tracker-demo\_gc-vm
#   .\process-video.ps1 3-13 "C:\Users\...\Coach_Tracker\_Coach_Video" 3 4 5

# 视频根目录：包含日期子目录（如 3-13）的文件夹，不是 3-13 本身。
# 默认 ~/coach-raw-video；本机示例（父目录）：
#   Git Bash:  export COACH_VIDEO_ROOT="/c/Users/vioyq/Desktop/Coach_Tracker/_Coach_Video"
#   WSL:       export COACH_VIDEO_ROOT="/mnt/c/Users/vioyq/Desktop/Coach_Tracker/_Coach_Video"
#
# 若未设置 COACH_VIDEO_ROOT，可用第二个参数传入根路径：
#   ./process-video.sh 3-13 "/c/Users/vioyq/Desktop/Coach_Tracker/_Coach_Video"
#
# 只跑部分机位（可选）：在日期 [ 根路径 ] 之后写 coach 编号 1–5
#   ./process-video.sh 3-13 "/c/.../_Coach_Video" 3 4 5
#   ./process-video.sh 3-13 3 4 5
#   （省略根路径时用 COACH_VIDEO_ROOT 或默认 ~/coach-raw-video）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/coach-raw-video-concat-hours.py"

# 1. 检查是否输入了日期参数
if [ -z "$1" ]; then
    echo "❌ Error: Please specify a date (e.g., ./process-video.sh 3-15)"
    exit 1
fi

TARGET_DATE=$1
shift

if [ -n "${COACH_VIDEO_ROOT:-}" ]; then
    VIDEO_ROOT="$COACH_VIDEO_ROOT"
else
    VIDEO_ROOT="$HOME/coach-raw-video"
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

echo "🚀 Starting processing for Date: $TARGET_DATE"
echo "   VIDEO_ROOT=$VIDEO_ROOT"
if [ ${#COACHES[@]} -gt 0 ]; then
    echo "   COACHES only: ${COACHES[*]}"
fi

# 2. 检查文件夹是否存在
if [ ! -d "$BASE_PATH" ]; then
    echo "❌ Error: Directory $BASE_PATH does not exist."
    exit 1
fi

# 3. 运行 Python 合并脚本（只处理 $TARGET_DATE，不会影响其它 3-x 目录）
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

# 4. 遍历 Coach-x 文件夹进行后续清理和搬运
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
        rmdir "${coach_dir}hourly"
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
        _cleanup_one_coach "$coach_dir"
    done
fi

echo "✨ All tasks for $TARGET_DATE are finished!"