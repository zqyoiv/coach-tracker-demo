#!/bin/bash

# 1. 检查是否输入了日期参数
if [ -z "$1" ]; then
    echo "❌ Error: Please specify a date (e.g., ./process_date.sh 3-15)"
    exit 1
fi

TARGET_DATE=$1
BASE_PATH="$HOME/coach-raw-video/$TARGET_DATE"
PYTHON_SCRIPT="$HOME/coach-tracker-demo/_gc-vm/coach-raw-video-concat-hours.py"

echo "🚀 Starting processing for Date: $TARGET_DATE"

# 2. 检查文件夹是否存在
if [ ! -d "$BASE_PATH" ]; then
    echo "❌ Error: Directory $BASE_PATH does not exist."
    exit 1
fi

# 3. 运行 Python 合并脚本
# 注意：这里假设你的 Python 脚本会处理指定日期文件夹下的所有 Coach-x
echo "🎬 Step 1: Merging videos into 1-hour chunks..."
python3 "$PYTHON_SCRIPT"

if [ $? -ne 0 ]; then
    echo "❌ Error: Python merge script failed. Stopping."
    exit 1
fi

# 4. 遍历 Coach-x 文件夹进行后续清理和搬运
echo "🧹 Step 2: Cleaning up and moving files..."

for coach_dir in "$BASE_PATH"/Coach-*/; do
    if [ -d "$coach_dir" ]; then
        echo "Processing $(basename "$coach_dir")..."
        
        # 检查 hourly 文件夹是否存在且不为空
        if [ -d "${coach_dir}hourly" ] && [ "$(ls -A "${coach_dir}hourly")" ]; then
            
            # A. 删除 Coach-x 目录下的原始视频 (不进子目录，不删 hourly)
            rm -f "$coach_dir"*.mp4
            
            # B. 把 hourly 里的视频移到上一层
            mv "${coach_dir}hourly"/*.mp4 "$coach_dir"
            
            # C. 删除现在已经变成空的 hourly 文件夹
            rmdir "${coach_dir}hourly"
            
            echo "✅ $(basename "$coach_dir") done."
        else
            echo "⚠️ Warning: No hourly folder or no merged videos found in $(basename "$coach_dir"). Skipping cleanup."
        fi
    fi
done

echo "✨ All tasks for $TARGET_DATE are finished!"