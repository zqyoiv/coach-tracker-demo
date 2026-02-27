import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# 1. 加载模型 (n 代表 nano, 运行速度最快，适合实时监控)
# model = YOLO("yolo11n.pt") 
model = YOLO("yolo11x.pt") 

# 使用绝对路径加载 tracker 配置，避免 cwd 或库内部加载错误
TRACKER_CFG = str(Path(__file__).resolve().parent / "vio-tracker.yaml")

# 2. 打开摄像头 (0 通常是内置摄像头，如果有多个可以试 1, 2)
cap = cv2.VideoCapture(0)
# Create a named window that can be resized
cv2.namedWindow("Store Monitor", cv2.WINDOW_NORMAL) 

# Set the window size (Width, Height) - adjust these numbers to fit your screen
cv2.resizeWindow("Store Monitor", 800, 450)

# 用于记录每个人首次出现的时间 {TrackID: StartTime}（不因暂时离开画面而清除，这样在 track_buffer 内回来会继续计时）
start_times = {}
# 上一帧出现的 track_id，用于检测谁“离开”画面
previous_track_ids = set()
# 已经离开过画面、尚未“再次出现”的 ID（用于打印 returned）
ids_who_left = set()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 水平镜像画面（像镜子一样）
    frame = cv2.flip(frame, 1)

    # 3. 实时追踪 (persist=True 保证跨帧锁定同一个人，classes=[0] 只识别 'person')
    results = model.track(
        frame, 
        persist=True,
        classes=[0], 
        tracker=TRACKER_CFG,
        verbose=False)

    # 当前帧出现的 track_id
    current_track_ids = set(
        results[0].boxes.id.int().cpu().tolist()
        if results[0].boxes.id is not None
        else []
    )
    # 检测本帧“离开”的人（上一帧在，本帧不在）并打印
    left_ids = previous_track_ids - current_track_ids
    for track_id in left_ids:
        duration = time.time() - start_times.get(track_id, time.time())
        print(f"Person ID:{track_id} left (was on screen for {duration:.1f}s)")
        ids_who_left.add(track_id)
    # 本帧又出现的人 = 之前离开过、现在又出现的 → 说明在 track_buffer 内回来，计时会继续
    returned_ids = ids_who_left & current_track_ids
    for track_id in returned_ids:
        print(f"Person ID:{track_id} returned (timer continues)")
        ids_who_left.discard(track_id)
    previous_track_ids = current_track_ids

    # 检查是否有检测结果
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy() # 坐标
        track_ids = results[0].boxes.id.int().cpu().tolist() # 每个人的唯一ID

        for box, track_id in zip(boxes, track_ids):
            # 如果是新面孔，记录当前时间；若在 track_buffer 内回来（同一 ID），不重置，计时继续
            if track_id not in start_times:
                start_times[track_id] = time.time()

            # 计算已存在时长
            duration = time.time() - start_times[track_id]

            # 4. 在视频画面上绘制
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id} Time:{duration:.1f}s", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示结果窗口
    cv2.imshow("Store Monitor", frame)

    # 按 'q' 键退出，或点击窗口 X 关闭
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    try:
        if cv2.getWindowProperty("Store Monitor", cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()