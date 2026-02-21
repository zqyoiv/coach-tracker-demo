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

# 用于记录每个人首次出现的时间 {TrackID: StartTime}
start_times = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. 实时追踪 (persist=True 保证跨帧锁定同一个人，classes=[0] 只识别 'person')
    results = model.track(
        frame, 
        persist=True,
        classes=[0], 
        tracker=TRACKER_CFG,
        verbose=False)

    # 检查是否有检测结果
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy() # 坐标
        track_ids = results[0].boxes.id.int().cpu().tolist() # 每个人的唯一ID

        for box, track_id in zip(boxes, track_ids):
            # 如果是新面孔，记录当前时间
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

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()