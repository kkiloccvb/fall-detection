"""
第 1 级：MediaPipe 人体骨架关键点显示
在摄像头实时画面上叠加 33 个关键点和连线骨架，按 q 退出。

原理：
  MediaPipe Pose 是预训练的姿态估计模型。
  每帧送进去 → 返回 33 个身体关键点的归一化坐标 (0~1) → 画在帧上。

依赖：mediapipe 0.10.x (task-based API, 非旧版 mp.solutions)
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarksConnections
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import drawing_utils

# 模型文件（已下载到项目目录）
MODEL_PATH = "pose_landmarker_full.task"  # 相对路径，避免中文目录传 C 库时乱码

# 初始化 PoseLandmarker（只做一次）
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,  # 视频流模式
    num_poses=1,                                # 最多检测几个人
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = PoseLandmarker.create_from_options(options)

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 时间戳计数器（detect_for_video 需要递增的时间戳，单位毫秒）
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收，退出...")
        break

    # OpenCV 读进来是 BGR → 转 RGB → 包成 MediaPipe Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    # 送进 PoseLandmarker 检测
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  # ~30fps，每帧约 33ms

    # 如果检测到人体，把关键点和连线画回 BGR 帧上（drawing_utils 自带 33 个关键点样式）
    if result.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks[0],                    # 第一个（也是唯一一个）人的关键点列表
            PoseLandmarksConnections.POSE_LANDMARKS,     # 骨架连线
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # 关键点：绿
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),                   # 连线：绿
        )

    cv2.imshow("Camera - 骨架", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
