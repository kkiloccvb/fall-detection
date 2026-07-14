"""
第 2 级：从关键点算特征 — 数据收集阶段 (2026-07-14)
基于第 1 级骨架代码，增加躯干中心 y 坐标的实时打印，
用于收集站立 / 蹲下 / 躺倒时的 y 坐标范围，为后续阈值设定提供数据依据。

今日目标：
  1. 跑起来，站着走几步，记录躯干中心 y 的稳定范围
  2. 蹲下，记录最低点的 y
  3. 躺倒（或坐地上），记录 y
  4. 对比三种状态的数值差异，确认"躯干中心 y"这个特征能区分站立和倒下

原理：
  躯干中心 = 两肩中点 和 两髋中点 的中间点
  - 两肩中点 y：(kpt[11].y + kpt[12].y) / 2
  - 两髋中点 y：(kpt[23].y + kpt[24].y) / 2
  - 躯干中心 y = 上述两个中点再平均
  - 人站立时躯干中心在画面偏上方（y 小），跌倒时躯干砸向地面（y 突然变大）

采集后要回答的问题：
  - 站立的 y 大致在什么区间？
  - 蹲下的 y 和站立差多少？
  - 躺倒的 y 变化幅度有多大？
  - 用 ln(当前y / 基准y) 的话，基准值取什么合理？

依赖：mediapipe 0.10.x (task-based API)
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

MODEL_PATH = "pose_landmarker_full.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
landmarker = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收，退出...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # === 第 2 级：计算躯干中心 y 坐标 ===
        # 关键点 11=左肩, 12=右肩, 23=左髋, 24=右髋
        shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
        hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
        torso_center_y = (shoulder_mid_y + hip_mid_y) / 2
        print(f"躯干中心 y: {torso_center_y:.3f}")

        # 画骨架
        drawing_utils.draw_landmarks(
            frame,
            landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

        # 画躯干中心点（红色圆点，方便肉眼对照）
        h, w = frame.shape[:2]
        cx = int(((landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) / 4) * w)
        cy = int(torso_center_y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Camera - 骨架 + 躯干中心", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
