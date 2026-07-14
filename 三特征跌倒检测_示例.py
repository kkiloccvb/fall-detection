"""
三特征跌倒检测 — 示例演示 (2026-07-14)
不覆盖现有文件，仅用于展示算法的完整思路。

跌倒 = 三个条件同时满足：
  特征 A：躯干中心 y 快速增大（人往下掉）
  特征 B：人体框高宽比反转（身体从竖变横）
  特征 C：倒下后保持不动一段时间（排除短暂晃动）

三者是"与"关系，不是"或"。缺一个不报警。

策略：
  不是每帧单独判断，而是维护一个滑动窗口。
  窗口内：躯干中心 y 的变化量 > 阈值A
          高宽比 < 阈值B（宽>高 = 横过来了）
          窗口内最后 N 帧的 y 稳定（不再漂 = 真倒了，不是弯腰）
  三个都通过 → 标记为跌倒

注意：
  - 阈值不是代码里写死的，是你自己采集数据后填进去的
  - 下面的阈值是占位符（TODO），你拿到真实数据后替换
  - 跑这个文件不会打开摄像头，纯粹演示算法结构
"""

import math
from collections import deque

# ============================================================
# 配置区（采集完数据后，改这里的值）
# ============================================================

# 特征 A：躯干中心 y 骤降阈值
# TODO: 从数据中确定——站立时平均 y 是多少，蹲下/躺倒时 y 是多少
BASELINE_Y = 0.65        # 站立基准 y（TODO: 采集后填入真实值）
DROP_THRESHOLD = 0.10    # y 增加超过这个量 = 躯干明显下移（TODO: 数据驱动）

# 特征 B：高宽比阈值
# 站立时高宽比 > 1（身高 > 肩宽），躺倒时 < 1（宽 > 高）
ASPECT_RATIO_THRESHOLD = 1.0  # 低于 1 = 身体变横（TODO: 数据驱动）

# 特征 C：时间窗口
WINDOW_SIZE = 30         # 窗口帧数（~1 秒 @ 30fps，TODO: 数据驱动）
STABLE_FRAMES = 10       # 窗口末尾连续稳定帧数（证明人没在动）

# ============================================================
# 数据结构：滑动窗口
# ============================================================
window = deque(maxlen=WINDOW_SIZE)  # 每个元素: (torso_y, aspect_ratio)


# ============================================================
# 特征 A：躯干中心 y
# ============================================================
def compute_torso_center_y(landmarks):
    """
    输入：MediaPipe 33 个关键点（归一化坐标 0~1）
    输出：躯干中心 y（0~1，越大越靠画面下方）

    躯干中心 = (两肩中点 + 两髋中点) / 2
    两肩中点 = (kpt[11] + kpt[12]) / 2
    两髋中点 = (kpt[23] + kpt[24]) / 2
    """
    shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
    return (shoulder_mid_y + hip_mid_y) / 2


# ============================================================
# 特征 B：人体框高宽比
# ============================================================
def compute_aspect_ratio(landmarks):
    """
    输入：MediaPipe 33 个关键点（归一化坐标 0~1）
    输出：高宽比 = 身高 / 肩宽

    身高 ≈ 鼻子(0) 到 两踝中点(27+28)/2 的 y 方向跨度
    肩宽 ≈ 左肩(11) 到 右肩(12) 的 x 方向跨度

    站立：高 >> 宽 → ratio > 1
    躺倒：宽 > 高 → ratio < 1
    """
    # 身高（y 方向跨度）
    nose_y = landmarks[0].y
    ankle_mid_y = (landmarks[27].y + landmarks[28].y) / 2
    height = abs(ankle_mid_y - nose_y)

    # 肩宽（x 方向跨度）
    shoulder_width = abs(landmarks[11].x - landmarks[12].x)

    # 防止除以零
    if shoulder_width < 0.001:
        return 999.0

    return height / shoulder_width


# ============================================================
# 特征 C：时间窗口 + 联合判断
# ============================================================
def is_falling(torso_y, aspect_ratio):
    """
    每帧调一次。把当前测值塞进窗口，然后检查三条件是否同时满足。
    返回：True = 检测到跌倒，False = 正常。
    """
    window.append((torso_y, aspect_ratio))

    # 窗口还没填满，不做判断
    if len(window) < WINDOW_SIZE:
        return False

    # ---- 条件 1：躯干中心 y 明显增大（人往下掉了） ----
    # 取窗口开头几帧的平均 y（"掉之前"的参考）
    early_y = sum(w[0] for w in list(window)[:5]) / 5
    # 取窗口末尾的 y（"现在"的位置）
    last_y = window[-1][0]
    drop_amount = last_y - early_y
    condition_a = drop_amount > DROP_THRESHOLD

    # ---- 条件 2：身体变横（高宽比 < 1） ----
    last_ratio = window[-1][1]
    condition_b = last_ratio < ASPECT_RATIO_THRESHOLD

    # ---- 条件 3：倒下后稳定不动（排除弯腰/短暂晃动） ----
    # 检查窗口末尾连续 STABLE_FRAMES 帧的 y 变化是否很小
    recent = list(window)[-STABLE_FRAMES:]
    y_values = [r[0] for r in recent]
    y_span = max(y_values) - min(y_values)
    # 如果连续帧之间 y 变化极小，说明人定住了
    condition_c = y_span < 0.02  # TODO: 数据驱动

    return condition_a and condition_b and condition_c


# ============================================================
# 演示：用模拟数据跑一遍算法
# ============================================================
def demo():
    """
    模拟一组从"站立"到"跌倒"的数据，展示三特征联合判断的逻辑。
    不依赖摄像头，纯演示。
    """
    print("=" * 60)
    print("三特征跌倒检测 — 模拟演示")
    print("=" * 60)
    print()

    # 模拟一个假的关键点对象（只用到的字段：y 和 x）
    class FakeLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    # 模拟 45 帧：前 30 帧站立，最后 15 帧模拟"倒下去"
    # 站立姿态：鼻子在上(y小)、脚踝在下(y大)、肩宽正常
    # 倒下姿态：鼻子和脚踝 y 接近（人横了）、肩宽"变宽"（实际是投影变横）
    frames = []
    for i in range(45):
        if i < 30:
            # 站立
            nose_y = 0.25
            ankle_y = 0.85
            shoulder_l_x, shoulder_r_x = 0.45, 0.55
            torso_shift = 0.0
        else:
            # 逐渐倒下（y 下移 + 身体变横）
            t = (i - 30) / 15  # 0 → 1
            nose_y = 0.25 + t * 0.40       # 鼻子往下
            ankle_y = 0.85 - t * 0.10      # 脚踝往上（人躺了）
            shoulder_l_x = 0.45 - t * 0.15
            shoulder_r_x = 0.55 + t * 0.15
            torso_shift = t * 0.12         # 躯干中心下移

        # 建完整的 33 点列表，用得到的填值，用不到的塞 None
        landmarks = [None] * 33
        landmarks[0]  = FakeLandmark(0.50, nose_y + torso_shift)          # 鼻子
        landmarks[11] = FakeLandmark(shoulder_l_x, 0.35 + torso_shift)    # 左肩
        landmarks[12] = FakeLandmark(shoulder_r_x, 0.35 + torso_shift)    # 右肩
        landmarks[23] = FakeLandmark(0.47, 0.60 + torso_shift)            # 左髋
        landmarks[24] = FakeLandmark(0.53, 0.60 + torso_shift)            # 右髋
        landmarks[27] = FakeLandmark(0.47, ankle_y + torso_shift)         # 左踝
        landmarks[28] = FakeLandmark(0.53, ankle_y + torso_shift)         # 右踝
        frames.append(landmarks)

    # 清空窗口
    window.clear()

    for i, lm in enumerate(frames):
        ty = compute_torso_center_y(lm)
        ar = compute_aspect_ratio(lm)
        falling = is_falling(ty, ar)

        marker = " [跌倒!]" if falling else ""
        status = ""
        if i == 29:
            status = " <-- 开始模拟倒下"
        elif i == 44:
            status = " <-- 三条件同时触发"

        print(f"帧 {i:3d} | 躯干y={ty:.3f} | 高宽比={ar:.2f}{marker}{status}")

    print()
    print('帧 30-44 模拟的是"人从站立到倒下"的过程。')
    print('三条件（y 骤降 + 身体变横 + 末尾稳定）同时满足时触发跌倒标记。')
    print()
    print("你的摄像头到位后：把上面的模拟数据换成真实关键点坐标，")
    print("调整三个阈值，就能跑真人了。")


if __name__ == "__main__":
    demo()
