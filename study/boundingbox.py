import cv2
import numpy as np

def find_circles(mask, MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append(((int(x), int(y)), r))
    return circles

# === パラメータ ===
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.7
MIN_AREA = 15

# === 入力画像読み込み ===
image = cv2.imread("study/a.png")
if image is None:
    raise FileNotFoundError("❌ 入力画像が見つかりません。")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 赤・緑マスク生成
# 赤色マスク（2つの範囲を統合）
red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
red_mask = cv2.bitwise_or(red1, red2)

# 緑色マスク
green_mask = cv2.inRange(hsv, (55, 60, 0), (80, 255, 255))

# === 円の取得 ===
red_circles = find_circles(red_mask, MIN_AREA)
green_circles = find_circles(green_mask, MIN_AREA)

centers = []  # 検出された中心点を格納

for center_r, r_r in red_circles:
    for center_g, r_g in green_circles:
        dist = np.linalg.norm(np.array(center_r) - np.array(center_g))
        ratio = r_r / r_g
        if dist < MAX_CENTER_DISTANCE and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX:
            cx = int((center_r[0] + center_g[0]) / 2)
            cy = int((center_r[1] + center_g[1]) / 2)
            centers.append((cx, cy))

# === バウンディングボックスの描画 + ローカル座標表示 ===
if centers:
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # 青枠

    for cx, cy in centers:
        # ローカル座標（原点：左下）
        local_x = cx - x_min
        local_y = y_max - cy
        cv2.circle(image, (cx, cy), 4, (0,165,255), -1)
        cv2.putText(image, f"({local_x}, {local_y})", (cx - 30, cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# === 結果表示 ===
cv2.imshow("Double Ring Marker Detection with Local Coordinates", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
