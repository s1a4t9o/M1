import cv2
import numpy as np

def find_circles(mask, MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue  # 面積が小さいものは無視（ノイズ除去）
        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append(((int(x), int(y)), r))
    return circles

# === パラメータ ===
MAX_CENTER_DISTANCE = 8    # 赤緑の中心距離
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.7
MIN_AREA = 15              # 面積の下限（px^2）

# === 入力画像を読み込み（適宜ファイル名変更） ===
image = cv2.imread("aaa.png")  # ← 対象の画像ファイル名に変更してください
if image is None:
    raise FileNotFoundError("❌ 入力画像が見つかりません。ファイル名を確認してください。")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 赤色マスク（2つの範囲を統合）
red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
red_mask = cv2.bitwise_or(red1, red2)

# 緑色マスク
green_mask = cv2.inRange(hsv, (55, 60, 0), (80, 255, 255))

# 複数の円を取得
red_circles = find_circles(red_mask, MIN_AREA)
green_circles = find_circles(green_mask, MIN_AREA)

for center_r, r_r in red_circles:
    for center_g, r_g in green_circles:
        dist = np.linalg.norm(np.array(center_r) - np.array(center_g))
        ratio = r_r / r_g
        if dist < MAX_CENTER_DISTANCE and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX:
            cx = int((center_r[0] + center_g[0]) / 2)
            cy = int((center_r[1] + center_g[1]) / 2)
            center = (cx, cy)

            # 可視化
            cv2.circle(image, center, 4, (0, 0, 255), -1)
            cv2.putText(image, f"{center}", (center[0]-40, center[1]+50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(image, center_r, int(r_r), (0, 0, 255), 1)
            cv2.circle(image, center_g, int(r_g), (0, 255, 0), 1)

# 結果表示
cv2.imshow("Double Ring Marker Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
