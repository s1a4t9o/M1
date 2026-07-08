#色の閾値を調整するためのコード（画像版）

import cv2
import numpy as np

# === パラメータ ===
MIN_AREA = 15  # ここ未満のピクセル数を持つ成分は捨てる

# === 入力画像 ===
input_path = "images/ttes.jpg"
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"画像が読み込めません: {input_path}")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === HSV 色範囲 ===
lower_red1 = np.array([0,   100,   0])
upper_red1 = np.array([5,   255, 255])
lower_red2 = np.array([177, 100,   0])
upper_red2 = np.array([180, 255, 255])   # 0〜180 の両端をまたぐため 170 から
lower_green = np.array([55,  60,   0])
upper_green = np.array([90, 255, 255])

# === マスク生成（色抽出）===
mask_red  = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                           cv2.inRange(hsv, lower_red2, upper_red2))
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# === 小領域除去（輪郭を塗りつぶさず形状維持）===
def filter_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for i in range(1, num):                      # 0番は背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    return filtered

mask_red_filtered   = filter_small_components(mask_red,   MIN_AREA)
mask_green_filtered = filter_small_components(mask_green, MIN_AREA)

# === 可視化用合成 ===
output = np.zeros_like(image)
output[mask_red_filtered   > 0] = [0,   0, 255]  # 赤領域 → 青チャンネル0, 緑0, 赤255
output[mask_green_filtered > 0] = [0, 255,   0]  # 緑領域

# === 表示 ===
#cv2.imshow("Red Mask (area-filtered)",   mask_red_filtered)
#cv2.imshow("Green Mask (area-filtered)", mask_green_filtered)
cv2.imshow("Combined Output",            output)
cv2.waitKey(0)
cv2.destroyAllWindows()
