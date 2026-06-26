#色の閾値を調整するためのコード（動画版）

import cv2
import numpy as np

# === パラメータ ===
MIN_AREA = 15

# === 入力動画 ===
input_video_path = "mp4_input/test3.mov"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"動画が開けません: {input_video_path}")


# === 小領域除去 ===
def filter_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered


# === HSV 色範囲 ===
lower_red1 = np.array([0, 100, 0])
upper_red1 = np.array([5, 255, 255])

lower_red2 = np.array([177, 100, 0])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([40, 60, 0])
upper_green = np.array([80, 255, 255])


while True:
    ret, image = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # === マスク生成 ===
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # === 小領域除去 ===
    mask_red_filtered = filter_small_components(mask_red, MIN_AREA)
    mask_green_filtered = filter_small_components(mask_green, MIN_AREA)

    # === 可視化用合成 ===
    output = np.zeros_like(image)
    output[mask_red_filtered > 0] = [0, 0, 255]
    output[mask_green_filtered > 0] = [0, 255, 0]

    # === 表示 ===
    cv2.imshow("Original Video", image)
    cv2.imshow("Combined Output", output)

    # ESCキーで終了
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()