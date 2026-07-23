# 色の閾値を調整するためのコード（Webカメラ版）

import cv2
import numpy as np

# === パラメータ ===
MIN_AREA = 15

# === Webカメラ設定 ===
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(f"Webカメラが開けません。CAMERA_INDEX={CAMERA_INDEX} を確認してください")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)


# === 小領域除去 ===
def filter_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered


# === HSV 色範囲 ===
lower_red1 = np.array([0,   80,   0])
upper_red1 = np.array([20,   255, 255])

lower_red2 = np.array([170, 80,   0])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([55,  60,   0])
upper_green = np.array([85, 255, 255])


while True:
    ret, image = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_red_filtered = filter_small_components(mask_red, MIN_AREA)
    mask_green_filtered = filter_small_components(mask_green, MIN_AREA)

    output = np.zeros_like(image)
    output[mask_red_filtered > 0] = [0, 0, 255]
    output[mask_green_filtered > 0] = [0, 255, 0]

    cv2.imshow("Original Camera", image)
    cv2.imshow("Combined Output", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()