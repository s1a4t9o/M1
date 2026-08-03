# 色の閾値を調整するためのコード（Raspberry Piカメラ版）

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform


# === パラメータ ===

# この面積より小さい領域を除去
MIN_AREA = 15

# カメラから取得する画像サイズ
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# PC画面に表示する画像サイズ
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


# === HSV 色範囲 ===

# 赤色の範囲1
lower_red1 = np.array([0, 80, 0])
upper_red1 = np.array([20, 255, 255])

# 赤色の範囲2
lower_red2 = np.array([170, 80, 0])
upper_red2 = np.array([180, 255, 255])

# 緑色の範囲
lower_green = np.array([55, 60, 0])
upper_green = np.array([85, 255, 255])


# === 小領域除去 ===
def filter_small_components(mask, min_area):
    """
    二値画像から面積がmin_area未満の領域を除去する
    """

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    filtered = np.zeros_like(mask)

    # 0番は背景なので、1番から調べる
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= min_area:
            filtered[labels == i] = 255

    return filtered


# === Raspberry Piカメラ設定 ===

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "format": "XRGB8888",
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT)
    },

    # 水平方向と垂直方向を反転して180度回転
    transform=Transform(
        hflip=True,
        vflip=True
    )
)

picam2.configure(config)
picam2.start()


try:
    while True:
        # 1920×1080で画像を取得
        captured_image = picam2.capture_array()

        # XRGB8888は4チャンネルなので、
        # OpenCVで扱いやすい3チャンネルBGR画像へ変換
        image = cv2.cvtColor(
            captured_image,
            cv2.COLOR_BGRA2BGR
        )

        # BGR画像からHSV画像へ変換
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # === 赤色抽出 ===

        mask_red1 = cv2.inRange(
            hsv,
            lower_red1,
            upper_red1
        )

        mask_red2 = cv2.inRange(
            hsv,
            lower_red2,
            upper_red2
        )

        mask_red = cv2.bitwise_or(
            mask_red1,
            mask_red2
        )

        # === 緑色抽出 ===

        mask_green = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # === 小領域除去 ===

        mask_red_filtered = filter_small_components(
            mask_red,
            MIN_AREA
        )

        mask_green_filtered = filter_small_components(
            mask_green,
            MIN_AREA
        )

        # === 赤と緑を描画した確認画像の作成 ===

        output = np.zeros_like(image)

        # 検出した赤色領域を赤色で表示
        output[mask_red_filtered > 0] = [0, 0, 255]

        # 検出した緑色領域を緑色で表示
        output[mask_green_filtered > 0] = [0, 255, 0]

        # === 表示用画像のみ縮小 ===

        display_original = cv2.resize(
            image,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        display_output = cv2.resize(
            output,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_NEAREST
        )

        # === 画面表示 ===

        cv2.imshow(
            "Original Camera",
            display_original
        )

        cv2.imshow(
            "Combined Output",
            display_output
        )

        # Escキーで終了
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

finally:
    # エラーが起きた場合でもカメラを確実に停止
    picam2.stop()
    cv2.destroyAllWindows()