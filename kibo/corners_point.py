import cv2
import numpy as np

# === 定数 ===
MARKER_SIZE = 45.0  # mm
PAPER_WIDTH = 297.0  # mm
PAPER_HEIGHT = 210.0  # mm

OFFSET_RIGHT = 26.0  # mm（マーカー右上端から紙の右端まで）
OFFSET_TOP = -42.5    # mm（マーカー右上端から紙の上端まで）

# === 入力画像読み込み ===
image = cv2.imread("kibo/input/test2.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === ArUcoマーカー検出 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)

if len(corners) == 0:
    raise ValueError("ARマーカーが検出できませんでした。")

# === マーカー画像上の4隅（TL, TR, BR, BL）を取得 ===
marker_corners_img = corners[0].reshape((4, 2)).astype(np.float32)

# === 実世界におけるマーカーの4隅座標（右上を原点）===
# マーカーの大きさは 34mm x 34mm
marker_corners_world = np.array([
    [-MARKER_SIZE, 0],         # top-left
    [0, 0],                    # top-right (原点)
    [0, MARKER_SIZE],          # bottom-right
    [-MARKER_SIZE, MARKER_SIZE]  # bottom-left
], dtype=np.float32)

# === 射影変換行列（実世界 → 画像座標）===
H, _ = cv2.findHomography(marker_corners_world, marker_corners_img)

# === 紙の実世界座標（右上角を基準）===
# 紙の右上角：マーカーの右上から左・上方向にある（※Yはマイナス方向に修正）
paper_ru = [OFFSET_RIGHT, -OFFSET_TOP]                     # 右上
paper_lu = [paper_ru[0] - PAPER_WIDTH, paper_ru[1]]         # 左上
paper_rd = [paper_ru[0], paper_ru[1] + PAPER_HEIGHT]        # 右下
paper_ld = [paper_lu[0], paper_lu[1] + PAPER_HEIGHT]        # 左下

# === 配列化して変換 (1, 4, 2) 形状にする ===
paper_pts_world = np.array([paper_lu, paper_ru, paper_rd, paper_ld], dtype=np.float32).reshape(1, 4, 2)

# === 射影変換 ===
paper_pts_img = cv2.perspectiveTransform(paper_pts_world, H)[0]

# === 赤いドットで描画 ===
for pt in paper_pts_img:
    pt_int = tuple(np.round(pt).astype(int))
    cv2.circle(image, pt_int, 8, (0, 0, 255), -1)

# === 保存 ===
cv2.imwrite("kibo/output/detected_corners_projected.png", image)
print("紙の4隅を赤ドットで描いた画像を 'detected_corners_projected.png' に保存しました。")