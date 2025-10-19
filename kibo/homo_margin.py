import cv2
import numpy as np

# === 定数 ===
MARKER_SIZE = 45.0  # マーカーの一辺（mm）
PAPER_WIDTH = 297.0  # A4紙の横サイズ（mm）
PAPER_HEIGHT = 210.0  # A4紙の縦サイズ（mm）

OFFSET_RIGHT = 26.0  # マーカー右上から紙右端までの距離（mm）
OFFSET_TOP = -42.5   # マーカー右上から紙上端までの距離（mm, 上方向はマイナス）

MARGIN = 0  # 余白サイズ（mm）

# === 画像読み込み ===
image = cv2.imread("kibo/input/img33.png")
if image is None:
    raise ValueError("画像が読み込めません。パスを確認してください。")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === ArUco検出 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)

if len(corners) == 0:
    raise ValueError("ARマーカーが検出できませんでした。")

# === マーカー4隅（画像座標）を取得 ===
marker_corners_img = corners[0].reshape((4, 2)).astype(np.float32)

# === マーカーの実世界座標（右上原点）===
marker_corners_world = np.array([
    [-MARKER_SIZE, 0],          # top-left
    [0, 0],                     # top-right
    [0, MARKER_SIZE],           # bottom-right
    [-MARKER_SIZE, MARKER_SIZE] # bottom-left
], dtype=np.float32)

# === 射影変換：マーカー座標（世界→画像） ===
H_marker_to_img, _ = cv2.findHomography(marker_corners_world, marker_corners_img)

# === 紙の4隅の実世界座標を定義（右上基準） ===
paper_ru = [OFFSET_RIGHT, OFFSET_TOP]
paper_lu = [paper_ru[0] - PAPER_WIDTH, paper_ru[1]]
paper_rd = [paper_ru[0], paper_ru[1] + PAPER_HEIGHT]
paper_ld = [paper_lu[0], paper_lu[1] + PAPER_HEIGHT]

paper_pts_world = np.array([paper_lu, paper_ru, paper_rd, paper_ld], dtype=np.float32).reshape(1, 4, 2)

# === 紙の4隅を画像座標へ ===
paper_pts_img = cv2.perspectiveTransform(paper_pts_world, H_marker_to_img)[0]

# === warp先の座標：余白ありの補正ビュー ===
output_width = int(PAPER_WIDTH + 2 * MARGIN)
output_height = int(PAPER_HEIGHT + 2 * MARGIN)

paper_pts_dst = np.array([
    [MARGIN, MARGIN],
    [PAPER_WIDTH + MARGIN, MARGIN],
    [PAPER_WIDTH + MARGIN, PAPER_HEIGHT + MARGIN],
    [MARGIN, PAPER_HEIGHT + MARGIN]
], dtype=np.float32)

# === 画像→正面ビュー変換行列 ===
H_img_to_flat, _ = cv2.findHomography(paper_pts_img, paper_pts_dst)

# === 射影変換（正面ビューに補正） ===
warped = cv2.warpPerspective(image, H_img_to_flat, (output_width, output_height))

# === 保存 ===
cv2.imwrite("kibo/output/warped_output.png", warped)
print("正面補正画像（余白付き）を 'output/warped_output.png' に保存しました。")