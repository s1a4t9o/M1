import cv2
import numpy as np
import os
import csv
from natsort import natsorted

# === 設定 ===
input_folder = "study/mp4_output/2_train"
csv_file = "study/2_train.csv"
MAX_MARKERS = 16
MIN_AREA = 15
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.7

# === 初回のみ：CSVヘッダー作成 ===
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Filename"] + [f"{axis}{i}" for i in range(1, MAX_MARKERS+1) for axis in ("x", "y")]
        writer.writerow(header)

# === 重複判定用セット ===
unique_coordinate_sets = set()

# === 画像処理ループ ===
for filename in natsorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    filepath = os.path.join(input_folder, filename)
    image = cv2.imread(filepath)
    if image is None:
        print(f"❌ 画像が読み込めませんでした: {filepath}")
        continue

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤・緑マスク生成
    # 赤色マスク（2つの範囲を統合）
    red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
    red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    
    # 緑色マスク
    green_mask = cv2.inRange(hsv, (55, 60, 0), (80, 255, 255))

    def find_circles(mask, min_area):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            circles.append(((int(x), int(y)), r))
        return circles

    red_circles = find_circles(red_mask, MIN_AREA)
    green_circles = find_circles(green_mask, MIN_AREA)

    # 赤緑ペアの中心を求める
    centers = []
    for center_r, r_r in red_circles:
        for center_g, r_g in green_circles:
            dist = np.linalg.norm(np.array(center_r) - np.array(center_g))
            ratio = r_r / r_g
            if dist < MAX_CENTER_DISTANCE and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX:
                cx = int((center_r[0] + center_g[0]) / 2)
                cy = int((center_r[1] + center_g[1]) / 2)
                centers.append((cx, cy))

    if not centers:
        print(f"⚠️ {filename} に有効なマーカーが見つかりませんでした。")
        continue

    # バウンディングボックスからローカル座標へ変換（左下原点）
    x_min = min(c[0] for c in centers)
    x_max = max(c[0] for c in centers)
    y_min = min(c[1] for c in centers)
    y_max = max(c[1] for c in centers)

    local_coords = []
    for cx, cy in centers:
        lx = cx - x_min
        ly = y_max - cy  # 画像座標のY軸を反転して左下原点に
        local_coords.append((lx, ly))

    # 16個に揃える（足りないものは -1 で埋める）
    coords_flat = []
    for i in range(MAX_MARKERS):
        if i < len(local_coords):
            coords_flat.extend(local_coords[i])
        else:
            coords_flat.extend([-1, -1])

    coord_tuple = tuple(coords_flat)
    if coord_tuple in unique_coordinate_sets:
        print(f"🚫 重複のためスキップ: {filename}")
        continue

    unique_coordinate_sets.add(coord_tuple)

    # === CSVに書き込み ===
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename] + coords_flat)
    print(f"✅ 保存完了: {filename}")