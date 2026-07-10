import cv2
import numpy as np
import os

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

# === 入出力フォルダ設定 ===
input_dir = "study/mp4_output/right1000"
output_dir = "study/mp4_output/output"
os.makedirs(output_dir, exist_ok=True)

# === 処理対象ファイルを取得（画像拡張子のみ）===
valid_exts = [".jpg", ".jpeg", ".png", ".bmp"]
image_files = sorted(f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in valid_exts))

for filename in image_files:
    input_path = os.path.join(input_dir, filename)
    image = cv2.imread(input_path)
    if image is None:
        print(f"⚠️ 読み込めませんでした: {filename}")
        continue

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 赤色マスク（2つの範囲を統合）
    red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
    red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)
    
    # 緑色マスク
    green_mask = cv2.inRange(hsv, (55, 60, 0), (80, 255, 255))

    # 円の取得
    red_circles = find_circles(red_mask, MIN_AREA)
    green_circles = find_circles(green_mask, MIN_AREA)

    # === 出力用黒背景画像を作成 ===
    result = np.zeros_like(image)

    for center_r, r_r in red_circles:
        for center_g, r_g in green_circles:
            dist = np.linalg.norm(np.array(center_r) - np.array(center_g))
            ratio = r_r / r_g
            if dist < MAX_CENTER_DISTANCE and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX:
                cx = int((center_r[0] + center_g[0]) / 2)
                cy = int((center_r[1] + center_g[1]) / 2)
                center = (cx, cy)

                # 結果を黒背景に描画
                cv2.circle(result, center, 4, (0, 0, 255), -1)
                #cv2.putText(result, f"{center}", (center[0]+10, center[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.circle(result, center_r, int(r_r), (0, 0, 255), 1)
                cv2.circle(result, center_g, int(r_g), (0, 255, 0), 1)

    # 出力ファイル保存
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result)
    print(f"✅ 保存: {output_path}")

print("🎉 すべての画像に処理を適用しました。")
