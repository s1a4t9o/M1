import cv2
import numpy as np

# === 入力画像 ===
img_path = ".png"       # ← ここを好きな画像ファイルに変える
#output_path = "kibo/output/sisei.png"

# === カメラ内部パラメータ ===
camera_matrix = np.array([
    [523.105750, 0.000000, 635.434258],
    [0.000000, 534.765913, 500.335102],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

dist_coeffs = np.array([
    -0.164787, 0.020375, -0.001572, -0.000369, 0.0
], dtype=np.float32)

marker_length = 0.045  # [m] ← マーカーの一辺の長さ（例：45mm）

# === 入力画像の読み込み ===
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"画像が読み込めません: {img_path}")

# === ArUcoマーカーの検出 ===
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
detector = cv2.aruco.ArucoDetector(dictionary)

corners, ids, _ = detector.detectMarkers(img)

if ids is None or len(ids) == 0:
    print("マーカーが見つかりませんでした。")
else:
    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # === 姿勢推定 ===
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs)

    for i, marker_id in enumerate(ids.flatten()):
        rvec, tvec = rvecs[i], tvecs[i]
        print(f"ID={marker_id} | tvec={tvec.flatten()} | rvec={rvec.flatten()}")

        # === 座標軸描画 ===
        #cv2.drawFrameAxes(img, camera_matrix, dist_coeffs,rvec, tvec, marker_length * 0.5, 2)

    # === 結果保存 ===
    #cv2.imwrite(output_path, img)
    #print(f"描画結果を保存しました → {output_path}")
