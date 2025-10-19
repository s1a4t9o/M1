import cv2
import numpy as np

# === シミュレータのカメラ行列 ===
camera_matrix = np.array([
    [523.105750, 0.0, 635.434258],
    [0.0, 534.765913, 500.335102],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# === 歪み係数 ===
dist_coeffs = np.array([
    [-0.164787, 0.020375, -0.001572, -0.000369, 0.000000]
], dtype=np.float32)

# === 画像読み込み ===
img = cv2.imread("kibo/input/undistorted.png")  # 画像パスを適宜変更

# === 歪み補正処理 ===
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# 歪み補正（undistort）
undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# === 表示または保存 ===
cv2.imshow("歪み補正前", img)
cv2.imshow("歪み補正後", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存したい場合：
cv2.imwrite("kibo/output/undistorted.png", undistorted)