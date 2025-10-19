import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# === カメラ内部パラメータ ===
camera_matrix = np.array([
    [523.105750, 0.000000, 635.434258],
    [0.000000, 534.765913, 500.335102],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

# === 歪み係数 ===
dist_coeffs = np.array([[-0.164787, 0.020375, -0.001572, -0.000369, 0.000000]], dtype=np.float32)

# === マーカーサイズ（m） ===
marker_length = 0.045  # 45mm

# === NavCamの位置 [m]（Astrobee中心基準）===
navcam_to_body = np.array([0.1177, -0.0422, -0.0826], dtype=np.float32).reshape(3, 1)

# === Astrobeeの仮想絶対位置・姿勢 ===
astrobee_abs_pos = np.array([11.267, -10.2, 5.147], dtype=np.float32).reshape(3, 1)
astrobee_quat = [0.0, 0.0, 0.707, -0.707]  # x, y, z, w

# === マーカー中心→紙中心の相対位置 ===
paper_center_local = np.array([[-0.1025, -0.0375, 0.0]], dtype=np.float32).T  # 3x1

# === 姿勢変換 ===
R_body = R.from_quat(astrobee_quat).as_matrix()  # 3x3
R_navcam = np.array([
    [0,  0,  1],
    [0,  1,  0],
    [-1, 0,  0]
])  # NavCamからAstrobee本体座標系への変換
R_z_90 = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])
R_navcam_to_body = R_navcam @ R_z_90

# === 入力画像読み込み ===
image = cv2.imread("kibo/input/test20.png")
if image is None:
    raise ValueError("画像が読み込めません。パスを確認してください。")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === ArUco検出 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None and len(ids) > 0:
    output = image.copy()
    cv2.aruco.drawDetectedMarkers(output, corners, ids)

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs
    )

    for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
        tvec = tvec.reshape(3, 1)
        cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        # === マーカー座標（絶対座標系） ===
        marker_in_body = R_navcam_to_body @ tvec + navcam_to_body
        marker_global = astrobee_abs_pos + R_body @ marker_in_body
        mx, my, mz = marker_global.ravel()

        # === 紙の中心座標（絶対座標系） ===
        marker_R, _ = cv2.Rodrigues(rvec)
        paper_in_marker = marker_R @ paper_center_local
        paper_in_body = R_navcam_to_body @ (tvec + paper_in_marker) + navcam_to_body
        paper_global = astrobee_abs_pos + R_body @ paper_in_body
        px, py, pz = paper_global.ravel()

        # === 投影（紙中心を赤丸で表示） ===
        image_points, _ = cv2.projectPoints(
            paper_center_local, rvec, tvec, camera_matrix, dist_coeffs
        )
        cx, cy = image_points[0][0]
        cv2.circle(output, (int(cx), int(cy)), 6, (0, 0, 255), -1)

        # === 表示 ===
        print(f"--- Marker ID: {marker_id} ---")
        print(f"[マーカー絶対座標] X={mx:.3f}, Y={my:.3f}, Z={mz:.3f} [m]")
        print(f"[紙の中心絶対座標] X={px:.3f}, Y={py:.3f}, Z={pz:.3f} [m]")

    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    raise ValueError("ARマーカーが検出されませんでした。")
