import cv2
import numpy as np
import scipy.spatial.transform as st

# === カメラ内部パラメータ ===
camera_matrix = np.array([
    [523.105750, 0.000000, 635.434258],
    [0.000000, 534.765913, 500.335102],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

# === 歪み係数 ===
dist_coeffs = np.array([[-0.164787, 0.020375, -0.001572, -0.000369, 0.000000]], dtype=np.float32)

# === マーカーサイズ（メートル）===
marker_length = 0.045  # 45mm

# === Astrobeeの姿勢（クオータニオン）と位置 ===
astrobee_quat = [0.0, 0.0, 1.0, 0.0]  # Y軸+90度回転
astrobee_pos = np.array([10.76698, -6.8525, 4.945])  # Astrobeeの絶対位置

# === NavCam補正行列 ===
# Step1: NavCamが -Z を向いている補正（Astrobee基準）
R_navcam_to_body = np.array([
    [ 0,  0,  1],
    [ 0,  1,  0],
    [-1,  0,  0]
])

# Step2: NavCam自身のローカルZ軸まわりに +90° 回転
R_z_90 = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])

# Step3: NavCamのローカル補正行列（順序に注意）
R_new = R_navcam_to_body @ R_z_90

# === Astrobeeの回転行列 ===
astrobee_R = st.Rotation.from_quat(astrobee_quat).as_matrix()

# === NavCamのワールド回転行列（Astrobee姿勢 × NavCam補正）===
R_cam = astrobee_R @ R_new

# === ArUco辞書と検出設定 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
aruco_params = cv2.aruco.DetectorParameters()

# === 画像読み込み ===
image = cv2.imread("kibo/input/test2.png")
if image is None:
    raise FileNotFoundError("画像が読み込めません。パスを確認してください。")

# === マーカー検出 ===
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
corners, ids, rejected = detector.detectMarkers(image)

if ids is not None:
    # === 各マーカーに対する姿勢推定と座標変換 ===
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs
    )

    for i in range(len(ids)):
        tvec_cam = tvecs[i].reshape(3, 1)                 # カメラ基準の位置
        tvec_world = R_cam @ tvec_cam                     # カメラ→ワールドへの回転
        marker_world = astrobee_pos.reshape(3, 1) + tvec_world  # 絶対座標

        print(f"マーカーID {ids[i][0]}: 絶対位置 X={marker_world[0][0]:.3f}, "
              f"Y={marker_world[1][0]:.3f}, Z={marker_world[2][0]:.3f}")

    # === マーカーを描画して表示 ===
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow("Detected ArUco Marker", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("マーカーが検出されませんでした。")
