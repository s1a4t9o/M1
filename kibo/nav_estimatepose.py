import cv2
import numpy as np

# === カメラ内部パラメータ ===
camera_matrix = np.array([
    [523.105750, 0.000000, 635.434258],
    [0.000000, 534.765913, 500.335102],
    [0.000000, 0.000000, 1.000000]
], dtype=np.float32)

# === 歪み係数 ===
dist_coeffs = np.array([[-0.164787, 0.020375, -0.001572, -0.000369, 0.000000]], dtype=np.float32)

# === マーカーサイズ（メートル） ===
marker_length = 0.045  # 45mm

# === Nav Cam の Astrobee中心からの位置 [m] ===
navcam_to_body = np.array([0.1177, -0.0422, -0.0826], dtype=np.float32)

# === Astrobee本体中心の絶対位置（仮定） ===
astrobee_abs_pos = np.array([10.925, -8.875, 4.66203], dtype=np.float32)

# === マーカー中心から見た紙の中心の相対位置 ===
# 左方向10.25cm, 上方向3.75cm → Yをマイナス
paper_center_local = np.array([[-0.1025, -0.0375, 0.0]], dtype=np.float32)

# === 入力画像 ===
image = cv2.imread("kibo/input/test2.png")  # パスを適宜変更
if image is None:
    raise ValueError("画像が読み込めません。ファイルパスを確認してください。")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === ArUco 検出 ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None and len(ids) > 0:
    output = image.copy()
    cv2.aruco.drawDetectedMarkers(output, corners, ids)

    # 姿勢推定
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_length, camera_matrix, dist_coeffs
    )

    for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
        cv2.drawFrameAxes(output, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        # === Astrobee中心から見たマーカー位置（相対） ===
        tvec_body = tvec[0] + navcam_to_body
        bx, by, bz = tvec_body

        # === マーカーの絶対座標 ===
        marker_abs_pos = astrobee_abs_pos + tvec_body
        mx, my, mz = marker_abs_pos

        # === 回転ベクトル → 回転行列 ===
        R, _ = cv2.Rodrigues(rvec)

        # === 紙の中心（カメラ → Astrobee → 絶対） ===
        paper_offset_global = (R @ paper_center_local.T).T
        paper_center_navcam = tvec[0] + paper_offset_global[0]
        paper_center_body = paper_center_navcam + navcam_to_body
        paper_abs_pos = astrobee_abs_pos + paper_center_body
        px, py, pz = paper_abs_pos

        # === 投影（赤点描画）===
        image_points, _ = cv2.projectPoints(
            paper_center_local, rvec, tvec, camera_matrix, dist_coeffs)
        center_px, center_py = image_points[0][0]
        cv2.circle(output, (int(center_px), int(center_py)), 6, (0, 0, 255), -1)

        # === 表示 ===
        print(f"--- Marker ID: {marker_id} ---")
        print(f"[Astrobee相対位置]    X={bx:.3f}, Y={by:.3f}, Z={bz:.3f} [m]")
        print(f"[紙の中心相対位置]    X={paper_center_body[0]:.3f}, Y={paper_center_body[1]:.3f}, Z={paper_center_body[2]:.3f} [m]")
        print(f"[マーカー絶対座標]    X={mx:.3f}, Y={my:.3f}, Z={mz:.3f} [m]")
        print(f"[紙の中心絶対座標]    X={px:.3f}, Y={py:.3f}, Z={pz:.3f} [m]")

    # === 結果表示 ===
    cv2.imshow("Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    raise ValueError("ARマーカーが検出されませんでした。")
