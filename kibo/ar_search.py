"""
1 枚の画像を全 ArUco/AprilTag 辞書で総当たりし、
ヒットした辞書名と ID を表示・ウィンドウに描画するだけの版
（ファイル保存は行わない）
"""

import cv2

# ---------- 設定 ----------
IMG_PATH = "kibo/input/undistorted.png"   # 入力画像
# --------------------------

# 画像読み込み
image = cv2.imread(IMG_PATH)
if image is None:
    raise ValueError(f"❌ 画像が読み込めません: {IMG_PATH}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# OpenCV が持つ辞書名（Aruco + AprilTag）
aruco_dict_names = [
    # ArUco
    "DICT_4X4_50",  "DICT_4X4_100",  "DICT_4X4_250",  "DICT_4X4_1000",
    "DICT_5X5_50",  "DICT_5X5_100",  "DICT_5X5_250",  "DICT_5X5_1000",
    "DICT_6X6_50",  "DICT_6X6_100",  "DICT_6X6_250",  "DICT_6X6_1000",
    "DICT_7X7_50",  "DICT_7X7_100",  "DICT_7X7_250",  "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    # AprilTag
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"
]

found = False

for name in aruco_dict_names:
    try:
        dict_const = getattr(cv2.aruco, name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)
        detector   = cv2.aruco.ArucoDetector(aruco_dict)

        # 検出
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            print(f"✅ 辞書: {name}  →  検出ID: {ids.flatten()}")
            found = True

            # 描画
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            for i, corner in enumerate(corners):
                pts    = corner[0].astype(int)
                center = tuple(pts.mean(axis=0).astype(int))
                cv2.circle(image, center, 5, (0, 0, 255), -1)
                cv2.putText(image, f"ID:{ids[i][0]}", (center[0]+10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 画面に表示するだけ
            cv2.imshow("Detected ArUco / AprilTag", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break   # 最初にヒットした辞書で終了
    except AttributeError:
        print(f"⚠️ {name}: この OpenCV には未実装")
    except Exception as e:
        print(f"⚠️ {name}: エラー {e}")

if not found:
    print("❌ どの辞書でも検出できませんでした。")
