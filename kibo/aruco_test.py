import cv2

# 画像読み込み
image = cv2.imread("kibo/input/img_1_test.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# マーカーディクショナリとパラメータ
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
parameters = cv2.aruco.DetectorParameters()

# マーカー検出
corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

# 結果表示
if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow("Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("ARマーカーが検出されませんでした")
