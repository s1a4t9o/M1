import cv2
from picamera2 import Picamera2
from libcamera import controls, Transform

picam2 = Picamera2()

# 180°回転してプレビュー
config = picam2.create_preview_configuration(
    #main={"format": "XRGB8888", "size": (640, 480)},
    main={"format": "XRGB8888", "size": (1920, 1080)},
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)

picam2.start()

# 連続オートフォーカス
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

while True:
    im = picam2.capture_array()
    cv2.imshow("Camera", im)

    key = cv2.waitKey(1)
    # Escキーで終了
    if key == 27:
        break

picam2.stop()
cv2.destroyAllWindows()