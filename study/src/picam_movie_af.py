import cv2
from picamera2 import Picamera2
from libcamera import controls, Transform

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "XRGB8888", "size": (1920, 1080)},
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)

picam2.start()

# 連続オートフォーカス
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

while True:
    # 元画像は1920×1080で取得
    im = picam2.capture_array()

    # 表示用だけ960×540に縮小
    display_im = cv2.resize(
        im,
        (960, 540),
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow("Camera", display_im)

    key = cv2.waitKey(1)
    if key == 27:
        break

picam2.stop()
cv2.destroyAllWindows()