from picamera2 import Picamera2
from libcamera import Transform
from time import sleep

# カメラ初期化
picam2 = Picamera2()

# 静止画設定（180°回転）
config = picam2.create_still_configuration(
    transform=Transform(hflip=True, vflip=True)
)
picam2.configure(config)

# カメラ開始
picam2.start()

# オートフォーカス・露出を安定させるため少し待つ
sleep(2)

# 撮影
picam2.capture_file("image.jpg")

# カメラ停止
picam2.stop()

print("image.jpg を180°回転して保存しました")