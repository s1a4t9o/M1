from datetime import datetime
from pathlib import Path

from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# record_video.py のある raspberrypi フォルダから1つ上の src に移動
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "mp4"


def main():
    while input("録画を開始するには y を入力: ").strip().lower() != "y":
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = OUTPUT_DIR / f"recording_{timestamp}.mp4"

    picam2 = Picamera2()
    try:
        config = picam2.create_video_configuration(
            main={
                "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                "format": "YUV420",
            },
            transform=Transform(hflip=True, vflip=True),
        )
        picam2.configure(config)

        encoder = H264Encoder(bitrate=10_000_000)
        picam2.start_recording(encoder, FfmpegOutput(str(output_path)))

        try:
            print(f"録画開始（{CAMERA_WIDTH}×{CAMERA_HEIGHT}）")
            while input("録画を終了するには f を入力: ").strip().lower() != "f":
                pass
        except KeyboardInterrupt:
            print("\n録画を終了します")
        finally:
            picam2.stop_recording()
    finally:
        picam2.close()

    print(f"保存完了: {output_path}")


if __name__ == "__main__":
    main()