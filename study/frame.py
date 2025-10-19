import cv2
import os
import math

video_path = "study/mp4_input/0.mp4"
output_dir = "study/mp4_output/0"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 取得失敗対策
interval_sec = 0.2                        # ★ ここを変えるだけでOK
eps = 0.5 / max(fps, 1.0)                 # 秒単位のゆるめ誤差

next_t = 0.0       # 次に保存したい“動画内時刻[秒]”
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # このフレームの時刻[秒]

    if t + eps >= next_t:
        filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
        next_t += interval_sec  # 次の保存ターゲット時刻へ

cap.release()
print(f"✅ {saved_count} 枚のフレームを {interval_sec} 秒間隔で保存しました。")
