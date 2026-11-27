import argparse
import csv
import math
from pathlib import Path
from statistics import median

import cv2
import numpy as np


MARKER_COUNT = 16
MARKER_DIAMETER_CM = 11.25
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 90


def find_circles(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        if cv2.contourArea(contour) >= MIN_AREA:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circles.append(((int(x), int(y)), radius))
    return circles


def detect_markers(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_low = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
    red_high = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
    red_circles = find_circles(cv2.bitwise_or(red_low, red_high))
    green_circles = find_circles(cv2.inRange(hsv, (40, 60, 0), (80, 255, 255)))

    candidates = []
    for red_center, red_radius in red_circles:
        for green_center, green_radius in green_circles:
            outer_radius = max(red_radius, green_radius)
            ratio = min(red_radius, green_radius) / outer_radius
            if (math.dist(red_center, green_center) < MAX_CENTER_DISTANCE
                    and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX):
                center = tuple(int((r + g) / 2) for r, g in zip(red_center, green_center))
                candidates.append((center, outer_radius))

    markers = []
    for center, radius in sorted(candidates, key=lambda marker: marker[1], reverse=True):
        if all(math.dist(center, saved_center) >= MAX_CENTER_DISTANCE
               for saved_center, _ in markers):
            markers.append((center, radius))
    return markers


def sample_video(path):
    if not path.is_file():
        raise FileNotFoundError(f"入力動画が見つかりません: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"入力動画を開けません: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps < 1:
            raise ValueError(f"1秒間隔の抽出に使えるFPSを取得できません: {fps}")

        samples = []
        second = 0
        frame_index = 0
        target_frame = 0
        while cap.grab():
            if frame_index == target_frame:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"フレーム {frame_index} を読み込めません")

                markers = detect_markers(frame)
                count = len(markers)
                center = None
                diameter = None
                if count == MARKER_COUNT:
                    center = np.mean([point for point, _ in markers], axis=0)
                    diameter = median(2 * radius for _, radius in markers)
                samples.append((second, frame_index / fps, count, center, diameter))
                print(f"動画 {second:4d} 秒  検出マーカー数 {count:2d}/16", flush=True)
                second += 1
                target_frame = int(second * fps + 0.5)

            frame_index += 1

        if not samples:
            raise RuntimeError("動画からフレームを読み込めませんでした")
        return samples
    finally:
        cap.release()


def write_csv(samples, output_path):
    valid_samples = [sample for sample in samples if sample[3] is not None]
    if valid_samples:
        reference_second, _, _, reference_center, _ = valid_samples[0]
        cm_per_px = MARKER_DIAMETER_CM / median(sample[4] for sample in valid_samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "time_s", "frame_time_s", "marker_count", "center_x_px", "center_y_px",
            "dx_cm", "dy_cm", "displacement_cm", "marker_diameter_px", "cm_per_px",
        ])
        for second, frame_time, count, center, diameter in samples:
            row = [second, round(frame_time, 6), count]
            if center is not None:
                dx, dy = (center - reference_center) * cm_per_px
                row.extend(round(float(value), 6) for value in
                           (*center, dx, dy, math.hypot(dx, dy), diameter, cm_per_px))
            else:
                row.extend([""] * 7)
            writer.writerow(row)

    print(f"CSV保存先: {output_path}")
    print(f"16個検出: {len(valid_samples)} / {len(samples)} フレーム")
    if valid_samples:
        print(f"変位の基準: {reference_second} s、換算係数: {cm_per_px:.6f} cm/px")
    else:
        print("16個検出できたフレームがなく、変位はすべて空欄です。")


def main():
    parser = argparse.ArgumentParser(description="HIDASのマーカー重心の変位を1秒間隔でCSVに出力")
    parser.add_argument("input_video", type=Path, nargs="?", default=Path("../mp4_input/112302.mov"))
    parser.add_argument("output_csv", type=Path, nargs="?", default=Path("../mp4_output/displacement_112302.csv"))
    args = parser.parse_args()
    if args.input_video.resolve() == args.output_csv.resolve():
        parser.error("入力動画と出力CSVには異なるパスを指定してください")
    write_csv(sample_video(args.input_video), args.output_csv)


if __name__ == "__main__":
    main()
