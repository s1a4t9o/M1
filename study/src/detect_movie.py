## マーカー検出コード（入力出力：動画　角度も描画）

import cv2
import numpy as np
import math


def find_circles(mask, MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append(((int(x), int(y)), r))

    return circles


def get_color_name(hsv_pixel):
    h, s, v = hsv_pixel

    if s < 50 or v < 50:
        return "unknown"

    if (0 <= h <= 5) or (177 <= h <= 180):
        return "red"

    if 55 <= h <= 80:
        return "green"

    return "unknown"


def majority_color(colors):
    red = colors.count("red")
    green = colors.count("green")

    if red > green:
        return "red"
    elif green > red:
        return "green"
    return "unknown"


def sample_color_at_radius(hsv, center, radius):
    cx, cy = center
    height, width = hsv.shape[:2]

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    colors = []

    for deg in angles:
        rad = math.radians(deg)
        x = int(cx + math.cos(rad) * radius)
        y = int(cy + math.sin(rad) * radius)

        if 0 <= x < width and 0 <= y < height:
            colors.append(get_color_name(hsv[y, x]))

    return majority_color(colors)


def judge_marker_type(hsv, center, outer_radius):
    inner_color = sample_color_at_radius(hsv, center, outer_radius * 0.18)
    middle_color = sample_color_at_radius(hsv, center, outer_radius * 0.48)
    outer_color = sample_color_at_radius(hsv, center, outer_radius * 0.78)

    # 基準ID9：内緑・中赤・外緑
    if inner_color == "green" and middle_color == "red" and outer_color == "green":
        marker_type = 9

    # 基準ID1：内緑・外赤
    elif inner_color == "green" and outer_color == "red":
        marker_type = 1

    # 通常マーカー：内赤・外緑
    elif inner_color == "red" and outer_color == "green":
        marker_type = 0

    else:
        marker_type = -1

    return marker_type, inner_color, middle_color, outer_color


def angle_from_center(point, hid_center):
    x, y = point
    cx, cy = hid_center

    return math.atan2(-(y - cy), x - cx)


def assign_ids_by_order(markers):
    if len(markers) == 0:
        return markers, None

    hid_cx = sum(m["center"][0] for m in markers) / len(markers)
    hid_cy = sum(m["center"][1] for m in markers) / len(markers)
    hid_center = (hid_cx, hid_cy)

    for m in markers:
        m["angle"] = angle_from_center(m["center"], hid_center)
        m["angle_deg"] = math.degrees(m["angle"])

    markers_sorted = sorted(markers, key=lambda m: m["angle"])

    base_index = None
    base_id = None

    for i, m in enumerate(markers_sorted):
        if m["type"] == 1:
            base_index = i
            base_id = 1
            break

    if base_index is None:
        for i, m in enumerate(markers_sorted):
            if m["type"] == 9:
                base_index = i
                base_id = 9
                break

    if base_index is None:
        for m in markers_sorted:
            m["assigned_id"] = -1
        return markers_sorted, hid_center

    n = len(markers_sorted)

    for i in range(n):
        m = markers_sorted[i]

        if m["type"] == 1:
            m["assigned_id"] = 1
            continue

        if m["type"] == 9:
            m["assigned_id"] = 9
            continue

        offset = (base_index - i) % n
        assigned_id = ((base_id - 1 + offset) % 16) + 1
        m["assigned_id"] = assigned_id

    return markers_sorted, hid_center


def process_frame(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
    red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    green_mask = cv2.inRange(hsv, (40, 60, 0), (80, 255, 255))

    red_circles = find_circles(red_mask, MIN_AREA)
    green_circles = find_circles(green_mask, MIN_AREA)

    candidates = []

    for center_r, r_r in red_circles:
        for center_g, r_g in green_circles:
            dist = np.linalg.norm(np.array(center_r) - np.array(center_g))

            small_r = min(r_r, r_g)
            large_r = max(r_r, r_g)
            ratio = small_r / large_r

            if dist < MAX_CENTER_DISTANCE and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX:
                cx = int((center_r[0] + center_g[0]) / 2)
                cy = int((center_r[1] + center_g[1]) / 2)

                candidates.append({
                    "center": (cx, cy),
                    "outer_radius": large_r,
                    "red": (center_r, r_r),
                    "green": (center_g, r_g)
                })

    final_candidates = []

    for cand in sorted(candidates, key=lambda c: c["outer_radius"], reverse=True):
        keep = True

        for saved in final_candidates:
            d = np.linalg.norm(np.array(cand["center"]) - np.array(saved["center"]))
            if d < MAX_CENTER_DISTANCE:
                keep = False
                break

        if keep:
            final_candidates.append(cand)

    markers = []

    for cand in final_candidates:
        center = cand["center"]
        outer_radius = cand["outer_radius"]

        marker_type, inner_color, middle_color, outer_color = judge_marker_type(
            hsv, center, outer_radius
        )

        markers.append({
            "center": center,
            "outer_radius": outer_radius,
            "red": cand["red"],
            "green": cand["green"],
            "type": marker_type,
            "inner_color": inner_color,
            "middle_color": middle_color,
            "outer_color": outer_color,
            "assigned_id": -1,
            "angle": 0,
            "angle_deg": 0
        })

    markers, hid_center = assign_ids_by_order(markers)

    if hid_center is not None:
        hx, hy = int(hid_center[0]), int(hid_center[1])

        cv2.circle(image, (hx, hy), 7, (0, 255, 255), -1)
        cv2.circle(image, (hx, hy), 12, (0, 255, 255), 2)

        cv2.putText(
            image,
            "HIDAS CENTER",
            (hx + 10, hy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    for m in markers:
        center = m["center"]
        center_r, r_r = m["red"]
        center_g, r_g = m["green"]

        if hid_center is not None:
            hx, hy = int(hid_center[0]), int(hid_center[1])
            cv2.line(image, (hx, hy), center, (0, 255, 255), 1)

        cv2.circle(image, center, 4, (255, 0, 255), -1)
        cv2.circle(image, center_r, int(r_r), (0, 0, 255), 1)
        cv2.circle(image, center_g, int(r_g), (0, 255, 0), 1)

        cv2.putText(
            image,
            f"ID:{m['assigned_id']}",
            (center[0] - 25, center[1] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"{m['angle_deg']:.1f}deg",
            (center[0] - 35, center[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    return image, markers


# =========================
# パラメータ
# =========================
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 15

# =========================
# 入出力動画
# =========================
input_video_path = "mp4_input/test4.mov"
output_video_path = "mp4_output/output_marker_result4.mp4"

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"動画が開けません: {input_video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    processed_frame, markers = process_frame(frame)

    writer.write(processed_frame)

    cv2.imshow("marker result", processed_frame)

    print(f"frame:{frame_count}  検出マーカー数:{len(markers)}")

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"出力動画を保存しました: {output_video_path}")