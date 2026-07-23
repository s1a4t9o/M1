#HIDASのマーカーを検出し、回転モードの指示を出すプログラム（入力：動画）

import cv2
import numpy as np
import math
import time


# =========================
# パラメータ
# =========================
MAX_CENTER_DISTANCE = 4
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 30

# === HSV 色範囲 ===
lower_red1 = np.array([0,   80,   0])
upper_red1 = np.array([20,   255, 255])

lower_red2 = np.array([170, 80,   0])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([50,  30,   20])
upper_green = np.array([90, 255, 255])

CIRCULARITY_THRESHOLD = 0.9
TARGET_ANGLE_DEG = -30

MAX_START_ID_JUMP = 3
MAX_TRACK_DISTANCE = 50

COMMAND_INTERVAL_SEC = 2.0

# 各区間の実測長[cm]
# 真下区間の検出順：ID1→ID16, ID16→ID15, ... , ID2→ID1
# 現在はすべて34 cm。実測後に各値を書き換えてください。
CELL_INTERVAL_LENGTH_CM = 34.0
NOMINAL_CELL_ANGLE_DEG = 360.0 / 16.0
MAX_FRAME_TRAVEL_INTERVALS = 0.30
MIN_COMMON_MARKERS_FOR_TRAVEL = 6

# HIDAS中心から見た画像下方向の角度
BOTTOM_ANGLE_DEG = -90.0

# =========================
# 入出力動画
# =========================
input_video_path = "../mp4_input/raw_camera.mp4"
output_video_path = "../mp4_output/output_marker_control_direction_fixed.mp4"


def find_circles(mask, MIN_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append(((int(x), int(y)), r, area))

    return circles


def get_color_name(hsv_pixel):
    h, s, v = hsv_pixel

    if s < 50 or v < 50:
        return "unknown"

    if (lower_red1[0] <= h <= upper_red1[0]) or (lower_red2[0] <= h <= upper_red2[0]):
        return "red"

    if lower_green[0] <= h <= upper_green[0]:
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

    if inner_color == "green" and middle_color == "red" and outer_color == "green":
        marker_type = 9
    elif inner_color == "green" and outer_color == "red":
        marker_type = 1
    elif inner_color == "red" and outer_color == "green":
        marker_type = 0
    else:
        marker_type = -1

    return marker_type, inner_color, middle_color, outer_color


def correct_id1_id9_by_red_area(markers):
    id1_candidates = [m for m in markers if m["type"] == 1]

    if len(id1_candidates) == 2:
        m1, m2 = id1_candidates

        if m1["red_area"] >= m2["red_area"]:
            m1["type"] = 1
            m2["type"] = 9
        else:
            m1["type"] = 9
            m2["type"] = 1

    return markers


def angle_from_center(point, hid_center):
    x, y = point
    cx, cy = hid_center
    return math.atan2(-(y - cy), x - cx)


def assign_ids_initial(markers):
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

    for i, m in enumerate(markers_sorted):
        offset = (base_index - i) % n
        assigned_id = ((base_id - 1 + offset) % 16) + 1

        if m["type"] == 1:
            assigned_id = 1
        elif m["type"] == 9:
            assigned_id = 9

        m["assigned_id"] = assigned_id

    return markers_sorted, hid_center


def build_id_positions(markers, prev_id_positions=None):
    if prev_id_positions is None:
        id_positions = {}
    else:
        id_positions = dict(prev_id_positions)

    for m in markers:
        if m["assigned_id"] != -1:
            id_positions[m["assigned_id"]] = m["center"]

    return id_positions


def stabilize_ids_by_previous_frame(markers, prev_id_positions):
    if prev_id_positions is None or len(prev_id_positions) == 0:
        return markers, build_id_positions(markers)

    used_indices = set()

    for m in markers:
        m["assigned_id"] = -1

    for prev_id, prev_center in prev_id_positions.items():
        best_index = None
        best_dist = None

        px, py = prev_center

        for i, m in enumerate(markers):
            if i in used_indices:
                continue

            cx, cy = m["center"]
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

            if dist <= MAX_TRACK_DISTANCE:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_index = i

        if best_index is not None:
            markers[best_index]["assigned_id"] = prev_id
            used_indices.add(best_index)

    new_id_positions = build_id_positions(markers, prev_id_positions)

    return markers, new_id_positions


def calculate_overall_circularity(markers):
    valid_markers = [m for m in markers if m["assigned_id"] != -1]

    if len(valid_markers) < 3:
        return None

    points = np.array(
        [m["center"] for m in sorted(valid_markers, key=lambda item: item["angle"])],
        dtype=np.float32
    )

    area = cv2.contourArea(points)
    perimeter = cv2.arcLength(points, True)

    if perimeter == 0:
        return None

    return 4 * math.pi * area / (perimeter ** 2)


def make_cells_from_start_id(start_id):
    pressurize_cells = [
        ((start_id - 1 - i) % 16) + 1
        for i in range(7)
    ]

    depressurize_cells = [
        cell_id for cell_id in range(1, 17)
        if cell_id not in pressurize_cells
    ]

    return pressurize_cells, depressurize_cells


def find_initial_start_id(markers):
    valid_markers = [
        m for m in markers
        if m["assigned_id"] != -1 and m["angle_deg"] > TARGET_ANGLE_DEG
    ]

    if len(valid_markers) == 0:
        return None

    start_marker = min(
        valid_markers,
        key=lambda m: abs(m["angle_deg"] - TARGET_ANGLE_DEG)
    )

    return start_marker["assigned_id"]


def find_new_over_start_id(markers, previous_over_ids):
    current_over_ids = set(
        m["assigned_id"] for m in markers
        if m["assigned_id"] != -1 and m["angle_deg"] > TARGET_ANGLE_DEG
    )

    new_over_ids = current_over_ids - previous_over_ids

    if len(new_over_ids) == 0:
        return None, current_over_ids

    new_over_markers = [
        m for m in markers
        if m["assigned_id"] in new_over_ids
    ]

    start_marker = min(
        new_over_markers,
        key=lambda m: abs(m["angle_deg"] - TARGET_ANGLE_DEG)
    )

    return start_marker["assigned_id"], current_over_ids


def cell_distance_circular(a, b):
    diff = abs(a - b)
    return min(diff, 16 - diff)


def normalize_angle_diff(diff):
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff



def get_bottom_continuous_position(markers):
    """
    真下を挟む隣接2マーカーと、その区間内での割合を求める。

    連続位置の定義:
      0.0～1.0   : ID1 → ID16 の区間
      1.0～2.0   : ID16 → ID15 の区間
      ...
      15.0～16.0 : ID2 → ID1 の区間

    fraction=0.0 は区間始点側、1.0 は区間終点側。
    """
    marker_by_id = {
        m["assigned_id"]: m
        for m in markers
        if m["assigned_id"] != -1
    }

    for interval_index in range(16):
        start_id = ((1 - interval_index - 1) % 16) + 1
        end_id = ((1 - (interval_index + 1) - 1) % 16) + 1

        if start_id not in marker_by_id or end_id not in marker_by_id:
            continue

        start_marker = marker_by_id[start_id]
        end_marker = marker_by_id[end_id]
        start_angle = start_marker["angle_deg"]
        end_angle = end_marker["angle_deg"]

        interval_angle = (end_angle - start_angle) % 360.0
        bottom_from_start = (BOTTOM_ANGLE_DEG - start_angle) % 360.0

        # 16セルなので通常は約22.5deg。変形を考慮して90degまで許容する。
        if interval_angle <= 0.0 or interval_angle > 90.0:
            continue

        if bottom_from_start <= interval_angle:
            fraction = bottom_from_start / interval_angle
            continuous_position = interval_index + fraction

            return {
                "interval_index": interval_index,
                "fraction": fraction,
                "position": continuous_position,
                "start_id": start_id,
                "end_id": end_id,
                "start_marker": start_marker,
                "end_marker": end_marker,
            }

    return None


def normalize_interval_delta(delta):
    """16区間の円環上の差を -8～+8区間に正規化する。"""
    while delta > 8.0:
        delta -= 16.0
    while delta <= -8.0:
        delta += 16.0
    return delta


def get_marker_angle_map(markers):
    return {
        m["assigned_id"]: m["angle_deg"]
        for m in markers
        if m["assigned_id"] != -1
    }


def update_travel_tracker(markers, previous_angle_by_id, accumulated_intervals):
    """
    各フレームにおける全マーカーの角度変化量の中央値から、
    符号付きの移動区間数を積算する。

    真下区間番号の飛びを直接加算しないため、ID再割当てや
    区間誤認識による1～2区間の誤加算を防ぐ。

    正方向:
      HIDASが画像上で左へ進む向き
    逆方向:
      HIDASが画像上で右へ進む向き
    """
    current_angle_by_id = get_marker_angle_map(markers)

    if len(current_angle_by_id) == 0:
        return previous_angle_by_id, accumulated_intervals

    if previous_angle_by_id is None:
        return current_angle_by_id, accumulated_intervals

    common_ids = sorted(
        set(previous_angle_by_id.keys()) &
        set(current_angle_by_id.keys())
    )

    if len(common_ids) < MIN_COMMON_MARKERS_FOR_TRAVEL:
        return current_angle_by_id, accumulated_intervals

    angle_deltas = []

    for cell_id in common_ids:
        delta_deg = normalize_angle_diff(
            current_angle_by_id[cell_id] -
            previous_angle_by_id[cell_id]
        )

        # ID付け替えなどで大きく飛んだマーカーは除外
        if abs(delta_deg) <= NOMINAL_CELL_ANGLE_DEG * 0.75:
            angle_deltas.append(delta_deg)

    if len(angle_deltas) < MIN_COMMON_MARKERS_FOR_TRAVEL:
        return current_angle_by_id, accumulated_intervals

    median_angle_delta_deg = float(np.median(angle_deltas))

    # 画像左方向への移動を正とする。
    # この角度定義では、左へ進むとマーカー角度変化が正になる。
    frame_interval_delta = median_angle_delta_deg / NOMINAL_CELL_ANGLE_DEG

    # 1フレームで0.30区間を超える値は検出・ID対応の異常として無視
    if abs(frame_interval_delta) <= MAX_FRAME_TRAVEL_INTERVALS:
        accumulated_intervals += frame_interval_delta

    return current_angle_by_id, accumulated_intervals


def capture_motion_reference(markers):
    valid_markers = [
        m for m in markers
        if m["assigned_id"] != -1
    ]

    if len(valid_markers) == 0:
        return None

    angle_by_id = {
        m["assigned_id"]: m["angle_deg"]
        for m in valid_markers
    }

    bottom_info = get_bottom_continuous_position(valid_markers)

    return {
        "bottom_info": bottom_info,
        "bottom_position": bottom_info["position"] if bottom_info is not None else None,
        "angle_by_id": angle_by_id
    }


def print_angle_changes(markers, motion_reference):
    if motion_reference is None:
        return

    valid_markers = [
        m for m in markers
        if m["assigned_id"] != -1
    ]

    if len(valid_markers) == 0:
        return

    print("角度変化量:")
    previous_angles = motion_reference["angle_by_id"]

    for m in sorted(valid_markers, key=lambda item: item["assigned_id"]):
        cell_id = m["assigned_id"]

        if cell_id not in previous_angles:
            continue

        angle_diff = normalize_angle_diff(
            m["angle_deg"] - previous_angles[cell_id]
        )
        print(f"セル{cell_id}: {angle_diff:.1f}deg")


def draw_motion_result(image, motion_result_text):
    if not motion_result_text:
        return image

    height, width = image.shape[:2]

    cv2.putText(
        image,
        motion_result_text,
        (20, height - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )

    return image


def start_new_sequence(start_id):
    pressurize_cells, depressurize_cells = make_cells_from_start_id(start_id)

    print(f"セル{depressurize_cells}を減圧してください")

    return {
        "pressurize_cells": pressurize_cells,
        "sequence_index": 0,
        "last_command_time": 0,
        "cycle_count": 0
    }


def run_pressurize_sequence(sequence_state):
    if sequence_state is None:
        return sequence_state, False

    pressurize_cells = sequence_state["pressurize_cells"]
    sequence_index = sequence_state["sequence_index"]
    last_command_time = sequence_state["last_command_time"]
    cycle_count = sequence_state["cycle_count"]

    if cycle_count == 0:
        current_cycle_cells = pressurize_cells
    else:
        current_cycle_cells = pressurize_cells[:4]

    if len(current_cycle_cells) == 0:
        return sequence_state, False

    now = time.time()

    if now - last_command_time < COMMAND_INTERVAL_SEC:
        return sequence_state, False

    cell_id = current_cycle_cells[sequence_index]
    print(f"セル{cell_id}を加圧してください")

    last_command_time = now
    sequence_index += 1

    one_cycle_finished = False

    if sequence_index >= len(current_cycle_cells):
        sequence_index = 0
        cycle_count += 1
        one_cycle_finished = True

    sequence_state["sequence_index"] = sequence_index
    sequence_state["last_command_time"] = last_command_time
    sequence_state["cycle_count"] = cycle_count

    return sequence_state, one_cycle_finished


def draw_neighbor_distances(image, markers):
    if len(markers) < 2:
        return image

    markers_sorted = sorted(markers, key=lambda m: m["angle"])
    n = len(markers_sorted)

    for i in range(n):
        m1 = markers_sorted[i]
        m2 = markers_sorted[(i + 1) % n]

        x1, y1 = m1["center"]
        x2, y2 = m2["center"]

        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 1)

    return image


def draw_circularity_label(image, circularity):
    if circularity is None:
        return image

    _, width = image.shape[:2]
    x = max(10, width - 185)
    y = 30

    cv2.putText(
        image,
        f"Circularity: {circularity:.3f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return image




def draw_bottom_reference(image, markers, hid_center):
    if hid_center is None:
        return image

    hx, hy = int(hid_center[0]), int(hid_center[1])
    h, _ = image.shape[:2]

    # HIDAS中心から画像下方向へ真下基準線を描画
    cv2.line(image, (hx, hy), (hx, h - 1), (0, 0, 255), 3)

    bottom_info = get_bottom_continuous_position(markers)
    if bottom_info is None:
        return image

    start_marker = bottom_info["start_marker"]
    end_marker = bottom_info["end_marker"]
    fraction = bottom_info["fraction"]

    for marker in (start_marker, end_marker):
        x, y = marker["center"]
        cv2.circle(image, (x, y), 12, (0, 255, 255), 3)
        cv2.line(image, (hx, hy), (x, y), (0, 255, 0), 2)

    cv2.line(
        image,
        start_marker["center"],
        end_marker["center"],
        (255, 0, 255),
        2
    )

    cv2.putText(
        image,
        f"BOTTOM: ID{bottom_info['start_id']}-ID{bottom_info['end_id']}  {fraction * 100.0:.1f}%",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    return image


def process_frame(image, prev_id_positions):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    red_circles = find_circles(red_mask, MIN_AREA)
    green_circles = find_circles(green_mask, MIN_AREA)

    candidates = []

    for center_r, r_r, area_r in red_circles:
        for center_g, r_g, area_g in green_circles:
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
                    "green": (center_g, r_g),
                    "red_area": area_r,
                    "green_area": area_g
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
            "red_area": cand["red_area"],
            "green_area": cand["green_area"],
            "type": marker_type,
            "inner_color": inner_color,
            "middle_color": middle_color,
            "outer_color": outer_color,
            "assigned_id": -1,
            "angle": 0,
            "angle_deg": 0
        })

    markers = correct_id1_id9_by_red_area(markers)

    markers, hid_center = assign_ids_initial(markers)

    if prev_id_positions is not None:
        markers, id_positions = stabilize_ids_by_previous_frame(markers, prev_id_positions)
    else:
        id_positions = build_id_positions(markers)

    circularity = calculate_overall_circularity(markers)

    image = draw_neighbor_distances(image, markers)
    image = draw_circularity_label(image, circularity)

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

        image = draw_bottom_reference(image, markers, hid_center)

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

    return image, markers, circularity, id_positions


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


step1_cleared = False
step1_instruction_printed = False
step2_first_output_done = False

previous_over_ids = set()
current_start_id = None

prev_id_positions = None

sequence_state = None
pending_start_id = None
motion_reference = None

waiting_for_motion_result = False
motion_result_wait_start = 0

motion_result_text = ""
travel_previous_angle_by_id = None
accumulated_travel_intervals = 0.0
travel_tracking_started = False


while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame, markers, circularity, prev_id_positions = process_frame(
        frame,
        prev_id_positions
    )

    if not step1_cleared:
        if circularity is not None and circularity > CIRCULARITY_THRESHOLD:
            print("回転開始")
            step1_cleared = True
        else:
            if not step1_instruction_printed:
                print("加圧してください")
                step1_instruction_printed = True

            processed_frame = draw_motion_result(processed_frame, motion_result_text)
            writer.write(processed_frame)
            cv2.imshow("marker result", processed_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            continue

    if not step2_first_output_done:
        current_start_id = find_initial_start_id(markers)

        if current_start_id is not None:
            sequence_state = start_new_sequence(current_start_id)
            motion_reference = capture_motion_reference(markers)
            travel_previous_angle_by_id = get_marker_angle_map(markers)
            accumulated_travel_intervals = 0.0
            travel_tracking_started = len(travel_previous_angle_by_id) >= MIN_COMMON_MARKERS_FOR_TRAVEL

            previous_over_ids = set(
                m["assigned_id"] for m in markers
                if m["assigned_id"] != -1 and m["angle_deg"] > TARGET_ANGLE_DEG
            )

            step2_first_output_done = True

    else:
        new_start_id, current_over_ids = find_new_over_start_id(
            markers,
            previous_over_ids
        )

        previous_over_ids = current_over_ids

        if new_start_id is not None and new_start_id != current_start_id:
            jump = cell_distance_circular(current_start_id, new_start_id)

            if jump < MAX_START_ID_JUMP:
                pending_start_id = new_start_id

    if travel_tracking_started:
        travel_previous_angle_by_id, accumulated_travel_intervals = update_travel_tracker(
            markers,
            travel_previous_angle_by_id,
            accumulated_travel_intervals
        )

    if not waiting_for_motion_result:
        sequence_state, one_cycle_finished = run_pressurize_sequence(sequence_state)
    else:
        one_cycle_finished = False

    if one_cycle_finished:
        waiting_for_motion_result = True
        motion_result_wait_start = time.time()

    if waiting_for_motion_result:
        now = time.time()

        if now - motion_result_wait_start >= COMMAND_INTERVAL_SEC:
            print_angle_changes(markers, motion_reference)

            if pending_start_id is not None:
                current_start_id = pending_start_id
                sequence_state = start_new_sequence(current_start_id)
                pending_start_id = None

            motion_reference = capture_motion_reference(markers)

            waiting_for_motion_result = False

    processed_frame = draw_motion_result(processed_frame, motion_result_text)

    writer.write(processed_frame)
    cv2.imshow("marker result", processed_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()
moved_intervals = accumulated_travel_intervals if travel_tracking_started else None
final_distance_cm = (
    moved_intervals * CELL_INTERVAL_LENGTH_CM
    if moved_intervals is not None
    else None
)

if moved_intervals is None or final_distance_cm is None:
    print("最終移動距離を計算できませんでした（マーカーの回転量を連続追跡できませんでした）")
else:
    print(f"移動区間数: {moved_intervals:.2f}区間")
    print(f"最終移動距離: {final_distance_cm:.2f}cm")
