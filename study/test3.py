import cv2
import numpy as np
import math
import time


# =========================
# パラメータ
# =========================
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 15

CIRCULARITY_THRESHOLD = 0.9
TARGET_ANGLE_DEG = -30

MAX_START_ID_JUMP = 3
MAX_TRACK_DISTANCE = 50

COMMAND_INTERVAL_SEC = 3.0

# =========================
# 入出力動画
# =========================
input_video_path = "mp4_input/test4.mov"
output_video_path = "mp4_output/output_marker_control.mp4"


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


def capture_motion_reference(markers):
    valid_markers = [
        m for m in markers
        if m["assigned_id"] != -1
    ]

    if len(valid_markers) == 0:
        return None

    leftmost = min(valid_markers, key=lambda m: m["center"][0])

    angle_by_id = {
        m["assigned_id"]: m["angle_deg"]
        for m in valid_markers
    }

    return {
        "leftmost_x": leftmost["center"][0],
        "leftmost_id": leftmost["assigned_id"],
        "angle_by_id": angle_by_id
    }


def print_motion_result(markers, motion_reference):
    if motion_reference is None:
        return

    valid_markers = [
        m for m in markers
        if m["assigned_id"] != -1
    ]

    if len(valid_markers) == 0:
        return

    leftmost = min(valid_markers, key=lambda m: m["center"][0])

    current_x = leftmost["center"][0]
    previous_x = motion_reference["leftmost_x"]
    dx = current_x - previous_x

    if dx < 0:
        print(f"左方向に{abs(dx):.1f}px移動しました")
    elif dx > 0:
        print(f"右方向に{dx:.1f}px移動しました")
    else:
        print("左右方向の移動はほぼありません")

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


def process_frame(image, prev_id_positions):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
    red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    green_mask = cv2.inRange(hsv, (40, 60, 0), (80, 255, 255))

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


while True:
    ret, frame = cap.read()

    if not ret:
        break

    processed_frame, markers, circularity, prev_id_positions = process_frame(
        frame,
        prev_id_positions
    )

    writer.write(processed_frame)
    cv2.imshow("marker result", processed_frame)

    if not step1_cleared:
        if circularity is not None and circularity > CIRCULARITY_THRESHOLD:
            print("回転開始")
            step1_cleared = True
        else:
            if not step1_instruction_printed:
                print("加圧してください")
                step1_instruction_printed = True

            if cv2.waitKey(1) & 0xFF == 27:
                break

            continue

    if not step2_first_output_done:
        current_start_id = find_initial_start_id(markers)

        if current_start_id is not None:
            sequence_state = start_new_sequence(current_start_id)
            motion_reference = capture_motion_reference(markers)

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
            print_motion_result(markers, motion_reference)

            if pending_start_id is not None:
                current_start_id = pending_start_id
                sequence_state = start_new_sequence(current_start_id)
                pending_start_id = None

            motion_reference = capture_motion_reference(markers)

            waiting_for_motion_result = False

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()