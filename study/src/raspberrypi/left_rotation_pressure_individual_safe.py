#HIDASの左回転自動化

import cv2
import numpy as np
import math
import time
import socket
import os
from picamera2 import Picamera2
from libcamera import Transform


# =========================
# パラメータ
# =========================
MAX_CENTER_DISTANCE = 5
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 18

# =========================
# HSV色閾値
# =========================
lower_red1 = np.array([0,   80,   0])
upper_red1 = np.array([20,   255, 255])

lower_red2 = np.array([170, 80,   0])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([55,  30,   0])
upper_green = np.array([90, 255, 255])


CIRCULARITY_THRESHOLD = 0.75
TARGET_ANGLE_DEG = -30

MAX_START_ID_JUMP = 3
MAX_TRACK_DISTANCE = 50

COMMAND_INTERVAL_SEC = 3.0  # 通常の加圧時間
FIRST_CELL_FROM_SECOND_SEQUENCE_SEC = 15.0  # 2サイクル目以降、各サイクル最初の加圧だけに使う時間
IDLE_INTERVAL_SEC = 1.0  # Idle指示後、次の加圧までの待ち時間
DEPRESSURIZE_INTERVAL_SEC = 3.0  # 減圧時間

# =========================
# 真下セル通過による移動距離推定
# =========================
CELL_INTERVAL_LENGTH_CM = 34.0
BOTTOM_ANGLE_DEG = -90.0
BOTTOM_INTERVAL_STABLE_FRAMES = 3

# =========================
# TCP通信設定
# =========================
HOST = "192.168.10.4"
PORT = 60001
SOCKET_TIMEOUT_SEC = 5.0

# 減圧を許可する最低圧力[kPa]
# 各ユニット内の3セルを個別に判定し、この値を超えるセルだけVENTする。
MIN_VENT_PRESSURE_KPA = 0.5

# 1セルあたりの3桁の動作番号
CELL_IDLE = "000"
CELL_INF = "111"
CELL_VENT = "222"

# 接続済みソケットを保持する
hidas_sock = None

# =========================
# Raspberry Piカメラ設定
# =========================
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

# 画面表示時だけ縮小するサイズ
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

# カメラ画像を180度回転する設定
CAMERA_HFLIP = True
CAMERA_VFLIP = True

# 動画保存時のFPS
OUTPUT_VIDEO_FPS = 30.0

# 処理結果を動画保存するか
SAVE_OUTPUT_VIDEO = False
output_video_path = "mp4_output/output_marker_control.mp4"


def connect_hidas():
    """HIDAS制御サーバーへTCP接続する。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT_SEC)

    print(f"{HOST}:{PORT} に接続します")
    sock.connect((HOST, PORT))
    print("TCP接続成功")

    return sock


def make_cmb_command(
    pressurize_cells=None,
    depressurize_cells=None,
    depressurize_states=None,
):
    """
    16ユニット分のCMBコマンドを作る。

    加圧ユニット         : 111
    通常の減圧ユニット   : 222
    個別減圧指定         : 例 220, 202, 020
    それ以外             : 000

    depressurize_statesは、
        {ユニット番号: 3桁の状態文字列}
    の辞書で指定する。
    """
    pressurize_set = set(pressurize_cells or [])
    depressurize_set = set(depressurize_cells or [])
    depressurize_states = dict(depressurize_states or {})

    duplicate_cells = pressurize_set & (depressurize_set | set(depressurize_states))
    if duplicate_cells:
        raise ValueError(
            f"加圧と減圧の両方に指定されたユニットがあります: "
            f"{sorted(duplicate_cells)}"
        )

    all_cells = pressurize_set | depressurize_set | set(depressurize_states)
    invalid_cells = sorted(
        cell_id for cell_id in all_cells
        if not 1 <= cell_id <= 16
    )
    if invalid_cells:
        raise ValueError(
            f"ユニット番号は1～16で指定してください: {invalid_cells}"
        )

    for unit_id, state in depressurize_states.items():
        if len(state) != 3 or any(char not in "02" for char in state):
            raise ValueError(
                f"ユニット{unit_id}の個別減圧状態が不正です: {state}"
            )

    states = []

    for cell_id in range(1, 17):
        if cell_id in pressurize_set:
            states.append(CELL_INF)
        elif cell_id in depressurize_states:
            states.append(depressurize_states[cell_id])
        elif cell_id in depressurize_set:
            states.append(CELL_VENT)
        else:
            states.append(CELL_IDLE)

    return "CMB " + " ".join(states)


def send_hidas_command(command):
    """コマンドを送信し、応答文字列を返す。"""
    global hidas_sock

    if hidas_sock is None:
        raise RuntimeError("HIDASとのTCP接続がありません")

    message = command.rstrip("\n") + "\n"
    hidas_sock.sendall(message.encode("utf-8"))
    print(f"送信: {command}")

    try:
        response = hidas_sock.recv(8192)
        if response:
            response_text = response.decode("utf-8", errors="replace").strip()
            print(f"受信: {response_text}")
            return response_text
    except socket.timeout:
        print("受信: タイムアウト（応答なし）")

    return None


def parse_sma_pressures(response):
    """
    CMAに対するSMA応答を解析する。

    戻り値:
        {
            1: [pressure1, pressure2, pressure3],
            ...
            16: [pressure1, pressure2, pressure3]
        }
    """
    if response is None:
        raise ValueError("SMA応答がありません")

    tokens = response.strip().split()

    if len(tokens) < 3 or tokens[0] != "SMA":
        raise ValueError(f"SMA応答ではありません: {response}")

    unit_count = int(tokens[1])
    expected_length = 3 + unit_count * 9

    if len(tokens) != expected_length:
        raise ValueError(
            f"SMA応答の項目数が不正です: "
            f"期待={expected_length}, 実際={len(tokens)}"
        )

    pressures_by_cell = {}
    index = 3

    for cell_id in range(1, unit_count + 1):
        # 各ユニットの並び:
        # acc_x acc_y acc_z mode1 pressure1 mode2 pressure2 mode3 pressure3
        pressure_1 = float(tokens[index + 4])
        pressure_2 = float(tokens[index + 6])
        pressure_3 = float(tokens[index + 8])

        pressures_by_cell[cell_id] = [
            pressure_1,
            pressure_2,
            pressure_3,
        ]

        index += 9

    return pressures_by_cell


def request_all_pressures():
    """CMAを送り、16セル分の3つの圧力を取得する。"""
    response = send_hidas_command("CMA")
    return parse_sma_pressures(response)


def make_depressurize_states_by_pressure(depressurize_cells):
    """
    減圧予定ユニットについてCMAで圧力を確認し、
    ユニット内の3セルを個別に判定した3桁状態を返す。

    各セルについて、
        圧力 > 1 kPa  : 2（VENT）
        圧力 <= 1 kPa : 0（Idle）

    例:
        [2.15, 1.45, 0.20] kPa -> "220"

    圧力取得に失敗した場合は、安全のため全セルをIdleにする。
    """
    requested_units = list(depressurize_cells or [])

    if len(requested_units) == 0:
        return {}

    try:
        pressures_by_cell = request_all_pressures()
    except (ValueError, TypeError, OSError) as exc:
        print(f"圧力取得に失敗したため減圧を中止します: {exc}")
        return {unit_id: CELL_IDLE for unit_id in requested_units}

    states_by_unit = {}

    for unit_id in requested_units:
        pressures = pressures_by_cell.get(unit_id)

        if pressures is None or len(pressures) != 3:
            print(f"ユニット{unit_id}: 圧力データなし → 000")
            states_by_unit[unit_id] = CELL_IDLE
            continue

        state_digits = []

        for cell_no, pressure in enumerate(pressures, start=1):
            if pressure > MIN_VENT_PRESSURE_KPA:
                state_digits.append("2")
                result = "VENT"
            else:
                state_digits.append("0")
                result = "Idle"

            print(
                f"ユニット{unit_id}・セル{cell_no}: "
                f"{pressure:.2f} kPa → {result}"
            )

        state = "".join(state_digits)
        states_by_unit[unit_id] = state
        print(f"ユニット{unit_id}: 個別減圧指示 → {state}")

    return states_by_unit


def send_cell_command(pressurize_cells=None, depressurize_cells=None):
    """
    ユニット指定からCMBコマンドを作成して送信する。

    減圧ユニットが指定された場合は、CMB送信前にCMAで圧力を取得し、
    各ユニット内の3セルを個別判定して3桁の指示を作る。
    """
    depressurize_states = make_depressurize_states_by_pressure(
        depressurize_cells
    )

    command = make_cmb_command(
        pressurize_cells=pressurize_cells,
        depressurize_states=depressurize_states,
    )
    send_hidas_command(command)


def send_all_idle():
    """全16セルをIdleにする。"""
    send_cell_command()


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



def circular_positive_angle_diff(start_deg, end_deg):
    """start_degから反時計回りにend_degまで進む正の角度差[0, 360)。"""
    return (end_deg - start_deg) % 360.0


def get_bottom_continuous_position(markers):
    """
    画像左方向への移動を正方向として、真下位置を連続値で求める。

    positionの範囲:
        0.0～1.0   : ID1 → ID2
        1.0～2.0   : ID2 → ID3
        ...
        15.0～16.0 : ID16 → ID1

    fraction=0.0は区間始点側、1.0は区間終点側。
    """
    marker_by_id = {
        m["assigned_id"]: m
        for m in markers
        if m["assigned_id"] != -1
    }

    for interval_index in range(16):
        start_id = interval_index + 1
        end_id = (start_id % 16) + 1

        if start_id not in marker_by_id or end_id not in marker_by_id:
            continue

        start_marker = marker_by_id[start_id]
        end_marker = marker_by_id[end_id]
        start_angle = start_marker["angle_deg"]
        end_angle = end_marker["angle_deg"]

        interval_angle = (start_angle - end_angle) % 360.0
        bottom_from_start = (start_angle - BOTTOM_ANGLE_DEG) % 360.0

        # 通常は約22.5deg。変形を考慮して90degまで許容する。
        if interval_angle <= 0.0 or interval_angle > 90.0:
            continue

        if bottom_from_start <= interval_angle:
            fraction = bottom_from_start / interval_angle
            return {
                "interval_index": interval_index,
                "fraction": fraction,
                "position": interval_index + fraction,
                "start_id": start_id,
                "end_id": end_id,
                "start_marker": start_marker,
                "end_marker": end_marker,
            }

    return None


def normalize_interval_step(step):
    """隣接区間への移動を、左方向+1・右方向-1として返す。"""
    if step == 1 or step == -15:
        return 1
    if step == -1 or step == 15:
        return -1
    return None


def initialize_boundary_tracker(markers):
    """開始時の区間番号と区間内割合を記録する。"""
    bottom_info = get_bottom_continuous_position(markers)
    if bottom_info is None:
        return None

    print(
        f"移動距離推定開始: 真下位置はセル{bottom_info['start_id']}-"
        f"セル{bottom_info['end_id']}間の{bottom_info['fraction'] * 100.0:.1f}%"
    )

    return {
        "start_fraction": bottom_info["fraction"],
        "stable_interval": bottom_info["interval_index"],
        "current_fraction": bottom_info["fraction"],
        "crossed_intervals": 0,
        "candidate_interval": None,
        "candidate_count": 0,
    }


def update_boundary_tracker(markers, tracker):
    """
    真下区間が隣接区間へ安定して移ったときだけ通過数を更新し、
    現在の区間内割合は毎フレーム更新する。
    """
    if tracker is None:
        return initialize_boundary_tracker(markers)

    bottom_info = get_bottom_continuous_position(markers)
    if bottom_info is None:
        return tracker

    current_interval = bottom_info["interval_index"]

    if current_interval == tracker["stable_interval"]:
        tracker["current_fraction"] = bottom_info["fraction"]
        tracker["candidate_interval"] = None
        tracker["candidate_count"] = 0
        return tracker

    if current_interval == tracker["candidate_interval"]:
        tracker["candidate_count"] += 1
    else:
        tracker["candidate_interval"] = current_interval
        tracker["candidate_count"] = 1

    if tracker["candidate_count"] < BOTTOM_INTERVAL_STABLE_FRAMES:
        return tracker

    raw_step = current_interval - tracker["stable_interval"]
    direction = normalize_interval_step(raw_step)

    # 隣接区間以外への飛びはID誤認識として採用しない。
    if direction is None:
        print(
            f"真下区間の急変を無視しました: "
            f"{tracker['stable_interval']} → {current_interval}"
        )
        tracker["candidate_interval"] = None
        tracker["candidate_count"] = 0
        return tracker

    previous_interval = tracker["stable_interval"]
    tracker["crossed_intervals"] += direction
    tracker["stable_interval"] = current_interval
    tracker["current_fraction"] = bottom_info["fraction"]
    tracker["candidate_interval"] = None
    tracker["candidate_count"] = 0

    print(
        f"真下区間通過: 区間{previous_interval + 1} → 区間{current_interval + 1}"
    )
    return tracker


def calculate_boundary_distance(tracker):
    """区間通過数と開始・現在の区間内割合から移動距離を計算する。"""
    if tracker is None:
        return None, None

    moved_intervals = (
        tracker["crossed_intervals"]
        + tracker["current_fraction"]
        - tracker["start_fraction"]
    )
    distance_cm = moved_intervals * CELL_INTERVAL_LENGTH_CM
    return moved_intervals, distance_cm


def print_boundary_distance(tracker, label="現在"):
    """真下位置の連続補間を使った移動距離を表示する。"""
    moved_intervals, distance_cm = calculate_boundary_distance(tracker)

    if moved_intervals is None or distance_cm is None:
        print(f"{label}の推定移動距離: まだ真下位置を確定できていません")
        return

    if moved_intervals > 0:
        direction = "画像左方向"
    elif moved_intervals < 0:
        direction = "画像右方向"
    else:
        direction = "移動なし"

    print(
        f"{label}の推定移動距離（真下位置補間法）: "
        f"{abs(distance_cm):.2f} cm "
        f"（{abs(moved_intervals):.2f}セル区間、{direction}）"
    )


def start_new_sequence(start_id, sequence_number):
    """
    新しいサイクルを開始する。

    ここでいうサイクルは、
    「セル[...]を減圧してください」
    と出力されてから、次の減圧指示が出るまでを指す。
    """
    pressurize_cells, depressurize_cells = make_cells_from_start_id(start_id)

    print(f"セル{depressurize_cells}を減圧してください")
    send_cell_command(depressurize_cells=depressurize_cells)

    return {
        "pressurize_cells": pressurize_cells,

        # 次の減圧指示が出るまで、この減圧状態を維持する
        "depressurize_cells": depressurize_cells,

        "sequence_index": 0,
        "cycle_count": 0,
        "waiting_after_depressurize": True,
        "depressurize_command_time": time.time(),
        "active_pressurize_cell": None,
        "pressurize_start_time": None,
        "waiting_after_idle": False,
        "idle_start_time": None,
        "active_pressurize_duration": None,

        # 減圧指示を基準に数えたサイクル番号（1, 2, 3, ...）
        "sequence_number": sequence_number,

        # このサイクル内で最初の加圧をすでに行ったか
        "first_pressurize_done": False
    }


def run_pressurize_sequence(sequence_state):
    """
    1セルずつ、次の順番で指示を出す。

    加圧指示
        ↓ 加圧時間
    同じセルをIdleにする指示
        ↓ IDLE_INTERVAL_SEC 秒間待機
    次のセルの加圧指示

    2サイクル目以降は、各サイクルの本当の最初の加圧だけを長くする。
    同じサイクル内の2回目以降の加圧は通常時間とする。

    戻り値のone_cycle_finishedは、内部の加圧セル列を1周し、
    最後のセルをIdleにした時点でTrueになる。
    """
    if sequence_state is None:
        return sequence_state, False

    now = time.time()

    # 最初に減圧指示を出してから、指定時間だけ待つ
    if sequence_state.get("waiting_after_depressurize", False):
        depressurize_command_time = sequence_state["depressurize_command_time"]

        if now - depressurize_command_time < DEPRESSURIZE_INTERVAL_SEC:
            return sequence_state, False

        sequence_state["waiting_after_depressurize"] = False

    pressurize_cells = sequence_state["pressurize_cells"]
    depressurize_cells = sequence_state["depressurize_cells"]
    sequence_index = sequence_state["sequence_index"]
    cycle_count = sequence_state["cycle_count"]
    active_cell = sequence_state["active_pressurize_cell"]
    pressurize_start_time = sequence_state["pressurize_start_time"]
    active_pressurize_duration = sequence_state["active_pressurize_duration"]

    # 内部の1周目は7セル、2周目以降は先頭4セルを対象にする
    if cycle_count == 0:
        current_cycle_cells = pressurize_cells
    else:
        current_cycle_cells = pressurize_cells[:4]

    if len(current_cycle_cells) == 0:
        return sequence_state, False

    # 現在加圧中のセルがある場合、加圧時間が終わるまで待つ
    if active_cell is not None:
        if now - pressurize_start_time < active_pressurize_duration:
            return sequence_state, False

        # 加圧時間終了後、必ず同じセルをIdleに戻す
        print(f"セル{active_cell}をアイドル状態にしてください")

        # 加圧していたセルだけをIdleに戻し、
        # 減圧対象セルは222のまま維持する
        send_cell_command(
            depressurize_cells=depressurize_cells
        )

        sequence_state["active_pressurize_cell"] = None
        sequence_state["pressurize_start_time"] = None
        sequence_state["active_pressurize_duration"] = None

        sequence_index += 1

        # 最後のセルをIdleにしたら、内部の1周は終了
        if sequence_index >= len(current_cycle_cells):
            sequence_state["sequence_index"] = 0
            sequence_state["cycle_count"] = cycle_count + 1
            sequence_state["waiting_after_idle"] = False
            sequence_state["idle_start_time"] = None
            return sequence_state, True

        # 次のセルへ進む前に、Idle状態で待つ
        sequence_state["sequence_index"] = sequence_index
        sequence_state["waiting_after_idle"] = True
        sequence_state["idle_start_time"] = now
        return sequence_state, False

    # Idle指示を出した後、指定時間が経つまで次の加圧をしない
    if sequence_state.get("waiting_after_idle", False):
        idle_start_time = sequence_state["idle_start_time"]

        if now - idle_start_time < IDLE_INTERVAL_SEC:
            return sequence_state, False

        sequence_state["waiting_after_idle"] = False
        sequence_state["idle_start_time"] = None

    # 次のセルを加圧する
    current_sequence_index = sequence_state["sequence_index"]
    cell_id = current_cycle_cells[current_sequence_index]

    # 減圧指示を基準とした2サイクル目以降で、
    # そのサイクル内でまだ一度も加圧していない場合だけ長くする。
    if (
        sequence_state["sequence_number"] >= 2
        and not sequence_state["first_pressurize_done"]
    ):
        pressurize_duration = FIRST_CELL_FROM_SECOND_SEQUENCE_SEC
    else:
        pressurize_duration = COMMAND_INTERVAL_SEC

    print(f"セル{cell_id}を加圧してください")

    # 現在の加圧セルを111にしながら、
    # 減圧対象セルは222のまま維持する
    send_cell_command(
        pressurize_cells=[cell_id],
        depressurize_cells=depressurize_cells
    )

    sequence_state["active_pressurize_cell"] = cell_id
    sequence_state["pressurize_start_time"] = now
    sequence_state["active_pressurize_duration"] = pressurize_duration
    sequence_state["first_pressurize_done"] = True

    return sequence_state, False

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

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

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

    # まず、そのフレーム単体でIDを付与する
    markers, hid_center = assign_ids_initial(markers)

    # 前フレームのID位置がある場合は、一度トラッキングでIDを安定化する
    # ただし、トラッキング結果に assigned_id == -1 が出た場合は、
    # トラッキング失敗として、そのフレーム単体のID付与に戻す。
    if prev_id_positions is not None:
        tracked_markers, tracked_id_positions = stabilize_ids_by_previous_frame(
            markers,
            prev_id_positions
        )

        tracking_failed = any(m["assigned_id"] == -1 for m in tracked_markers)

        if tracking_failed:
            # 追跡でIDが付かなかった場合は、前フレーム追跡を捨ててIDを付け直す
            markers, hid_center = assign_ids_initial(markers)
            id_positions = build_id_positions(markers)

            cv2.putText(
                image,
                "TRACKING FAILED -> REASSIGN ID",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        else:
            markers = tracked_markers
            id_positions = tracked_id_positions
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


picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "format": "XRGB8888",
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT)
    },
    transform=Transform(
        hflip=CAMERA_HFLIP,
        vflip=CAMERA_VFLIP
    )
)

picam2.configure(camera_config)
picam2.start()

writer = None

if SAVE_OUTPUT_VIDEO:
    output_dir = os.path.dirname(output_video_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_video_path,
        fourcc,
        OUTPUT_VIDEO_FPS,
        (CAMERA_WIDTH, CAMERA_HEIGHT)
    )

    if not writer.isOpened():
        picam2.stop()
        cv2.destroyAllWindows()
        raise RuntimeError(f"出力動画を開けません: {output_video_path}")

print(
    f"Raspberry Piカメラ準備完了: "
    f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}, "
    f"保存FPS={OUTPUT_VIDEO_FPS:.1f}"
)
print("Enterキーを押すと回転モードを開始します")
input()

try:
    hidas_sock = connect_hidas()
except OSError as exc:
    picam2.stop()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    raise RuntimeError(f"HIDASにTCP接続できませんでした: {exc}") from exc

print("回転モード開始")
print("ESCキーで終了します")


step1_cleared = False
step1_instruction_printed = False
step2_first_output_done = False

current_start_id = None

prev_id_positions = None

sequence_state = None
pending_start_id = None
motion_reference = None

waiting_for_motion_result = False
motion_result_wait_start = 0

waiting_for_next_cycle = False
next_cycle_wait_start = 0

# 減圧指示が出た回数を基準にしたサイクル番号
sequence_number = 0

# 真下セル通過法の移動距離推定状態
boundary_tracker = None
latest_markers = []


try:
    while True:
        # Raspberry PiカメラからXRGB8888画像を取得
        captured_frame = picam2.capture_array()

        # 4チャンネル画像をOpenCV用BGR画像へ変換
        frame = cv2.cvtColor(
            captured_frame,
            cv2.COLOR_BGRA2BGR
        )

        processed_frame, markers, circularity, prev_id_positions = process_frame(
            frame,
            prev_id_positions
        )

        latest_markers = markers

        # 回転開始後は、真下を通過したセル区間を毎フレーム監視する。
        if step1_cleared:
            boundary_tracker = update_boundary_tracker(markers, boundary_tracker)

        if writer is not None:
            writer.write(processed_frame)

        display_frame = cv2.resize(
            processed_frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        cv2.imshow("marker result", display_frame)

        if not step1_cleared:
            valid_marker_count = sum(
                1 for m in markers
                if m["assigned_id"] != -1
            )

            # 検出マーカーが5個未満の場合はHIDASとして扱わない
            if valid_marker_count < 16:
                cv2.putText(
                    processed_frame,
                    "HIDAS NOT FOUND",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                if not step1_instruction_printed:
                    print("HIDASが見つかりません")
                    step1_instruction_printed = True

                cv2.imshow("marker result", display_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                continue

            # 5個以上検出できた場合は、従来どおり円形度で加圧/回転開始を判断
            if circularity is None:
                cv2.putText(
                    processed_frame,
                    "CIRCULARITY ERROR",
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                cv2.imshow("marker result", display_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                continue

            if circularity > CIRCULARITY_THRESHOLD:
                if step1_instruction_printed:
                    print("全セルをアイドル状態にします")
                    send_all_idle()

                print("回転開始")
                step1_cleared = True
            else:
                if not step1_instruction_printed:
                    print("HIDASを加圧してください")
                    send_cell_command(pressurize_cells=list(range(1, 17)))
                    step1_instruction_printed = True

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                continue

        if not step2_first_output_done:
            current_start_id = find_initial_start_id(markers)

            if current_start_id is not None:
                sequence_number += 1
                sequence_state = start_new_sequence(
                    current_start_id,
                    sequence_number
                )
                motion_reference = capture_motion_reference(markers)

                step2_first_output_done = True

        if not waiting_for_motion_result and not waiting_for_next_cycle:
            sequence_state, one_cycle_finished = run_pressurize_sequence(sequence_state)
        else:
            one_cycle_finished = False

        if one_cycle_finished:
            # 内部の加圧セル列を1周した時点の最新角度から、次の開始セル候補を決める
            latest_start_id = find_initial_start_id(markers)

            if latest_start_id is not None and latest_start_id != current_start_id:
                jump = cell_distance_circular(current_start_id, latest_start_id)

                if jump < MAX_START_ID_JUMP:
                    pending_start_id = latest_start_id

            waiting_for_motion_result = True
            motion_result_wait_start = time.time()

        if waiting_for_motion_result:
            now = time.time()

            if now - motion_result_wait_start >= COMMAND_INTERVAL_SEC:
                print_motion_result(markers, motion_reference)
                print_boundary_distance(boundary_tracker, label="回転開始から現在まで")

                motion_reference = capture_motion_reference(markers)

                waiting_for_motion_result = False
                waiting_for_next_cycle = True
                next_cycle_wait_start = time.time()

        if waiting_for_next_cycle:
            now = time.time()

            if now - next_cycle_wait_start >= COMMAND_INTERVAL_SEC:
                if pending_start_id is not None:
                    current_start_id = pending_start_id
                    sequence_number += 1
                    sequence_state = start_new_sequence(
                        current_start_id,
                        sequence_number
                    )
                    pending_start_id = None

                waiting_for_next_cycle = False

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("\nCtrl+Cによる中断を受け付けました")
    print_boundary_distance(boundary_tracker, label="中断時点")

finally:
    # ESC・Ctrl+C・例外終了のいずれでも最終結果を表示する。
    print_boundary_distance(boundary_tracker, label="終了時点")

    # 終了前に、加圧・減圧中のセルをすべてIdleへ戻す。
    if hidas_sock is not None:
        try:
            print("全セルをアイドル状態にします")
            send_all_idle()
        except Exception as exc:
            print(f"終了時のIdle送信に失敗しました: {exc}")

    picam2.stop()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    if hidas_sock is not None:
        try:
            hidas_sock.close()
        except OSError:
            pass
