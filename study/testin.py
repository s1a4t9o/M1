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

    # 画像座標はy軸が下向きなので、数学座標っぽく扱う
    return math.atan2(-(y - cy), x - cx)


def assign_ids_by_order(markers):
    if len(markers) == 0:
        return markers

    # HIDAS全体中心を全マーカー中心の平均で推定
    hid_cx = sum(m["center"][0] for m in markers) / len(markers)
    hid_cy = sum(m["center"][1] for m in markers) / len(markers)
    hid_center = (hid_cx, hid_cy)

    for m in markers:
        m["angle"] = angle_from_center(m["center"], hid_center)

    # 反時計回り順に並べる
    markers_sorted = sorted(markers, key=lambda m: m["angle"])

    # ID1を優先して基準にする
    base_index = None
    base_id = None

    for i, m in enumerate(markers_sorted):
        if m["type"] == 1:
            base_index = i
            base_id = 1
            break

    # ID1が見えなければID9を基準にする
    if base_index is None:
        for i, m in enumerate(markers_sorted):
            if m["type"] == 9:
                base_index = i
                base_id = 9
                break

    # どちらも見えない場合
    if base_index is None:
        for m in markers_sorted:
            m["assigned_id"] = -1
        return markers_sorted

    n = len(markers_sorted)

    for i in range(n):
        offset = (i - base_index) % n
        assigned_id = ((base_id - 1 + offset) % 16) + 1
        m = markers_sorted[i]
        m["assigned_id"] = assigned_id

    return markers_sorted


# === パラメータ ===
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 15

# === 入力画像 ===
image = cv2.imread("q7c.png")
if image is None:
    raise FileNotFoundError("入力画像が見つかりません。")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 赤色マスク
red1 = cv2.inRange(hsv, (0, 100, 0), (5, 255, 255))
red2 = cv2.inRange(hsv, (177, 100, 0), (180, 255, 255))
red_mask = cv2.bitwise_or(red1, red2)

# 緑色マスク
green_mask = cv2.inRange(hsv, (55, 60, 0), (80, 255, 255))

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


# 中心が近い候補は、外径が大きいものだけ残す
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
        "assigned_id": -1
    })


# 円周順にIDを付与
markers = assign_ids_by_order(markers)


# === 可視化 ===
for m in markers:
    center = m["center"]
    center_r, r_r = m["red"]
    center_g, r_g = m["green"]

    cv2.circle(image, center, 4, (255, 0, 255), -1)
    cv2.circle(image, center_r, int(r_r), (0, 0, 255), 1)
    cv2.circle(image, center_g, int(r_g), (0, 255, 0), 1)

    text = f"ID:{m['assigned_id']}"

    cv2.putText(
        image,
        text,
        (center[0] - 25, center[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    debug_text = f"type:{m['type']} in:{m['inner_color']} mid:{m['middle_color']} out:{m['outer_color']}"

    cv2.putText(
        image,
        debug_text,
        (center[0] - 80, center[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1
    )


cv2.imshow("HIDAS Marker ID Assignment", image)
cv2.waitKey(0)
cv2.destroyAllWindows()