import cv2
import numpy as np
import math


def find_circles(mask, min_area):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    circles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
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
    if green > red:
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
    if (
        inner_color == "green"
        and middle_color == "red"
        and outer_color == "green"
    ):
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

    # 画像座標はy軸が下向きなので、数学座標として扱う
    return math.atan2(-(y - cy), x - cx)


def assign_ids_by_order(markers):
    if len(markers) == 0:
        return markers, None

    # HIDAS中心
    hid_cx = sum(m["center"][0] for m in markers) / len(markers)
    hid_cy = sum(m["center"][1] for m in markers) / len(markers)
    hid_center = (hid_cx, hid_cy)

    for marker in markers:
        marker["angle"] = angle_from_center(marker["center"], hid_center)
        marker["angle_deg"] = math.degrees(marker["angle"])

    # 角度順に並べる
    markers_sorted = sorted(markers, key=lambda m: m["angle"])

    base_index = None
    base_id = None

    # ID1を優先して基準にする
    for i, marker in enumerate(markers_sorted):
        if marker["type"] == 1:
            base_index = i
            base_id = 1
            break

    # ID1が見えなければID9を基準にする
    if base_index is None:
        for i, marker in enumerate(markers_sorted):
            if marker["type"] == 9:
                base_index = i
                base_id = 9
                break

    # ID1もID9も見えない場合
    if base_index is None:
        for marker in markers_sorted:
            marker["assigned_id"] = -1
        return markers_sorted, hid_center

    marker_count = len(markers_sorted)

    for i in range(marker_count):
        marker = markers_sorted[i]

        # 検出できたID1とID9は絶対固定
        if marker["type"] == 1:
            marker["assigned_id"] = 1
            continue

        if marker["type"] == 9:
            marker["assigned_id"] = 9
            continue

        # 時計回り（CW）にIDを補完
        offset = (base_index - i) % marker_count
        assigned_id = ((base_id - 1 + offset) % 16) + 1
        marker["assigned_id"] = assigned_id

    return markers_sorted, hid_center


# =========================
# パラメータ
# =========================
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8
MIN_AREA = 15

# HSV色抽出範囲
lower_red1 = np.array([0,   100,   0])
upper_red1 = np.array([5,   255, 255])

lower_red2 = np.array([177, 100,   0])
upper_red2 = np.array([180, 255, 255])  # 0〜180の両端をまたぐため177から

lower_green = np.array([30,  30,   0])
upper_green = np.array([90, 255, 255])

# =========================
# 入力画像
# =========================
input_path = "../images/4.jpg"

image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"入力画像が見つかりません: {input_path}")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 赤色マスク
red1 = cv2.inRange(hsv, lower_red1, upper_red1)
red2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(red1, red2)

# 緑色マスク
green_mask = cv2.inRange(hsv, lower_green, upper_green)

red_circles = find_circles(red_mask, MIN_AREA)
green_circles = find_circles(green_mask, MIN_AREA)

candidates = []

for center_r, r_r in red_circles:
    for center_g, r_g in green_circles:
        dist = np.linalg.norm(np.array(center_r) - np.array(center_g))

        small_r = min(r_r, r_g)
        large_r = max(r_r, r_g)
        ratio = small_r / large_r

        if (
            dist < MAX_CENTER_DISTANCE
            and RADIUS_RATIO_MIN < ratio < RADIUS_RATIO_MAX
        ):
            cx = int((center_r[0] + center_g[0]) / 2)
            cy = int((center_r[1] + center_g[1]) / 2)

            candidates.append({
                "center": (cx, cy),
                "outer_radius": large_r,
                "red": (center_r, r_r),
                "green": (center_g, r_g)
            })


# 中心が近い候補は外径が大きいものだけ残す
final_candidates = []

for candidate in sorted(
    candidates,
    key=lambda c: c["outer_radius"],
    reverse=True
):
    keep = True

    for saved in final_candidates:
        distance = np.linalg.norm(
            np.array(candidate["center"]) - np.array(saved["center"])
        )
        if distance < MAX_CENTER_DISTANCE:
            keep = False
            break

    if keep:
        final_candidates.append(candidate)


markers = []

for candidate in final_candidates:
    center = candidate["center"]
    outer_radius = candidate["outer_radius"]

    marker_type, inner_color, middle_color, outer_color = judge_marker_type(
        hsv,
        center,
        outer_radius
    )

    markers.append({
        "center": center,
        "outer_radius": outer_radius,
        "red": candidate["red"],
        "green": candidate["green"],
        "type": marker_type,
        "inner_color": inner_color,
        "middle_color": middle_color,
        "outer_color": outer_color,
        "assigned_id": -1,
        "angle": 0,
        "angle_deg": 0
    })


# 時計回りにIDを付与
markers, hid_center = assign_ids_by_order(markers)


# =========================
# HIDAS中心を描画・出力
# =========================
if hid_center is not None:
    hx, hy = int(hid_center[0]), int(hid_center[1])

    # HIDAS中心の座標をターミナルへ出力
    print(
        f"HIDAS中心座標: "
        f"x={hid_center[0]:.2f}, y={hid_center[1]:.2f}"
    )

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
else:
    print("HIDAS中心座標を算出できませんでした")


# =========================
# マーカー描画
# =========================
for marker in markers:
    center = marker["center"]
    center_r, r_r = marker["red"]
    center_g, r_g = marker["green"]

    # HIDAS中心からマーカー中心への線
    if hid_center is not None:
        hx, hy = int(hid_center[0]), int(hid_center[1])
        cv2.line(image, (hx, hy), center, (0, 255, 255), 1)

    cv2.circle(image, center, 4, (255, 0, 255), -1)
    cv2.circle(image, center_r, int(r_r), (0, 0, 255), 1)
    cv2.circle(image, center_g, int(r_g), (0, 255, 0), 1)

    # ID表示
    cv2.putText(
        image,
        f"ID:{marker['assigned_id']}",
        (center[0] - 25, center[1] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    # 角度表示
    cv2.putText(
        image,
        f"{marker['angle_deg']:.1f}deg",
        (center[0] - 35, center[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )


# =========================
# 結果画像を画面に表示（保存はしない）
# =========================
print(f"検出マーカー数: {len(markers)}")

for marker in markers:
    print(
        f"ID:{marker['assigned_id']} "
        f"type:{marker['type']} "
        f"center:{marker['center']} "
        f"angle:{marker['angle_deg']:.1f}deg "
        f"colors:{marker['inner_color']}-"
        f"{marker['middle_color']}-{marker['outer_color']}"
    )

cv2.imshow("Marker Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
