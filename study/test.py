import cv2
import numpy as np
import math

# === パラメータ ===
MIN_AREA = 15
MAX_CENTER_DISTANCE = 8
RADIUS_RATIO_MIN = 0.2
RADIUS_RATIO_MAX = 0.8

# === 入力画像 ===
input_path = "images/q19.png"
image = cv2.imread(input_path)
if image is None:
    raise FileNotFoundError(f"画像が読み込めません: {input_path}")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === HSV 色範囲 ===
lower_red1 = np.array([0,   100,   0])
upper_red1 = np.array([5,   255, 255])
lower_red2 = np.array([177, 100,   0])
upper_red2 = np.array([180, 255, 255])

lower_green = np.array([55,  60,   0])
upper_green = np.array([80, 255, 255])

# === マスク生成 ===
mask_red = cv2.bitwise_or(
    cv2.inRange(hsv, lower_red1, upper_red1),
    cv2.inRange(hsv, lower_red2, upper_red2)
)

mask_green = cv2.inRange(hsv, lower_green, upper_green)


# === 小領域除去 ===
def filter_small_components(mask, min_area):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    return filtered


mask_red_filtered = filter_small_components(mask_red, MIN_AREA)
mask_green_filtered = filter_small_components(mask_green, MIN_AREA)


# === 円検出 ===
def find_circles(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        circles.append(((int(x), int(y)), r))

    return circles


# === 色判定 ===
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


# === 指定半径の8方向の色を見る + 点の座標も返す ===
def sample_color_at_radius(hsv, center, radius):
    cx, cy = center
    height, width = hsv.shape[:2]

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    colors = []
    points = []

    for deg in angles:
        rad = math.radians(deg)

        x = int(cx + math.cos(rad) * radius)
        y = int(cy + math.sin(rad) * radius)

        if 0 <= x < width and 0 <= y < height:
            color_name = get_color_name(hsv[y, x])
            colors.append(color_name)
            points.append((x, y, color_name))

    return majority_color(colors), points


def judge_marker_type(hsv, center, outer_radius):
    inner_color, inner_points = sample_color_at_radius(
        hsv, center, outer_radius * 0.18
    )

    middle_color, middle_points = sample_color_at_radius(
        hsv, center, outer_radius * 0.48
    )

    outer_color, outer_points = sample_color_at_radius(
        hsv, center, outer_radius * 0.78
    )

    if inner_color == "green" and middle_color == "red" and outer_color == "green":
        marker_type = 9

    elif inner_color == "green" and outer_color == "red":
        marker_type = 1

    elif inner_color == "red" and outer_color == "green":
        marker_type = 0

    else:
        marker_type = -1

    return {
        "type": marker_type,
        "inner_color": inner_color,
        "middle_color": middle_color,
        "outer_color": outer_color,
        "inner_points": inner_points,
        "middle_points": middle_points,
        "outer_points": outer_points,
    }


# === 可視化用画像 ===
output = np.zeros_like(image)

output[mask_red_filtered > 0] = [0, 0, 255]
output[mask_green_filtered > 0] = [0, 255, 0]

# 元画像の上に描画したい場合はこっち
# output = image.copy()


# === 赤円・緑円の検出 ===
red_circles = find_circles(mask_red_filtered, MIN_AREA)
green_circles = find_circles(mask_green_filtered, MIN_AREA)

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


# === 重複除去 ===
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


# === 色を見ている点を描画 ===
for cand in final_candidates:
    center = cand["center"]
    outer_radius = cand["outer_radius"]

    result = judge_marker_type(hsv, center, outer_radius)

    # 中心
    cv2.circle(output, center, 4, (255, 0, 255), -1)

    # 赤円・緑円の検出円
    center_r, r_r = cand["red"]
    center_g, r_g = cand["green"]

    cv2.circle(output, center_r, int(r_r), (0, 0, 255), 1)
    cv2.circle(output, center_g, int(r_g), (0, 255, 0), 1)

    # 色を見る半径の円
    cv2.circle(output, center, int(outer_radius * 0.18), (255, 255, 255), 1)
    cv2.circle(output, center, int(outer_radius * 0.48), (255, 255, 255), 1)
    cv2.circle(output, center, int(outer_radius * 0.78), (255, 255, 255), 1)

    # 内側を見る点
    for x, y, color_name in result["inner_points"]:
        cv2.circle(output, (x, y), 4, (255, 255, 255), -1)

    # 中間を見る点
    for x, y, color_name in result["middle_points"]:
        cv2.circle(output, (x, y), 4, (255, 255, 0), -1)

    # 外側を見る点
    for x, y, color_name in result["outer_points"]:
        cv2.circle(output, (x, y), 4, (0, 255, 255), -1)

    # 判定結果を表示
    cv2.putText(
        output,
        f"type:{result['type']}",
        (center[0] - 30, center[1] - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"in:{result['inner_color']} mid:{result['middle_color']} out:{result['outer_color']}",
        (center[0] - 80, center[1] + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


# === 表示・保存 ===
cv2.imshow("Sample Points Output", output)
cv2.imwrite("images/test.png", output)

cv2.waitKey(0)
cv2.destroyAllWindows()