import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ==== 表示設定（Trueで表示、Falseで非表示）====
SHOW_OASIS = True #オアシスゾーン
SHOW_ITEM  = True #itemゾーン
SHOW_KIZ   = True #KIZ
SHOW_TRAJECTORY = True #移動場所(P)
SHOW_INTERSECTIONS = False #衝突場所

# ==== 任意の新しい点 ====
custom_points = [
    np.array([10.842, -8.917, 4.544]),
    np.array([9.867, -6.824, 4.709]),
    #np.array([11.142, -8.651, 3.762])
]

# ==== 各点定義 ====
points = [
    np.array([9.815, -9.806, 4.293]),#Start
    np.array([10.951, -9.681, 5.195]),
    np.array([10.925, -8.875, 4.66203]),
    np.array([10.76698, -6.8525, 4.945]),
    np.array([11.143, -6.7607, 4.9654])#Goal
]

# ==== オアシスゾーン定義 ====
oasis_zones = [
    {"name": "oasiszone 1", "x_min": 10.425, "x_max": 11.425, "y_min": -10.2, "y_max": -9.5, "z_min": 4.445, "z_max": 4.945},
    {"name": "oasiszone 2", "x_min": 10.925, "x_max": 11.425, "y_min": -9.5,  "y_max": -8.45,"z_min": 4.945, "z_max": 5.445},
    {"name": "oasiszone 3", "x_min": 10.425, "x_max": 10.975, "y_min": -8.45, "y_max": -7.4, "z_min": 4.945, "z_max": 5.445},
    {"name": "oasiszone 4", "x_min": 10.925, "x_max": 11.425, "y_min": -7.4,  "y_max": -6.35,"z_min": 4.425, "z_max": 4.945}
]

# ==== アイテムゾーン定義 ====
item_zones = [
    {"name": "item 1", "x_min": 10.42, "x_max": 11.48, "y_min": -10.58, "y_max": -10.58, "z_min": 4.82, "z_max": 5.57},
    {"name": "item 2", "x_min": 10.3,  "x_max": 11.55, "y_min": -9.25,  "y_max": -8.5,  "z_min": 3.76203, "z_max": 3.76203},
    {"name": "item 3", "x_min": 10.3,  "x_max": 11.55, "y_min": -8.4,   "y_max": -7.45, "z_min": 3.76093, "z_max": 3.76093},
    {"name": "item 4", "x_min": 9.866984,"x_max": 9.866984,"y_min": -7.34,"y_max": -6.365,"z_min": 4.32,    "z_max": 5.57}
]

# ==== KIZゾーン定義 ====
kiz_zones = [
    {"name": "KIZ 1", "x_min": 10.3, "x_max": 11.55, "y_min": -10.2, "y_max": -6.0, "z_min": 4.32, "z_max": 5.57},
    {"name": "KIZ 2", "x_min": 9.5,  "x_max": 10.5,  "y_min": -10.5, "y_max": -9.6, "z_min": 4.02, "z_max": 4.8}
]

# ==== 交差判定（Slabs法） ====
def check_entry_exit(p1, p2, box):
    direction = p2 - p1
    tmin, tmax = -np.inf, np.inf
    for i, axis in enumerate(['x', 'y', 'z']):
        d = direction[i]
        if d != 0:
            t1 = (box[f'{axis}_min'] - p1[i]) / d
            t2 = (box[f'{axis}_max'] - p1[i]) / d
            t1, t2 = min(t1, t2), max(t1, t2)
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
        elif not (box[f'{axis}_min'] <= p1[i] <= box[f'{axis}_max']):
            return None, None
    if tmin > tmax or tmax < 0 or tmin > 1:
        return None, None
    entry_point = p1 + tmin * direction if 0 <= tmin <= 1 else None
    exit_point  = p1 + tmax * direction if 0 <= tmax <= 1 else None
    return entry_point, exit_point

# ==== オアシスゾーンとの交点リスト ====
intersections = []
for i in range(len(points) - 1):
    p1, p2 = points[i], points[i + 1]
    for box in oasis_zones:
        entry, exit = check_entry_exit(p1, p2, box)
        if entry is not None:
            intersections.append({"from_to": f"{i}{i+1}", "zone": box["name"], "type": "entry", "intersection": entry.tolist()})
        if exit is not None:
            intersections.append({"from_to": f"{i}{i+1}", "zone": box["name"], "type": "exit",  "intersection": exit.tolist()})

# ==== 描画補助 ====
def draw_cube(ax, box, color='green', alpha=0.1):
    x = [box['x_min'], box['x_max']]
    y = [box['y_min'], box['y_max']]
    z = [box['z_min'], box['z_max']]
    vertices = np.array([
        [x[0], y[0], z[0]],
        [x[1], y[0], z[0]],
        [x[1], y[1], z[0]],
        [x[0], y[1], z[0]],
        [x[0], y[0], z[1]],
        [x[1], y[0], z[1]],
        [x[1], y[1], z[1]],
        [x[0], y[1], z[1]]
    ])
    faces = [
        [vertices[j] for j in [0,1,2,3]],
        [vertices[j] for j in [4,5,6,7]],
        [vertices[j] for j in [0,1,5,4]],
        [vertices[j] for j in [2,3,7,6]],
        [vertices[j] for j in [1,2,6,5]],
        [vertices[j] for j in [4,7,3,0]]
    ]
    ax.add_collection3d(
        Poly3DCollection(faces, facecolors=color, linewidths=0.5,
                         edgecolors='k', alpha=alpha)
    )

# ==== 描画 ====
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 軌道描画
if SHOW_TRAJECTORY:
    for i, pt in enumerate(points):
        ax.scatter(*pt, color='blue')
        ax.text(*pt, f"P{i}", fontsize=10)
    for i in range(len(points) - 1):
        line = np.vstack((points[i], points[i+1]))
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color='gray', linestyle='--')

# 交点描画
if SHOW_INTERSECTIONS:
    for hit in intersections:
        p = hit["intersection"]
        ax.scatter(*p, color='red', s=50)
        ax.text(*p, f"{hit['from_to']} {hit['type']}∩{hit['zone']}", fontsize=8, color='red')
        

# ==== 任意の点を赤色で表示 ====
for i, pt in enumerate(custom_points):
    ax.scatter(*pt, color='red', s=60, marker='o')  # 大きめの赤丸で表示
    ax.text(*pt, f"Custom{i}", fontsize=10, color='red')

# ゾーン描画
if SHOW_OASIS:
    for box in oasis_zones:
        draw_cube(ax, box, color='green', alpha=0.1)
        ax.text(*(np.mean([[box['x_min'], box['x_max']],
                           [box['y_min'], box['y_max']],
                           [box['z_min'], box['z_max']]], axis=1)),
                box['name'], color='green', fontsize=9)

if SHOW_ITEM:
    for box in item_zones:
        draw_cube(ax, box, color='orange', alpha=0.1)
        ax.text(*(np.mean([[box['x_min'], box['x_max']],
                           [box['y_min'], box['y_max']],
                           [box['z_min'], box['z_max']]], axis=1)),
                box['name'], color='orange', fontsize=9)

if SHOW_KIZ:
    for box in kiz_zones:
        draw_cube(ax, box, color='purple', alpha=0.15)
        ax.text(*(np.mean([[box['x_min'], box['x_max']],
                           [box['y_min'], box['y_max']],
                           [box['z_min'], box['z_max']]], axis=1)),
                box['name'], color='purple', fontsize=9)
# ==== itemゾーンとの交点リスト表示用 ====
itemCheck = True  # ← ON/OFF切り替え

item_intersections = []
if itemCheck:
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        for box in item_zones:
            entry, exit = check_entry_exit(p1, p2, box)
            if entry is not None:
                item_intersections.append({"from_to": f"{i}{i+1}", "zone": box["name"], "type": "entry", "intersection": entry.tolist()})
            if exit is not None:
                item_intersections.append({"from_to": f"{i}{i+1}", "zone": box["name"], "type": "exit",  "intersection": exit.tolist()})

    # ==== item交点描画 ====
    for hit in item_intersections:
        p = hit["intersection"]
        ax.scatter(*p, color='red', s=60, marker='^')  # item交点は▲で区別
        ax.text(*p, f"[item] {hit['from_to']} {hit['type']}∩{hit['zone']}", fontsize=8, color='red')

# 表示設定
ax.view_init(elev=-170, azim=-20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D View with KIZ Zones")

plt.tight_layout()
plt.show()