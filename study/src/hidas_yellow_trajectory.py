#!/usr/bin/env python3
"""
複数のHIDAS画像から黄色マーカーの重心を検出し、移動軌跡を可視化する。

必要なライブラリ:
    pip install opencv-python numpy matplotlib

実行例:
    python hidas_yellow_trajectory.py --input images --output results

画像を個別に指定する場合:
    python hidas_yellow_trajectory.py --input img01.jpg img02.jpg img03.jpg

主な出力:
    trajectory_graph.png      : 各画像の形状と重心の矢印軌跡を重ねたグラフ
    trajectory_overlay.png    : 1枚目の画像に形状と重心の矢印軌跡を重ねた画像
    yellow_centroids.csv      : 各黄色マーカーの重心座標
    hidas_centroid.csv        : 全黄色マーカーの平均座標
    resized/                  : 1枚目の解像度に統一した各入力画像
    detected/                 : 検出位置と番号を描画した各画像
    masks/                    : 黄色領域の二値画像

注意:
    ・画像は時系列順のファイル名にする（例: image_001.jpg, image_002.jpg）。
    ・2枚目以降の画像は、ファイル名順で最初の画像（image_001など）の
      解像度に自動でリサイズしてから処理する。
    ・正確に移動を比較するため、カメラ位置は全画像でそろえる。
    ・マーカー番号は、1枚目の最上部にある黄色マーカーを1番として、
      そこから時計回りに自動で割り当てる。HIDASのセルIDとは別の番号である。
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Detection:
    x: float
    y: float
    area: float
    circularity: float
    contour: np.ndarray


def natural_sort_key(path: Path) -> list[object]:
    """数字を数値として扱い、image_2をimage_10より前に並べる。"""
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def collect_image_paths(inputs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []

    for input_text in inputs:
        path = Path(input_text)
        if path.is_dir():
            paths.extend(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
        else:
            raise FileNotFoundError(f"画像または画像フォルダが見つかりません: {input_text}")

    # 同じ画像が重複指定された場合は1回だけ処理する。
    unique_paths = list(dict.fromkeys(p.resolve() for p in paths))
    unique_paths.sort(key=natural_sort_key)

    if not unique_paths:
        raise FileNotFoundError("入力画像がありません。")

    return unique_paths


def read_image(path: Path) -> np.ndarray:
    """日本語を含むWindowsパスでも読み込めるようにする。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"画像を読み込めません: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    """日本語を含むWindowsパスでも保存できるようにする。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"画像を保存できません: {path}")
    encoded.tofile(str(path))


def detect_yellow_centroids(
    image: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    min_area: float,
    max_area: float,
    min_circularity: float,
    expected_markers: int,
    roi: tuple[int, int, int, int] | None,
) -> tuple[list[Detection], np.ndarray]:
    """黄色領域を抽出し、各領域の重心を返す。"""
    height, width = image.shape[:2]
    x_offset = 0
    y_offset = 0
    target = image

    if roi is not None:
        x, y, w, h = roi
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ValueError(
                f"ROIが画像範囲外です: roi={roi}, image_size=({width}, {height})"
            )
        target = image[y:y + h, x:x + w]
        x_offset = x
        y_offset = y

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    mask_roi = cv2.inRange(
        hsv,
        np.array(hsv_lower, dtype=np.uint8),
        np.array(hsv_upper, dtype=np.uint8),
    )

    # JPEGノイズを除去し、黄色領域内の小さな欠けを埋める。
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, open_kernel)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        x = moments["m10"] / moments["m00"] + x_offset
        y = moments["m01"] / moments["m00"] + y_offset
        shifted_contour = contour + np.array([[[x_offset, y_offset]]], dtype=contour.dtype)
        detections.append(Detection(x, y, area, circularity, shifted_contour))

    # 背景の小さな黄色ノイズが残った場合は、面積の大きいものを優先する。
    if len(detections) > expected_markers:
        detections.sort(key=lambda d: d.area, reverse=True)
        detections = detections[:expected_markers]

    full_mask = np.zeros((height, width), dtype=np.uint8)
    if roi is None:
        full_mask = mask_roi
    else:
        x, y, w, h = roi
        full_mask[y:y + h, x:x + w] = mask_roi

    return detections, full_mask


def order_clockwise_from_top(points: np.ndarray) -> np.ndarray:
    """全点の平均位置を中心とし、最上部から時計回りに並べる。"""
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    # 画像ではyが下向きなので、角度の増加方向が時計回りになる。
    angles_from_top = (angles + np.pi / 2.0) % (2.0 * np.pi)
    return points[np.argsort(angles_from_top)]


def match_to_previous(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """
    円周上の順番を維持したまま、前画像との総移動量が最小となる対応を選ぶ。
    """
    ordered = order_clockwise_from_top(current)
    best_points: np.ndarray | None = None
    best_cost = float("inf")

    for shift in range(len(ordered)):
        candidate = np.roll(ordered, shift, axis=0)
        cost = float(np.sum((candidate - previous) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_points = candidate

    if best_points is None:
        raise RuntimeError("マーカー対応付けに失敗しました。")
    return best_points


def draw_detection(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    result = image.copy()
    center = np.mean(points, axis=0)

    for marker_index, (x, y) in enumerate(points, start=1):
        position = (int(round(x)), int(round(y)))
        cv2.circle(result, position, 12, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(result, position, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            result,
            str(marker_index),
            (position[0] + 8, position[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            str(marker_index),
            (position[0] + 8, position[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    center_position = (int(round(center[0])), int(round(center[1])))
    cv2.drawMarker(
        result,
        center_position,
        (255, 255, 0),
        cv2.MARKER_CROSS,
        28,
        3,
        cv2.LINE_AA,
    )
    return result


def save_csv_files(
    output_dir: Path,
    image_paths: Sequence[Path],
    trajectories: np.ndarray,
    overall_centroids: np.ndarray,
) -> None:
    marker_csv = output_dir / "yellow_centroids.csv"
    with marker_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "image_name", "marker_id", "x_px", "y_px"])
        for frame_index, (image_path, points) in enumerate(
            zip(image_paths, trajectories), start=1
        ):
            for marker_id, (x, y) in enumerate(points, start=1):
                writer.writerow([frame_index, image_path.name, marker_id, f"{x:.3f}", f"{y:.3f}"])

    center_csv = output_dir / "hidas_centroid.csv"
    with center_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_index", "image_name", "x_px", "y_px"])
        for frame_index, (image_path, center) in enumerate(
            zip(image_paths, overall_centroids), start=1
        ):
            writer.writerow(
                [frame_index, image_path.name, f"{center[0]:.3f}", f"{center[1]:.3f}"]
            )


def save_shape_transition_graph(
    output_path: Path,
    trajectories: np.ndarray,
) -> None:
    """
    各画像の16個の重心を閉じた線でつなぎ、画像ごとのHIDAS形状を重ねる。

    色は時系列に沿って青から赤へ変化する。個々のマーカーの時系列軌跡は
    描かず、各時刻におけるHIDAS全体の形状変化を比較できるようにする。
    """
    frame_count = trajectories.shape[0]
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, frame_count))
    figure, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for points, color in zip(trajectories, colors):
        # 16番目から1番目にも線を引いて円環を閉じる。
        closed_points = np.vstack([points, points[0]])

        # 黒い下線を加えることで、複数形状が重なっても輪郭を見やすくする。
        ax.plot(
            closed_points[:, 0],
            closed_points[:, 1],
            color="black",
            linewidth=2.7,
            alpha=0.35,
            zorder=1,
        )
        ax.plot(
            closed_points[:, 0],
            closed_points[:, 1],
            color=color,
            linewidth=1.6,
            alpha=0.90,
            zorder=2,
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=46,
            color=[color],
            edgecolors="black",
            linewidths=0.65,
            zorder=3,
        )

    # 16個のマーカー座標の平均を各画像におけるHIDASの重心とする。
    centers = np.mean(trajectories, axis=1)

    # 形状自体には矢印を付けず、重心の移動だけを撮影順に矢印で示す。
    for start, end in zip(centers[:-1], centers[1:]):
        ax.annotate(
            "",
            xy=(end[0], end[1]),
            xytext=(start[0], start[1]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "linewidth": 2.2,
                "mutation_scale": 16,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=4,
        )

    for center, color in zip(centers, colors):
        ax.scatter(
            center[0],
            center[1],
            s=95,
            marker="X",
            color=[color],
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )

    # 画像全体ではなく、すべての検出点が収まる範囲を拡大表示する。
    all_x = trajectories[:, :, 0]
    all_y = trajectories[:, :, 1]
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    x_margin = max(15.0, (x_max - x_min) * 0.07)
    # 上部に時系列の色順を表示できる余白を少し多めに確保する。
    y_margin = max(15.0, (y_max - y_min) * 0.13)

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_max + y_margin, y_min - y_margin)  # 画像と同じくyは下向き。
    ax.set_aspect("equal", adjustable="box")
    # タイトル、凡例、座標軸、目盛、pixel表記は表示しない。
    ax.axis("off")

    # 画像名は表示せず、グラフ内に色と撮影順だけを示す。
    # 例: 青い1 → 緑の2 → 黄色の3 → 赤い4
    key_width = min(0.78, max(0.24, 0.075 * frame_count + 0.06))
    key_ax = ax.inset_axes([0.03, 0.875, key_width, 0.10])
    key_ax.set_xlim(0.0, float(frame_count))
    key_ax.set_ylim(0.0, 1.0)
    key_ax.set_axis_off()
    key_ax.patch.set_visible(True)
    key_ax.patch.set_facecolor((1.0, 1.0, 1.0, 0.88))
    key_ax.patch.set_edgecolor((0.0, 0.0, 0.0, 0.22))
    key_ax.patch.set_linewidth(0.8)

    for index, color in enumerate(colors):
        x = index + 0.5
        key_ax.scatter(
            x,
            0.64,
            s=72,
            color=[color],
            edgecolors="black",
            linewidths=0.7,
            zorder=2,
        )
        key_ax.text(
            x,
            0.17,
            str(index + 1),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

        if index < frame_count - 1:
            key_ax.annotate(
                "",
                xy=(index + 1.37, 0.64),
                xytext=(index + 0.63, 0.64),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "linewidth": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=1,
            )

    figure.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def matplotlib_color_to_bgr(color: Iterable[float]) -> tuple[int, int, int]:
    rgba = list(color)
    red, green, blue = rgba[:3]
    return int(blue * 255), int(green * 255), int(red * 255)


def save_shape_transition_overlay(
    output_path: Path,
    background: np.ndarray,
    trajectories: np.ndarray,
) -> None:
    """1枚目の画像上に、各時刻の16点と閉じた形状を重ねる。"""
    overlay = background.copy()
    frame_count = trajectories.shape[0]
    colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, frame_count))

    for frame_points, plot_color in zip(trajectories, colors):
        points = np.rint(frame_points).astype(np.int32)
        color = matplotlib_color_to_bgr(plot_color)
        cv2.polylines(overlay, [points], True, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(overlay, [points], True, color, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(overlay, tuple(point), 7, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, tuple(point), 7, (0, 0, 0), 1, cv2.LINE_AA)

    centers = np.mean(trajectories, axis=1)
    center_points = np.rint(centers).astype(np.int32)

    # 重心間だけに矢印を描画する。黒い外縁と白い内線で背景上でも見やすくする。
    for start, end in zip(center_points[:-1], center_points[1:]):
        cv2.arrowedLine(
            overlay,
            tuple(start),
            tuple(end),
            (0, 0, 0),
            7,
            cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.arrowedLine(
            overlay,
            tuple(start),
            tuple(end),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
            tipLength=0.18,
        )

    for center, plot_color in zip(center_points, colors):
        color = matplotlib_color_to_bgr(plot_color)
        cv2.circle(overlay, tuple(center), 10, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, tuple(center), 7, color, -1, cv2.LINE_AA)

    write_image(output_path, overlay)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="複数画像の黄色マーカー重心からHIDASの形状変化を重ね描きします。"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="入力画像、または画像を格納したフォルダ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="出力フォルダ（既定値: results）",
    )
    parser.add_argument("--expected-markers", type=int, default=16)
    parser.add_argument(
        "--hsv-lower",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(15, 100, 120),
        help="黄色抽出HSV下限（既定値: 15 100 120）",
    )
    parser.add_argument(
        "--hsv-upper",
        type=int,
        nargs=3,
        metavar=("H", "S", "V"),
        default=(40, 255, 255),
        help="黄色抽出HSV上限（既定値: 40 255 255）",
    )
    parser.add_argument("--min-area", type=float, default=20.0)
    parser.add_argument("--max-area", type=float, default=5000.0)
    parser.add_argument("--min-circularity", type=float, default=0.50)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        default=None,
        help="HIDASが写る範囲だけを処理する場合のROI",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = collect_image_paths(args.input)
    output_dir: Path = args.output.resolve()
    resized_dir = output_dir / "resized"
    detected_dir = output_dir / "detected"
    masks_dir = output_dir / "masks"
    resized_dir.mkdir(parents=True, exist_ok=True)
    detected_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    all_points: list[np.ndarray] = []
    first_image: np.ndarray | None = None
    image_size: tuple[int, int] | None = None

    for frame_index, image_path in enumerate(image_paths, start=1):
        image = read_image(image_path)
        height, width = image.shape[:2]

        if first_image is None:
            first_image = image.copy()
            image_size = (width, height)
        elif image_size != (width, height):
            target_width, target_height = image_size
            original_width, original_height = width, height

            original_aspect = original_width / original_height
            target_aspect = target_width / target_height
            if not np.isclose(original_aspect, target_aspect, rtol=0.0, atol=1e-3):
                print(
                    f"[warning] {image_path.name}: 1枚目と縦横比が異なります。"
                    " リサイズ後の画像は縦または横に変形します。"
                )

            # 縮小時はINTER_AREA、拡大時はINTER_LINEARを使用する。
            interpolation = (
                cv2.INTER_AREA
                if original_width * original_height > target_width * target_height
                else cv2.INTER_LINEAR
            )
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=interpolation,
            )
            height, width = image.shape[:2]
            print(
                f"[resize] {image_path.name}: "
                f"{original_width}x{original_height} -> {target_width}x{target_height}"
            )

        # 1枚目を含め、基準解像度に統一された処理用画像をすべて保存する。
        stem = f"{frame_index:03d}_{image_path.stem}"
        write_image(resized_dir / f"{stem}.png", image)

        detections, mask = detect_yellow_centroids(
            image=image,
            hsv_lower=tuple(args.hsv_lower),
            hsv_upper=tuple(args.hsv_upper),
            min_area=args.min_area,
            max_area=args.max_area,
            min_circularity=args.min_circularity,
            expected_markers=args.expected_markers,
            roi=tuple(args.roi) if args.roi is not None else None,
        )

        if len(detections) != args.expected_markers:
            raise RuntimeError(
                f"{image_path.name}: 黄色マーカーを{len(detections)}個検出しました。"
                f" 必要数は{args.expected_markers}個です。"
                " HSV範囲、面積範囲、円形度、またはROIを調整してください。"
            )

        raw_points = np.array([[d.x, d.y] for d in detections], dtype=np.float64)
        if not all_points:
            tracked_points = order_clockwise_from_top(raw_points)
        else:
            tracked_points = match_to_previous(all_points[-1], raw_points)
        all_points.append(tracked_points)

        annotated = draw_detection(image, tracked_points)
        write_image(detected_dir / f"{stem}.png", annotated)
        write_image(masks_dir / f"{stem}_mask.png", mask)

        print(
            f"[{frame_index:03d}/{len(image_paths):03d}] "
            f"{image_path.name}: {len(detections)} markers"
        )

    if first_image is None or image_size is None:
        raise RuntimeError("処理可能な画像がありません。")

    trajectories = np.stack(all_points, axis=0)
    overall_centroids = np.mean(trajectories, axis=1)

    save_csv_files(output_dir, image_paths, trajectories, overall_centroids)
    save_shape_transition_graph(
        output_dir / "trajectory_graph.png",
        trajectories,
    )
    save_shape_transition_overlay(
        output_dir / "trajectory_overlay.png",
        first_image,
        trajectories,
    )

    total_displacement = overall_centroids[-1] - overall_centroids[0]
    print("\n処理が完了しました。")
    print(f"入力画像数: {len(image_paths)}")
    print(f"HIDAS重心の変位: dx={total_displacement[0]:.3f} px, dy={total_displacement[1]:.3f} px")
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()
