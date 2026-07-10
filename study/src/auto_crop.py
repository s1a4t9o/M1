#インフレータブル構造物画像の指定範囲切り出し

from pathlib import Path
import cv2

INPUT_DIR = Path(r"C:/20251211/100GOPRO")   # ← 写真が入っているフォルダ
OUTPUT_DIR = Path(r"C:/20251211/100_crop")  # ← 切り出し保存先

# (x_center, y_center, width, height) 正規化座標
XC, YC, WN, HN = 0.460219, 0.469007, 0.345007, 0.424466


# jpgで保存
JPG_QUALITY = 100

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalized_crop_box(img_w, img_h, xc, yc, wn, hn):
    crop_w = int(round(wn * img_w))
    crop_h = int(round(hn * img_h))
    cx = int(round(xc * img_w))
    cy = int(round(yc * img_h))

    x0 = cx - crop_w // 2
    y0 = cy - crop_h // 2
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    x0 = clamp(x0, 0, img_w - 1)
    y0 = clamp(y0, 0, img_h - 1)
    x1 = clamp(x1, x0 + 1, img_w)
    y1 = clamp(y1, y0 + 1, img_h)

    return x0, y0, x1, y1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in EXTS]
    )

    if not files:
        print("画像が見つかりません")
        return

    for i, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[SKIP] 読み込み失敗: {p.name}")
            continue

        h, w = img.shape[:2]
        x0, y0, x1, y1 = normalized_crop_box(w, h, XC, YC, WN, HN)
        crop = img[y0:y1, x0:x1]

        out_path = OUTPUT_DIR / f"{p.stem}_crop.jpg"
        cv2.imwrite(
            str(out_path),
            crop,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY]
        )

        print(f"[{i}/{len(files)}] 保存: {out_path.name}")


if __name__ == "__main__":
    main()
