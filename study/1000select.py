import os
import random
import shutil

# === 入出力フォルダ ===
input_dir = "study/mp4_output/4_ok_ver2"   # 元の画像フォルダ
output_dir = "study/mp4_output/4_add"      # コピー先フォルダ
os.makedirs(output_dir, exist_ok=True)

# === 対象となる拡張子 ===
valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

# === フォルダ内の画像ファイル一覧を取得 ===
all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
total = len(all_images)

# === ランダムに1030枚選ぶ ===
sample_size = 228
if sample_size > total:
    raise ValueError(f"フォルダ内の画像が {total} 枚しかありません（1030枚選べません）")

selected = random.sample(all_images, sample_size)

# === 選ばれた画像をコピー ===
for filename in selected:
    src_path = os.path.join(input_dir, filename)
    dst_path = os.path.join(output_dir, filename)
    shutil.copy2(src_path, dst_path)

print(f"🎉 {sample_size} 枚をランダムにコピーしました！")
print(f"コピー元: {input_dir}")
print(f"コピー先: {output_dir}")
