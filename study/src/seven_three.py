import os
import random
import shutil

# 入力フォルダと出力フォルダの設定
photo_folder = "study/mp4_output/4_ok_ver1"  # 画像が入っているフォルダ
output2_folder = "study/mp4_output/4_train"  # 70%が移動するフォルダ
output1_folder = "study/mp4_output/4_valid"  # 30%が移動するフォルダ

# 出力フォルダが存在しない場合は作成
os.makedirs(output1_folder, exist_ok=True)
os.makedirs(output2_folder, exist_ok=True)

# photoフォルダ内のファイルリストを取得
all_files = [f for f in os.listdir(photo_folder) if os.path.isfile(os.path.join(photo_folder, f))]

# ファイルをランダムにシャッフル
random.shuffle(all_files)

# 30%と70%に分割
split_point = int(len(all_files) * 0.3)
output1_files = all_files[:split_point]
output2_files = all_files[split_point:]

# ファイルを移動
for file_name in output1_files:
    shutil.move(os.path.join(photo_folder, file_name), os.path.join(output1_folder, file_name))

for file_name in output2_files:
    shutil.move(os.path.join(photo_folder, file_name), os.path.join(output2_folder, file_name))

print(f"Moved {len(output1_files)} files to {output1_folder}")
print(f"Moved {len(output2_files)} files to {output2_folder}")