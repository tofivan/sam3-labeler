"""
YOLO OBB 資料集準備工具
將標註資料分割為訓練集和驗證集

使用方法:
    python prepare_dataset.py --input labeled_dataset --output yolo_dataset --split 0.8
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def prepare_yolo_obb_dataset(
    input_folder,
    output_folder,
    train_ratio=0.8,
    seed=42
):
    """
    準備 YOLO OBB 訓練資料集

    參數:
        input_folder: 標註資料夾 (含 images/, labels/, classes.txt)
        output_folder: 輸出資料夾
        train_ratio: 訓練集比例 (預設 0.8)
        seed: 隨機種子
    """
    random.seed(seed)

    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # 讀取類別
    classes_file = input_path / "classes.txt"
    if not classes_file.exists():
        print(f"錯誤: 找不到 {classes_file}")
        return

    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"類別數量: {len(classes)}")
    print(f"類別: {classes}")

    # 取得所有標註檔
    labels_folder = input_path / "labels"
    images_folder = input_path / "images"

    if not labels_folder.exists() or not images_folder.exists():
        print(f"錯誤: 找不到 images/ 或 labels/ 資料夾")
        return

    # 找出所有有標註的圖片
    label_files = list(labels_folder.glob("*.txt"))
    valid_samples = []

    for label_file in label_files:
        # 檢查標註檔是否有內容
        with open(label_file, 'r') as f:
            content = f.read().strip()
        if not content:
            continue

        # 找對應的圖片
        stem = label_file.stem
        img_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            candidate = images_folder / f"{stem}{ext}"
            if candidate.exists():
                img_file = candidate
                break

        if img_file:
            valid_samples.append((img_file, label_file))

    print(f"\n有效樣本數: {len(valid_samples)}")

    if len(valid_samples) < 10:
        print(f"\n警告: 樣本數太少 ({len(valid_samples)} 張)，建議至少 100 張以上")
        print("繼續處理，但訓練效果可能不佳...")

    # 隨機分割
    random.shuffle(valid_samples)
    split_idx = int(len(valid_samples) * train_ratio)
    train_samples = valid_samples[:split_idx]
    val_samples = valid_samples[split_idx:]

    print(f"訓練集: {len(train_samples)} 張")
    print(f"驗證集: {len(val_samples)} 張")

    # 建立輸出目錄
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"

    for folder in [train_images, train_labels, val_images, val_labels]:
        folder.mkdir(parents=True, exist_ok=True)

    # 複製檔案
    def copy_samples(samples, img_dest, label_dest, desc):
        for img_file, label_file in samples:
            shutil.copy2(img_file, img_dest / img_file.name)
            shutil.copy2(label_file, label_dest / label_file.name)
        print(f"已複製 {len(samples)} 個 {desc} 樣本")

    copy_samples(train_samples, train_images, train_labels, "訓練")
    copy_samples(val_samples, val_images, val_labels, "驗證")

    # 建立 data.yaml
    yaml_content = f"""# YOLO OBB Dataset Configuration
# 自動產生於 {Path(__file__).name}

path: {output_path.absolute()}
train: train/images
val: val/images

# 類別
names:
"""
    for i, cls_name in enumerate(classes):
        yaml_content += f"  {i}: {cls_name}\n"

    yaml_file = output_path / "data.yaml"
    with open(yaml_file, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n已建立資料集配置: {yaml_file}")

    # 統計各類別標註數量
    print("\n" + "=" * 50)
    print("標註統計:")
    print("=" * 50)

    class_counts = {i: 0 for i in range(len(classes))}
    for _, label_file in valid_samples:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    for class_id, count in class_counts.items():
        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
        status = "✓" if count >= 100 else "⚠ 不足"
        print(f"  {class_name}: {count} 個標註 {status}")

    print("\n" + "=" * 50)
    print("資料集準備完成!")
    print(f"輸出目錄: {output_path}")
    print(f"配置檔: {yaml_file}")
    print("=" * 50)

    return str(yaml_file)


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB 資料集準備工具")
    parser.add_argument("--input", "-i", default="F:/yolov8_env/labeled_dataset",
                        help="標註資料夾路徑")
    parser.add_argument("--output", "-o", default="F:/yolov8_env/yolo_obb_dataset",
                        help="輸出資料夾路徑")
    parser.add_argument("--split", "-s", type=float, default=0.8,
                        help="訓練集比例 (預設: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="隨機種子")

    args = parser.parse_args()

    prepare_yolo_obb_dataset(
        input_folder=args.input,
        output_folder=args.output,
        train_ratio=args.split,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
