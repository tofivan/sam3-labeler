"""
SAM 3 自動標註工具
使用 SAM 3 文字提示分割，自動產生 YOLO OBB 格式標註

使用方法:
    # 基本用法
    python auto_label_obb.py -i images_folder -o output_folder -t "trash,diver,boat"

    # 指定信心閾值
    python auto_label_obb.py -i images_folder -o output_folder -t "trash,diver,boat" --conf 0.3

    # 處理單張圖片
    python auto_label_obb.py -i image.jpg -o output_folder -t "trash,diver,boat"

    # 顯示視覺化結果
    python auto_label_obb.py -i images_folder -o output_folder -t "trash,diver,boat" --visualize
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


# 支援的圖片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def find_images(input_path, recursive=False):
    """尋找資料夾中的所有圖片檔案"""
    input_path = Path(input_path)
    images = []

    if input_path.is_file():
        if input_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [input_path]
        return []

    if recursive:
        for ext in IMAGE_EXTENSIONS:
            images.extend(input_path.rglob(f"*{ext}"))
            images.extend(input_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in IMAGE_EXTENSIONS:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def mask_to_obb(mask, img_width, img_height):
    """
    將二值化 Mask 轉換為 YOLO OBB 格式

    參數:
        mask: numpy array, 二值化遮罩
        img_width: 圖片寬度
        img_height: 圖片高度

    回傳:
        list: [x1, y1, x2, y2, x3, y3, x4, y4] 歸一化座標，或 None
    """
    # 確保 mask 是 numpy array 且為 uint8
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)

    # 如果 mask 是 3D，取第一個通道
    if len(mask.shape) == 3:
        mask = mask[0]

    # 確保是二值化
    mask = (mask > 0).astype(np.uint8) * 255

    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 取最大輪廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 如果輪廓太小，跳過
    if cv2.contourArea(largest_contour) < 100:
        return None

    # 計算最小外接旋轉矩形
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)

    # 歸一化座標
    normalized_box = []
    for point in box:
        x_norm = point[0] / img_width
        y_norm = point[1] / img_height
        # 確保座標在 0-1 範圍內
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))
        normalized_box.extend([x_norm, y_norm])

    return normalized_box


def save_yolo_obb(labels, output_path):
    """
    儲存 YOLO OBB 格式標註檔

    參數:
        labels: list of (class_id, [x1, y1, x2, y2, x3, y3, x4, y4])
        output_path: 輸出檔案路徑
    """
    with open(output_path, 'w') as f:
        for class_id, coords in labels:
            # YOLO OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
            coords_str = ' '.join([f"{c:.6f}" for c in coords])
            f.write(f"{class_id} {coords_str}\n")


def visualize_obb(image, labels, class_names, output_path):
    """
    視覺化 OBB 標註結果

    參數:
        image: numpy array, 原始圖片
        labels: list of (class_id, [x1, y1, x2, y2, x3, y3, x4, y4])
        class_names: list, 類別名稱
        output_path: 輸出圖片路徑
    """
    img_h, img_w = image.shape[:2]
    vis_image = image.copy()

    # 顏色列表
    colors = [
        (0, 255, 0),    # 綠
        (255, 0, 0),    # 藍
        (0, 0, 255),    # 紅
        (255, 255, 0),  # 青
        (255, 0, 255),  # 洋紅
        (0, 255, 255),  # 黃
    ]

    for class_id, coords in labels:
        color = colors[class_id % len(colors)]

        # 反歸一化座標
        points = []
        for i in range(0, 8, 2):
            x = int(coords[i] * img_w)
            y = int(coords[i + 1] * img_h)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        # 繪製旋轉矩形
        cv2.drawContours(vis_image, [points], 0, color, 2)

        # 繪製類別名稱
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        cv2.putText(vis_image, class_name, (points[0][0], points[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(str(output_path), vis_image)


def auto_label_with_sam3(
    input_path,
    output_folder,
    text_prompts,
    model_path="F:/yolov8_env/sam3.pt",
    conf_threshold=0.25,
    visualize=False,
    recursive=False
):
    """
    使用 SAM 3 自動標註圖片

    參數:
        input_path: 輸入圖片或資料夾路徑
        output_folder: 輸出資料夾路徑
        text_prompts: 文字提示列表 (類別名稱)
        model_path: SAM 3 模型路徑
        conf_threshold: 信心閾值
        visualize: 是否儲存視覺化結果
        recursive: 是否遞迴處理子資料夾
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    # 建立輸出資料夾
    output_folder = Path(output_folder)
    images_folder = output_folder / "images"
    labels_folder = output_folder / "labels"
    images_folder.mkdir(parents=True, exist_ok=True)
    labels_folder.mkdir(parents=True, exist_ok=True)

    if visualize:
        vis_folder = output_folder / "visualize"
        vis_folder.mkdir(parents=True, exist_ok=True)

    # 尋找圖片
    images = find_images(input_path, recursive=recursive)

    if not images:
        print(f"錯誤: 在 '{input_path}' 中找不到圖片檔案")
        return

    print(f"\n找到 {len(images)} 張圖片")
    print(f"文字提示: {text_prompts}")
    print(f"信心閾值: {conf_threshold}")

    # 載入 SAM 3 模型
    print(f"\n載入 SAM 3 模型...")
    predictor = SAM3SemanticPredictor(overrides=dict(
        conf=conf_threshold,
        model=model_path,
        half=True,
        verbose=False
    ))
    print("模型載入完成!")

    # 建立 classes.txt
    classes_file = output_folder / "classes.txt"
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in text_prompts:
            f.write(f"{class_name}\n")

    # 統計
    total_labels = 0
    processed_images = 0

    # 處理每張圖片
    print(f"\n開始自動標註...")
    for img_path in tqdm(images, desc="標註進度", unit="張"):
        try:
            # 讀取圖片
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"警告: 無法讀取 {img_path}")
                continue

            img_h, img_w = image.shape[:2]

            # 設定圖片並執行推理
            predictor.set_image(str(img_path))
            results = predictor(text=text_prompts)

            # 處理結果
            labels = []

            if results and len(results) > 0 and results[0].masks is not None:
                masks = results[0].masks.data
                boxes = results[0].boxes

                for i, mask in enumerate(masks):
                    # 取得類別 ID
                    if boxes is not None and boxes.cls is not None:
                        class_id = int(boxes.cls[i].item())
                    else:
                        class_id = 0

                    # Mask 轉 OBB
                    obb_coords = mask_to_obb(mask, img_w, img_h)

                    if obb_coords is not None:
                        labels.append((class_id, obb_coords))

            # 儲存標註檔
            label_filename = img_path.stem + ".txt"
            label_path = labels_folder / label_filename

            if labels:
                save_yolo_obb(labels, label_path)
                total_labels += len(labels)
            else:
                # 建立空的標註檔
                open(label_path, 'w').close()

            # 複製圖片到 images 資料夾
            import shutil
            shutil.copy2(img_path, images_folder / img_path.name)

            # 視覺化
            if visualize and labels:
                vis_path = vis_folder / f"vis_{img_path.name}"
                visualize_obb(image, labels, text_prompts, vis_path)

            processed_images += 1

        except Exception as e:
            print(f"錯誤處理 {img_path}: {e}")
            continue

    # 總結
    print(f"\n{'='*60}")
    print(f"自動標註完成!")
    print(f"  處理圖片數: {processed_images}")
    print(f"  總標註數量: {total_labels}")
    print(f"  輸出目錄: {output_folder}")
    print(f"\n輸出結構:")
    print(f"  {output_folder}/")
    print(f"  ├── images/     (圖片)")
    print(f"  ├── labels/     (YOLO OBB 標註)")
    print(f"  ├── classes.txt (類別名稱)")
    if visualize:
        print(f"  └── visualize/  (視覺化結果)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 自動標註工具 - 產生 YOLO OBB 格式標註",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本用法
  python auto_label_obb.py -i images_folder -o output -t "trash,diver,boat"

  # 指定信心閾值
  python auto_label_obb.py -i images_folder -o output -t "trash,diver,boat" --conf 0.3

  # 顯示視覺化結果
  python auto_label_obb.py -i images_folder -o output -t "trash,diver,boat" --visualize

  # 遞迴處理子資料夾
  python auto_label_obb.py -i images_folder -o output -t "trash,diver,boat" --recursive
        """
    )

    parser.add_argument("-i", "--input", required=True,
                        help="輸入圖片或資料夾路徑")
    parser.add_argument("-o", "--output", required=True,
                        help="輸出資料夾路徑")
    parser.add_argument("-t", "--text", required=True,
                        help="文字提示 (用逗號分隔)，例如: 'trash,diver,boat'")
    parser.add_argument("--model", default="F:/yolov8_env/sam3.pt",
                        help="SAM 3 模型路徑 (預設: F:/yolov8_env/sam3.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="信心閾值 (預設: 0.25)")
    parser.add_argument("--visualize", action="store_true",
                        help="儲存視覺化結果")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="遞迴處理子資料夾")

    args = parser.parse_args()

    # 解析文字提示
    text_prompts = [t.strip() for t in args.text.split(',')]

    # 檢查輸入路徑
    if not os.path.exists(args.input):
        print(f"錯誤: 找不到路徑 '{args.input}'")
        return 1

    # 執行自動標註
    auto_label_with_sam3(
        input_path=args.input,
        output_folder=args.output,
        text_prompts=text_prompts,
        model_path=args.model,
        conf_threshold=args.conf,
        visualize=args.visualize,
        recursive=args.recursive
    )

    return 0


if __name__ == "__main__":
    exit(main())
