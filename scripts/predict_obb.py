"""
YOLO OBB 推論腳本
使用訓練好的模型進行物件偵測

使用方法:
    # 偵測單張圖片
    python predict_obb.py --source image.jpg

    # 偵測資料夾
    python predict_obb.py --source images_folder/

    # 偵測影片
    python predict_obb.py --source video.mp4

    # 即時攝影機
    python predict_obb.py --source 0
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def predict_obb(
    source,
    model_path="runs/obb/train/weights/best.pt",
    conf=0.25,
    iou=0.45,
    save=True,
    show=False,
    save_txt=False,
    save_crop=False,
    project="runs/obb/predict",
    name="exp",
    device=0
):
    """
    使用 YOLO OBB 模型進行推論

    參數:
        source: 輸入來源 (圖片/資料夾/影片/攝影機)
        model_path: 模型路徑
        conf: 信心閾值
        iou: NMS IoU 閾值
        save: 儲存結果圖片
        show: 顯示結果
        save_txt: 儲存標註檔
        save_crop: 儲存裁切圖
        project: 輸出目錄
        name: 實驗名稱
        device: GPU 裝置
    """
    print("\n" + "=" * 60)
    print("YOLO OBB 推論")
    print("=" * 60)

    # 檢查模型
    if not Path(model_path).exists():
        print(f"錯誤: 找不到模型 {model_path}")
        print("請先訓練模型或指定正確的模型路徑")
        return None

    print(f"模型: {model_path}")
    print(f"輸入: {source}")
    print(f"信心閾值: {conf}")
    print("=" * 60 + "\n")

    # 載入模型
    model = YOLO(model_path)

    # 執行推論
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        save=save,
        show=show,
        save_txt=save_txt,
        save_crop=save_crop,
        project=project,
        name=name,
        device=device,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("推論完成!")
    if save:
        print(f"結果已儲存至: {project}/{name}/")
    print("=" * 60)

    return results


def batch_predict_and_label(
    source_folder,
    output_folder,
    model_path="runs/obb/train/weights/best.pt",
    conf=0.25,
    device=0
):
    """
    批次預測並產生標註檔 (可用於半自動標註)

    這個功能可以用訓練過的模型自動標註新圖片，
    然後你只需要檢查和修正結果，大幅提升標註效率。
    """
    import cv2
    from pathlib import Path

    print("\n" + "=" * 60)
    print("批次自動標註")
    print("=" * 60)

    source_path = Path(source_folder)
    output_path = Path(output_folder)

    # 建立輸出目錄
    images_out = output_path / "images"
    labels_out = output_path / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # 載入模型
    model = YOLO(model_path)

    # 找出所有圖片
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [f for f in source_path.iterdir() if f.suffix.lower() in extensions]

    print(f"找到 {len(images)} 張圖片")
    print(f"模型: {model_path}")
    print(f"信心閾值: {conf}")
    print("=" * 60 + "\n")

    labeled_count = 0
    total_detections = 0

    for img_path in images:
        # 讀取圖片尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 執行推論
        results = model.predict(source=str(img_path), conf=conf, device=device, verbose=False)

        if results and len(results) > 0:
            result = results[0]

            # 檢查是否有 OBB 偵測結果
            if hasattr(result, 'obb') and result.obb is not None:
                obb = result.obb

                labels = []
                for i in range(len(obb)):
                    # 取得類別 ID
                    class_id = int(obb.cls[i].item())

                    # 取得 OBB 座標 (xyxyxyxy 格式)
                    if hasattr(obb, 'xyxyxyxy'):
                        box = obb.xyxyxyxy[i].cpu().numpy().flatten()
                        # 歸一化
                        normalized = []
                        for j in range(0, 8, 2):
                            x = box[j] / img_w
                            y = box[j + 1] / img_h
                            normalized.extend([x, y])
                        labels.append((class_id, normalized))
                        total_detections += 1

                # 儲存標註檔
                if labels:
                    label_path = labels_out / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        for class_id, coords in labels:
                            coords_str = ' '.join([f"{c:.6f}" for c in coords])
                            f.write(f"{class_id} {coords_str}\n")

                    # 複製圖片
                    import shutil
                    shutil.copy2(img_path, images_out / img_path.name)
                    labeled_count += 1

        print(f"已處理: {img_path.name}")

    print("\n" + "=" * 60)
    print("批次自動標註完成!")
    print(f"已標註圖片: {labeled_count} 張")
    print(f"總偵測數: {total_detections} 個")
    print(f"輸出目錄: {output_path}")
    print("=" * 60)
    print("\n提示: 請使用 sam3_labeler.py 檢查和修正自動標註結果")


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB 推論腳本")

    parser.add_argument("--source", required=True,
                        help="輸入來源 (圖片/資料夾/影片/攝影機編號)")
    parser.add_argument("--model", default="runs/obb/train/weights/best.pt",
                        help="模型路徑")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="信心閾值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 閾值")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU 裝置編號")
    parser.add_argument("--save", action="store_true", default=True,
                        help="儲存結果")
    parser.add_argument("--show", action="store_true",
                        help="顯示結果")
    parser.add_argument("--save-txt", action="store_true",
                        help="儲存標註檔")
    parser.add_argument("--save-crop", action="store_true",
                        help="儲存裁切圖")
    parser.add_argument("--project", default="runs/obb/predict",
                        help="輸出目錄")
    parser.add_argument("--name", default="exp",
                        help="實驗名稱")

    # 批次自動標註模式
    parser.add_argument("--auto-label", action="store_true",
                        help="批次自動標註模式")
    parser.add_argument("--output", default="auto_labeled",
                        help="自動標註輸出目錄")

    args = parser.parse_args()

    if args.auto_label:
        batch_predict_and_label(
            source_folder=args.source,
            output_folder=args.output,
            model_path=args.model,
            conf=args.conf,
            device=args.device
        )
    else:
        predict_obb(
            source=args.source,
            model_path=args.model,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show,
            save_txt=args.save_txt,
            save_crop=args.save_crop,
            project=args.project,
            name=args.name,
            device=args.device
        )


if __name__ == "__main__":
    main()
