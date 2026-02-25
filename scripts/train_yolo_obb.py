"""
YOLO OBB 訓練腳本
使用標註資料訓練旋轉邊界框物件偵測模型

使用方法:
    # 基本訓練
    python train_yolo_obb.py

    # 指定參數
    python train_yolo_obb.py --epochs 100 --batch 8 --model yolov8n-obb.pt

    # 從檢查點繼續訓練
    python train_yolo_obb.py --resume runs/obb/train/weights/last.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_yolo_obb(
    data_yaml,
    model="yolov8n-obb.pt",
    epochs=100,
    batch_size=8,
    img_size=640,
    device=0,
    project="runs/obb",
    name="train",
    resume=None,
    patience=20,
    workers=4
):
    """
    訓練 YOLO OBB 模型

    參數:
        data_yaml: 資料集配置檔路徑
        model: 預訓練模型 (yolov8n-obb, yolov8s-obb, yolov8m-obb, yolov8l-obb, yolov8x-obb)
        epochs: 訓練輪數
        batch_size: 批次大小
        img_size: 輸入圖片大小
        device: GPU 裝置 (0 = 第一張 GPU)
        project: 輸出目錄
        name: 實驗名稱
        resume: 繼續訓練的檢查點路徑
        patience: 早停耐心值 (幾輪沒改善就停止)
        workers: 資料載入執行緒數
    """
    print("\n" + "=" * 60)
    print("YOLO OBB 訓練")
    print("=" * 60)

    # 檢查資料集配置
    if not Path(data_yaml).exists():
        print(f"錯誤: 找不到資料集配置檔 {data_yaml}")
        print("請先執行 prepare_dataset.py 準備資料集")
        return None

    # 顯示訓練參數
    print(f"資料集: {data_yaml}")
    print(f"模型: {model}")
    print(f"訓練輪數: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"圖片大小: {img_size}")
    print(f"裝置: cuda:{device}")
    print(f"早停耐心值: {patience}")
    print("=" * 60 + "\n")

    # 載入模型
    if resume:
        print(f"從檢查點繼續訓練: {resume}")
        model_instance = YOLO(resume)
    else:
        print(f"載入預訓練模型: {model}")
        model_instance = YOLO(model)

    # 開始訓練
    results = model_instance.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        # 資料增強
        hsv_h=0.015,      # 色調變化
        hsv_s=0.7,        # 飽和度變化
        hsv_v=0.4,        # 亮度變化
        degrees=15.0,     # 旋轉角度 (OBB 重要)
        translate=0.1,    # 平移
        scale=0.5,        # 縮放
        flipud=0.5,       # 垂直翻轉 (水下適用)
        fliplr=0.5,       # 水平翻轉
        mosaic=1.0,       # Mosaic 增強
        mixup=0.1,        # MixUp 增強
        # 其他設定
        plots=True,       # 產生訓練圖表
        save=True,        # 儲存模型
        save_period=10,   # 每 N 輪儲存檢查點
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("訓練完成!")
    print("=" * 60)
    print(f"最佳模型: {project}/{name}/weights/best.pt")
    print(f"最後模型: {project}/{name}/weights/last.pt")
    print(f"訓練結果: {project}/{name}/")
    print("=" * 60)

    return results


def validate_model(model_path, data_yaml):
    """驗證模型效能"""
    print("\n驗證模型...")
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results


def export_model(model_path, format="onnx"):
    """匯出模型為其他格式"""
    print(f"\n匯出模型為 {format} 格式...")
    model = YOLO(model_path)
    model.export(format=format)


def main():
    parser = argparse.ArgumentParser(description="YOLO OBB 訓練腳本")

    # 資料設定
    parser.add_argument("--data", default="F:/yolov8_env/yolo_obb_dataset/data.yaml",
                        help="資料集配置檔路徑")

    # 模型設定
    parser.add_argument("--model", default="yolov8m-obb.pt",
                        choices=["yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt",
                                 "yolov8l-obb.pt", "yolov8x-obb.pt"],
                        help="預訓練模型 (n=最小, x=最大)")

    # 訓練參數
    parser.add_argument("--epochs", type=int, default=100, help="訓練輪數")
    parser.add_argument("--batch", type=int, default=8, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="圖片大小")
    parser.add_argument("--device", type=int, default=0, help="GPU 裝置編號")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    parser.add_argument("--workers", type=int, default=4, help="資料載入執行緒數")

    # 輸出設定
    parser.add_argument("--project", default="runs/obb", help="輸出目錄")
    parser.add_argument("--name", default="train", help="實驗名稱")

    # 繼續訓練
    parser.add_argument("--resume", default=None, help="從檢查點繼續訓練")

    # 其他動作
    parser.add_argument("--validate", action="store_true", help="只執行驗證")
    parser.add_argument("--export", default=None, help="匯出格式 (onnx, torchscript, etc.)")

    args = parser.parse_args()

    if args.validate:
        # 只驗證
        model_path = args.resume or "runs/obb/train/weights/best.pt"
        validate_model(model_path, args.data)
    elif args.export:
        # 只匯出
        model_path = args.resume or "runs/obb/train/weights/best.pt"
        export_model(model_path, args.export)
    else:
        # 訓練
        train_yolo_obb(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            patience=args.patience,
            workers=args.workers
        )


if __name__ == "__main__":
    main()
