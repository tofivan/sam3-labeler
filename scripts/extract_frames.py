r"""
影片擷取工具
從 AVI/MP4 影片中擷取單張照片

使用方法:
   # 單一影片，每秒擷取一張
python extract_frames.py video.mp4 -o frames -m seconds -i 1

# 整個資料夾的所有影片
python extract_frames.py "F:\yolov8_env\鼻頭資料\GOPRO\20251211" -o output_frames -m seconds -i 1

# 遞迴處理子資料夾
python extract_frames.py "F:\yolov8_env\鼻頭資料\GOPRO" -o output_frames -m seconds -i 1 --recursive

# 查看資料夾中所有影片的資訊
python extract_frames.py "F:\yolov8_env\鼻頭資料\GOPRO\20251211" --info

# 處理多個影片，全部輸出到同一資料夾，連續編號
python extract_frames.py "F:\yolov8_env\鼻頭資料\GOPRO" -o output_frames1 -m seconds -i 1 --recursive

# 自訂前綴
python extract_frames.py "F:\yolov8_env\鼻頭資料\GOPRO" -o output_frames --prefix img

# 指定起始編號 (從 1000 開始編號) 單一影片：從指定編號開始 多個影片：從指定編號開始，後續影片自動連續編號
python extract_frames.py video.mp4 -o frames -m seconds -i 1 --start-number 1000

"""

import os
# 解決 GoPro 等多串流影片的讀取警告
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "65536"

import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg'}


def find_videos(input_path, recursive=False):
    """尋找資料夾中的所有影片檔案"""
    input_path = Path(input_path)
    videos = []

    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            return [input_path]
        return []

    if recursive:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_path.rglob(f"*{ext}"))
            videos.extend(input_path.rglob(f"*{ext.upper()}"))
    else:
        for ext in VIDEO_EXTENSIONS:
            videos.extend(input_path.glob(f"*{ext}"))
            videos.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(set(videos))


def get_video_info(video_path):
    """取得影片基本資訊"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_frames(
    video_path,
    output_folder,
    mode="frames",
    interval=1,
    output_format="jpg",
    quality=95,
    resize=None,
    start_time=0,
    end_time=None,
    prefix="frame",
    start_number=0
):
    """
    從影片中擷取幀

    參數:
        video_path: 影片檔案路徑
        output_folder: 輸出資料夾
        mode: 擷取模式 - "frames"(按幀間隔) 或 "seconds"(按秒間隔)
        interval: 間隔值 (幀數或秒數)
        output_format: 輸出格式 (jpg, png, bmp)
        quality: JPEG 品質 (1-100)
        resize: 調整大小 (width, height) 或 None
        start_time: 開始時間(秒)
        end_time: 結束時間(秒), None 表示到影片結尾
        prefix: 輸出檔名前綴
        start_number: 起始編號 (用於多影片連續編號)
    """
    # 建立輸出資料夾
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")

    # 取得影片資訊
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # 計算幀間隔
    if mode == "seconds":
        frame_interval = int(fps * interval)
    else:
        frame_interval = int(interval)

    frame_interval = max(1, frame_interval)

    # 計算開始和結束幀
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)

    # 設定起始位置
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 設定輸出參數
    if output_format.lower() == "jpg":
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        ext = ".jpg"
    elif output_format.lower() == "png":
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ext = ".png"
    else:
        encode_param = []
        ext = f".{output_format}"

    # 計算預計擷取數量
    estimated_count = (end_frame - start_frame) // frame_interval

    print(f"\n{'='*50}")
    print(f"影片資訊:")
    print(f"  檔案: {video_path}")
    print(f"  解析度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"  FPS: {fps:.2f}")
    print(f"  總幀數: {total_frames}")
    print(f"  時長: {duration:.2f} 秒")
    print(f"\n擷取設定:")
    print(f"  模式: 每 {interval} {'秒' if mode == 'seconds' else '幀'}")
    print(f"  幀間隔: {frame_interval} 幀")
    print(f"  時間範圍: {start_time:.2f}s - {end_time if end_time else duration:.2f}s")
    print(f"  預計擷取: ~{estimated_count} 張圖片")
    print(f"  輸出格式: {output_format.upper()}")
    print(f"  輸出目錄: {output_folder}")
    print(f"{'='*50}\n")

    # 計算需要擷取的幀位置列表
    target_frames = list(range(start_frame, end_frame, frame_interval))

    print(f"使用 seek 模式擷取 {len(target_frames)} 幀...")

    current_number = start_number  # 使用起始編號
    saved_count = 0
    failed_count = 0

    for target_frame in tqdm(target_frames, desc="擷取進度", unit="張"):
        # 使用時間定位 (比幀定位更可靠)
        target_time_ms = (target_frame / fps) * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time_ms)

        ret, frame = cap.read()
        if not ret:
            failed_count += 1
            continue

        # 調整大小
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

        # 產生檔名（使用連續編號）
        timestamp = target_frame / fps
        filename = output_folder / f"{prefix}_{current_number:06d}_t{timestamp:.2f}s{ext}"

        # 儲存圖片
        cv2.imwrite(str(filename), frame, encode_param)
        current_number += 1
        saved_count += 1

    cap.release()

    if failed_count > 0:
        print(f"警告: {failed_count} 幀讀取失敗")

    print(f"\n完成! 共擷取 {saved_count} 張圖片 (編號 {start_number} - {current_number - 1})")
    print(f"儲存位置: {output_folder}")

    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="影片擷取工具 - 從影片中擷取單張照片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 單一影片，每秒擷取一張
  python extract_frames.py video.mp4 -o frames -m seconds -i 1

  # 整個資料夾的所有影片
  python extract_frames.py /path/to/videos -o frames -m seconds -i 1

  # 遞迴處理子資料夾
  python extract_frames.py /path/to/videos -o frames -m seconds -i 1 --recursive

  # 每30幀擷取一張
  python extract_frames.py video.mp4 -o frames -m frames -i 30

  # 擷取影片10-60秒之間，每5秒一張
  python extract_frames.py video.mp4 -o frames -m seconds -i 5 --start 10 --end 60

  # 擷取並調整大小為 640x480
  python extract_frames.py video.mp4 -o frames --resize 640 480

  # 指定起始編號 (從 1000 開始)
  python extract_frames.py video.mp4 -o frames -m seconds -i 1 --start-number 1000
        """
    )

    parser.add_argument("input", help="輸入影片檔案或資料夾路徑 (支援 mp4, avi, mov 等)")
    parser.add_argument("-o", "--output", default="output_frames", help="輸出資料夾 (預設: output_frames)")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="遞迴處理子資料夾中的影片")
    parser.add_argument("-m", "--mode", choices=["frames", "seconds"], default="frames",
                        help="擷取模式: frames=按幀間隔, seconds=按秒間隔 (預設: frames)")
    parser.add_argument("-i", "--interval", type=float, default=1,
                        help="間隔值: 幀數或秒數 (預設: 1)")
    parser.add_argument("-f", "--format", choices=["jpg", "png", "bmp"], default="jpg",
                        help="輸出圖片格式 (預設: jpg)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="JPEG 品質 1-100 (預設: 95)")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"),
                        help="調整輸出圖片大小，例如: --resize 640 480")
    parser.add_argument("--start", type=float, default=0,
                        help="開始時間(秒) (預設: 0)")
    parser.add_argument("--end", type=float, default=None,
                        help="結束時間(秒) (預設: 影片結尾)")
    parser.add_argument("--prefix", default="frame",
                        help="輸出檔名前綴 (預設: frame)")
    parser.add_argument("--info", action="store_true",
                        help="只顯示影片資訊，不擷取")
    parser.add_argument("--start-number", type=int, default=0,
                        help="輸出圖檔起始編號 (預設: 0)")

    args = parser.parse_args()

    # 檢查輸入路徑是否存在
    if not os.path.exists(args.input):
        print(f"錯誤: 找不到路徑 '{args.input}'")
        return 1

    # 尋找所有影片檔案
    videos = find_videos(args.input, recursive=args.recursive)

    if not videos:
        print(f"錯誤: 在 '{args.input}' 中找不到影片檔案")
        print(f"支援的格式: {', '.join(VIDEO_EXTENSIONS)}")
        return 1

    print(f"\n找到 {len(videos)} 個影片檔案")

    # 只顯示資訊
    if args.info:
        for video in videos:
            try:
                info = get_video_info(str(video))
                print(f"\n影片: {video.name}")
                print(f"  路徑: {video}")
                print(f"  解析度: {info['width']}x{info['height']}")
                print(f"  FPS: {info['fps']:.2f}")
                print(f"  總幀數: {info['total_frames']}")
                print(f"  時長: {info['duration']:.2f} 秒 ({info['duration']/60:.2f} 分鐘)")
            except Exception as e:
                print(f"\n影片: {video.name} - 錯誤: {e}")
        return 0

    # 擷取幀
    resize = tuple(args.resize) if args.resize else None
    total_extracted = 0
    current_number = args.start_number  # 使用指定的起始編號
    failed_videos = []

    for idx, video in enumerate(videos, 1):
        print(f"\n{'#'*60}")
        print(f"處理影片 [{idx}/{len(videos)}]: {video.name}")
        print(f"{'#'*60}")

        try:
            count = extract_frames(
                video_path=str(video),
                output_folder=args.output,  # 統一輸出到同一資料夾
                mode=args.mode,
                interval=args.interval,
                output_format=args.format,
                quality=args.quality,
                resize=resize,
                start_time=args.start,
                end_time=args.end,
                prefix=args.prefix,
                start_number=current_number  # 使用指定或連續編號
            )
            total_extracted += count
            current_number += count  # 更新編號供下一個影片使用
        except Exception as e:
            print(f"錯誤: {e}")
            failed_videos.append(video.name)

    # 總結
    print(f"\n{'='*60}")
    print(f"批次處理完成!")
    print(f"  處理影片數: {len(videos)}")
    print(f"  成功: {len(videos) - len(failed_videos)}")
    print(f"  失敗: {len(failed_videos)}")
    print(f"  總擷取圖片: {total_extracted} 張")
    print(f"  輸出目錄: {args.output}")
    if failed_videos:
        print(f"\n失敗的影片:")
        for v in failed_videos:
            print(f"  - {v}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
