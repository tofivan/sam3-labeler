"""
SAM 3 互動式標註工具
使用網頁介面進行人工標註，產生 YOLO OBB 格式資料集

使用方法:
    python sam3_labeler.py

    # 指定圖片資料夾
    python sam3_labeler.py --images F:/path/to/images

    # 指定輸出資料夾
    python sam3_labeler.py --output F:/path/to/output
"""

import os
import json
import cv2
import numpy as np
import gradio as gr
from pathlib import Path


CLASSES_STORE = Path(__file__).with_name("sam3_classes.txt")
PROGRESS_FILE = Path(__file__).with_name("sam3_progress.json")
CONFIG_FILE = Path(__file__).with_name("sam3_config.json")

# 預設配置
DEFAULT_CONFIG = {
    "images_folder": "./sample_images",
    "output_folder": "./output"
}


class LabelingState:
    """標註狀態管理"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_image = None
        self.current_image_path = None
        self.current_masks = []
        # 標註資料: [(class_id, obb_coords, polygon_coords, mask_binary), ...]
        # - obb_coords: OBB 座標 (8個數字)
        # - polygon_coords: 多邊形座標 (可變長度)
        # - mask_binary: 二值化 mask (numpy array 或 None)
        self.current_labels = []
        self.image_list = []
        self.current_index = 0
        self.classes = ["debris"]
        self.sam_predictor = None
        self.sam_model = None  # SAM 模型 (點擊/框選分割用)
        self.output_folder = Path("labeled_dataset")
        # 矩形框選狀態
        self.box_first_point = None  # 儲存第一個點擊位置 (x, y)
        # 標註選取狀態（多選）
        self.selected_labels = set()  # 選中的標註索引集合
        # 輸出格式設定
        self.output_formats = {
            "obb": True,      # OBB 格式 (預設開啟)
            "seg": True,      # YOLO-Seg 多邊形格式
            "mask": False,    # PNG mask
            "coco": False     # COCO JSON
        }
        # 多邊形簡化參數
        self.polygon_epsilon = 0.005
        # 重疊檢測閾值 (0.1 = 10% 重疊即視為衝突)
        self.overlap_threshold = 0.1
        # 顯示模式: "outline" (框線), "mask" (遮罩), "both" (框線+遮罩)
        self.display_mode = "outline"


state = LabelingState()


# ============================================================
# 工具函數
# ============================================================

def load_config():
    """讀取配置檔案"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 合併預設值（確保新增的配置項有預設值）
                return {**DEFAULT_CONFIG, **config}
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"讀取配置失敗: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(images_folder=None, output_folder=None):
    """儲存配置到檔案"""
    try:
        config = load_config()
        if images_folder:
            config["images_folder"] = images_folder
        if output_folder:
            config["output_folder"] = output_folder

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"儲存配置失敗: {e}")


def save_progress(folder_path, index):
    """儲存進度到 JSON 檔案"""
    try:
        progress = {}
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)

        # 使用資料夾路徑作為 key
        folder_key = str(Path(folder_path).resolve())
        progress[folder_key] = {
            "last_index": index,
            "last_image": state.image_list[index].name if state.image_list else ""
        }

        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"儲存進度失敗: {e}")


def load_progress(folder_path):
    """讀取進度，返回上次的索引"""
    try:
        if not PROGRESS_FILE.exists():
            return 0

        with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
            progress = json.load(f)

        folder_key = str(Path(folder_path).resolve())
        if folder_key in progress:
            return progress[folder_key].get("last_index", 0)
        return 0
    except Exception as e:
        print(f"讀取進度失敗: {e}")
        return 0


def mask_to_obb(mask, img_width, img_height):
    """將二值化 Mask 轉換為 OBB 座標"""
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[0]

    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return None

    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)

    normalized_box = []
    for point in box:
        x_norm = max(0, min(1, point[0] / img_width))
        y_norm = max(0, min(1, point[1] / img_height))
        normalized_box.extend([x_norm, y_norm])

    return normalized_box


def mask_to_polygon(mask, img_width, img_height, epsilon_ratio=0.005):
    """將二值化 Mask 轉換為多邊形座標 (YOLO-Seg 格式)

    Args:
        mask: 二值化 mask
        img_width: 圖片寬度
        img_height: 圖片高度
        epsilon_ratio: 多邊形簡化比例 (越小越精確，越大頂點越少)

    Returns:
        list: 正規化的多邊形座標 [x1, y1, x2, y2, ...]
    """
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[0]

    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return None

    # 簡化多邊形
    epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 確保至少有 3 個頂點
    if len(approx) < 3:
        return None

    # 正規化座標
    normalized_polygon = []
    for point in approx:
        x_norm = max(0, min(1, point[0][0] / img_width))
        y_norm = max(0, min(1, point[0][1] / img_height))
        normalized_polygon.extend([x_norm, y_norm])

    return normalized_polygon


def mask_to_binary_image(mask):
    """將 mask 轉換為二值化圖像 (用於儲存 PNG)

    Args:
        mask: SAM 輸出的 mask

    Returns:
        numpy.ndarray: 二值化圖像 (0 或 255)
    """
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[0]

    return (mask > 0).astype(np.uint8) * 255


def polygon_to_mask(polygon_coords, img_width, img_height):
    """將多邊形座標轉換回 mask 圖像

    Args:
        polygon_coords: 正規化的多邊形座標 [x1, y1, x2, y2, ...]
        img_width: 圖片寬度
        img_height: 圖片高度

    Returns:
        numpy.ndarray: 二值化 mask 圖像
    """
    # 轉換為像素座標
    points = []
    for i in range(0, len(polygon_coords), 2):
        x = int(polygon_coords[i] * img_width)
        y = int(polygon_coords[i + 1] * img_height)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)

    # 創建空白 mask 並填充多邊形
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    return mask


def check_mask_overlap(new_mask, existing_labels, img_width, img_height, threshold=None):
    """檢測新 mask 是否與現有標註重疊

    Args:
        new_mask: 新標註的二值化 mask
        existing_labels: 現有標註列表 [(class_id, obb_coords, polygon_coords, mask_binary), ...]
        img_width: 圖片寬度
        img_height: 圖片高度
        threshold: 重疊閾值，若為 None 則使用 state.overlap_threshold

    Returns:
        tuple: (is_overlapping, overlapping_label_idx, overlap_ratio)
    """
    if new_mask is None:
        return False, None, 0

    # 使用 state 中的閾值或傳入的值
    if threshold is None:
        threshold = state.overlap_threshold

    # 閾值為 0 時允許重疊，直接返回
    if threshold <= 0:
        return False, None, 0

    # 確保 new_mask 是正確格式
    if hasattr(new_mask, 'cpu'):
        new_mask = new_mask.cpu().numpy()
    new_mask = new_mask.astype(np.uint8)
    if len(new_mask.shape) == 3:
        new_mask = new_mask[0]

    # 確保尺寸正確
    if new_mask.shape != (img_height, img_width):
        new_mask = cv2.resize(new_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

    new_mask_binary = (new_mask > 0)
    new_mask_area = new_mask_binary.sum()

    if new_mask_area == 0:
        return False, None, 0

    for idx, label in enumerate(existing_labels):
        existing_mask = label[3] if len(label) > 3 and label[3] is not None else None

        if existing_mask is None:
            # 如果沒有 mask，從多邊形座標生成
            polygon_coords = label[2] if len(label) > 2 and label[2] else label[1]
            if polygon_coords:
                existing_mask = polygon_to_mask(polygon_coords, img_width, img_height)

        if existing_mask is None:
            continue

        # 確保尺寸一致
        if existing_mask.shape != (img_height, img_width):
            existing_mask = cv2.resize(existing_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

        existing_mask_binary = (existing_mask > 0)

        # 計算交集
        intersection = np.logical_and(new_mask_binary, existing_mask_binary)
        intersection_area = intersection.sum()

        if intersection_area > 0:
            # 計算重疊比例 (相對於新 mask)
            overlap_ratio = intersection_area / new_mask_area
            if overlap_ratio > threshold:
                return True, idx, overlap_ratio

    return False, None, 0


def create_coco_annotation(ann_id, image_id, category_id, polygon_coords, img_width, img_height):
    """建立單一 COCO annotation 物件

    Args:
        ann_id: annotation ID
        image_id: 圖片 ID
        category_id: 類別 ID
        polygon_coords: 正規化的多邊形座標
        img_width: 圖片寬度
        img_height: 圖片高度

    Returns:
        dict: COCO annotation 格式
    """
    # 轉換為像素座標
    segmentation = []
    for i in range(0, len(polygon_coords), 2):
        x = polygon_coords[i] * img_width
        y = polygon_coords[i + 1] * img_height
        segmentation.extend([x, y])

    # 計算 bounding box
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min

    # 計算面積 (使用 Shoelace formula)
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    area = abs(area) / 2.0

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [segmentation],
        "area": area,
        "bbox": [x_min, y_min, width, height],
        "iscrowd": 0
    }


def create_coco_dataset(image_list, labels_dict, classes, output_path):
    """建立完整的 COCO 資料集 JSON

    Args:
        image_list: 圖片路徑列表
        labels_dict: {image_name: [(class_id, polygon_coords), ...]}
        classes: 類別名稱列表
        output_path: 輸出路徑
    """
    coco_data = {
        "info": {
            "description": "SAM3 Labeler Dataset",
            "version": "1.0",
            "year": 2026
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 建立類別
    for idx, class_name in enumerate(classes):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })

    ann_id = 1
    for img_id, img_path in enumerate(image_list):
        img_name = img_path.name

        # 讀取圖片尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 添加圖片資訊
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": img_w,
            "height": img_h
        })

        # 添加標註
        if img_name in labels_dict:
            for class_id, polygon_coords in labels_dict[img_name]:
                if polygon_coords and len(polygon_coords) >= 6:
                    ann = create_coco_annotation(
                        ann_id, img_id, class_id,
                        polygon_coords, img_w, img_h
                    )
                    coco_data["annotations"].append(ann)
                    ann_id += 1

    # 儲存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)


def box_to_obb(x1, y1, x2, y2, img_w, img_h):
    """將矩形框轉換為軸向 OBB 座標"""
    if img_w <= 0 or img_h <= 0:
        return None

    box_x1 = max(0, min(img_w, min(x1, x2)))
    box_x2 = max(0, min(img_w, max(x1, x2)))
    box_y1 = max(0, min(img_h, min(y1, y2)))
    box_y2 = max(0, min(img_h, max(y1, y2)))

    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return None

    points = [
        (box_x1, box_y1),
        (box_x2, box_y1),
        (box_x2, box_y2),
        (box_x1, box_y2),
    ]

    normalized_box = []
    for px, py in points:
        x_norm = max(0, min(1, px / img_w))
        y_norm = max(0, min(1, py / img_h))
        normalized_box.extend([x_norm, y_norm])

    return normalized_box


def point_in_obb(x, y, obb_coords, img_w, img_h):
    """判斷點 (x, y) 是否在 OBB 內"""
    # 將正規化座標轉換為像素座標
    points = []
    for i in range(0, 8, 2):
        px = int(obb_coords[i] * img_w)
        py = int(obb_coords[i + 1] * img_h)
        points.append([px, py])

    polygon = np.array(points, dtype=np.int32)
    # 使用 OpenCV 的 pointPolygonTest
    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
    return result >= 0  # >= 0 表示在多邊形內或邊上


def find_clicked_label(x, y, labels, img_w, img_h):
    """找出點擊位置對應的標註索引，如果點擊多個重疊標註則返回最小的"""
    clicked_indices = []
    for idx, label in enumerate(labels):
        coords = label[1]  # OBB 座標
        if point_in_obb(x, y, coords, img_w, img_h):
            clicked_indices.append(idx)

    if clicked_indices:
        return clicked_indices[0]  # 返回第一個（最早建立的）
    return None


def obb_intersects_box(obb_coords, box_x1, box_y1, box_x2, box_y2, img_w, img_h):
    """判斷 OBB 是否與矩形框相交或被包含"""
    # 將 OBB 正規化座標轉為像素座標
    obb_points = []
    for i in range(0, 8, 2):
        px = obb_coords[i] * img_w
        py = obb_coords[i + 1] * img_h
        obb_points.append((px, py))

    # 計算 OBB 的中心點
    center_x = sum(p[0] for p in obb_points) / 4
    center_y = sum(p[1] for p in obb_points) / 4

    # 檢查 OBB 中心是否在框選範圍內
    if box_x1 <= center_x <= box_x2 and box_y1 <= center_y <= box_y2:
        return True

    # 檢查 OBB 的任一頂點是否在框選範圍內
    for px, py in obb_points:
        if box_x1 <= px <= box_x2 and box_y1 <= py <= box_y2:
            return True

    return False


def find_labels_in_box(x1, y1, x2, y2, labels, img_w, img_h):
    """找出框選範圍內的所有標註索引"""
    # 確保座標順序正確
    box_x1, box_x2 = min(x1, x2), max(x1, x2)
    box_y1, box_y2 = min(y1, y2), max(y1, y2)

    found_indices = []
    for idx, label in enumerate(labels):
        coords = label[1]  # OBB 座標
        if obb_intersects_box(coords, box_x1, box_y1, box_x2, box_y2, img_w, img_h):
            found_indices.append(idx)

    return found_indices


def draw_labels_on_image(image, labels, classes, pending_box_point=None, selected_indices=None, preview_box=None, display_mode=None):
    """在圖片上繪製標註

    Args:
        display_mode: 顯示模式 - "outline" (框線), "mask" (遮罩), "both" (框線+遮罩)
    """
    if image is None:
        return None

    vis_image = image.copy()
    img_h, img_w = vis_image.shape[:2]

    # 使用傳入的 display_mode 或全域設定
    if display_mode is None:
        display_mode = state.display_mode

    # 確保 selected_indices 是集合
    if selected_indices is None:
        selected_indices = set()

    colors = [
        (0, 255, 0),    # 綠
        (255, 0, 0),    # 藍
        (0, 0, 255),    # 紅
        (255, 255, 0),  # 青
        (255, 0, 255),  # 洋紅
        (0, 255, 255),  # 黃
    ]

    # 先繪製所有遮罩 (如果需要)
    if display_mode in ("mask", "both"):
        for idx, label in enumerate(labels):
            class_id = label[0]
            is_selected = idx in selected_indices
            color = colors[class_id % len(colors)]

            # 取得 mask 資料
            mask_binary = label[3] if len(label) > 3 and label[3] is not None else None

            if mask_binary is not None:
                # 確保 mask 尺寸正確
                if mask_binary.shape[:2] != (img_h, img_w):
                    mask_binary = cv2.resize(mask_binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

                # 建立彩色遮罩
                overlay = vis_image.copy()
                mask_color = (0, 255, 255) if is_selected else color
                overlay[mask_binary > 0] = mask_color

                # 半透明疊加
                alpha = 0.5 if is_selected else 0.35
                cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)
            else:
                # 如果沒有 mask，使用多邊形填充
                polygon_coords = label[2] if len(label) > 2 and label[2] else label[1]
                if polygon_coords:
                    poly_points = []
                    for i in range(0, len(polygon_coords), 2):
                        x = int(polygon_coords[i] * img_w)
                        y = int(polygon_coords[i + 1] * img_h)
                        poly_points.append([x, y])
                    poly_points = np.array(poly_points, dtype=np.int32)

                    overlay = vis_image.copy()
                    mask_color = (0, 255, 255) if is_selected else color
                    cv2.fillPoly(overlay, [poly_points], mask_color)
                    alpha = 0.5 if is_selected else 0.35
                    cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0, vis_image)

    # 繪製框線 (如果需要)
    if display_mode in ("outline", "both"):
        for idx, label in enumerate(labels):
            class_id, coords = label[0], label[1]  # 取 class_id 和 OBB 座標
            is_selected = idx in selected_indices
            color = colors[class_id % len(colors)]

            points = []
            for i in range(0, 8, 2):
                x = int(coords[i] * img_w)
                y = int(coords[i + 1] * img_h)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            if is_selected:
                cv2.drawContours(vis_image, [points], 0, (0, 255, 255), 4)  # 黃色粗框
            else:
                cv2.drawContours(vis_image, [points], 0, color, 2)

    # 繪製標籤文字
    for idx, label in enumerate(labels):
        class_id, coords = label[0], label[1]
        is_selected = idx in selected_indices
        color = colors[class_id % len(colors)]

        # 取得第一個點作為文字位置
        x = int(coords[0] * img_w)
        y = int(coords[1] * img_h)

        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
        label_text = f"{idx+1}. {class_name}"
        text_color = (0, 255, 255) if is_selected else color
        cv2.putText(vis_image, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # 繪製待選框的第一個點
    if pending_box_point is not None:
        x, y = pending_box_point
        # 繪製十字標記
        cross_size = 15
        cv2.line(vis_image, (x - cross_size, y), (x + cross_size, y), (255, 165, 0), 3)
        cv2.line(vis_image, (x, y - cross_size), (x, y + cross_size), (255, 165, 0), 3)
        # 繪製圓圈
        cv2.circle(vis_image, (x, y), 8, (255, 165, 0), 2)
        # 提示文字
        cv2.putText(vis_image, "點擊第二角落", (x + 15, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    # 繪製預覽框（用於即時顯示框選範圍）
    if preview_box is not None:
        x1, y1, x2, y2 = preview_box
        # 半透明填充
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 165, 0), -1)
        cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)
        # 橘色邊框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 165, 0), 2)

    return vis_image


def get_label_choices():
    """取得標註選項列表"""
    if not state.current_labels:
        return []

    choices = []
    for idx, label in enumerate(state.current_labels):
        class_id = label[0]
        class_name = state.classes[class_id] if class_id < len(state.classes) else f"class_{class_id}"
        choices.append(f"{idx+1}. {class_name}")

    return choices


def format_labels_text():
    """格式化標註文字顯示"""
    if not state.current_labels:
        return "目前沒有標註"

    lines = []
    for idx, label in enumerate(state.current_labels):
        class_id = label[0]
        class_name = state.classes[class_id] if class_id < len(state.classes) else f"class_{class_id}"
        lines.append(f"{idx+1}. {class_name}")

    return "\n".join(lines)

# ============================================================
# 類別管理
# ============================================================

def persist_classes():
    """保存類別清單到檔案"""
    try:
        with open(CLASSES_STORE, "w", encoding="utf-8") as f:
            for class_name in state.classes:
                f.write(f"{class_name}\n")
    except Exception as exc:
        print(f"寫入類別檔失敗: {exc}")


def load_persisted_classes():
    """載入已保存的類別清單"""
    try:
        with open(CLASSES_STORE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        classes = [line for line in lines if line]
        if classes:
            state.classes = classes
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"讀取類別檔失敗: {exc}")


def build_class_dropdown_update(value=None):
    """建立類別下拉選單更新"""
    choices = list(state.classes)
    if value is not None and value in choices:
        return gr.update(choices=choices, value=value)
    elif choices:
        return gr.update(choices=choices, value=choices[0])
    else:
        return gr.update(choices=[], value=None)


def build_class_radio_update(value=None):
    """建立類別 Radio 選項更新"""
    choices = list(state.classes)
    if value is not None and value in choices:
        return gr.update(choices=choices, value=value)
    elif choices:
        return gr.update(choices=choices, value=choices[0])
    else:
        return gr.update(choices=[], value=None)


# ============================================================
# SAM 3 模型
# ============================================================

def load_sam3_model():
    """載入 SAM 3 語義模型 (文字提示用)"""
    if state.sam_predictor is None:
        from ultralytics.models.sam import SAM3SemanticPredictor
        state.sam_predictor = SAM3SemanticPredictor(overrides=dict(
            conf=0.25,
            model="F:/yolov8_env/sam3.pt",
            device="cuda:0",  # 明確使用 GPU
            half=True,        # FP16 加速
            verbose=False
        ))
        print("✓ SAM 3 語義模型已載入 (GPU: cuda:0)")
    return state.sam_predictor


def load_sam_model():
    """載入 SAM 模型 (點擊/框選分割用)"""
    if state.sam_model is None:
        from ultralytics import SAM
        state.sam_model = SAM("F:/yolov8_env/sam3.pt")
        print("✓ SAM 點擊/框選模型已載入 (GPU: cuda:0)")
    return state.sam_model


# ============================================================
# 圖片管理
# ============================================================

def build_label_checkbox_update(selected=None, choices=None):
    """更新標註勾選清單"""
    if choices is None:
        choices = get_label_choices()
    if selected is None:
        selected = []
    return gr.update(choices=choices, value=selected)


def load_images_from_folder(folder_path, output_folder):
    """載入資料夾中的圖片"""
    if not folder_path or not os.path.exists(folder_path):
        return "請輸入有效的資料夾路徑", None, "", build_label_checkbox_update(choices=[]), ""

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    for f in Path(folder_path).iterdir():
        if f.suffix.lower() in extensions:
            images.append(f)

    images = sorted(images)

    if not images:
        return "資料夾中沒有找到圖片", None, "", build_label_checkbox_update(choices=[]), ""

    state.image_list = images
    state.output_folder = Path(output_folder)  # 設定輸出資料夾

    # 儲存路徑到配置（下次啟動自動填入）
    save_config(images_folder=folder_path, output_folder=output_folder)

    # 讀取上次進度
    last_index = load_progress(folder_path)
    if last_index >= len(images):
        last_index = 0
    state.current_index = last_index
    state.current_labels = []

    status, vis_image, labels_text, checkbox_update = load_current_image()

    # 返回跳轉輸入框的值
    jump_value = str(state.current_index + 1)

    if last_index > 0:
        status = f"從上次位置繼續 | {status}"

    return status, vis_image, labels_text, checkbox_update, jump_value


def count_labeled_images():
    """計算已標註的圖片數量"""
    if not state.image_list or not state.output_folder:
        return 0

    labels_folder = state.output_folder / "labels"
    if not labels_folder.exists():
        return 0

    count = 0
    for img_path in state.image_list:
        label_file = labels_folder / f"{img_path.stem}.txt"
        if label_file.exists() and label_file.stat().st_size > 0:
            count += 1
    return count


def load_current_image():
    """載入當前圖片"""
    if not state.image_list:
        return "沒有圖片可載入", None, "", build_label_checkbox_update(choices=[])

    img_path = state.image_list[state.current_index]
    image = cv2.imread(str(img_path))
    if image is None:
        return f"無法讀取圖片: {img_path}", None, "", build_label_checkbox_update(choices=[])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    state.current_image = image
    state.current_image_path = img_path
    state.current_labels = []
    state.box_first_point = None  # 清除框選狀態
    state.selected_labels.clear()  # 清除選取狀態

    # 檢查是否已有標註
    label_path = state.output_folder / "labels" / f"{img_path.stem}.txt"
    if label_path.exists():
        load_existing_labels(label_path)

    # 增強狀態列資訊
    labeled_count = count_labeled_images()
    current_labels_count = len(state.current_labels)
    status = f"📷 {state.current_index + 1}/{len(state.image_list)} | ✅ 已標註: {labeled_count} | 🏷️ 本張: {current_labels_count} 個"

    vis_image = draw_labels_on_image(image, state.current_labels, state.classes)
    labels_text = format_labels_text()

    return status, vis_image, labels_text, build_label_checkbox_update()


def load_existing_labels(label_path, seg_label_path=None):
    """載入已存在的標註

    Args:
        label_path: OBB 標註檔路徑
        seg_label_path: YOLO-Seg 標註檔路徑 (可選)
    """
    state.current_labels = []

    # 優先載入 YOLO-Seg 格式 (有完整多邊形)
    if seg_label_path and seg_label_path.exists():
        img_h, img_w = state.current_image.shape[:2]
        with open(seg_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:  # 至少 class_id + 3 個頂點
                    class_id = int(parts[0])
                    polygon_coords = [float(x) for x in parts[1:]]
                    # 從多邊形生成 OBB
                    mask = polygon_to_mask(polygon_coords, img_w, img_h)
                    obb_coords = mask_to_obb(mask, img_w, img_h)
                    if obb_coords:
                        state.current_labels.append((class_id, obb_coords, polygon_coords, mask))
        return

    # 載入 OBB 格式 (舊格式相容)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 9:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                # 舊格式：OBB 作為多邊形，生成對應的 mask
                img_h, img_w = state.current_image.shape[:2]
                mask = polygon_to_mask(coords, img_w, img_h)
                state.current_labels.append((class_id, coords, coords, mask))


def auto_save_current_labels():
    """自動儲存當前標註，支援多格式輸出。若無標註則刪除對應檔案"""
    if state.current_image is None or state.current_image_path is None:
        return None

    output_path = state.output_folder
    img_stem = state.current_image_path.stem

    # 如果沒有標註，刪除對應的標註檔案
    if not state.current_labels:
        deleted_files = []
        # 刪除 OBB 標註檔
        obb_label_path = output_path / "labels" / f"{img_stem}.txt"
        if obb_label_path.exists():
            obb_label_path.unlink()
            deleted_files.append("OBB")
        # 刪除 Seg 標註檔
        seg_label_path = output_path / "labels_seg" / f"{img_stem}.txt"
        if seg_label_path.exists():
            seg_label_path.unlink()
            deleted_files.append("Seg")
        # 刪除 Mask 檔案
        mask_path = output_path / "masks" / f"{img_stem}.png"
        if mask_path.exists():
            mask_path.unlink()
            deleted_files.append("Mask")
        # 刪除圖片副本（可選，保留圖片副本可能更好）
        # img_copy_path = output_path / "images" / state.current_image_path.name
        # if img_copy_path.exists():
        #     img_copy_path.unlink()

        if deleted_files:
            return f"🗑️ 已刪除標註檔 ({', '.join(deleted_files)})"
        return None

    # 有標註時，儲存標註
    images_folder = output_path / "images"
    images_folder.mkdir(parents=True, exist_ok=True)

    # 儲存圖片
    import shutil
    shutil.copy2(state.current_image_path, images_folder / state.current_image_path.name)

    saved_formats = []

    # 儲存 OBB 格式 (保持 labels 目錄名稱，向後兼容)
    if state.output_formats.get("obb", True):
        labels_folder = output_path / "labels"
        labels_folder.mkdir(parents=True, exist_ok=True)
        label_path = labels_folder / f"{img_stem}.txt"
        with open(label_path, 'w') as f:
            for label in state.current_labels:
                class_id, obb_coords = label[0], label[1]
                coords_str = ' '.join([f"{c:.6f}" for c in obb_coords])
                f.write(f"{class_id} {coords_str}\n")
        saved_formats.append("OBB")

    # 儲存 YOLO-Seg 多邊形格式
    if state.output_formats.get("seg", True):
        labels_seg_folder = output_path / "labels_seg"
        labels_seg_folder.mkdir(parents=True, exist_ok=True)
        label_path = labels_seg_folder / f"{img_stem}.txt"
        with open(label_path, 'w') as f:
            for label in state.current_labels:
                class_id = label[0]
                polygon_coords = label[2] if len(label) > 2 and label[2] else label[1]
                if polygon_coords:
                    coords_str = ' '.join([f"{c:.6f}" for c in polygon_coords])
                    f.write(f"{class_id} {coords_str}\n")
        saved_formats.append("Seg")

    # 儲存 PNG Mask
    if state.output_formats.get("mask", False):
        masks_folder = output_path / "masks"
        masks_folder.mkdir(parents=True, exist_ok=True)
        img_h, img_w = state.current_image.shape[:2]

        # 建立合併的 mask (每個類別用不同值)
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for idx, label in enumerate(state.current_labels):
            mask_binary = label[3] if len(label) > 3 and label[3] is not None else None
            if mask_binary is not None:
                # 確保 mask 尺寸正確
                if mask_binary.shape != (img_h, img_w):
                    mask_binary = cv2.resize(mask_binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                combined_mask[mask_binary > 0] = idx + 1  # 用標註索引+1 作為像素值

        mask_path = masks_folder / f"{img_stem}.png"
        cv2.imwrite(str(mask_path), combined_mask)
        saved_formats.append("Mask")

    # 更新 classes.txt
    classes_file = output_path / "classes.txt"
    with open(classes_file, 'w', encoding='utf-8') as f:
        for class_name in state.classes:
            f.write(f"{class_name}\n")
    persist_classes()

    formats_str = "+".join(saved_formats) if saved_formats else "無"
    return f"💾 已自動儲存 {len(state.current_labels)} 個標註 ({formats_str})"


def next_image(output_folder):
    """下一張圖片（自動儲存當前標註）"""
    if not state.image_list:
        return "請先載入圖片資料夾", None, "", build_label_checkbox_update(choices=[]), ""

    # 更新輸出資料夾
    state.output_folder = Path(output_folder)

    # 自動儲存當前標註
    save_msg = auto_save_current_labels()

    if state.current_index < len(state.image_list) - 1:
        state.current_index += 1
        # 儲存進度
        save_progress(state.image_list[0].parent, state.current_index)
        status, vis_image, labels_text, checkbox_update = load_current_image()
        if save_msg:
            status = f"{save_msg} | {status}"
        return status, vis_image, labels_text, checkbox_update, str(state.current_index + 1)
    else:
        if save_msg:
            return f"{save_msg} | 已經是最後一張圖片", None, format_labels_text(), build_label_checkbox_update(), str(state.current_index + 1)
        return "已經是最後一張圖片", None, format_labels_text(), build_label_checkbox_update(), str(state.current_index + 1)


def prev_image(output_folder):
    """上一張圖片（自動儲存當前標註）"""
    if not state.image_list:
        return "請先載入圖片資料夾", None, "", build_label_checkbox_update(choices=[]), ""

    # 更新輸出資料夾
    state.output_folder = Path(output_folder)

    # 自動儲存當前標註
    save_msg = auto_save_current_labels()

    if state.current_index > 0:
        state.current_index -= 1
        # 儲存進度
        save_progress(state.image_list[0].parent, state.current_index)
        status, vis_image, labels_text, checkbox_update = load_current_image()
        if save_msg:
            status = f"{save_msg} | {status}"
        return status, vis_image, labels_text, checkbox_update, str(state.current_index + 1)
    else:
        if save_msg:
            return f"{save_msg} | 已經是第一張圖片", None, format_labels_text(), build_label_checkbox_update(), str(state.current_index + 1)
        return "已經是第一張圖片", None, format_labels_text(), build_label_checkbox_update(), str(state.current_index + 1)


def jump_to_image(jump_index, output_folder):
    """跳轉到指定圖片"""
    if not state.image_list:
        return "請先載入圖片資料夾", None, "", build_label_checkbox_update(choices=[]), ""

    try:
        target_index = int(jump_index) - 1  # 轉換為 0-based 索引
    except (ValueError, TypeError):
        return f"請輸入有效數字 (1-{len(state.image_list)})", None, format_labels_text(), build_label_checkbox_update(), str(state.current_index + 1)

    if target_index < 0 or target_index >= len(state.image_list):
        return f"超出範圍，請輸入 1-{len(state.image_list)}", None, format_labels_text(), build_label_checkbox_update(), str(state.current_index + 1)

    # 更新輸出資料夾
    state.output_folder = Path(output_folder)

    # 自動儲存當前標註
    save_msg = auto_save_current_labels()

    # 跳轉
    state.current_index = target_index
    # 儲存進度
    save_progress(state.image_list[0].parent, state.current_index)

    status, vis_image, labels_text, checkbox_update = load_current_image()
    if save_msg:
        status = f"{save_msg} | {status}"

    return status, vis_image, labels_text, checkbox_update, str(state.current_index + 1)


# ============================================================
# 分割功能
# ============================================================

def segment_with_text(text_prompt, _default_class):
    """使用文字提示進行分割"""
    if state.current_image is None:
        return ("請先載入圖片", None, "", build_label_checkbox_update(choices=[]),
                build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())

    if not text_prompt.strip():
        return ("請輸入文字提示", None, "", build_label_checkbox_update(choices=[]),
                build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())

    try:
        predictor = load_sam3_model()

        # 暫存圖片
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(state.current_image, cv2.COLOR_RGB2BGR))

        predictor.set_image(temp_path)

        # 解析提示詞
        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            os.remove(temp_path)
            return ("請輸入文字提示", None, "", build_label_checkbox_update(choices=[]),
                    build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())

        # 自動新增不存在的類別
        new_classes = [p for p in prompts if p not in state.classes]
        if new_classes:
            state.classes.extend(new_classes)
            persist_classes()

        results = predictor(text=prompts)
        os.remove(temp_path)

        if results and len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data
            boxes = results[0].boxes
            img_h, img_w = state.current_image.shape[:2]

            added_count = 0
            skipped_count = 0

            for i, mask in enumerate(masks):
                if boxes is not None and boxes.cls is not None:
                    class_id = int(boxes.cls[i].item())
                else:
                    class_id = 0

                if class_id >= len(prompts):
                    class_id = 0

                obb_coords = mask_to_obb(mask, img_w, img_h)
                if obb_coords is not None:
                    prompt_class = prompts[min(class_id, len(prompts)-1)]
                    if prompt_class in state.classes:
                        class_id = state.classes.index(prompt_class)
                    # 同時計算多邊形座標和儲存 mask
                    polygon_coords = mask_to_polygon(mask, img_w, img_h, state.polygon_epsilon)
                    mask_binary = mask_to_binary_image(mask)

                    # 檢查是否與現有標註重疊
                    is_overlap, _, _ = check_mask_overlap(
                        mask_binary, state.current_labels, img_w, img_h
                    )
                    if is_overlap:
                        skipped_count += 1
                        continue  # 跳過重疊的標註

                    state.current_labels.append((class_id, obb_coords, polygon_coords, mask_binary))
                    added_count += 1

            vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
            if skipped_count > 0:
                return (f"偵測到 {len(masks)} 個物體，已加入 {added_count} 個，跳過 {skipped_count} 個 (重疊)", vis_image, format_labels_text(), build_label_checkbox_update(),
                        build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())
            return (f"偵測到 {len(masks)} 個物體", vis_image, format_labels_text(), build_label_checkbox_update(),
                    build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())
        else:
            vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
            return ("沒有偵測到物體", vis_image, format_labels_text(), build_label_checkbox_update(),
                    build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())

    except Exception as e:
        return (f"錯誤: {str(e)}", None, "", build_label_checkbox_update(choices=[]),
                build_class_radio_update(), build_class_dropdown_update(), build_class_dropdown_update())


def segment_with_point_internal(x, y, default_class_id):
    """內部點擊分割函數"""
    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(state.current_image, cv2.COLOR_RGB2BGR))

    sam_model = load_sam_model()
    results = sam_model.predict(source=temp_path, points=[[x, y]], labels=[1], device="cuda:0")

    os.remove(temp_path)

    if results and len(results) > 0 and results[0].masks is not None:
        masks_data = results[0].masks.data
        # 檢查 masks.data 是否為空
        if len(masks_data) > 0:
            mask = masks_data[0]
            img_h, img_w = state.current_image.shape[:2]

            obb_coords = mask_to_obb(mask, img_w, img_h)
            if obb_coords is not None:
                # 同時計算多邊形座標和儲存 mask
                polygon_coords = mask_to_polygon(mask, img_w, img_h, state.polygon_epsilon)
                mask_binary = mask_to_binary_image(mask)

                # 檢查是否與現有標註重疊
                is_overlap, overlap_idx, overlap_ratio = check_mask_overlap(
                    mask_binary, state.current_labels, img_w, img_h
                )
                if is_overlap:
                    overlap_label = state.current_labels[overlap_idx]
                    overlap_class = state.classes[overlap_label[0]] if overlap_label[0] < len(state.classes) else f"class_{overlap_label[0]}"
                    return False, f"⚠️ 此區域與 [{overlap_idx + 1}. {overlap_class}] 重疊 ({overlap_ratio*100:.0f}%)，無法建立標註"

                state.current_labels.append((default_class_id, obb_coords, polygon_coords, mask_binary))
                return True, f"在 ({x}, {y}) 偵測到物體"

    return False, f"在 ({x}, {y}) 沒有偵測到物體"


def segment_with_box_internal(x1, y1, x2, y2, default_class_id):
    """內部矩形框選分割函數"""
    # 確保座標順序正確 (左上到右下)
    box_x1 = min(x1, x2)
    box_y1 = min(y1, y2)
    box_x2 = max(x1, x2)
    box_y2 = max(y1, y2)

    temp_path = "temp_image.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(state.current_image, cv2.COLOR_RGB2BGR))

    sam_model = load_sam_model()
    # 使用 bboxes 參數進行框選分割
    results = sam_model.predict(source=temp_path, bboxes=[[box_x1, box_y1, box_x2, box_y2]], device="cuda:0")

    os.remove(temp_path)

    if results and len(results) > 0 and results[0].masks is not None:
        masks_data = results[0].masks.data
        # 檢查 masks.data 是否為空
        if len(masks_data) > 0:
            mask = masks_data[0]
            img_h, img_w = state.current_image.shape[:2]

            obb_coords = mask_to_obb(mask, img_w, img_h)
            if obb_coords is not None:
                # 同時計算多邊形座標和儲存 mask
                polygon_coords = mask_to_polygon(mask, img_w, img_h, state.polygon_epsilon)
                mask_binary = mask_to_binary_image(mask)

                # 檢查是否與現有標註重疊
                is_overlap, overlap_idx, overlap_ratio = check_mask_overlap(
                    mask_binary, state.current_labels, img_w, img_h
                )
                if is_overlap:
                    overlap_label = state.current_labels[overlap_idx]
                    overlap_class = state.classes[overlap_label[0]] if overlap_label[0] < len(state.classes) else f"class_{overlap_label[0]}"
                    return False, f"⚠️ 此區域與 [{overlap_idx + 1}. {overlap_class}] 重疊 ({overlap_ratio*100:.0f}%)，無法建立標註"

                state.current_labels.append((default_class_id, obb_coords, polygon_coords, mask_binary))
                return True, f"在框選區域 ({box_x1}, {box_y1}) - ({box_x2}, {box_y2}) 偵測到物體"

    return False, f"在框選區域沒有偵測到物體"


def add_box_label(x1, y1, x2, y2, class_id):
    """使用軸向矩形框建立標註"""
    img_h, img_w = state.current_image.shape[:2]
    if img_w <= 0 or img_h <= 0:
        return False, "圖片尺寸錯誤"

    if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
        return False, "框選範圍太小，無法建立標註"

    obb_coords = box_to_obb(x1, y1, x2, y2, img_w, img_h)
    if obb_coords is None:
        return False, "框選範圍太小，無法建立標註"

    # 為方框建立對應的多邊形座標 (4個頂點)
    polygon_coords = obb_coords.copy()  # 方框的 OBB 就是多邊形

    # 建立方框對應的 mask
    mask_binary = polygon_to_mask(polygon_coords, img_w, img_h)

    # 檢查是否與現有標註重疊
    is_overlap, overlap_idx, overlap_ratio = check_mask_overlap(
        mask_binary, state.current_labels, img_w, img_h
    )
    if is_overlap:
        overlap_label = state.current_labels[overlap_idx]
        overlap_class = state.classes[overlap_label[0]] if overlap_label[0] < len(state.classes) else f"class_{overlap_label[0]}"
        return False, f"⚠️ 此區域與 [{overlap_idx + 1}. {overlap_class}] 重疊 ({overlap_ratio*100:.0f}%)，無法建立標註"

    state.current_labels.append((class_id, obb_coords, polygon_coords, mask_binary))
    return True, "已用方框建立標註"


def segment_with_box_fallback(x1, y1, x2, y2, default_class_id, fallback_to_box):
    """矩形框選分割，必要時以方框做 fallback"""
    success, msg = segment_with_box_internal(x1, y1, x2, y2, default_class_id)
    if success:
        return True, msg, False

    if fallback_to_box:
        added, fallback_msg = add_box_label(x1, y1, x2, y2, default_class_id)
        if added:
            return True, "SAM 未偵測到物體，已用方框建立標註", True
        return False, fallback_msg, True

    return False, msg, False


def format_selected_label_info():
    """格式化選中標註的資訊（支援多選）"""
    # 清理無效的索引
    valid_indices = {idx for idx in state.selected_labels if idx < len(state.current_labels)}
    state.selected_labels = valid_indices

    if not state.selected_labels:
        return "尚未選取（點擊標註框切換選取）"

    # 顯示選中的標註
    selected_info = []
    for idx in sorted(state.selected_labels):
        class_id = state.current_labels[idx][0]
        class_name = state.classes[class_id] if class_id < len(state.classes) else f"class_{class_id}"
        selected_info.append(f"[{idx + 1}]{class_name}")

    return f"已選 {len(state.selected_labels)} 個: {', '.join(selected_info)}"


def handle_image_click(_image_with_points, default_class, click_mode, fallback_to_box, evt: gr.SelectData):
    """處理圖片點擊事件 (統一處理點擊、框選、選取模式)"""
    if state.current_image is None:
        return "請先載入圖片", None, "", build_label_checkbox_update(choices=[]), format_selected_label_info()

    if evt is None or evt.index is None:
        vis_image = draw_labels_on_image(
            state.current_image, state.current_labels, state.classes,
            selected_indices=state.selected_labels
        )
        return "無效的點擊座標", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    if not isinstance(evt.index, (list, tuple)) or len(evt.index) != 2:
        vis_image = draw_labels_on_image(
            state.current_image, state.current_labels, state.classes,
            selected_indices=state.selected_labels
        )
        return "無效的點擊座標", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    # 取得預設類別 ID
    if default_class and default_class in state.classes:
        default_class_id = state.classes.index(default_class)
    else:
        default_class_id = 0

    x, y = evt.index
    img_h, img_w = state.current_image.shape[:2]

    try:
        if click_mode == "選取標註":
            # 選取模式：點擊切換選取狀態 或 框選多個標註
            clicked_idx = find_clicked_label(x, y, state.current_labels, img_w, img_h)

            if state.box_first_point is not None:
                # 已有第一個點，完成框選
                x1, y1 = state.box_first_point
                state.box_first_point = None

                # 找出框選範圍內的所有標註
                found_indices = find_labels_in_box(x1, y1, x, y, state.current_labels, img_w, img_h)

                if found_indices:
                    # 將找到的標註加入選取（切換模式）
                    for idx in found_indices:
                        if idx in state.selected_labels:
                            state.selected_labels.discard(idx)
                        else:
                            state.selected_labels.add(idx)

                    vis_image = draw_labels_on_image(
                        state.current_image, state.current_labels, state.classes,
                        selected_indices=state.selected_labels
                    )
                    return f"框選切換了 {len(found_indices)} 個標註", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()
                else:
                    vis_image = draw_labels_on_image(
                        state.current_image, state.current_labels, state.classes,
                        selected_indices=state.selected_labels
                    )
                    return "框選範圍內沒有標註", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

            elif clicked_idx is not None:
                # 點擊到標註，切換選取狀態
                if clicked_idx in state.selected_labels:
                    state.selected_labels.discard(clicked_idx)
                    action = "取消選取"
                else:
                    state.selected_labels.add(clicked_idx)
                    action = "加入選取"

                vis_image = draw_labels_on_image(
                    state.current_image, state.current_labels, state.classes,
                    selected_indices=state.selected_labels
                )
                class_id = state.current_labels[clicked_idx][0]
                class_name = state.classes[class_id] if class_id < len(state.classes) else f"class_{class_id}"
                return f"{action} [{clicked_idx + 1}] {class_name}", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()
            else:
                # 點擊空白處，開始框選
                state.box_first_point = (x, y)
                vis_image = draw_labels_on_image(
                    state.current_image, state.current_labels, state.classes,
                    pending_box_point=(x, y),
                    selected_indices=state.selected_labels
                )
                return f"框選起點 ({x}, {y})，點擊第二角落完成框選", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

        elif click_mode == "點擊分割":
            # 點擊模式：直接分割
            state.box_first_point = None  # 清除任何待處理的框選
            state.selected_labels.clear()  # 清除選取
            success, msg = segment_with_point_internal(x, y, default_class_id)

            vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
            class_name = state.classes[default_class_id]
            if success:
                return f"{msg}，類別: {class_name}", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()
            else:
                return msg, vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

        else:
            # 矩形框選模式
            state.selected_labels.clear()  # 清除選取
            if state.box_first_point is None:
                # 第一次點擊：記錄位置，顯示標記
                state.box_first_point = (x, y)
                vis_image = draw_labels_on_image(
                    state.current_image, state.current_labels, state.classes,
                    pending_box_point=(x, y)
                )
                return f"已設定第一個角落 ({x}, {y})，請點擊第二個角落", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()
            else:
                # 第二次點擊：執行框選分割
                x1, y1 = state.box_first_point
                state.box_first_point = None  # 清除狀態

                success, msg, _used_fallback = segment_with_box_fallback(
                    x1, y1, x, y, default_class_id, fallback_to_box
                )

                vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
                class_name = state.classes[default_class_id]
                if success:
                    return f"{msg}，類別: {class_name}", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()
                else:
                    return msg, vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    except Exception as e:
        state.box_first_point = None  # 發生錯誤時清除狀態
        state.selected_labels.clear()
        vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
        return f"錯誤: {str(e)}", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()


def handle_drag_box(bbox_text, default_class, fallback_to_box):
    """處理拖曳矩形框選"""
    if state.current_image is None:
        return "請先載入圖片", None, "", build_label_checkbox_update(choices=[]), format_selected_label_info()

    # 取得預設類別 ID
    if default_class and default_class in state.classes:
        default_class_id = state.classes.index(default_class)
    else:
        default_class_id = 0

    try:
        raw = str(bbox_text or "").strip()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError("bbox parts")

        x1, y1, x2, y2 = [float(p) for p in parts]
    except Exception:
        vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
        return "框選資料格式錯誤", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    img_h, img_w = state.current_image.shape[:2]
    x1 = max(0, min(img_w, int(round(x1))))
    y1 = max(0, min(img_h, int(round(y1))))
    x2 = max(0, min(img_w, int(round(x2))))
    y2 = max(0, min(img_h, int(round(y2))))

    state.box_first_point = None
    state.selected_labels.clear()

    success, msg, _used_fallback = segment_with_box_fallback(
        x1, y1, x2, y2, default_class_id, fallback_to_box
    )
    vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
    class_name = state.classes[default_class_id]
    if success:
        return f"{msg}，類別: {class_name}", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    return msg, vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()


def delete_selected_label():
    """刪除選中的標註（支援多選）"""
    if state.current_image is None:
        return "請先載入圖片", None, "", build_label_checkbox_update(choices=[]), format_selected_label_info()

    if not state.selected_labels:
        vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
        return "請先選取要刪除的標註", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    # 按索引從大到小排序刪除，避免索引偏移問題
    indices_to_delete = sorted(state.selected_labels, reverse=True)
    deleted_count = 0

    for idx in indices_to_delete:
        if idx < len(state.current_labels):
            del state.current_labels[idx]
            deleted_count += 1

    state.selected_labels.clear()
    vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
    return f"已刪除 {deleted_count} 個標註", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()


def change_selected_label_class(new_class):
    """更改選中標註的類別（支援多選）"""
    if state.current_image is None:
        return "請先載入圖片", None, "", build_label_checkbox_update(choices=[]), format_selected_label_info()

    if not state.selected_labels:
        vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
        return "請先選取要修改的標註", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    if not new_class or new_class not in state.classes:
        vis_image = draw_labels_on_image(
            state.current_image, state.current_labels, state.classes,
            selected_indices=state.selected_labels
        )
        return "請選擇有效的類別", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()

    new_class_id = state.classes.index(new_class)
    modified_count = 0

    for idx in state.selected_labels:
        if idx < len(state.current_labels):
            label = state.current_labels[idx]
            # 保留其他字段，只更新 class_id
            state.current_labels[idx] = (new_class_id,) + label[1:]
            modified_count += 1

    vis_image = draw_labels_on_image(
        state.current_image, state.current_labels, state.classes,
        selected_indices=state.selected_labels
    )
    return f"已將 {modified_count} 個標註改為 {new_class}", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()


def deselect_label():
    """取消所有選取和框選狀態"""
    if state.current_image is None:
        return "請先載入圖片", None, "", build_label_checkbox_update(choices=[]), format_selected_label_info()

    state.selected_labels.clear()
    state.box_first_point = None  # 清除框選狀態
    vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
    return "已取消所有選取", vis_image, format_labels_text(), build_label_checkbox_update(), format_selected_label_info()


# ============================================================
# 標註編輯
# ============================================================

def change_single_label_class(label_index, new_class):
    """修改單一標註的類別"""
    if not state.current_labels:
        return "沒有標註可修改", None, "", build_label_checkbox_update(choices=[])

    if not new_class:
        return "請選擇類別", None, format_labels_text(), build_label_checkbox_update()

    try:
        idx = int(label_index) - 1
        if 0 <= idx < len(state.current_labels):
            if new_class not in state.classes:
                state.classes.append(new_class)
                persist_classes()

            class_id = state.classes.index(new_class)
            label = state.current_labels[idx]
            # 保留其他字段，只更新 class_id
            state.current_labels[idx] = (class_id,) + label[1:]

            vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
            return f"已將標註 {label_index} 改為 {new_class}", vis_image, format_labels_text(), build_label_checkbox_update()
        else:
            return f"無效的標註編號: {label_index}", None, format_labels_text(), build_label_checkbox_update()
    except Exception as e:
        return f"錯誤: {str(e)}", None, "", build_label_checkbox_update(choices=[])


def delete_single_label(label_index):
    """刪除單一標註"""
    if not state.current_labels:
        return "沒有標註可刪除", None, "", build_label_checkbox_update(choices=[])

    try:
        idx = int(label_index) - 1
        if 0 <= idx < len(state.current_labels):
            del state.current_labels[idx]
            vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
            return f"已刪除標註 {label_index}", vis_image, format_labels_text(), build_label_checkbox_update()
        else:
            return f"無效的標註編號: {label_index}", None, format_labels_text(), build_label_checkbox_update()
    except Exception as e:
        return f"錯誤: {str(e)}", None, "", build_label_checkbox_update(choices=[])


def batch_change_class(selected_labels, new_class):
    """批次修改標註的類別"""
    if not state.current_labels:
        return "沒有標註可修改", None, "", build_label_checkbox_update(choices=[])

    if not selected_labels:
        return "請先選擇要修改的標註", None, format_labels_text(), build_label_checkbox_update()

    if not new_class:
        return "請選擇類別", None, format_labels_text(), build_label_checkbox_update(selected=selected_labels)

    try:
        if new_class not in state.classes:
            state.classes.append(new_class)
            persist_classes()

        class_id = state.classes.index(new_class)
        modified_count = 0

        for label_str in selected_labels:
            idx = int(label_str.split('.')[0]) - 1
            if 0 <= idx < len(state.current_labels):
                label = state.current_labels[idx]
                # 保留其他字段，只更新 class_id
                state.current_labels[idx] = (class_id,) + label[1:]
                modified_count += 1

        vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
        return f"已將 {modified_count} 個標註改為 {new_class}", vis_image, format_labels_text(), build_label_checkbox_update()

    except Exception as e:
        return f"錯誤: {str(e)}", None, "", build_label_checkbox_update(choices=[])


def batch_delete_labels(selected_labels):
    """批次刪除標註"""
    if not state.current_labels:
        return "沒有標註可刪除", None, "", build_label_checkbox_update(choices=[])

    if not selected_labels:
        return "請先選擇要刪除的標註", None, format_labels_text(), build_label_checkbox_update()

    try:
        indices_to_delete = []
        for label_str in selected_labels:
            idx = int(label_str.split('.')[0]) - 1
            if 0 <= idx < len(state.current_labels):
                indices_to_delete.append(idx)

        indices_to_delete.sort(reverse=True)

        for idx in indices_to_delete:
            del state.current_labels[idx]

        vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
        return f"已刪除 {len(indices_to_delete)} 個標註", vis_image, format_labels_text(), build_label_checkbox_update()

    except Exception as e:
        return f"錯誤: {str(e)}", None, "", build_label_checkbox_update(choices=[])


def select_all_labels():
    """全選所有標註"""
    choices = get_label_choices()
    return build_label_checkbox_update(selected=choices, choices=choices)


def deselect_all_labels():
    """取消全選"""
    return build_label_checkbox_update()


def clear_all_labels():
    """清除所有標註"""
    state.current_labels = []
    vis_image = draw_labels_on_image(state.current_image, state.current_labels, state.classes)
    return "已清除所有標註", vis_image, format_labels_text(), build_label_checkbox_update(choices=[])


# ============================================================
# 儲存功能
# ============================================================

def save_current_labels(output_folder):
    """儲存當前圖片的標註 (使用多格式輸出)"""
    if state.current_image is None or state.current_image_path is None:
        return "請先載入圖片"

    if not state.current_labels:
        return "目前沒有標註可儲存"

    state.output_folder = Path(output_folder)
    result = auto_save_current_labels()
    return result if result else "儲存失敗"


# ============================================================
# Gradio 介面
# ============================================================

def create_app():
    default_class = state.classes[0] if state.classes else ""

    # 讀取配置
    config = load_config()
    default_images_folder = config.get("images_folder", DEFAULT_CONFIG["images_folder"])
    default_output_folder = config.get("output_folder", DEFAULT_CONFIG["output_folder"])

    # 自訂暗色主題
    dark_theme = gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="#12121a",
        body_background_fill_dark="#0a0a0f",
        block_background_fill="#1a1a2e",
        block_background_fill_dark="#14141f",
        block_border_color="#2d3748",
        block_border_color_dark="#252538",
        block_label_background_fill="#1e1e2f",
        block_label_background_fill_dark="#16162a",
        block_label_text_color="#a0aec0",
        block_title_text_color="#e2e8f0",
        input_background_fill="#2d3748",
        input_background_fill_dark="#252538",
        input_border_color="#4a5568",
        input_border_color_dark="#3d3d5c",
        button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%)",
        button_secondary_background_fill="#2d3748",
        button_secondary_background_fill_hover="#3d4a5c",
        checkbox_background_color="#2d3748",
        checkbox_background_color_selected="#667eea",
        checkbox_border_color="#4a5568",
        checkbox_border_color_focus="#667eea",
        checkbox_border_color_selected="#667eea",
        checkbox_label_text_color="#e2e8f0",
        checkbox_label_text_color_selected="#ffffff",
    )

    # 自訂 CSS 樣式 (暗色系)
    custom_css = """
    /* 全域暗色背景 */
    .gradio-container {
        background: #12121a !important;
    }

    .nav-bar {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%) !important;
        border: 1px solid #2d3748 !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
    }

    .nav-row {
        align-items: center !important;
        gap: 8px !important;
        flex-wrap: nowrap !important;
    }

    .nav-row > * {
        min-width: 0 !important;
    }

    .nav-title {
        font-weight: 700 !important;
        font-size: 16px !important;
        color: #e2e8f0 !important;
        white-space: nowrap !important;
    }

    .main-container {
        display: flex !important;
        align-items: stretch !important;
        gap: 12px !important;
        flex-wrap: nowrap !important;
    }

    .main-container > * {
        min-width: 0 !important;
    }

    #tool-panel {
        flex: 0 0 90px !important;
        max-width: 110px !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
        padding: 8px 6px !important;
        background: #1a1a2e !important;
    }

    #tool-panel .gr-radio {
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
    }

    #tool-panel .gr-radio label {
        justify-content: center !important;
        text-align: center !important;
        padding: 6px 4px !important;
        border-radius: 8px !important;
        color: #a0aec0 !important;
    }

    #tool-panel .gr-radio label::before {
        display: block;
        font-size: 16px;
        margin-bottom: 2px;
    }

    #tool-panel .gr-radio label:nth-of-type(1)::before {
        content: "\\1F50D";
    }

    #tool-panel .gr-radio label:nth-of-type(2)::before {
        content: "\\1F4E6";
    }

    #tool-panel .gr-radio label:nth-of-type(3)::before {
        content: "\\270F";
    }

    #tool-panel input:checked + label {
        background: rgba(102, 126, 234, 0.3) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    #image-panel {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
        padding: 8px !important;
        position: relative !important;
        overflow: hidden !important;
        background: #0f0f1a !important;
    }

    #image-panel .image-container {
        height: 900px !important;
        max-height: 900px !important;
        max-width: 100% !important;
    }

    #image-panel .image-preview,
    #image-panel canvas,
    #image-panel img {
        max-height: 100% !important;
        max-width: 100% !important;
        height: 100% !important;
        width: 100% !important;
        object-fit: contain !important;
        user-select: none !important;
        -webkit-user-drag: none !important;
    }

    #image-nav {
        position: absolute !important;
        inset: 0 !important;
        pointer-events: none !important;
    }

    #box-drag-overlay {
        position: absolute !important;
        inset: 0 !important;
        z-index: 4 !important;
        pointer-events: none !important;
    }

    #box-drag-overlay.active {
        cursor: crosshair !important;
    }

    #box-drag-rect {
        position: absolute !important;
        border: 2px solid #ffa500 !important;
        background: rgba(255, 165, 0, 0.2) !important;
        box-sizing: border-box !important;
        display: none;
    }

    #prev_btn,
    #next_btn {
        position: absolute !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        pointer-events: auto !important;
        z-index: 5 !important;
    }

    #prev_btn {
        left: 6px !important;
    }

    #next_btn {
        right: 6px !important;
    }

    #prev_btn button,
    #next_btn button {
        width: 32px !important;
        height: 32px !important;
        border-radius: 999px !important;
        padding: 0 !important;
        background: rgba(26, 26, 46, 0.9) !important;
        border: 1px solid #4a5568 !important;
        color: #e2e8f0 !important;
    }

    #prev_btn button:hover,
    #next_btn button:hover {
        background: rgba(45, 55, 72, 0.95) !important;
        border-color: #667eea !important;
    }

    #control-panel {
        flex: 0 0 30% !important;
        min-width: 360px !important;
        max-width: 720px !important;
        border: 1px solid #2d3748 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        background: #1a1a2e !important;
    }

    #control-panel input,
    #control-panel textarea,
    #control-panel select {
        max-width: 100% !important;
        box-sizing: border-box !important;
        background: #2d3748 !important;
        color: #e2e8f0 !important;
        border-color: #4a5568 !important;
    }

    #control-panel button {
        max-width: 100% !important;
        box-sizing: border-box !important;
    }

    #control-panel .gr-button-group,
    #control-panel .button-row {
        flex-wrap: wrap !important;
    }

    #labels-list,
    #labels-list .wrap {
        max-height: 560px !important;
        overflow-y: auto !important;
    }

    #status-box textarea {
        font-weight: bold !important;
        font-size: 14px !important;
        background: #1e1e2f !important;
        color: #a0aec0 !important;
    }

    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: #ffffff !important;
    }

    .gr-button-primary:hover {
        background: linear-gradient(135deg, #5a6fd6 0%, #6a4190 100%) !important;
    }

    .gr-button-secondary {
        background: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        color: #e2e8f0 !important;
    }

    .gr-button-secondary:hover {
        background: #3d4a5c !important;
    }

    /* 下拉選單樣式 */
    .gr-dropdown {
        background: #2d3748 !important;
        color: #e2e8f0 !important;
    }

    /* 勾選框樣式 */
    .gr-checkbox-group label {
        color: #a0aec0 !important;
    }

    /* 單一 Checkbox 修復 - 強制顯示原生樣式 */
    input[type="checkbox"] {
        -webkit-appearance: checkbox !important;
        -moz-appearance: checkbox !important;
        appearance: checkbox !important;
        width: 18px !important;
        height: 18px !important;
        min-width: 18px !important;
        min-height: 18px !important;
        cursor: pointer !important;
        accent-color: #667eea !important;
        background-color: #2d3748 !important;
        border: 2px solid #4a5568 !important;
        border-radius: 4px !important;
        opacity: 1 !important;
        visibility: visible !important;
        position: relative !important;
    }

    input[type="checkbox"]:checked {
        background-color: #667eea !important;
        border-color: #667eea !important;
    }

    /* Checkbox 容器修復 */
    .gr-checkbox,
    [data-testid="checkbox"] {
        pointer-events: auto !important;
    }

    .gr-checkbox label,
    [data-testid="checkbox"] label {
        pointer-events: auto !important;
        cursor: pointer !important;
        color: #e2e8f0 !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    /* 確保 control-panel 內的 checkbox 完全可用 */
    #control-panel input[type="checkbox"] {
        pointer-events: auto !important;
        cursor: pointer !important;
    }

    /* 快速選擇類別 Radio 樣式 */
    #class-radio {
        background: #1e1e2f !important;
        border: 1px solid #3d3d5c !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    #class-radio label {
        cursor: pointer !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        margin: 2px !important;
        transition: all 0.2s ease !important;
    }

    #class-radio label:hover {
        background: rgba(102, 126, 234, 0.2) !important;
    }

    #class-radio input[type="radio"]:checked + label,
    #class-radio label.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* 顯示模式 Radio 樣式 */
    #display-mode {
        background: #1e1e2f !important;
        border: 1px solid #3d3d5c !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }

    #display-mode label {
        cursor: pointer !important;
        padding: 6px 12px !important;
        border-radius: 6px !important;
        margin: 2px !important;
        transition: all 0.2s ease !important;
        color: #a0aec0 !important;
    }

    #display-mode label:hover {
        background: rgba(59, 130, 246, 0.2) !important;
    }

    #display-mode input[type="radio"]:checked + label,
    #display-mode label.selected {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* 標籤文字 */
    label, .label-wrap {
        color: #a0aec0 !important;
    }

    /* 滾動條樣式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }
    """



    with gr.Blocks(title="SAM 3 標註工具", theme=dark_theme, css=custom_css) as app:
        with gr.Column(elem_classes="nav-bar"):
            with gr.Row(elem_classes="nav-row"):
                gr.HTML("<div class='nav-title'>SAM3</div>")
                folder_input = gr.Textbox(
                    label="圖片資料夾",
                    value=default_images_folder,
                    placeholder="輸入圖片資料夾路徑",
                    show_label=False,
                    scale=4,
                )
                output_folder = gr.Textbox(
                    label="輸出資料夾",
                    value=default_output_folder,
                    show_label=False,
                    scale=3,
                )
                load_btn = gr.Button("載入資料夾", variant="primary", scale=1)
                save_btn = gr.Button("儲存標註", variant="primary", scale=1)
            with gr.Row(elem_classes="nav-row"):
                jump_input = gr.Textbox(
                    label="跳到第 N 張",
                    placeholder="輸入數字",
                    scale=2,
                    show_label=False,
                )
                jump_btn = gr.Button("跳轉", scale=1)

        with gr.Row(elem_classes="main-container"):
            with gr.Column(scale=1, elem_id="tool-panel"):
                click_mode_radio = gr.Radio(
                    choices=["點擊分割", "矩形框選", "選取標註"],
                    value="點擊分割",
                    label="互動模式",
                    elem_id="click_mode_radio",
                )
            with gr.Column(scale=6, elem_id="image-panel"):
                image_display = gr.Image(
                    label="📷 圖片預覽 (點擊進行分割)",
                    interactive=True,
                    type="numpy",
                    height=900
                )
                with gr.Row(elem_id="image-nav"):
                    prev_btn = gr.Button("\u25C0", size="sm", elem_id="prev_btn")
                    next_btn = gr.Button("\u25B6", size="sm", elem_id="next_btn")
            with gr.Column(scale=3, elem_id="control-panel"):
                text_prompt = gr.Textbox(
                    label="文字提示 (逗號分隔)",
                    value=default_class,
                    placeholder="例如: debris, diver, boat"
                )
                with gr.Row():
                    click_class_dropdown = gr.Dropdown(
                        choices=state.classes,
                        label="分割使用的類別",
                        value=default_class,
                        allow_custom_value=False
                    )
                    segment_btn = gr.Button("執行文字分割", variant="primary")
                sam_fallback_checkbox = gr.Checkbox(
                    label="SAM 失敗改用方框",
                    value=True
                )

                class_radio = gr.Radio(
                    choices=state.classes,
                    label="快速選擇類別 (點擊切換)",
                    value=default_class,
                    elem_id="class-radio"
                )
                with gr.Row():
                    new_class_input = gr.Textbox(
                        label="新類別",
                        placeholder="輸入名稱",
                        scale=2
                    )
                    add_class_btn = gr.Button("新增", scale=1)
                with gr.Row():
                    delete_class_dropdown = gr.Dropdown(
                        choices=state.classes,
                        label="刪除類別",
                        value=None,
                        allow_custom_value=False,
                        scale=2
                    )
                    delete_class_btn = gr.Button("刪除", variant="stop", scale=1)

                # 顯示模式
                display_mode_radio = gr.Radio(
                    choices=["框線", "遮罩", "框線+遮罩"],
                    value="框線",
                    label="顯示模式",
                    elem_id="display-mode"
                )

                # 輸出格式設定
                with gr.Accordion("輸出格式設定", open=False):
                    output_format_checkboxes = gr.CheckboxGroup(
                        choices=["OBB (旋轉框)", "YOLO-Seg (多邊形)", "PNG Mask", "COCO JSON"],
                        value=["OBB (旋轉框)", "YOLO-Seg (多邊形)"],
                        label="選擇輸出格式",
                        elem_id="output-formats"
                    )
                    polygon_epsilon_slider = gr.Slider(
                        minimum=0.001,
                        maximum=0.02,
                        value=0.005,
                        step=0.001,
                        label="多邊形簡化程度 (越小越精確)",
                        elem_id="polygon-epsilon"
                    )
                    overlap_threshold_slider = gr.Slider(
                        minimum=0.0,
                        maximum=0.5,
                        value=0.1,
                        step=0.05,
                        label="重疊禁止閾值 (0=允許重疊, 0.5=50%重疊即禁止)",
                        elem_id="overlap-threshold"
                    )

                labels_display = gr.Textbox(
                    label="標註摘要",
                    interactive=False,
                    lines=2
                )
                selected_label_info = gr.Textbox(
                    label="選中",
                    value="尚未選取",
                    interactive=False,
                    lines=1
                )
                with gr.Row():
                    selected_class_dropdown = gr.Dropdown(
                        choices=state.classes,
                        label="改為",
                        value=default_class,
                        allow_custom_value=False,
                        scale=2
                    )
                    change_selected_class_btn = gr.Button("套用", scale=1)
                with gr.Row():
                    delete_selected_btn = gr.Button("刪除選中 (Del)", variant="stop", size="sm", elem_id="delete_selected_btn")
                    deselect_btn = gr.Button("取消選取", variant="secondary", size="sm")
                label_checkboxes = gr.CheckboxGroup(
                    choices=[],
                    label="勾選標註",
                    interactive=True,
                    elem_id="labels-list",
                )
                with gr.Row():
                    select_all_btn = gr.Button("全選", size="sm")
                    deselect_all_btn = gr.Button("取消全選", size="sm")
                    clear_btn = gr.Button("清除全部", variant="stop", size="sm")
                with gr.Row():
                    batch_class_dropdown = gr.Dropdown(
                        choices=state.classes,
                        label="改為類別",
                        value=default_class,
                        allow_custom_value=False,
                        scale=2
                    )
                    batch_change_btn = gr.Button("批次改", scale=1)
                batch_delete_btn = gr.Button("批次刪除勾選", variant="stop")
                single_class_dropdown = gr.Dropdown(choices=state.classes, value=default_class, visible=False)

        status_text = gr.Textbox(
            label="📊 狀態",
            interactive=False,
            elem_id="status-box"
        )
        bbox_input = gr.Textbox(label="bbox", visible=False, elem_id="bbox_input")
        bbox_submit = gr.Button("bbox_submit", visible=False, elem_id="bbox_submit")
        # 圖片載入與導覽
        load_btn.click(
            load_images_from_folder,
            inputs=[folder_input, output_folder],
            outputs=[status_text, image_display, labels_display, label_checkboxes, jump_input]
        )

        prev_btn.click(
            prev_image,
            inputs=[output_folder],
            outputs=[status_text, image_display, labels_display, label_checkboxes, jump_input]
        )

        next_btn.click(
            next_image,
            inputs=[output_folder],
            outputs=[status_text, image_display, labels_display, label_checkboxes, jump_input]
        )

        jump_btn.click(
            jump_to_image,
            inputs=[jump_input, output_folder],
            outputs=[status_text, image_display, labels_display, label_checkboxes, jump_input]
        )

        # 文字提示分割
        segment_btn.click(
            segment_with_text,
            inputs=[text_prompt, click_class_dropdown],
            outputs=[status_text, image_display, labels_display, label_checkboxes,
                     class_radio, click_class_dropdown, batch_class_dropdown]
        )

        # 點擊/框選/選取分割
        image_display.select(
            handle_image_click,
            inputs=[image_display, click_class_dropdown, click_mode_radio, sam_fallback_checkbox],
            outputs=[status_text, image_display, labels_display, label_checkboxes, selected_label_info]
        )

        bbox_submit.click(
            handle_drag_box,
            inputs=[bbox_input, click_class_dropdown, sam_fallback_checkbox],
            outputs=[status_text, image_display, labels_display, label_checkboxes, selected_label_info]
        )

        # 選取標註操作
        delete_selected_btn.click(
            delete_selected_label,
            outputs=[status_text, image_display, labels_display, label_checkboxes, selected_label_info]
        )
        change_selected_class_btn.click(
            change_selected_label_class,
            inputs=[selected_class_dropdown],
            outputs=[status_text, image_display, labels_display, label_checkboxes, selected_label_info]
        )
        deselect_btn.click(
            deselect_label,
            outputs=[status_text, image_display, labels_display, label_checkboxes, selected_label_info]
        )

        # 雙向同步：Radio → Dropdown + 文字提示
        def sync_radio_to_all(selected_class):
            if selected_class:
                return gr.update(value=selected_class), gr.update(value=selected_class)
            return gr.update(), gr.update()

        class_radio.change(
            sync_radio_to_all,
            inputs=[class_radio],
            outputs=[click_class_dropdown, text_prompt]
        )

        # 雙向同步：Dropdown → Radio + 文字提示
        def sync_dropdown_to_all(selected_class):
            if selected_class:
                return gr.update(value=selected_class), gr.update(value=selected_class)
            return gr.update(), gr.update()

        click_class_dropdown.change(
            sync_dropdown_to_all,
            inputs=[click_class_dropdown],
            outputs=[class_radio, text_prompt]
        )

        # 輸出格式設定事件
        def update_output_formats(selected_formats):
            state.output_formats["obb"] = "OBB (旋轉框)" in selected_formats
            state.output_formats["seg"] = "YOLO-Seg (多邊形)" in selected_formats
            state.output_formats["mask"] = "PNG Mask" in selected_formats
            state.output_formats["coco"] = "COCO JSON" in selected_formats
            formats_str = ", ".join(selected_formats) if selected_formats else "無"
            return f"輸出格式: {formats_str}"

        output_format_checkboxes.change(
            update_output_formats,
            inputs=[output_format_checkboxes],
            outputs=[status_text]
        )

        # 多邊形簡化參數事件
        def update_polygon_epsilon(epsilon):
            state.polygon_epsilon = epsilon
            return f"多邊形簡化參數: {epsilon:.3f}"

        polygon_epsilon_slider.change(
            update_polygon_epsilon,
            inputs=[polygon_epsilon_slider],
            outputs=[status_text]
        )

        # 重疊閾值事件
        def update_overlap_threshold(threshold):
            state.overlap_threshold = threshold
            if threshold == 0:
                return "重疊禁止: 關閉 (允許重疊)"
            return f"重疊禁止閾值: {threshold*100:.0f}%"

        overlap_threshold_slider.change(
            update_overlap_threshold,
            inputs=[overlap_threshold_slider],
            outputs=[status_text]
        )

        # 顯示模式切換事件
        def update_display_mode(mode):
            mode_map = {"框線": "outline", "遮罩": "mask", "框線+遮罩": "both"}
            state.display_mode = mode_map.get(mode, "outline")
            # 重新繪製當前圖片
            if state.current_image is not None:
                vis_image = draw_labels_on_image(
                    state.current_image, state.current_labels, state.classes,
                    selected_indices=state.selected_labels
                )
                return f"顯示模式: {mode}", vis_image
            return f"顯示模式: {mode}", None

        display_mode_radio.change(
            update_display_mode,
            inputs=[display_mode_radio],
            outputs=[status_text, image_display]
        )

        # 類別管理 - 新增類別（同步更新所有下拉選單和 Radio）
        def handle_add_class(new_class_name):
            name = str(new_class_name).strip() if new_class_name else ""
            if not name:
                return ("請輸入類別名稱", build_class_radio_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update(), "")

            if name in state.classes:
                return (f"類別已存在: {name}", build_class_radio_update(name),
                        build_class_dropdown_update(name), build_class_dropdown_update(name),
                        build_class_dropdown_update(), build_class_dropdown_update(name),
                        build_class_dropdown_update(name), "")

            state.classes.append(name)
            persist_classes()

            return (f"已新增類別: {name}", build_class_radio_update(name),
                    build_class_dropdown_update(name), build_class_dropdown_update(name),
                    build_class_dropdown_update(), build_class_dropdown_update(name),
                    build_class_dropdown_update(name), "")

        add_class_btn.click(
            handle_add_class,
            inputs=[new_class_input],
            outputs=[status_text, class_radio, click_class_dropdown, batch_class_dropdown,
                     delete_class_dropdown, single_class_dropdown, selected_class_dropdown, new_class_input]
        )

        # 類別管理 - 刪除類別（同步更新所有下拉選單和 Radio）
        def handle_delete_class(class_to_delete):
            if not class_to_delete:
                return ("請選擇要刪除的類別", build_class_radio_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update())

            if class_to_delete not in state.classes:
                return (f"類別不存在: {class_to_delete}", build_class_radio_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update())

            if len(state.classes) <= 1:
                return ("至少需要保留一個類別", build_class_radio_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update())

            # 檢查是否有標註使用此類別
            class_id = state.classes.index(class_to_delete)
            using_count = sum(1 for label in state.current_labels if label[0] == class_id)
            if using_count > 0:
                return (f"有 {using_count} 個標註使用此類別，無法刪除", build_class_radio_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update(), build_class_dropdown_update(),
                        build_class_dropdown_update())

            state.classes.remove(class_to_delete)
            persist_classes()

            return (f"已刪除類別: {class_to_delete}", build_class_radio_update(),
                    build_class_dropdown_update(), build_class_dropdown_update(),
                    build_class_dropdown_update(), build_class_dropdown_update(),
                    build_class_dropdown_update())

        delete_class_btn.click(
            handle_delete_class,
            inputs=[delete_class_dropdown],
            outputs=[status_text, class_radio, click_class_dropdown, batch_class_dropdown,
                     delete_class_dropdown, single_class_dropdown, selected_class_dropdown]
        )

        # 批次標註操作
        select_all_btn.click(
            select_all_labels,
            outputs=[label_checkboxes]
        )

        deselect_all_btn.click(
            deselect_all_labels,
            outputs=[label_checkboxes]
        )

        batch_change_btn.click(
            batch_change_class,
            inputs=[label_checkboxes, batch_class_dropdown],
            outputs=[status_text, image_display, labels_display, label_checkboxes]
        )

        batch_delete_btn.click(
            batch_delete_labels,
            inputs=[label_checkboxes],
            outputs=[status_text, image_display, labels_display, label_checkboxes]
        )

        clear_btn.click(
            clear_all_labels,
            outputs=[status_text, image_display, labels_display, label_checkboxes]
        )

        # 儲存
        save_btn.click(
            save_current_labels,
            inputs=[output_folder],
            outputs=[status_text]
        )

        # 注入 JavaScript 監聽 Delete 鍵與拖曳框選
        app.load(None, None, None, js="""
        () => {
            const BOX_MODE_LABEL = '矩形框選';

            function getImagePanel() {
                return document.getElementById('image-panel');
            }

            function getImageElement() {
                const panel = getImagePanel();
                if (!panel) {
                    return null;
                }
                return panel.querySelector('img') || panel.querySelector('canvas');
            }

            function disableImageDrag() {
                const panel = getImagePanel();
                if (panel && panel.dataset.dragDisabled !== '1') {
                    panel.dataset.dragDisabled = '1';
                    panel.addEventListener('dragstart', (e) => e.preventDefault());
                }

                const imgEl = getImageElement();
                if (imgEl && imgEl.dataset.dragDisabled !== '1') {
                    imgEl.dataset.dragDisabled = '1';
                    imgEl.draggable = false;
                    imgEl.addEventListener('dragstart', (e) => e.preventDefault());
                }
            }

            function ensureOverlay() {
                const panel = getImagePanel();
                if (!panel) {
                    return null;
                }
                let overlay = document.getElementById('box-drag-overlay');
                if (!overlay) {
                    overlay = document.createElement('div');
                    overlay.id = 'box-drag-overlay';
                    panel.appendChild(overlay);
                }
                let rect = overlay.querySelector('#box-drag-rect');
                if (!rect) {
                    rect = document.createElement('div');
                    rect.id = 'box-drag-rect';
                    overlay.appendChild(rect);
                }
                if (overlay.dataset.dragDisabled !== '1') {
                    overlay.dataset.dragDisabled = '1';
                    overlay.addEventListener('dragstart', (e) => e.preventDefault());
                }
                return overlay;
            }

            function getCheckedMode() {
                const checked = document.querySelector('#tool-panel input[type="radio"]:checked');
                return checked ? checked.value : '';
            }

            function updateOverlayMode() {
                const overlay = ensureOverlay();
                if (!overlay) {
                    return;
                }
                if (getCheckedMode() === BOX_MODE_LABEL) {
                    overlay.style.pointerEvents = 'auto';
                    overlay.classList.add('active');
                } else {
                    overlay.style.pointerEvents = 'none';
                    overlay.classList.remove('active');
                }
                disableImageDrag();
            }

            function submitBox(x1, y1, x2, y2) {
                const inputWrap = document.getElementById('bbox_input');
                if (!inputWrap) {
                    return;
                }
                const input = inputWrap.querySelector('input, textarea');
                if (!input) {
                    return;
                }
                input.value = [x1, y1, x2, y2].join(',');
                input.dispatchEvent(new Event('input', { bubbles: true }));
                const btnWrap = document.getElementById('bbox_submit');
                if (btnWrap) {
                    const btn = btnWrap.querySelector('button') || btnWrap;
                    btn.click();
                }
            }

            const overlay = ensureOverlay();
            if (overlay && overlay.dataset.bound !== '1') {
                overlay.dataset.bound = '1';
                let dragActive = false;
                let startX = 0;
                let startY = 0;
                let ctx = null;
                const rectEl = overlay.querySelector('#box-drag-rect');

                const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

                overlay.addEventListener('mousedown', (e) => {
                    if (getCheckedMode() !== BOX_MODE_LABEL) {
                        return;
                    }
                    const imgEl = getImageElement();
                    if (!imgEl) {
                        return;
                    }

                    const imgRect = imgEl.getBoundingClientRect();
                    if (e.clientX < imgRect.left || e.clientX > imgRect.right ||
                        e.clientY < imgRect.top || e.clientY > imgRect.bottom) {
                        return;
                    }

                    const panelRect = overlay.getBoundingClientRect();
                    const naturalW = imgEl.naturalWidth || imgEl.width;
                    const naturalH = imgEl.naturalHeight || imgEl.height;
                    if (!naturalW || !naturalH || imgRect.width === 0 || imgRect.height === 0) {
                        return;
                    }

                    dragActive = true;
                    startX = clamp(e.clientX - imgRect.left, 0, imgRect.width);
                    startY = clamp(e.clientY - imgRect.top, 0, imgRect.height);

                    ctx = {
                        imgRect: imgRect,
                        panelRect: panelRect,
                        naturalW: naturalW,
                        naturalH: naturalH,
                        offsetX: imgRect.left - panelRect.left,
                        offsetY: imgRect.top - panelRect.top
                    };

                    rectEl.style.display = 'block';
                    rectEl.style.left = (ctx.offsetX + startX) + 'px';
                    rectEl.style.top = (ctx.offsetY + startY) + 'px';
                    rectEl.style.width = '0px';
                    rectEl.style.height = '0px';

                    e.preventDefault();
                    e.stopPropagation();
                });

                window.addEventListener('mousemove', (e) => {
                    if (!dragActive || !ctx) {
                        return;
                    }
                    const imgRect = ctx.imgRect;
                    const curX = clamp(e.clientX - imgRect.left, 0, imgRect.width);
                    const curY = clamp(e.clientY - imgRect.top, 0, imgRect.height);

                    const left = Math.min(startX, curX) + ctx.offsetX;
                    const top = Math.min(startY, curY) + ctx.offsetY;
                    const width = Math.abs(curX - startX);
                    const height = Math.abs(curY - startY);

                    rectEl.style.left = left + 'px';
                    rectEl.style.top = top + 'px';
                    rectEl.style.width = width + 'px';
                    rectEl.style.height = height + 'px';
                });

                window.addEventListener('mouseup', (e) => {
                    if (!dragActive || !ctx) {
                        return;
                    }
                    dragActive = false;

                    const imgRect = ctx.imgRect;
                    const endX = clamp(e.clientX - imgRect.left, 0, imgRect.width);
                    const endY = clamp(e.clientY - imgRect.top, 0, imgRect.height);

                    rectEl.style.display = 'none';

                    const width = Math.abs(endX - startX);
                    const height = Math.abs(endY - startY);
                    const naturalW = ctx.naturalW;
                    const naturalH = ctx.naturalH;
                    const scaleX = imgRect.width > 0 ? (naturalW / imgRect.width) : 1;
                    const scaleY = imgRect.height > 0 ? (naturalH / imgRect.height) : 1;

                    ctx = null;

                    if (width < 4 || height < 4) {
                        return;
                    }

                    const x1 = Math.round(Math.min(startX, endX) * scaleX);
                    const y1 = Math.round(Math.min(startY, endY) * scaleY);
                    const x2 = Math.round(Math.max(startX, endX) * scaleX);
                    const y2 = Math.round(Math.max(startY, endY) * scaleY);

                    submitBox(x1, y1, x2, y2);
                    e.preventDefault();
                    e.stopPropagation();
                });
            }

            document.querySelectorAll('#tool-panel input[type="radio"]').forEach((radio) => {
                radio.addEventListener('change', updateOverlayMode);
            });
            setTimeout(updateOverlayMode, 0);
            setInterval(updateOverlayMode, 800);

            document.addEventListener('keydown', function(e) {
                if (e.target.matches('input, textarea')) {
                    return;
                }

                if (e.key === 'Delete') {
                    e.preventDefault();
                    const btnContainer = document.getElementById('delete_selected_btn');
                    if (btnContainer) {
                        const btn = btnContainer.querySelector('button') || btnContainer;
                        btn.click();
                    }
                }

                if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                    e.preventDefault();
                    const targetId = e.key === 'ArrowLeft' ? 'prev_btn' : 'next_btn';
                    const btnContainer = document.getElementById(targetId);
                    if (btnContainer) {
                        const btn = btnContainer.querySelector('button') || btnContainer;
                        btn.click();
                    }
                }
            });
            console.log('Hotkeys: Delete, Left/Right');
        }
        """)

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SAM 3 標註工具")
    parser.add_argument("--images", default="F:/yolov8_env/output_frames", help="圖片資料夾路徑")
    parser.add_argument("--output", default="F:/yolov8_env/labeled_dataset", help="輸出資料夾路徑")
    parser.add_argument("--port", type=int, default=7860, help="伺服器埠號")
    parser.add_argument("--share", action="store_true", help="建立公開連結")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SAM 3 互動式標註工具")
    print("="*60)
    print(f"圖片資料夾: {args.images}")
    print(f"輸出資料夾: {args.output}")
    print(f"伺服器埠號: {args.port}")
    print("="*60 + "\n")

    load_persisted_classes()
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        inbrowser=True
    )
