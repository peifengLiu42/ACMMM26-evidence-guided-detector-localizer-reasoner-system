import os
import cv2
import numpy as np
import json
import argparse
from prettytable import PrettyTable
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = [0, 1]
        if labels != [0, 1]:
            raise ValueError("Fallback confusion_matrix only supports labels=[0, 1]")

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        return np.array([[tn, fp], [fn, tp]])

FORGED_LABELS = {"forged", "fake", "tampered", "1", "true", "positive"}
IOU_AT_THRESHOLDS = (0.5, 0.75, 0.90)

def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0

def normalize_key(path_or_name):
    name = os.path.basename(str(path_or_name).strip().replace("\\", "/"))
    stem = os.path.splitext(name)[0]
    if stem.endswith("_mask"):
        stem = stem[:-5]
    return stem

def load_json_or_jsonl(path):
    if path.lower().endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("predictions", "results", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    return data

def load_detector_predictions(detector_json):
    data = load_json_or_jsonl(detector_json)
    if isinstance(data, dict):
        items = []
        for key, value in data.items():
            if isinstance(value, dict):
                item = dict(value)
                item.setdefault("image", key)
                items.append(item)
            else:
                items.append({"image": key, "pred_label": value})
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unsupported detector JSON format: {detector_json}")

    detector_map = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        image_name = (
            item.get("image")
            or item.get("image_path")
            or item.get("image_name")
            or item.get("filename")
            or item.get("file_name")
            or item.get("name")
            or item.get("id")
        )
        if image_name is None:
            continue
        detector_map[normalize_key(image_name)] = item
    return detector_map

def detector_is_forged(item, threshold):
    if "pred_label_id" in item:
        return int(item["pred_label_id"]) == 1
    if "pred_label" in item:
        value = item["pred_label"]
        if isinstance(value, (int, float)):
            return int(value) == 1
        return str(value).strip().lower() in FORGED_LABELS
    if "label" in item:
        value = item["label"]
        if isinstance(value, (int, float)):
            return int(value) == 1
        return str(value).strip().lower() in FORGED_LABELS
    if "prob_forged" in item:
        return float(item["prob_forged"]) >= threshold
    if "score" in item:
        return float(item["score"]) >= threshold
    if "prob" in item:
        return float(item["prob"]) >= threshold
    raise ValueError(f"Cannot determine detector label from item: {item}")

def collect_pairs_flat_pred_nested_gt(pred_dir, gt_dir, pred_ext='.png', gt_ext=None):
    """按文件名（不含路径）匹配扁平 pred 和嵌套 gt"""
    if gt_ext is None:
        gt_ext = pred_ext
    pred_files = {}
    for f in os.listdir(pred_dir):
        if f.lower().endswith(pred_ext):
            name = os.path.splitext(f)[0]
            if "Forensic" in name:
                if "_mask" in name:
                    pred_files[name] = os.path.join(pred_dir, f)
                else:
                    pred_files[name+"_mask"] = os.path.join(pred_dir, f)
            else:
                pred_files[name.replace("_mask", "")] = os.path.join(pred_dir, f)
    
    gt_files = {}
    for root, _, files in os.walk(gt_dir):
        for f in files:
            if f.lower().endswith(gt_ext):
                name = os.path.splitext(f)[0]
                if name in gt_files:
                    raise ValueError(f"重复的 GT 文件名: {name}")
                gt_files[name] = os.path.join(root, f)
    
    pairs = []
    for name in pred_files:
        if name in gt_files:
            pairs.append((pred_files[name], gt_files[name]))
        else:
            print(f"⚠️ 未找到 GT: {pred_files[name]}")
    
    for name in gt_files:
        if name not in pred_files:
            print(f"{name} ⚠️ 未找到 Pred: {gt_files[name]}")
    
    print(f"✅ 成功匹配 {len(pairs)} 对 (pred, gt)")
    return pairs

def f_measure(tp, fp, fn):
    return safe_div(2 * tp, 2 * tp + fp + fn)

def iou_measure(tp, fp, fn): 
    return safe_div(tp, tp + fp + fn)

def precision_measure(tp, fp): 
    return safe_div(tp, tp + fp)

def recall_measure(tp, fn): 
    return safe_div(tp, tp + fn)

def binarize_prediction_mask(mask, threshold):
    scale = 1.0 if float(np.max(mask)) <= 1.0 else 255.0
    return (mask > (threshold * scale)).astype(np.uint8)

def binary_image_metrics(tp, fp, fn, tn):
    """Image-level binary classification metrics, positive class = tampered/forged."""
    total = tp + fp + fn + tn
    support_forged = tp + fn
    support_authentic = tn + fp

    precision_forged = precision_measure(tp, fp)
    recall_forged = recall_measure(tp, fn)
    f1_forged = f_measure(tp, fp, fn)

    # Treat authentic as the positive class: TP=TN, FP=FN, FN=FP.
    precision_authentic = precision_measure(tn, fn)
    recall_authentic = recall_measure(tn, fp)
    f1_authentic = f_measure(tn, fn, fp)

    return {
        "accuracy": safe_div(tp + tn, total),
        "balanced_acc": (recall_forged + recall_authentic) / 2,
        "positive_f1": f1_forged,
        "macro_f1": (f1_forged + f1_authentic) / 2,
        "weighted_f1": safe_div(
            support_forged * f1_forged + support_authentic * f1_authentic,
            total,
        ),
        "forged": {
            "precision": precision_forged,
            "recall": recall_forged,
            "f1": f1_forged,
            "support": support_forged,
        },
        "authentic": {
            "precision": precision_authentic,
            "recall": recall_authentic,
            "f1": f1_authentic,
            "support": support_authentic,
        },
    }

def iou_at_rates(iou_values, thresholds=IOU_AT_THRESHOLDS):
    if not iou_values:
        return {thr: 0.0 for thr in thresholds}
    iou_values = np.asarray(iou_values, dtype=np.float32)
    return {thr: float(np.mean(iou_values >= thr)) for thr in thresholds}

def load_gt_boxes(gt_txt_path):
    boxes = []
    with open(gt_txt_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", ",").replace(" ", ",").split(",")
            parts = [p for p in parts if p != ""]
            if len(parts) < 4:
                raise ValueError(f"Invalid GT box line {gt_txt_path}:{line_no}: {line}")
            x1, y1, x2, y2 = [float(v) for v in parts[:4]]
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            boxes.append((x1, y1, x2, y2))
    return boxes

def clip_box_to_image(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def best_component_iou_for_box(labels, component_areas, box):
    x1, y1, x2, y2 = box
    gt_area = (x2 - x1) * (y2 - y1)
    crop = labels[y1:y2, x1:x2]
    component_ids, intersections = np.unique(crop[crop > 0], return_counts=True)
    if len(component_ids) == 0:
        return 0.0, None

    best_iou = 0.0
    best_component_id = None
    for component_id, intersection in zip(component_ids, intersections):
        component_area = int(component_areas[int(component_id)])
        union = gt_area + component_area - int(intersection)
        iou = int(intersection) / (union + 1e-12)
        if iou > best_iou:
            best_iou = iou
            best_component_id = int(component_id)
    return float(best_iou), best_component_id

def process_single_image(args):
    """处理单个图像对的函数（用于多进程）"""
    pred_path, gt_path, threshold = args
    
    try:
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            print(f"❌ 跳过无效掩码: {pred_path} 或 {gt_path}")
            return None
        
        target_size = (1344, 894)
        pred = cv2.resize(pred, target_size, interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_NEAREST)
        
        pred_bin = binarize_prediction_mask(pred, threshold)
        gt_bin = (gt > 0).astype(np.uint8)
        
        is_tampered = bool(np.any(gt_bin))
        pred_positive = bool(np.any(pred_bin))
        
        tn, fp, fn, tp = confusion_matrix(gt_bin.flatten(), pred_bin.flatten(), labels=[0,1]).ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        
        iou = iou_measure(tp, fp, fn)
        f1 = f_measure(tp, fp, fn)
        precision = precision_measure(tp, fp)
        recall = recall_measure(tp, fn)
        
        return {
            'pred_file': os.path.basename(pred_path),
            'gt_file': gt_path,
            'is_tampered': is_tampered,
            'pred_positive': pred_positive,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'iou': float(iou),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
        }
    except Exception as e:
        print(f"❌ 处理出错 {pred_path}: {str(e)}")
        return None

def process_single_image_box_iou(args):
    """GT box vs best matching predicted connected-component mask IoU."""
    pred_path, gt_txt_path, threshold, component_min_area = args

    try:
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print(f"❌ 跳过无效预测掩码: {pred_path}")
            return None

        gt_boxes_raw = load_gt_boxes(gt_txt_path)
        height, width = pred.shape[:2]
        gt_boxes = [
            clipped for clipped in
            (clip_box_to_image(box, width, height) for box in gt_boxes_raw)
            if clipped is not None
        ]

        pred_bin = binarize_prediction_mask(pred, threshold)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
        component_areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
        if component_min_area > 1:
            small_labels = np.where(component_areas < component_min_area)[0]
            small_labels = small_labels[small_labels != 0]
            if len(small_labels):
                labels[np.isin(labels, small_labels)] = 0
                pred_bin = (labels > 0).astype(np.uint8)
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
                component_areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)

        box_details = []
        box_ious = []
        for box in gt_boxes:
            best_iou, best_component_id = best_component_iou_for_box(labels, component_areas, box)
            box_ious.append(best_iou)
            box_details.append({
                "box": [int(v) for v in box],
                "best_iou": best_iou,
                "best_component_id": best_component_id,
            })

        return {
            "pred_file": os.path.basename(pred_path),
            "gt_file": gt_txt_path,
            "num_gt_boxes": len(gt_boxes),
            "num_pred_components": int(num_labels - 1),
            "box_ious": box_ious,
            "box_details": box_details,
        }
    except Exception as e:
        print(f"❌ 处理框 IoU 出错 {pred_path}: {str(e)}")
        return None

def evaluate_box_iou(
    pred_dir,
    gt_box_dir,
    save_dir,
    method_name,
    threshold=0.5,
    num_workers=None,
    component_min_area=1,
):
    os.makedirs(save_dir, exist_ok=True)
    pairs = collect_pairs_flat_pred_nested_gt(pred_dir, gt_box_dir, pred_ext=".png", gt_ext=".txt")
    if not pairs:
        raise ValueError("未找到有效的 pred mask / GT box 配对！")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"🔄 使用 {num_workers} 个进程计算 GT-box vs pred-component IoU...")
    process_args = [
        (pred_path, gt_path, threshold, component_min_area)
        for pred_path, gt_path in pairs
    ]

    all_results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_pair = {
            executor.submit(process_single_image_box_iou, args): args
            for args in process_args
        }
        for future in tqdm(as_completed(future_to_pair), total=len(pairs), desc="框 IoU 评估中"):
            result = future.result()
            if result is not None:
                all_results.append(result)

    if not all_results:
        raise ValueError("没有成功处理的框 IoU 图像！")

    all_box_ious = [
        iou
        for result in all_results
        for iou in result["box_ious"]
    ]
    iou_at = iou_at_rates(all_box_ious)
    mean_iou = float(np.mean(all_box_ious)) if all_box_ious else 0.0
    total_gt_boxes = int(sum(result["num_gt_boxes"] for result in all_results))
    total_pred_components = int(sum(result["num_pred_components"] for result in all_results))

    tb = PrettyTable([
        "Metric", "Images", "GT boxes", "Pred comps",
        "Box-IoU (mean)", "Box-IoU@0.5", "Box-IoU@0.75", "Box-IoU@0.90"
    ])
    tb.add_row([
        "GT-box Best Component",
        len(all_results),
        total_gt_boxes,
        total_pred_components,
        f"{mean_iou*100:.2f}%",
        f"{iou_at[0.5]*100:.2f}%",
        f"{iou_at[0.75]*100:.2f}%",
        f"{iou_at[0.90]*100:.2f}%",
    ])

    print("\n" + "="*120)
    print(tb)
    print("="*120 + "\n")

    suffix = f"{method_name}_box_iou"
    with open(os.path.join(save_dir, f"result_{suffix}.json"), "w") as f:
        json.dump({
            "metric_definition": "For each GT box, convert it to a rectangle mask and compute IoU with the best overlapping predicted connected-component mask.",
            "threshold": threshold,
            "component_min_area": component_min_area,
            "results": all_results,
            "summary": {
                "num_images": len(all_results),
                "num_gt_boxes": total_gt_boxes,
                "num_pred_components": total_pred_components,
                "box_iou_mean": mean_iou,
                "box_iou_at_0.5": iou_at[0.5],
                "box_iou_at_0.75": iou_at[0.75],
                "box_iou_at_0.90": iou_at[0.90],
            },
        }, f, indent=2)
    with open(os.path.join(save_dir, f"ans_{suffix}.txt"), "w") as f:
        f.write(str(tb))
    print(f"✅ 框 IoU 结果已保存至: {save_dir}")

def evaluate(
    pred_dir,
    gt_dir,
    save_dir,
    method_name,
    threshold=0.5,
    num_workers=None,
    detector_json=None,
    detector_threshold=0.5,
):
    os.makedirs(save_dir, exist_ok=True)
    pairs = collect_pairs_flat_pred_nested_gt(pred_dir, gt_dir)
    if not pairs:
        raise ValueError("未找到有效配对！")

    detector_map = None
    if detector_json:
        detector_map = load_detector_predictions(detector_json)
        print(f"✅ 加载 detector 预测 {len(detector_map)} 条，用于 Img/Wtd 指标: {detector_json}")
    
    # 设置默认进程数
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"🔄 使用 {num_workers} 个进程进行并行计算...")
    
    # 准备参数列表
    process_args = [(pred_path, gt_path, threshold) for pred_path, gt_path in pairs]
    
    all_results = []
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_pair = {executor.submit(process_single_image, args): args 
                         for args in process_args}
        
        # 收集结果（带进度条）
        for future in tqdm(as_completed(future_to_pair), total=len(pairs), desc="评估中"):
            result = future.result()
            if result is not None:
                all_results.append(result)
    
    if not all_results:
        raise ValueError("没有成功处理的图像！")
    
    # 聚合结果
    all_cm = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    tamp_cm = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    
    tp_img = tn_img = fp_img = fn_img = 0
    loc_iou_tamp, loc_f1_tamp = [], []
    loc_prec_tamp, loc_rec_tamp = [], []
    
    for r in all_results:
        tp, fp, fn, tn = r['tp'], r['fp'], r['fn'], r['tn']
        is_tampered = r['is_tampered']
        pred_positive = r['pred_positive']
        if detector_map is not None:
            key = normalize_key(r['pred_file'])
            detector_item = detector_map.get(key)
            if detector_item is None:
                raise KeyError(f"Detector prediction not found for {r['pred_file']} (key={key})")
            pred_positive = detector_is_forged(detector_item, detector_threshold)
        
        all_cm['tp'] += tp
        all_cm['fp'] += fp
        all_cm['fn'] += fn
        all_cm['tn'] += tn
        
        if is_tampered:
            if pred_positive:
                tp_img += 1
            else:
                fn_img += 1
            loc_iou_tamp.append(r['iou'])
            loc_f1_tamp.append(r['f1'])
            loc_prec_tamp.append(r['precision'])
            loc_rec_tamp.append(r['recall'])
            tamp_cm['tp'] += tp
            tamp_cm['fp'] += fp
            tamp_cm['fn'] += fn
            tamp_cm['tn'] += tn
        else:
            if not pred_positive:
                tn_img += 1
            else:
                fp_img += 1
    
    # 计算所有图像的平均 IoU 和 F1
    loc_iou_all = [r['iou'] for r in all_results]
    loc_prec_all = [r['precision'] for r in all_results]
    loc_rec_all = [r['recall'] for r in all_results]
    avg_iou_all = np.mean(loc_iou_all) if all_results else 0.0
    avg_f1_all = np.mean([r['f1'] for r in all_results]) if all_results else 0.0
    avg_prec_all = np.mean(loc_prec_all) if all_results else 0.0
    avg_rec_all = np.mean(loc_rec_all) if all_results else 0.0
    
    avg_iou_tamp = np.mean(loc_iou_tamp) if loc_iou_tamp else 0.0
    avg_f1_tamp = np.mean(loc_f1_tamp) if loc_f1_tamp else 0.0
    avg_prec_tamp = np.mean(loc_prec_tamp) if loc_prec_tamp else 0.0
    avg_rec_tamp = np.mean(loc_rec_tamp) if loc_rec_tamp else 0.0
    iou_at_all = iou_at_rates(loc_iou_all)
    iou_at_tamp = iou_at_rates(loc_iou_tamp)
    
    image_metrics = binary_image_metrics(tp_img, fp_img, fn_img, tn_img)
    acc_img = image_metrics["accuracy"]
    f1_img = image_metrics["positive_f1"]
    balanced_acc = image_metrics["balanced_acc"]
    weighted_f1 = image_metrics["weighted_f1"]
    
    iou_all = iou_measure(all_cm['tp'], all_cm['fp'], all_cm['fn'])
    f1_all = f_measure(all_cm['tp'], all_cm['fp'], all_cm['fn'])
    prec_all = precision_measure(all_cm['tp'], all_cm['fp'])
    rec_all = recall_measure(all_cm['tp'], all_cm['fn'])
    
    iou_tamp = iou_measure(tamp_cm['tp'], tamp_cm['fp'], tamp_cm['fn'])
    f1_tamp_cm = f_measure(tamp_cm['tp'], tamp_cm['fp'], tamp_cm['fn'])
    prec_tamp = precision_measure(tamp_cm['tp'], tamp_cm['fp'])
    rec_tamp = recall_measure(tamp_cm['tp'], tamp_cm['fn'])
    
    tb = PrettyTable([
        'Metric', 'IoU (mean)', 'Prec (mean)', 'Rec (mean)', 'F1 (mean)',
        'IoU@0.5', 'IoU@0.75', 'IoU@0.90',
        'IoU (CM)', 'Prec (CM)', 'Rec (CM)', 'F1 (CM)',
        'Img-F1', 'Img-Acc', 'BalAcc', 'Wtd-F1'
    ])
    tb.add_row([
        'All Images',
        f"{avg_iou_all*100:.2f}%",
        f"{avg_prec_all*100:.2f}%",
        f"{avg_rec_all*100:.2f}%",
        f"{avg_f1_all*100:.2f}%",
        f"{iou_at_all[0.5]*100:.2f}%",
        f"{iou_at_all[0.75]*100:.2f}%",
        f"{iou_at_all[0.90]*100:.2f}%",
        f"{iou_all*100:.2f}%", f"{prec_all*100:.2f}%",
        f"{rec_all*100:.2f}%", f"{f1_all*100:.2f}%",
        f"{f1_img*100:.2f}%", f"{acc_img*100:.2f}%",
        f"{balanced_acc*100:.2f}%", f"{weighted_f1*100:.2f}%"
    ])
    tb.add_row([
        'Tampered Only',
        f"{avg_iou_tamp*100:.2f}%",
        f"{avg_prec_tamp*100:.2f}%",
        f"{avg_rec_tamp*100:.2f}%",
        f"{avg_f1_tamp*100:.2f}%",
        f"{iou_at_tamp[0.5]*100:.2f}%",
        f"{iou_at_tamp[0.75]*100:.2f}%",
        f"{iou_at_tamp[0.90]*100:.2f}%",
        f"{iou_tamp*100:.2f}%", f"{prec_tamp*100:.2f}%",
        f"{rec_tamp*100:.2f}%", f"{f1_tamp_cm*100:.2f}%",
        '-', '-', '-', '-'
    ])
    
    print("\n" + "="*130)
    print(tb)
    print("="*130 + "\n")
    
    with open(os.path.join(save_dir, f"result_{method_name}.json"), 'w') as f:
        json.dump({
            'results': all_results,
            'confusion_all': all_cm,
            'confusion_tamp': tamp_cm,
            'image_confusion': {
                'tp': tp_img,
                'fp': fp_img,
                'fn': fn_img,
                'tn': tn_img,
                'source': detector_json or 'pred_mask_non_empty',
            },
            'image_metrics': image_metrics,
            'summary': {
                'all_images': {
                    'iou_mean': float(avg_iou_all),
                    'precision_mean': float(avg_prec_all),
                    'recall_mean': float(avg_rec_all),
                    'f1_mean': float(avg_f1_all),
                    'iou_at_0.5': iou_at_all[0.5],
                    'iou_at_0.75': iou_at_all[0.75],
                    'iou_at_0.90': iou_at_all[0.90],
                    'iou_cm': float(iou_all),
                    'precision_cm': float(prec_all),
                    'recall_cm': float(rec_all),
                    'f1_cm': float(f1_all),
                    'img_f1': float(f1_img),
                    'img_acc': float(acc_img),
                    'balanced_acc': float(balanced_acc),
                    'weighted_f1': float(weighted_f1),
                    'macro_f1': float(image_metrics["macro_f1"]),
                },
                'tampered_only': {
                    'iou_mean': float(avg_iou_tamp),
                    'precision_mean': float(avg_prec_tamp),
                    'recall_mean': float(avg_rec_tamp),
                    'f1_mean': float(avg_f1_tamp),
                    'iou_at_0.5': iou_at_tamp[0.5],
                    'iou_at_0.75': iou_at_tamp[0.75],
                    'iou_at_0.90': iou_at_tamp[0.90],
                    'iou_cm': float(iou_tamp),
                    'precision_cm': float(prec_tamp),
                    'recall_cm': float(rec_tamp),
                    'f1_cm': float(f1_tamp_cm),
                },
            },
        }, f, indent=2)
    with open(os.path.join(save_dir, f"ans_{method_name}.txt"), 'w') as f:
        f.write(str(tb))
    print(f"✅ 结果已保存至: {save_dir}")

# ==================== 命令行入口 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="评估篡改检测结果（扁平 pred + 嵌套 gt）")
    parser.add_argument('--pred_dir', required=True, help='预测掩码目录（扁平结构）')
    parser.add_argument('--gt_dir', default=None, help='真实掩码目录（可含子文件夹）')
    parser.add_argument('--gt_box_dir', default=None, help='可选：GT 框 txt 目录，每行 x1,y1,x2,y2[,class]')
    parser.add_argument('--save_dir', default='./eval_results', help='结果保存目录')
    parser.add_argument('--method_name', default='method', help='方法名称（用于命名结果文件）')
    parser.add_argument('--threshold', type=float, default=0.5, help='二值化阈值（0.0~1.0）')
    parser.add_argument('--component_min_area', type=int, default=1, help='框 IoU 模式下忽略小于该面积的预测连通域')
    parser.add_argument('--workers', type=int, default=None, help='并行进程数（默认：CPU核心数）')
    parser.add_argument('--detector_json', default=None, help='可选：使用 detector 预测计算 Img/Wtd 指标')
    parser.add_argument('--detector_threshold', type=float, default=0.5, help='detector 概率阈值（默认：0.5）')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.method_name = os.path.basename(args.pred_dir)
    if args.gt_dir is None and args.gt_box_dir is None:
        raise ValueError("请至少提供 --gt_dir 或 --gt_box_dir")
    if args.gt_dir is not None:
        evaluate(
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            save_dir=args.save_dir,
            method_name=args.method_name,
            threshold=args.threshold,
            num_workers=args.workers,
            detector_json=args.detector_json,
            detector_threshold=args.detector_threshold,
        )
    if args.gt_box_dir is not None:
        evaluate_box_iou(
            pred_dir=args.pred_dir,
            gt_box_dir=args.gt_box_dir,
            save_dir=args.save_dir,
            method_name=args.method_name,
            threshold=args.threshold,
            num_workers=args.workers,
            component_min_area=args.component_min_area,
        )
