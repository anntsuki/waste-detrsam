import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# --- 評估指標類 (無變動) ---
class StructureMeasure:
    def __init__(self, alpha=0.5):
        self.alpha = alpha; self.eps = np.finfo(np.double).eps

    def _object(self, pred, gt):
        fg = (gt > 0.5);
        if np.sum(fg) == 0: return 0.0
        x, sigma_x = np.mean(pred[fg]), np.std(pred[fg]);
        return 2.0 * x / (x ** 2 + 1.0 + sigma_x + self.eps)

    def _s_region(self, pred, gt):
        gt_bool = (gt > 0.5);
        x, y = self._centroid(gt_bool);
        rows, cols = gt.shape;
        area = float(rows * cols)
        w1, w2, w3 = (x * y) / area, ((cols - x) * y) / area, (x * (rows - y)) / area;
        w4 = 1.0 - w1 - w2 - w3
        pred_LT, pred_RT, pred_LB, pred_RB = self._divide_pred(pred, x, y)
        gt_LT, gt_RT, gt_LB, gt_RB = self._divide_pred(gt, x, y)
        q1, q2, q3, q4 = [self._ssim(p, g) for p, g in
                          zip([pred_LT, pred_RT, pred_LB, pred_RB], [gt_LT, gt_RT, gt_LB, gt_RB])]
        return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    def _ssim(self, pred, gt):
        if np.sum(gt) == 0 and np.sum(pred) == 0: return 1.0
        if np.sum(gt) == 0 or np.sum(pred) == 0: return 0.0
        N = gt.size;
        x, y = np.mean(pred), np.mean(gt)
        sigma_x2 = np.sum((pred - x) ** 2) / (N - 1 + self.eps);
        sigma_y2 = np.sum((gt - y) ** 2) / (N - 1 + self.eps)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1 + self.eps)
        alpha, beta = 2 * x * y, x ** 2 + y ** 2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (alpha + C1) * (2 * sigma_xy + C2) / ((beta + C1) * (sigma_x2 + sigma_y2 + C2))
        return np.mean(ssim_map)

    def _centroid(self, gt_bool):
        rows, cols = gt_bool.shape
        if np.sum(gt_bool) == 0: return round(cols / 2), round(rows / 2)
        y_coords, x_coords = np.where(gt_bool);
        return int(round(np.mean(x_coords))), int(round(np.mean(y_coords)))

    def _divide_pred(self, pred, x, y):
        rows, cols = pred.shape;
        return pred[0:y, 0:x], pred[0:y, x:cols], pred[y:rows, 0:x], pred[y:rows, x:cols]

    def __call__(self, pred, gt):
        pred, gt = pred.astype(np.float64), gt.astype(np.float64)
        if np.mean(gt) == 0: return 1.0 - np.mean(pred)
        if np.mean(gt) == 1: return np.mean(pred)
        S_object = self.alpha * self._object(pred, gt) + (1 - self.alpha) * self._object(1.0 - pred, 1.0 - gt)
        S_region = self._s_region(pred, gt)
        return self.alpha * S_object + (1 - self.alpha) * S_region


class EnhancedAlignmentMeasure:
    def __init__(self):
        self.eps = np.finfo(np.double).eps

    def __call__(self, pred, gt):
        pred, gt = pred.astype(np.float64), gt.astype(np.float64)
        if np.sum(gt) == 0: return 1.0 - np.mean(pred)
        if np.sum(gt) == gt.size: return np.mean(pred)
        align_matrix = 2.0 * (gt * pred) / (gt ** 2 + pred ** 2 + self.eps)
        enhanced_matrix = (align_matrix + 1) ** 2 / 4
        return np.sum(enhanced_matrix) / (gt.size - 1 + self.eps)


# --- 輔助函數 ---
def calculate_standard_metrics(pred_mask, gt_mask):
    pred_mask_bool = pred_mask.astype(bool);
    gt_mask_bool = gt_mask.astype(bool)
    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union = np.logical_or(pred_mask_bool, gt_mask_bool).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred_mask_bool.sum() + gt_mask_bool.sum() + 1e-6)
    tp = float(intersection);
    fp = float(pred_mask_bool.sum() - tp);
    fn = float(gt_mask_bool.sum() - tp)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def load_yolo_txt_as_mask(txt_path, image_shape):
    h, w = image_shape[:2];
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(txt_path): return gt_mask
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 2:
                polygon_norm = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                polygon_pixels = (polygon_norm * np.array([w, h])).astype(np.int32)
                cv2.fillPoly(gt_mask, [polygon_pixels], 1)
    return gt_mask


# --- 新增：保存YOLO檢測框的可視化函數 ---
def save_yolo_detection_visualization(img_path, yolo_results, output_path):
    # yolo_results[0].plot() 返回的是一张 RGB 格式的 NumPy 数组图像
    res_plotted_rgb = yolo_results[0].plot()

    # --- 核心修正：在保存前，将RGB格式转换回OpenCV的BGR格式 ---
    res_plotted_bgr = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)

    # 使用转换后的BGR图像进行保存
    cv2.imwrite(output_path, res_plotted_bgr)


def save_segmentation_visualization(img_path, gt_mask, pred_mask, metrics, output_path):
    image = cv2.imread(img_path);
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 6));
    plt.subplot(1, 3, 1);
    plt.imshow(image_rgb);
    plt.title("Original Image");
    plt.axis('off')
    gt_overlay = image_rgb.copy();
    gt_overlay[gt_mask.astype(bool)] = (gt_overlay[gt_mask.astype(bool)] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(
        np.uint8)
    plt.subplot(1, 3, 2);
    plt.imshow(gt_overlay);
    plt.title("Ground Truth (Green)");
    plt.axis('off')
    pred_overlay = image_rgb.copy();
    pred_overlay[pred_mask.astype(bool)] = (
                pred_overlay[pred_mask.astype(bool)] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    plt.subplot(1, 3, 3);
    plt.imshow(pred_overlay);
    plt.title("Prediction (Red)");
    plt.axis('off')
    plt.suptitle(
        f"IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}",
        fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]);
    plt.savefig(output_path);
    plt.close()


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"eval_{'refined' if args.refine else 'baseline'}_{timestamp}")

    # --- 新增：為不同類型的可視化創建不同的子文件夾 ---
    vis_seg_dir = os.path.join(run_output_dir, "segmentation_results")  # 分割結果圖
    vis_det_dir = os.path.join(run_output_dir, "detection_results")  # 檢測結果圖
    os.makedirs(vis_seg_dir, exist_ok=True)
    os.makedirs(vis_det_dir, exist_ok=True)
    # --- 修改結束 ---

    print(f"本次運行的所有結果將保存到: {run_output_dir}")
    if args.refine: print("模式: 已啟用迭代式精煉 (Refinement Mode: ON)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(args.yolo_weights).to(device)
    sam2_model = build_sam2(args.sam_config, args.sam_checkpoint, device=device)
    sam_predictor = SAM2ImagePredictor(sam2_model)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    metrics_log = {'iou': [], 'dice': [], 'precision': [], 'recall': [], 's_measure': [], 'e_measure': []}
    s_measure_calc = StructureMeasure()
    e_measure_calc = EnhancedAlignmentMeasure()

    csv_path = os.path.join(run_output_dir, "per_image_results.csv")
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    header = ['image_name', 'iou', 'dice', 'precision', 'recall', 's_measure', 'e_measure']
    csv_writer.writerow(header)

    for i, img_name in enumerate(tqdm(image_files, desc="正在評估")):
        img_path = os.path.join(args.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yolo_results = yolo_model.predict(image_rgb, conf=args.conf_threshold, verbose=False)

        # --- 新增：保存YOLO檢測框的可視化 ---
        if i % args.visualize_every == 0:
            det_vis_path = os.path.join(vis_det_dir, f"detection_{img_name}")
            save_yolo_detection_visualization(img_path, yolo_results, det_vis_path)
        # --- 新增結束 ---

        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if len(boxes) > 0:
            sam_predictor.set_image(image_rgb)
            masks, _, _ = sam_predictor.predict(box=boxes, multimask_output=False)
            if args.refine:
                refined_boxes = []
                if masks.size > 0:
                    for mask in masks:
                        mask_2d = np.squeeze(mask).astype(np.uint8)
                        x, y, w, h = cv2.boundingRect(mask_2d)
                        if w > 0 and h > 0: refined_boxes.append([x, y, x + w, y + h])
                if refined_boxes:
                    masks, _, _ = sam_predictor.predict(box=np.array(refined_boxes), multimask_output=False)
            if masks.size > 0:
                final_mask = np.zeros(image.shape[:2], dtype=bool)
                for mask_instance in masks:
                    current_mask = np.squeeze(mask_instance).astype(bool)
                    if current_mask.shape == final_mask.shape:
                        final_mask = np.logical_or(final_mask, current_mask)
                pred_mask = final_mask.astype(np.uint8)

        base_name = os.path.splitext(img_name)[0]
        gt_txt_path = os.path.join(args.gt_txt_dir, base_name + '.txt')
        gt_mask = load_yolo_txt_as_mask(gt_txt_path, image.shape)

        standard_metrics = calculate_standard_metrics(pred_mask, gt_mask)
        for key, value in standard_metrics.items():
            metrics_log[key].append(value)

        pred_mask_norm = pred_mask.astype(float);
        gt_mask_norm = gt_mask.astype(float)
        if np.max(pred_mask_norm) > 0: pred_mask_norm /= np.max(pred_mask_norm)
        if np.max(gt_mask_norm) > 0: gt_mask_norm /= np.max(gt_mask_norm)

        s_measure_score = s_measure_calc(pred_mask_norm, gt_mask_norm)
        e_measure_score = e_measure_calc(pred_mask_norm, gt_mask_norm)
        metrics_log['s_measure'].append(s_measure_score)
        metrics_log['e_measure'].append(e_measure_score)

        csv_row = [img_name, standard_metrics['iou'], standard_metrics['dice'], standard_metrics['precision'],
                   standard_metrics['recall'], s_measure_score, e_measure_score]
        csv_writer.writerow([f"{val:.4f}" if isinstance(val, float) else val for val in csv_row])

        if i % args.visualize_every == 0:
            seg_vis_path = os.path.join(vis_seg_dir, f"segmentation_{img_name}")
            save_segmentation_visualization(img_path, gt_mask, pred_mask, standard_metrics, seg_vis_path)

    csv_file.close()

    if not metrics_log['iou']: print("\n沒有圖像被成功評估。"); return

    report_lines = ["--- 最終評估結果 (Final Evaluation Results) ---"]
    if args.refine: report_lines.append("模式: 已啟用迭代式精煉 (Refinement Mode: ON)")
    for key, values in metrics_log.items():
        mean_value = np.mean(values);
        report_line = f"平均 {key.replace('_', ' ').title()}: {mean_value:.4f}"
        print(report_line);
        report_lines.append(report_line)
    report_lines.append(f"\n處理圖片總數: {len(metrics_log['iou'])}")

    results_txt_path = os.path.join(run_output_dir, "evaluation_results.txt")
    with open(results_txt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"\n評估報告已保存到: {results_txt_path}")
    print(f"每張圖片的詳細指標已保存到: {csv_path}")
    print(f"YOLO檢測框可視化圖已保存到: {vis_det_dir}")
    print(f"最終分割可視化圖已保存到: {vis_seg_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 + SAM 2 最終分割評估腳本 (帶雙重可視化)")
    parser.add_argument('--refine', action='store_true', help='啟用迭代式提示精煉策略。')
    parser.add_argument('--image_dir', type=str, required=True, help='測試圖片文件夾路徑')
    parser.add_argument('--gt_txt_dir', type=str, required=True, help='YOLO格式.txt分割标注文件夾路徑')
    parser.add_argument('--yolo_weights', type=str, required=True, help='YOLOv8模型權重路徑')
    parser.add_argument('--output_dir', type=str, default='./evaluation_runs', help='保存所有評估結果的基礎文件夾')
    parser.add_argument('--sam-checkpoint', type=str, default='./checkpoints/sam2_hiera_large.pt',
                        help='SAM 2模型權重路徑')
    parser.add_argument('--sam-config', type=str, default='sam2_hiera_l.yaml', help='SAM 2的模型配置文件')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='YOLOv8检测的置信度阈值')
    parser.add_argument('--visualize_every', type=int, default=1,
                        help='每隔多少張圖像保存一次可視化結果 (設為1則全部保存)')
    args = parser.parse_args()
    main(args)