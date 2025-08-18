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

from ultralytics import RTDETR
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# --- 評估指標類 (無變動) ---
class StructureMeasure:
    def __init__(self, alpha=0.5):
        self.alpha = alpha;
        self.eps = np.finfo(np.double).eps

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


# --- 輔助函數 (無變動) ---
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


def fill_holes(mask):
    """
    填充二值掩码内部的孔洞。
    :param mask: 输入的二值掩码 (H, W), np.uint8
    :return: 填充孔洞后的掩码
    """
    # 查找掩码的轮廓
    # cv2.RETR_EXTERNAL 表示只检测最外层轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与输入掩码同样大小的黑色背景
    filled_mask = np.zeros_like(mask)

    # 将找到的所有外层轮廓绘制并填充在黑色背景上
    # -1 表示填充所有找到的轮廓
    # (255) 表示用白色填充
    # cv2.FILLED 表示填充轮廓内部
    cv2.drawContours(filled_mask, contours, -1, (255), cv2.FILLED)

    return filled_mask

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


def save_yolo_detection_visualization(img_path, yolo_results, output_path):
    res_plotted_rgb = yolo_results[0].plot()
    res_plotted_bgr = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)
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


# --- 新增：不確定性校正輔助函數 ---
# --- 新增：不確定性校正輔助函數 (V2 - 引入缓冲区和多点策略) ---
# --- 最终版：融合智能采样与几何缓冲区的混合策略函数 ---
def generate_final_prompts(masks, iou_scores,
                           num_positive_points=20,
                           num_negative_points=5,
                           buffer_zone_factor=0.3):
    if masks.shape[0] < 1:
        return None, None
    masks_bool = masks.astype(bool)

    # --- 1. 智能正向点：在“共识区”（交集）放置 ---
    positive_points = []
    consensus_mask = np.logical_and.reduce(masks_bool) if masks.shape[0] > 1 else masks_bool[0]
    positive_coords = np.argwhere(consensus_mask)
    if len(positive_coords) > 0:
        num_samples = min(num_positive_points, len(positive_coords))
        sampled_indices = np.random.choice(len(positive_coords), num_samples, replace=False)
        positive_points = positive_coords[sampled_indices][:, ::-1].tolist()

    # --- 2. 几何负向点：在“最大范围”（并集）外的缓冲区放置 ---
    negative_points = []
    union_mask = np.logical_or.reduce(masks_bool)
    object_size = np.sqrt(np.sum(union_mask))
    kernel_size = max(5, int(object_size * buffer_zone_factor))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_union_mask = cv2.dilate(union_mask.astype(np.uint8), kernel, iterations=1)
    negative_sampling_zone = dilated_union_mask - union_mask
    negative_coords = np.argwhere(negative_sampling_zone > 0)
    if len(negative_coords) > 0:
        num_samples = min(num_negative_points, len(negative_coords))
        sampled_indices = np.random.choice(len(negative_coords), num_samples, replace=False)
        negative_points = negative_coords[sampled_indices][:, ::-1].tolist()

    if not positive_points and not negative_points:
        return None, None

    all_points = positive_points + negative_points
    positive_labels = [1] * len(positive_points)
    negative_labels = [0] * len(negative_points)
    all_labels = positive_labels + negative_labels
    point_coords = np.array(all_points, dtype=np.float32)
    point_labels = np.array(all_labels, dtype=np.float32)
    return point_coords, point_labels


# --- 新增結束 ---


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 修改：根據模式自動命名輸出檔案夾 ---
    mode_name = "baseline"
    if args.refine:
        mode_name = "refined"
    elif args.rectify:
        mode_name = "rectified"
    run_output_dir = os.path.join(args.output_dir, f"eval_{mode_name}_{timestamp}")
    # --- 修改結束 ---

    vis_seg_dir = os.path.join(run_output_dir, "segmentation_results")
    vis_det_dir = os.path.join(run_output_dir, "detection_results")
    os.makedirs(vis_seg_dir, exist_ok=True)
    os.makedirs(vis_det_dir, exist_ok=True)

    print(f"本次運行的所有結果將保存到: {run_output_dir}")
    # --- 修改：打印當前運行模式 ---
    if args.refine:
        print("模式: 已啟用迭代式精煉 (Refinement Mode: ON)")
    elif args.rectify:
        print("模式: 已啟用不確定性校正 (Uncertainty Rectification Mode: ON)")
    else:
        print("模式: 基礎模式 (Baseline Mode: ON)")
    # --- 修改結束 ---

    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = RTDETR(args.yolo_weights).to(device)
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

        if i % args.visualize_every == 0:
            det_vis_path = os.path.join(vis_det_dir, f"detection_{img_name}")
            save_yolo_detection_visualization(img_path, yolo_results, det_vis_path)

        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if len(boxes) > 0:
            sam_predictor.set_image(image_rgb)

            # --- 核心修改：根據模式執行不同的分割策略 ---
            # --- 核心修改：根據模式執行不同的分割策略 ---
            # --- 核心修改：根據模式執行不同的分割策略 ---
            if args.rectify:
                rectified_masks_list = []
                for i, box in enumerate(boxes):
                    object_id = f"Image_{img_name}_Box_{i}"

                    # 1. 第一轮预测：对单个box获取多个候选Mask
                    initial_masks, iou_scores, _ = sam_predictor.predict(
                        box=np.array([box]),
                        multimask_output=True
                    )

                    # 2. 调用最终的混合策略函数生成点提示
                    point_coords, point_labels = generate_final_prompts(initial_masks, iou_scores)

                    # 3. 第二轮预测：用生成的点或“框+中心点”组合进行精炼
                    if point_coords is not None:
                        # 如果混合策略成功生成点，则使用这些点
                        rectified_mask, _, _ = sam_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=False
                        )
                    else:
                        # 如果混合策略仍无法生成点(极罕见情况)，启用最终回退：框+中心点
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        rectified_mask, _, _ = sam_predictor.predict(
                            box=np.array([box]),
                            point_coords=np.array([[center_x, center_y]]),
                            point_labels=np.array([1]),
                            multimask_output=False
                        )

                    # 确保维度统一
                    if rectified_mask.ndim == 4:
                        rectified_mask = rectified_mask.squeeze(axis=1)
                    rectified_masks_list.append(rectified_mask)

                # 拼接所有mask
                if rectified_masks_list:
                    masks = np.concatenate(rectified_masks_list, axis=0)
                else:
                    masks = np.empty(0)

            elif args.refine:
                # 模式2: 迭代式精煉 (您的原始代碼)
                masks, _, _ = sam_predictor.predict(box=boxes, multimask_output=False)
                refined_boxes = []
                if masks.size > 0:
                    for mask in masks:
                        mask_2d = np.squeeze(mask).astype(np.uint8)
                        x, y, w, h = cv2.boundingRect(mask_2d)
                        if w > 0 and h > 0: refined_boxes.append([x, y, x + w, y + h])
                if refined_boxes:
                    masks, _, _ = sam_predictor.predict(box=np.array(refined_boxes), multimask_output=False)

            else:
                # 模式1: 基礎模式 (Baseline)
                masks, _, _ = sam_predictor.predict(box=boxes, multimask_output=False)
            # --- 核心修改結束 ---
            # --- 核心修改結束 ---

            if masks.size > 0:
                final_mask = np.zeros(image.shape[:2], dtype=bool)
                for mask_instance in masks:
                    current_mask = np.squeeze(mask_instance).astype(bool)
                    if current_mask.shape == final_mask.shape:
                        final_mask = np.logical_or(final_mask, current_mask)
                pred_mask = final_mask.astype(np.uint8)
                pred_mask = fill_holes(pred_mask)

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
    # --- 修改：在報告中也註明模式 ---
    if args.refine:
        report_lines.append("模式: 已啟用迭代式精煉 (Refinement Mode: ON)")
    elif args.rectify:
        report_lines.append("模式: 已啟用不確定性校正 (Uncertainty Rectification Mode: ON)")
    # --- 修改結束 ---

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
    parser = argparse.ArgumentParser(description="YOLOv8 + SAM 2 分割評估腳本 (支持多種精煉策略)")

    # --- 修改：將 refine 和 rectify 放入互斥組，確保一次只能運行一種精煉模式 ---
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--refine', action='store_true', help='啟用迭代式精煉策略 (通過重算 Bounding Box)。')
    group.add_argument('--rectify', action='store_true', help='啟用不確定性校正策略 (通過動態生成 Point Prompts)。')
    # --- 修改結束 ---

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