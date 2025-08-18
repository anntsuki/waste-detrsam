import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import shutil
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import sys
from prettytable import PrettyTable
# 这部分代码是为了处理可能的模块导入问题
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from ultralytics import RTDETR
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ==============================================================================
#  [配置区域] - 您需要编辑的就是这里
# ==============================================================================
DATASETS_TO_TEST = {
    # 'CVC-300': {
    #     'image_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test22\\CVC-300\\images',
    #     'gt_txt_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test22\\CVC-300/gtlabels'
    # },
    # 'CVC-ClinicDB': {
    #     'image_dir': 'E:\\stv\\ml\\paper\\yolo11\\ultralytics-yolo11-main\\dataset\\polyp_yolo\\test\\CVC-ClinicDB\\images',
    #     'gt_txt_dir': 'E:\\stv\\ml\\paper\\yolo11\\ultralytics-yolo11-main\\dataset\\polyp_yolo\\test\\CVC-ClinicDB\\gtlabels'
    # },
    # 'CVC-ColonDB': {
    #     'image_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test22\\CVC-ColonDB\\images',
    #     'gt_txt_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test22\\CVC-ColonDB\\gtlabels'
    # },
    # 'ETIS-LaribPolypDB': {
    #     'image_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test22\\ETIS-LaribPolypDB\\images',
    #     'gt_txt_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test22\\ETIS-LaribPolypDB\\gtlabels'
    # # },
    # 'Kvasir': {
    #     'image_dir': 'E:\\stv\\ml\\paper\\yolo11\\ultralytics-yolo11-main\\dataset\\polyp_yolo\\test\\Kvasir\\images',
    #     'gt_txt_dir': 'E:\\stv\\ml\\paper\\yolo11\\ultralytics-yolo11-main\\dataset\\polyp_yolo\\test\\Kvasir\\gtlabels'
    # },
    'mix':{
        'image_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test\\images',
        'gt_txt_dir': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\test\\gtlabels'

    },
}


# ==============================================================================


# --- 评估指标类 (无变动) ---
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


# --- 辅助函数 (无变动) ---
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


def save_detection_visualization(det_results, output_path):
    res_plotted_rgb = det_results[0].plot()
    res_plotted_bgr = cv2.cvtColor(res_plotted_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, res_plotted_bgr)


def save_segmentation_visualization(image, gt_mask, pred_mask, metrics, output_path):
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
    overall_run_dir = os.path.join(args.output_dir, f"eval_run_{timestamp}")
    print(f"本次所有数据集的评估结果将保存到主目录: {overall_run_dir}")
    os.makedirs(overall_run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"正在加载 RT-DETR 模型: {args.rtdetr_weights}")
    detector_model = RTDETR(args.rtdetr_weights).to(device)
    print(f"正在加载 SAM2 模型: {args.sam_checkpoint}")
    sam2_model = build_sam2(args.sam_config, args.sam_checkpoint, device=device)
    sam_predictor = SAM2ImagePredictor(sam2_model)

    # ==============================================================================
    # <<< 关键改动 1: 初始化一个总的结果字典 >>>
    # ==============================================================================
    overall_results = {}
    metric_keys = ['iou', 'dice', 'precision', 'recall', 's_measure', 'e_measure']

    for dataset_name, paths in DATASETS_TO_TEST.items():
        image_dir = paths['image_dir']
        gt_txt_dir = paths['gt_txt_dir']

        print("\n" + "=" * 80)
        print(f"开始评估数据集: {dataset_name}")
        print("=" * 80)

        dataset_output_dir = os.path.join(overall_run_dir, dataset_name)
        vis_seg_dir = os.path.join(dataset_output_dir, "segmentation_results")
        vis_det_dir = os.path.join(dataset_output_dir, "detection_results")
        os.makedirs(vis_seg_dir, exist_ok=True)
        os.makedirs(vis_det_dir, exist_ok=True)

        if not os.path.isdir(image_dir):
            print(f"警告: 数据集 '{dataset_name}' 的图片目录不存在，已跳过: {image_dir}")
            continue

        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        metrics_log = {key: [] for key in metric_keys}
        s_measure_calc = StructureMeasure()
        e_measure_calc = EnhancedAlignmentMeasure()
        csv_path = os.path.join(dataset_output_dir, f"{dataset_name}_results.csv")
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        header = ['image_name'] + metric_keys
        csv_writer.writerow(header)

        for i, img_name in enumerate(tqdm(image_files, desc=f"正在评估 {dataset_name}")):
            img_path = os.path.join(image_dir, img_name)

            det_results = list(
                detector_model.predict(img_path, conf=args.conf_threshold, verbose=False, imgsz=args.imgsz))

            if not det_results:
                print(f"警告: 模型未能处理或未在图片 {img_name} 中找到任何结果，已跳过。")
                continue

            image = det_results[0].orig_img
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if i % args.visualize_every == 0 and len(det_results[0].boxes) > 0:
                det_vis_path = os.path.join(vis_det_dir, f"detection_{img_name}")
                save_detection_visualization(det_results, det_vis_path)

            pred_mask = np.zeros(image.shape[:2], dtype=np.uint8)

            # ==============================================================================
            # <<< 唯一修改点：从“单框提示”改为“多框提示” >>>
            # ==============================================================================
            if len(det_results[0].boxes) > 0:
                # 1. 直接获取所有检测到的框 (不再需要寻找最佳框)
                all_boxes = det_results[0].boxes.xyxy.cpu().numpy()

                # 2. 设置图像并使用所有框作为提示来预测
                sam_predictor.set_image(image_rgb)
                masks, _, _ = sam_predictor.predict(
                    box=all_boxes,  # 传入所有框的 NumPy 数组
                    multimask_output=False  # 让 SAM 智能合并结果，返回一个掩码
                )

                # 3. 处理 SAM 返回的单个最优掩码
                if masks.size > 0:
                    pred_mask = np.squeeze(masks[0]).astype(np.uint8)
            # ==============================================================================
            # <<< 修改结束 >>>
            # ==============================================================================

            base_name = os.path.splitext(img_name)[0]
            gt_txt_path = os.path.join(gt_txt_dir, base_name + '.txt')
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

            csv_row = [img_name] + [metrics_log[key][-1] for key in metric_keys]
            csv_writer.writerow([f"{val:.4f}" if isinstance(val, float) else val for val in csv_row])

            if i % args.visualize_every == 0:
                seg_vis_path = os.path.join(vis_seg_dir, f"segmentation_{img_name}")
                save_segmentation_visualization(image, gt_mask, pred_mask, standard_metrics, seg_vis_path)

        csv_file.close()

        if not metrics_log['iou']:
            print(f"\n数据集 '{dataset_name}' 没有图像被成功评估。")
            continue

        # --- 关键改动 2: 计算并存储当前数据集的平均结果，但先不打印 ---
        dataset_summary = {}
        report_lines = [f"--- 数据集: {dataset_name} 详细评估报告 ---"]
        for key, values in metrics_log.items():
            mean_value = np.mean(values)
            dataset_summary[key] = mean_value
            report_lines.append(f"平均 {key.replace('_', ' ').title()}: {mean_value:.4f}")
        report_lines.append(f"\n处理图片总数: {len(metrics_log['iou'])}")

        overall_results[dataset_name] = dataset_summary

        # 将单个数据集的详细报告保存到其自己的文件夹中
        results_txt_path = os.path.join(dataset_output_dir, f"{dataset_name}_evaluation_summary.txt")
        with open(results_txt_path, "w", encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"数据集 '{dataset_name}' 的详细报告已保存到: {results_txt_path}")

    # ==============================================================================
    # <<< 关键改动 3: 所有数据集都评估完后，打印和保存总的对比表格 >>>
    # ==============================================================================
    print("\n\n" + "#" * 30)
    print("### 所有数据集评估完成，生成总览报告 ###")
    print("#" * 30 + "\n")

    summary_table = PrettyTable()
    table_headers = ["数据集 (Dataset)"] + [key.replace('_', ' ').title() for key in metric_keys]
    summary_table.field_names = table_headers

    for dataset_name, metrics in overall_results.items():
        row = [dataset_name] + [f"{metrics.get(key, 0.0):.4f}" for key in metric_keys]
        summary_table.add_row(row)

    print(summary_table)

    # 将总览表格也保存到文件中
    summary_report_path = os.path.join(overall_run_dir, "overall_summary_report.txt")
    with open(summary_report_path, 'w', encoding='utf-8') as f:
        f.write(f"评估模型: {args.rtdetr_weights}\n")
        f.write(f"评估时间: {timestamp}\n\n")
        f.write(str(summary_table))

    print(f"\n总览评估报告已保存到: {summary_report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RT-DETR + SAM 2 批量分割评估脚本")
    # 移除了 --image_dir 和 --gt_txt_dir, 因为它们现在从代码顶部的字典中读取
    parser.add_argument('--rtdetr_weights', type=str, required=True, help='RT-DETR 模型權重路徑 (.pt)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_runs', help='保存所有评估结果的基礎文件夾')
    parser.add_argument('--sam-checkpoint', type=str, required=True, help='SAM 2模型權重路徑')
    parser.add_argument('--sam-config', type=str, required=True, help='SAM 2的模型配置文件')
    parser.add_argument('--conf-threshold', type=float, default=0.1,
                        help='RT-DETR 检测的置信度阈值 (建议从0.1开始尝试)')
    parser.add_argument('--imgsz', type=int, default=640, help='验证时使用的图像尺寸 (请确保与训练时一致)')
    parser.add_argument('--visualize_every', type=int, default=1,
                        help='每隔多少張圖像保存一次可視化結果 (設為1則全部保存)')
    args = parser.parse_args()
    main(args)