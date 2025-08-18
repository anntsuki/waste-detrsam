# 文件名: viaualizetojson.py (最终完整版)

import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import json
import base64
import random

# 导入所需库
import hydra
from omegaconf import OmegaConf
import pandas as pd

from ultralytics import YOLO
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_sam2_model(cfg_path, ckpt_path, device):
    """一个可靠的、用于加载SAM2模型和权重的函数。"""
    print("正在加载 SAM2 配置文件...")
    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    print("正在根据配置创建 SAM2 模型实例...")
    model = hydra.utils.instantiate(cfg.trainer.model)
    print(f"正在从 '{ckpt_path}' 加载权重...")
    loaded_state_dict = torch.load(ckpt_path, map_location="cpu")
    model_state_dict = loaded_state_dict.get("model", loaded_state_dict)
    model.load_state_dict(model_state_dict, strict=True)
    print("模型权重加载成功！")
    model.to(device)
    model.eval()
    return model


def calculate_iou(mask1, mask2):
    """计算两个二值掩码的交并比 (IoU)"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return intersection / union


def main(args):
    # 1. 准备工作
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        args.viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(args.viz_dir, exist_ok=True)
        print(f"可视化结果将保存到: {args.viz_dir}")
    print(f"生成的 LabelMe JSON 文件将保存到: {args.output_dir}")

    # 2. 加载模型
    print("正在加载 YOLOv8 模型...")
    yolo_model = YOLO(args.yolo_weights).to(device)
    sam2_model = load_sam2_model(args.sam_config, args.sam_checkpoint, device=device)
    print("正在创建 SAM2 预测器...")
    sam_predictor = SAM2ImagePredictor(sam2_model)
    print("所有模型已成功加载！")

    # 3. 获取待处理的图片列表
    image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    # 用于存储评估结果到Excel的列表
    evaluation_records = []

    # 4. 循环处理每张图片
    for img_name in tqdm(image_files, desc="正在处理图片"):
        img_path = os.path.join(args.image_dir, img_name)
        base_filename = os.path.splitext(img_name)[0]

        # =======================================================
        # <<< 修正：确保在这里读取图片并定义 img_height 和 img_width >>>
        # =======================================================
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 無法讀取圖像 {img_path}，已跳過。")
            continue
        img_height, img_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # =======================================================

        # a. YOLOv8 目标检测
        yolo_results = yolo_model.predict(image_rgb, conf=args.conf_threshold, verbose=False)
        if not yolo_results or not hasattr(yolo_results[0], 'boxes') or len(yolo_results[0].boxes) == 0:
            print(f"警告: 在图像 {img_name} 中未检测到任何物体，已跳過。")
            continue

        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        class_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = yolo_results[0].names

        # b. SAM2 分割
        sam_predictor.set_image(image_rgb)
        masks, _, _ = sam_predictor.predict(box=boxes, multimask_output=False)
        if masks is None or masks.size == 0:
            print(f"警告: SAM2 未能为图像 {img_name} 生成任何掩码，已跳過。")
            continue
        bool_masks = (masks > 0.5)

        # c. (可选) 计算评估指标
        if args.gt_dir:
            gt_mask_path = os.path.join(args.gt_dir, base_filename + '.png')
            if os.path.exists(gt_mask_path):
                gt_mask_image = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                gt_instance_ids = np.unique(gt_mask_image)
                gt_instance_ids = gt_instance_ids[gt_instance_ids != 0]
                gt_masks = [{'id': obj_id, 'mask': (gt_mask_image == obj_id)} for obj_id in gt_instance_ids]

                for i in range(len(bool_masks)):
                    pred_mask = bool_masks[i].squeeze()
                    best_iou = -1.0
                    best_gt_id = -1
                    for gt_data in gt_masks:
                        iou = calculate_iou(pred_mask, gt_data['mask'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_id = gt_data['id']
                    if best_gt_id != -1:
                        record = {'image_name': img_name, 'predicted_object_index': i, 'best_match_gt_id': best_gt_id,
                                  'iou': best_iou}
                        evaluation_records.append(record)
            else:
                print(f"警告: 找不到对应的Ground Truth mask: {gt_mask_path}")

        # d. (可选) 生成可视化图片
        if args.visualize:
            viz_image = image.copy()
            for i in range(len(bool_masks)):
                mask_np = bool_masks[i].squeeze()
                color = [random.randint(100, 255), random.randint(100, 255), random.randint(50, 150)]
                overlay = viz_image.copy()
                overlay[mask_np] = color
                viz_image = cv2.addWeighted(overlay, 0.5, viz_image, 0.5, 0)
                x1, y1, x2, y2 = boxes[i].astype(int)
                label = class_names.get(class_ids[i], '')
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(viz_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            output_viz_path = os.path.join(args.viz_dir, f"{base_filename}_visualized.jpg")
            cv2.imwrite(output_viz_path, viz_image)

        # e. 创建并保存 LabelMe JSON 文件
        with open(img_path, "rb") as img_file:
            image_data_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        labelme_data = {
            "version": "5.4.1", "flags": {}, "shapes": [],
            "imagePath": img_name, "imageData": image_data_base64,
            "imageHeight": img_height, "imageWidth": img_width
        }
        for i in range(len(bool_masks)):
            mask_2d = bool_masks[i].squeeze().astype(np.uint8)
            class_id = class_ids[i]
            label_name = class_names.get(class_id, f"class_{class_id}")
            contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = contour.squeeze().tolist()
                if len(points) < 3: continue
                shape = {"label": label_name, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}}
                labelme_data["shapes"].append(shape)
        if labelme_data["shapes"]:
            output_json_path = os.path.join(args.output_dir, f"{base_filename}.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！")
    if os.path.exists(args.output_dir) and any(f.endswith('.json') for f in os.listdir(args.output_dir)):
        print(f"JSON文件已保存至 '{args.output_dir}'")
    if args.visualize and 'viz_dir' in args and os.path.exists(args.viz_dir) and any(
            f.endswith('.jpg') for f in os.listdir(args.viz_dir)):
        print(f"可视化图片已保存至 '{args.viz_dir}'")

    if evaluation_records:
        df = pd.DataFrame(evaluation_records)
        mean_iou = df['iou'].mean()
        excel_path = os.path.join(args.output_dir, "evaluation_results.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"详细评估结果已保存到: {excel_path}")
        print("\n" + "=" * 30)
        print("          评 估 结 果")
        print("=" * 30)
        print(f"已评估的物体数量 (Total instances evaluated): {len(df)}")
        print(f"平均交并比 (Mean IoU): {mean_iou:.4f}")
        print("=" * 30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用YOLOv8+SAM2生成LabelMe格式的分割JSON文件，并可选进行可视化和评估。")
    parser.add_argument('--image_dir', type=str, required=True, help='需要进行分割的图像文件夹路径')
    parser.add_argument('--yolo_weights', type=str, required=True, help='YOLOv8檢測模型权重路径')
    parser.add_argument('--output_dir', type=str, default='./labelme_json_output',
                        help='保存生成的LabelMe .json文件的文件夹')
    parser.add_argument('--sam-checkpoint', type=str, required=True, help='您自己训练好的 SAM 2 模型权重路径 (.pt)')
    parser.add_argument('--sam-config', type=str, required=True,
                        help='与您训练时使用的 SAM 2 模型配置文件完全一致的 .yaml 文件')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='YOLOv8檢測的置信度閾值')
    parser.add_argument('--refine', action='store_true', help='啟用迭代式提示精煉策略。')
    parser.add_argument('--visualize', action='store_true', help='生成并保存带有分割掩码的可视化图片。')
    parser.add_argument('--gt_dir', type=str, default=None, help='(可选) 提供Ground Truth掩码文件夹路径以进行评估。')
    args = parser.parse_args()
    main(args)