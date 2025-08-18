import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import argparse
from pathlib import Path
from prettytable import PrettyTable
# 注意：我们在这里导入整个ultralytics库，而不是具体的类
from ultralytics.utils.torch_utils import model_info
# 关键改动：导入 increment_path 工具，用于生成唯一的文件夹名，如 exp, exp2, exp3...
from ultralytics.utils.files import increment_path

# ==============================================================================
#  [配置区域]
#  请在此字典中管理您需要验证的数据集
#  格式为：'数据集名称': '数据集yaml文件路径'
# ==============================================================================
DATASETS_TO_TEST = {
    # 'ETIS-LaribPolypDB': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\ETIS-LaribPolypDB.yaml',
    # 'CVC-ColonDB': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\CVC-ColonDB.yaml',
    # 'CVC-300': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\CVC-300.yaml',
    # 'kvasir' : 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\kvasir.yaml',
    # 'CVC-ClinicDB':'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\CVC-ClinicDB.yaml'
    # 如果还有其他测试集，继续在这里添加
    'cdwdata': "E:\\stv\\ml\\paper\\yolo11\\ultralytics-yolo11-main\\ultralytics\\cfg\\datasets\\cdwdata.yaml",
    # 'ClinicDB + Kvasir‑SEG': 'E:\\stv\\ml\\paper\\RTDETR-pl\\results\\DATESET\\POLYP\\1.yaml'
}
# ==============================================================================

def get_weight_size(path):
    """计算模型文件的大小（MB）"""
    if not os.path.exists(path):
        return "N/A"
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

def validate_model(args):
    """主验证函数"""
    
    # ==============================================================================
    # 关键改动：在所有验证开始前，创建本次运行专属的、唯一的父文件夹
    # 例如，如果 runs/val_results/exp 已存在，则会自动创建 runs/val_results/exp2
    # ==============================================================================
    project_dir = increment_path(Path(args.project) / 'exp', exist_ok=False)
    
    # 根据命令行参数决定加载哪个模型类
    if args.model_type.lower() == 'yolo':
        from ultralytics import YOLO as ModelClass
    elif args.model_type.lower() == 'rtdetr':
        from ultralytics import RTDETR as ModelClass
    else:
        print(f"错误: 不支持的模型类型 '{args.model_type}'。请选择 'yolo' 或 'rtdetr'。")
        return

    # 遍历所有预先定义好的数据集进行验证
    for dataset_name, dataset_yaml in DATASETS_TO_TEST.items():
        if not os.path.exists(dataset_yaml):
            print(f"\n警告: 数据集配置文件不存在，跳过: {dataset_yaml}")
            continue
            
        print("\n" + "="*80)
        print(f"正在加载 {args.model_type.upper()} 模型: {args.weights}")
        try:
            model = ModelClass(args.weights)
        except Exception as e:
            print(f"错误: 无法加载模型权重 {args.weights}。请检查路径和模型类型是否匹配。")
            print(e)
            return
            
        print(f"开始使用 {args.model_type.upper()} 验证数据集: {dataset_name} ({dataset_yaml})")
        print("="*80 + "\n")

        weights_basename = os.path.splitext(os.path.basename(args.weights))[0]
        run_name = f"{weights_basename}_on_{dataset_name}"

        try:
            # 关键改动：使用上面创建的唯一父文件夹作为 project 路径
            result = model.val(data=dataset_yaml,
                               split=args.split,
                               imgsz=args.imgsz,
                               batch=args.batch,
                               project=project_dir,  # 指定本次运行的总目录
                               name=run_name,        # 在总目录下创建单个验证结果的子目录
                               augment = True
                               )
        except Exception as e:
            print(f"在验证数据集 {dataset_yaml} 时发生错误: {e}")
            continue

        if model.task == 'detect':
            
            # --- 模型信息表 ---
            n_l, n_p, n_g, flops = model_info(model.model)
            preprocess_time = result.speed.get('preprocess', 0.0)
            inference_time = result.speed.get('inference', 0.0)
            postprocess_time = result.speed.get('postprocess', 0.0)
            all_time_per_image = preprocess_time + inference_time + postprocess_time
            
            model_info_table = PrettyTable()
            model_info_table.title = f"模型信息 ({os.path.basename(args.weights)})"
            model_info_table.field_names = ["GFLOPs", "参数量(M)", "前处理/图(ms)", "推理/图(ms)", "后处理/图(ms)", "FPS(总)", "FPS(仅推理)", "模型大小(MB)"]
            model_info_table.add_row([
                f'{flops:.1f}', f'{n_p / 1e6:.2f}', f'{preprocess_time:.2f}', 
                f'{inference_time:.2f}', f'{postprocess_time:.2f}', 
                f'{1000 / all_time_per_image:.2f}' if all_time_per_image > 0 else 'N/A', 
                f'{1000 / inference_time:.2f}' if inference_time > 0 else 'N/A', 
                f'{get_weight_size(args.weights)}'
            ])
            print(model_info_table)

            # --- 详细指标表 ---
            length = result.box.p.size
            class_names = list(result.names.values())
            
            model_metrice_table = PrettyTable()
            model_metrice_table.title = f"模型在 {dataset_name} 数据集上的性能指标"
            model_metrice_table.field_names = ["类别", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
            
            for idx in range(length):
                model_metrice_table.add_row([
                    class_names[idx], f"{result.box.p[idx]:.4f}", f"{result.box.r[idx]:.4f}", 
                    f"{result.box.f1[idx]:.4f}", f"{result.box.ap50[idx]:.4f}", 
                    f"{result.box.all_ap[idx, 5]:.4f}", f"{result.box.ap[idx]:.4f}"
                ])
            
            model_metrice_table.add_row([
                "all (平均)", f"{result.results_dict['metrics/precision(B)']:.4f}", 
                f"{result.results_dict['metrics/recall(B)']:.4f}", f"{np.mean(result.box.f1[:length]):.4f}", 
                f"{result.results_dict['metrics/mAP50(B)']:.4f}", f"{np.mean(result.box.all_ap[:length, 5]):.4f}",
                f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
            ])
            print(model_metrice_table)

            # --- 保存结果到文件 ---
            output_file_path = result.save_dir / 'paper_ready_results.txt'
            try:
                with open(output_file_path, 'w', errors="ignore", encoding="utf-8") as f:
                    f.write(f"模型类型: {args.model_type.upper()}\n")
                    f.write(f"模型权重: {os.path.basename(args.weights)}\n")
                    f.write(f"验证数据集: {dataset_name} ({dataset_yaml})\n\n")
                    f.write(str(model_info_table))
                    f.write('\n\n')
                    f.write(str(model_metrice_table))
                print('-'*20, f'结果已保存至 {output_file_path}', '-'*20)
            except Exception as e:
                print(f"保存结果文件时发生错误: {e}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO/RT-DETR 通用模型批量验证脚本")
    parser.add_argument('--model-type', type=str, required=True, choices=['yolo', 'rtdetr'], help="必须项: 指定模型类型, 'yolo' 或 'rtdetr'")
    parser.add_argument('--weights', type=str, required=True, help='必须项: 指向模型权重文件(.pt)的路径')
    parser.add_argument('--split', type=str, default='test', help="要验证的数据集划分 ('train', 'val', 'test')")
    parser.add_argument('--imgsz', type=int, default=640, help='验证时使用的图像尺寸')
    parser.add_argument('--batch', type=int, default=8, help='验证时的批处理大小')
    parser.add_argument('--project', type=str, default='runs/val_results', help='保存验证结果的主目录')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    validate_model(args)