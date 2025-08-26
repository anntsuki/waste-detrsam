# waste-detrsam

将 **RT-DETR(ultralytics)** 的目标检测与 **SAM(分割)** 流水线串联，面向垃圾/可回收物等场景的一体化检测、分割与可视化工具集。仓库内包含训练、验证、推理、性能评估、导出与可视化脚本，支持 COCO 指标评估与 FPS 基准测试。


## ✨ 功能特性
- **检测 + 分割**：先用 RT-DETR 做检测，再用 SAM 根据检测框做细粒度分割。
- **训练 / 验证 / 推理**：`train.py`、`val.py | val-pl.py`、`detect.py`。
- **性能评估**：`get_COCO_metrice.py`（COCO 指标）与 `get_FPS.py`（吞吐/FPS）。
- **可视化与分析**：`plot_result.py`、`heatmap.py`、`get_model_erf.py` 等。
- **模型导出**：`export.py`（例如导出到 ONNX）。
- **Profiling**：`main_profile.py`。
- **跟踪**：`track.py`（简易多目标跟踪管线）。

## 📦 目录结构（节选）
```
waste-detrsam/
├── ultralytics/           # 内置/定制的 ultralytics 相关代码（含 RT-DETR 等）
├── sam2-main/             # SAM / SAM2 相关实现或子模块
├── result/                # 结果输出目录（示例）
├── train.py               # 训练入口
├── val.py                 # 验证入口（标准）
├── val-pl.py              # 验证入口（变体 / PyTorch Lightning 风格）
├── detect.py              # 单/批量图片或视频推理
├── track.py               # 简易跟踪脚本
├── export.py              # 模型导出（如 ONNX）
├── get_COCO_metrice.py    # 计算 COCO 指标
├── get_FPS.py             # FPS/吞吐测试
├── get_all_yaml_param_and_flops.py  # 统计模型配置与 FLOPs
├── get_model_erf.py       # 有效感受野(ERF)分析
├── heatmap.py             # 热力图可视化
├── plot_result.py         # 绘制检测/分割结果
├── test_env.py            # 环境自检
├── setup.py / setup.cfg   # 打包&项目元信息
└── requirements.txt       # 依赖清单
```

## 🧰 环境准备
- **Python**： 3.9
- **PyTorch 2.3.1/ TorchVision0.16.0 + cu118
- **系统依赖**：Windows/Linux/macOS 

```bash
# 1) 先装好 PyTorch（示例：CUDA 12.1；请去 pytorch.org 选择与你环境匹配的命令）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) 克隆仓库并进入
git clone https://github.com/anntsuki/waste-detrsam
cd waste-detrsam

# 3) 安装 Python 依赖
pip install -r requirements.txt

# 4)（可选）本地可编辑安装
pip install -e .
```


## 📂 数据准备
建议采用 **COCO 标注格式**（
```
datasets/
├── waste/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/           # YOLO 标注或 COCO json（按你的配置）
│       ├── train/
│       ├── val/
│       └── test/
└── coco/annotations/instances_{split}.json  # 若使用 COCO json
```
你可以在 `data/*.yaml` 中定义数据路径、类别名等（若仓库已有对应配置文件）。

## 🚀 快速开始

### 训练
```bash
# 例：使用 RT-DETR（ultralytics）训练
python train.py   --data ./data/waste.yaml   --model ./ultralytics/cfg/models/rtdetr/rtdetr-l.yaml   --epochs 100   --batch 16   --imgsz 640   --project runs/train --name rtdetr_waste
```

### 验证
```bash
python val.py   --data ./data/waste.yaml   --weights ./runs/train/rtdetr_waste/weights/best.pt   --imgsz 640
# 或使用 val-pl.py（如果你偏好对应风格/依赖）
python val-pl.py --data ./data/waste.yaml --weights ./runs/train/.../best.pt
```

### 推理（检测 + 分割）
```bash
# 1) 纯检测（RT-DETR）
python detect.py   --weights ./runs/train/rtdetr_waste/weights/best.pt   --source ./demo/images   --imgsz 640   --save-txt --save-conf

# 2) 检测后分割（SAM/SAM2），根据 detect 的输出对每个 bbox 执行分割
#   具体参数因你的实现而异，这里给出示例占位：
python detect.py   --weights ./runs/train/rtdetr_waste/weights/best.pt   --source ./demo/images   --use_sam   --sam_checkpoint ./sam2-main/checkpoints/sam2_hiera_large.pt   --imgsz 640
```

### COCO 指标与 FPS
```bash
# 计算 COCO 指标
python get_COCO_metrice.py   --pred ./runs/val/preds.json   --gt ./datasets/coco/annotations/instances_val.json

# 测试 FPS（摄像头/视频/图片序列）
python get_FPS.py   --weights ./runs/train/rtdetr_waste/weights/best.pt   --source ./demo/video.mp4   --imgsz 640
```

### 导出（ONNX 等）
```bash
python export.py   --weights ./runs/train/rtdetr_waste/weights/best.pt   --include onnx   --imgsz 640
```

### 跟踪（可选）
```bash
python track.py   --weights ./runs/train/rtdetr_waste/weights/best.pt   --source ./demo/video.mp4   --imgsz 640
```

## 🔧 常见问题
- **ImportError / 算子不匹配**：核对 `torch/torchvision` 与 CUDA 的版本是否匹配。
- **`pycocotools` 安装报错**：Windows 上请先装好 VS Build Tools / 或使用预编译轮子；Linux 建议确保 `gcc` 与 Python dev 头文件齐全。
- **显存不足**：降低 `--imgsz`、`--batch`，或切换到更小的 RT-DETR 模型配置。

## 🙏 致谢
- [Ultralytics/RT-DETR](https://github.com/ultralytics/ultralytics)
- [Meta AI - Segment Anything / SAM / SAM2](https://github.com/facebookresearch/segment-anything)
---

