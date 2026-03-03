<!-- PAGE_ID: monai-segmentation-examples -->
# MONAI 语义分割实战实例

> **目的**: 让 Claude Code 掌握 MONAI 语义分割的完整工作流
> **生成时间**: 2026-03-03
> **源仓库**: https://github.com/Project-MONAI/MONAI

---

## 一、完整分割工作流（3D 脑肿瘤分割）

### 1.1 项目结构

```
brain_tumor_segmentation/
├── data/
│   ├── images/          # 原始图像
│   └── labels/          # 分割标签
├── configs/
│   └── config.yaml      # 配置文件
├── scripts/
│   ├── train.py         # 训练脚本
│   ├── infer.py         # 推理脚本
│   └── preprocess.py    # 预处理脚本
└── outputs/
    ├── models/          # 保存的模型
    └── predictions/     # 预测结果
```

---

### 1.2 完整训练代码

```python
"""
3D 脑肿瘤分割完整训练脚本
基于 MONAI 的最佳实践
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from monai.data import CacheDataset, PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityRanged, RandCropByPosNegLabeld, RandFlipd,
    RandRotated, EnsureTyped, Activationsd, AsDiscreted,
    KeepLargestConnectedComponentd, FillHolesd
)
from monai.networks.nets import UNet, DynUNet, SwinUNETR
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.handlers import (
    CheckpointSaver, StatsHandler, ValidationHandler,
    EarlyStopHandler, LrScheduleHandler
)
from monai.engines import SupervisedTrainer, SupervisedEvaluator
import os
from datetime import datetime

# ==================== 1. 配置 ====================

config = {
    # 数据
    "data_dir": "./data",
    "train_files": "./data/train.json",  # [{"image": "...", "label": "..."}, ...]
    
    # 模型
    "model": "swin_unetr",  # "unet", "dynunet", "swin_unetr"
    "spatial_dims": 3,
    "in_channels": 4,  # 多模态：T1, T1ce, T2, FLAIR
    "out_channels": 4,  # 背景 + 3 个肿瘤子区域
    
    # 预处理
    "pixdim": (1.0, 1.0, 1.0),
    "roi_size": (96, 96, 96),
    
    # 训练
    "batch_size": 2,
    "num_epochs": 500,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 4,
    "cache_rate": 1.0,  # 1.0 = 缓存所有数据
    
    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==================== 2. 数据变换 ====================

# 训练变换
train_transforms = Compose([
    # 加载
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    
    # 空间预处理
    Spacingd(keys=["image", "label"], pixdim=config["pixdim"], 
             mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    
    # 强度归一化（BraTS 专用）
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=1000,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    
    # 数据增强：基于标签的裁剪
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=config["roi_size"],
        pos=1, neg=1,  # 50% 前景，50% 背景
        num_samples=4,  # 每个 volume 生成 4 个 patch
        image_key="image",
        image_threshold=0.0
    ),
    
    # 随机增强
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotated(
        keys=["image", "label"],
        range_x=[-0.3, 0.3],
        range_y=[-0.3, 0.3],
        range_z=[-0.3, 0.3],
        prob=0.5,
        mode=("bilinear", "nearest")
    ),
    
    # 转换为 tensor
    EnsureTyped(keys=["image", "label"]),
])

# 验证变换
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=config["pixdim"],
             mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=1000,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    EnsureTyped(keys=["image", "label"]),
])

# ==================== 3. 数据加载 ====================

# 加载数据列表
import json
with open(config["train_files"]) as f:
    data_dicts = json.load(f)

# 划分训练集和验证集
train_files, val_files = data_dicts[:80], data_dicts[80:]

# 创建数据集
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=config["cache_rate"],
    num_workers=config["num_workers"]
)

val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=config["num_workers"]
)

# 创建 DataLoader
train_loader = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=1,  # 验证时 batch_size=1
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True
)

# ==================== 4. 模型定义 ====================

def create_model(config):
    if config["model"] == "unet":
        return UNet(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.1,
        )
    
    elif config["model"] == "dynunet":
        return DynUNet(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            norm_name="instance",
            deep_supervision=True,
            deep_supr_num=2,
        )
    
    elif config["model"] == "swin_unetr":
        return SwinUNETR(
            img_size=config["roi_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            use_v2=True,
        )
    
    else:
        raise ValueError(f"Unknown model: {config['model']}")

model = create_model(config).to(config["device"])

# ==================== 5. 损失函数与优化器 ====================

# 推荐使用 DiceCELoss（Dice + CrossEntropy）
loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,
    lambda_ce=1.0,
    include_background=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
)

# 学习率调度器
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["num_epochs"],
    eta_min=1e-6,
)

# ==================== 6. 评估指标 ====================

# Dice 指标（按类别计算）
dice_metric = DiceMetric(
    include_background=True,
    reduction="mean",
    get_not_nans=False,
)

# Hausdorff 距离（评估边界）
hd_metric = HausdorffDistanceMetric(
    include_background=True,
    reduction="mean",
)

# ==================== 7. 训练引擎 ====================

# 混合精度训练
scaler = GradScaler()

# 训练循环
def train_step(engine, batch):
    model.train()
    images = batch["image"].to(config["device"])
    labels = batch["label"].to(config["device"])
    
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = loss_function(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return {"loss": loss.item()}

# ==================== 8. 验证引擎 ====================

# 滑动窗口推理（处理大体积数据）
def val_step(engine, batch):
    model.eval()
    with torch.no_grad():
        images = batch["image"].to(config["device"])
        labels = batch["label"].to(config["device"])
        
        # 滑动窗口推理
        roi_size = config["roi_size"]
        sw_batch_size = 2
        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            model,
            overlap=0.25,
            mode="gaussian",
        )
        
        # 计算 Dice
        dice_metric(y_pred=outputs, y=labels)
        
        # 计算 Hausdorff
        hd_metric(y_pred=outputs, y=labels)
    
    return {"pred": outputs, "label": labels}

# ==================== 9. Handlers ====================

# 检查点保存
checkpoint_handler = CheckpointSaver(
    save_dir="./outputs/models",
    save_dict={"net": model, "optimizer": optimizer},
    save_interval=10,
    n_saved=5,  # 只保存最近的 5 个
)

# 训练统计
stats_handler = StatsHandler(
    name="training_stats",
    log_interval=10,
)

# 验证处理
val_handler = ValidationHandler(
    interval=1,  # 每个 epoch 验证一次
)

# 早停
early_stop_handler = EarlyStopHandler(
    patience=50,
    score_function=lambda engine: engine.state.metrics["mean_dice"],
)

# 学习率调度
lr_handler = LrScheduleHandler(
    lr_scheduler=lr_scheduler,
    print_lr=True,
)

# ==================== 10. 创建训练器 ====================

trainer = SupervisedTrainer(
    device=config["device"],
    max_epochs=config["num_epochs"],
    train_data_loader=train_loader,
    network=model,
    optimizer=optimizer,
    loss_function=loss_function,
    train_step_function=train_step,
    amp=True,  # 启用混合精度
)

# 添加 handlers
trainer.add_handler(checkpoint_handler)
trainer.add_handler(stats_handler)
trainer.add_handler(val_handler)
trainer.add_handler(early_stop_handler)
trainer.add_handler(lr_handler)

# ==================== 11. 创建验证器 ====================

evaluator = SupervisedEvaluator(
    device=config["device"],
    val_data_loader=val_loader,
    network=model,
    val_step_function=val_step,
    amp=True,
)

# 验证完成后计算指标
@evaluator.on(Completes)
def compute_metrics(engine):
    mean_dice = dice_metric.aggregate().item()
    mean_hd = hd_metric.aggregate().item()
    
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Hausdorff: {mean_hd:.4f}")
    
    # 重置指标
    dice_metric.reset()
    hd_metric.reset()
    
    # 保存到 trainer 的 state
    engine.state.metrics["mean_dice"] = mean_dice
    engine.state.metrics["mean_hd"] = mean_hd

# 将验证器附加到训练器
trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: evaluator.run())

# ==================== 12. 开始训练 ====================

if __name__ == "__main__":
    print(f"Starting training with {config['model']}...")
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Device: {config['device']}")
    
    trainer.run()
    
    print("Training completed!")
```

---

## 二、推理脚本

### 2.1 单张图像推理

```python
"""
单张图像推理脚本
"""

import torch
from monai.transforms import (
    Compose, LoadImaged, AddChanneld, Spacingd, Orientationd,
    ScaleIntensityRanged, EnsureTyped, Activationsd, AsDiscreted,
    KeepLargestConnectedComponentd, SaveImaged
)
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
import os

# ==================== 1. 配置 ====================

config = {
    "model_path": "./outputs/models/net_epoch_500.pth",
    "input_image": "./data/test/image_001.nii.gz",
    "output_dir": "./outputs/predictions",
    "roi_size": (96, 96, 96),
    "sw_batch_size": 2,
    "overlap": 0.25,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ==================== 2. 加载模型 ====================

model = SwinUNETR(
    img_size=config["roi_size"],
    in_channels=4,
    out_channels=4,
    feature_size=48,
)
model.load_state_dict(torch.load(config["model_path"]))
model.to(config["device"])
model.eval()

# ==================== 3. 推理变换 ====================

infer_transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=1000,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    EnsureTyped(keys=["image"]),
])

# ==================== 4. 后处理变换 ====================

post_transforms = Compose([
    Activationsd(keys=["pred"], softmax=True),
    AsDiscreted(keys=["pred"], argmax=True, to_onehot=4),
    KeepLargestConnectedComponentd(keys=["pred"], applied_labels=[1, 2, 3]),
    FillHolesd(keys=["pred"], applied_labels=[1, 2, 3]),
    SaveImaged(
        keys=["pred"],
        output_dir=config["output_dir"],
        output_postfix="seg",
        resample=True,
    ),
])

# ==================== 5. 推理 ====================

# 加载图像
data = {"image": config["input_image"]}
data = infer_transforms(data)
image = data["image"].unsqueeze(0).to(config["device"])

print(f"Input shape: {image.shape}")

# 滑动窗口推理
with torch.no_grad():
    pred = sliding_window_inference(
        image,
        roi_size=config["roi_size"],
        sw_batch_size=config["sw_batch_size"],
        predictor=model,
        overlap=config["overlap"],
        mode="gaussian",
        padding_mode="constant",
    )

print(f"Output shape: {pred.shape}")

# 后处理
data["pred"] = pred
data = post_transforms(data)

print(f"Prediction saved to {config['output_dir']}")
```

---

### 2.2 批量推理

```python
"""
批量推理脚本
"""

import json
from torch.utils.data import DataLoader
from monai.data import Dataset

# 加载测试数据列表
with open("./data/test.json") as f:
    test_files = json.load(f)

# 创建数据集
test_ds = Dataset(data=test_files, transform=infer_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

# 批量推理
results = []
for batch in test_loader:
    image = batch["image"].to(config["device"])
    
    with torch.no_grad():
        pred = sliding_window_inference(
            image,
            config["roi_size"],
            config["sw_batch_size"],
            model,
            overlap=config["overlap"],
        )
    
    # 后处理
    batch["pred"] = pred
    batch = post_transforms(batch)
    
    # 记录结果
    results.append({
        "filename": batch["image_meta_dict"]["filename_or_obj"][0],
        "tumor_volume": batch["pred"].sum().item(),
    })

# 保存结果统计
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("./outputs/predictions/statistics.csv", index=False)
```

---

## 三、评估脚本

### 3.1 计算 Dice 和 Hausdorff

```python
"""
评估脚本：计算 Dice、Hausdorff 等指标
"""

import torch
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import LoadImaged, AddChanneld, Spacingd, EnsureTyped
from monai.data import Dataset
from torch.utils.data import DataLoader
import json

# ==================== 1. 配置 ====================

config = {
    "val_files": "./data/val.json",
    "pred_dir": "./outputs/predictions",
    "device": "cuda",
}

# ==================== 2. 加载数据 ====================

with open(config["val_files"]) as f:
    val_files = json.load(f)

# 加载变换
load_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0)),
    EnsureTyped(keys=["image", "label"]),
])

# 创建数据集
val_ds = Dataset(data=val_files, transform=load_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

# ==================== 3. 加载预测 ====================

def load_prediction(pred_path):
    """加载预测的 NIfTI 文件"""
    import nibabel as nib
    pred_nii = nib.load(pred_path)
    pred = torch.from_numpy(pred_nii.get_fdata()).unsqueeze(0).unsqueeze(0)
    return pred

# ==================== 4. 计算指标 ====================

# 初始化指标
dice_metric = DiceMetric(include_background=True, reduction="mean")
hd_metric = HausdorffDistanceMetric(include_background=True, reduction="mean")
sd_metric = SurfaceDistanceMetric(include_background=True, reduction="mean")

# 存储每个样本的结果
results = []

for batch in val_loader:
    label = batch["label"]
    filename = batch["image_meta_dict"]["filename_or_obj"][0]
    
    # 加载对应的预测
    pred_path = os.path.join(
        config["pred_dir"],
        os.path.basename(filename).replace(".nii.gz", "_seg.nii.gz")
    )
    pred = load_prediction(pred_path)
    
    # 计算 Dice（按类别）
    dice_metric(y_pred=pred, y=label)
    dice_scores = dice_metric.aggregate()
    
    # 计算 Hausdorff
    hd_metric(y_pred=pred, y=label)
    hd_scores = hd_metric.aggregate()
    
    # 计算表面距离
    sd_metric(y_pred=pred, y=label)
    sd_scores = sd_metric.aggregate()
    
    # 记录结果
    results.append({
        "filename": filename,
        "dice_background": dice_scores[0, 0].item(),
        "dice_tumor_core": dice_scores[0, 1].item(),
        "dice_peritumoral": dice_scores[0, 2].item(),
        "dice_enhancing": dice_scores[0, 3].item(),
        "hd_background": hd_scores[0, 0].item(),
        "hd_tumor_core": hd_scores[0, 1].item(),
        "hd_peritumoral": hd_scores[0, 2].item(),
        "hd_enhancing": hd_scores[0, 3].item(),
    })
    
    # 重置指标
    dice_metric.reset()
    hd_metric.reset()
    sd_metric.reset()

# ==================== 5. 保存结果 ====================

import pandas as pd

df = pd.DataFrame(results)

# 计算平均值
mean_results = df.mean(numeric_only=True)
print("=== 平均指标 ===")
print(f"Mean Dice (Background): {mean_results['dice_background']:.4f}")
print(f"Mean Dice (Tumor Core): {mean_results['dice_tumor_core']:.4f}")
print(f"Mean Dice (Peritumoral): {mean_results['dice_peritumoral']:.4f}")
print(f"Mean Dice (Enhancing): {mean_results['dice_enhancing']:.4f}")
print(f"Mean Hausdorff (Tumor Core): {mean_results['hd_tumor_core']:.4f} mm")

# 保存到 CSV
df.to_csv("./outputs/evaluation_results.csv", index=False)
print("\nResults saved to ./outputs/evaluation_results.csv")
```

---

## 四、数据预处理脚本

### 4.1 BraTS 数据预处理

```python
"""
BraTS 数据预处理脚本
将原始 BraTS 数据转换为 MONAI 格式
"""

import os
import json
import nibabel as nib
import numpy as np
from pathlib import Path

# ==================== 1. 配置 ====================

config = {
    "raw_data_dir": "./data/BraTS2021_Training",
    "output_dir": "./data/processed",
    "train_ratio": 0.8,
}

# ==================== 2. 扫描所有病例 ====================

patient_ids = sorted(os.listdir(config["raw_data_dir"]))

data_list = []
for patient_id in patient_ids:
    patient_dir = os.path.join(config["raw_data_dir"], patient_id)
    
    # 检查文件是否存在
    t1_path = os.path.join(patient_dir, f"{patient_id}_t1.nii.gz")
    t1ce_path = os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz")
    t2_path = os.path.join(patient_dir, f"{patient_id}_t2.nii.gz")
    flair_path = os.path.join(patient_dir, f"{patient_id}_flair.nii.gz")
    label_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
    
    if all(os.path.exists(p) for p in [t1_path, t1ce_path, t2_path, flair_path, label_path]):
        data_list.append({
            "patient_id": patient_id,
            "t1": t1_path,
            "t1ce": t1ce_path,
            "t2": t2_path,
            "flair": flair_path,
            "label": label_path,
        })

print(f"Found {len(data_list)} valid patients")

# ==================== 3. 合并多模态 ====================

def merge_modalities(patient_data, output_dir):
    """将 4 个模态合并为一个 4 通道的 NIfTI"""
    
    # 加载所有模态
    t1 = nib.load(patient_data["t1"]).get_fdata()
    t1ce = nib.load(patient_data["t1ce"]).get_fdata()
    t2 = nib.load(patient_data["t2"]).get_fdata()
    flair = nib.load(patient_data["flair"]).get_fdata()
    label = nib.load(patient_data["label"]).get_fdata()
    
    # 合并为 4 通道
    merged = np.stack([t1, t1ce, t2, flair], axis=0)
    
    # 保存
    output_image_path = os.path.join(
        output_dir,
        f"{patient_data['patient_id']}_image.nii.gz"
    )
    output_label_path = os.path.join(
        output_dir,
        f"{patient_data['patient_id']}_label.nii.gz"
    )
    
    nib.save(nib.Nifti1Image(merged, nib.load(patient_data["t1"]).affine), output_image_path)
    nib.save(nib.Nifti1Image(label, nib.load(patient_data["label"]).affine), output_label_path)
    
    return {
        "image": output_image_path,
        "label": output_label_path,
    }

# 创建输出目录
os.makedirs(config["output_dir"], exist_ok=True)

# 处理所有病例
processed_data = []
for patient_data in data_list:
    processed = merge_modalities(patient_data, config["output_dir"])
    processed["patient_id"] = patient_data["patient_id"]
    processed_data.append(processed)

# ==================== 4. 划分训练集和验证集 ====================

import random
random.shuffle(processed_data)

split_idx = int(len(processed_data) * config["train_ratio"])
train_data = processed_data[:split_idx]
val_data = processed_data[split_idx:]

print(f"Training: {len(train_data)}, Validation: {len(val_data)}")

# ==================== 5. 保存数据列表 ====================

with open(os.path.join(config["output_dir"], "train.json"), "w") as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(config["output_dir"], "val.json"), "w") as f:
    json.dump(val_data, f, indent=2)

with open(os.path.join(config["output_dir"], "test.json"), "w") as f:
    json.dump(processed_data[-10:], f, indent=2)  # 最后 10 个作为测试集

print("Data preprocessing completed!")
```

---

## 五、配置文件示例

### 5.1 YAML 配置

```yaml
# configs/config.yaml

# 数据
data:
  train_json: "./data/processed/train.json"
  val_json: "./data/processed/val.json"
  cache_rate: 1.0
  num_workers: 4

# 预处理
preprocessing:
  pixdim: [1.0, 1.0, 1.0]
  roi_size: [96, 96, 96]
  intensity_min: -1000
  intensity_max: 1000

# 模型
model:
  name: "swin_unetr"
  in_channels: 4
  out_channels: 4
  feature_size: 48
  dropout_rate: 0.0

# 训练
training:
  batch_size: 2
  num_epochs: 500
  learning_rate: 0.0001
  weight_decay: 0.00001
  amp: true

# 验证
validation:
  interval: 1
  sw_batch_size: 2
  overlap: 0.25

# 损失函数
loss:
  name: "DiceCELoss"
  lambda_dice: 1.0
  lambda_ce: 1.0

# 检查点
checkpoint:
  save_dir: "./outputs/models"
  save_interval: 10
  n_saved: 5

# 早停
early_stopping:
  enabled: true
  patience: 50
```

---

## 六、常见问题

### 6.1 显存不足怎么办？

```python
# 方法 1: 减小 batch size
batch_size = 1

# 方法 2: 减小 roi_size
roi_size = (64, 64, 64)  # 而不是 (96, 96, 96)

# 方法 3: 使用梯度检查点
model = SwinUNETR(..., use_checkpoint=True)

# 方法 4: 减小 sw_batch_size
sw_batch_size = 1  # 滑动窗口批大小
```

---

### 6.2 如何处理类别不平衡？

```python
# 方法 1: 使用加权 Dice Loss
from monai.losses import DiceLoss

loss_func = DiceLoss(
    include_background=True,
    weight=[0.1, 1.0, 1.0, 1.0],  # 背景权重低，前景权重高
)

# 方法 2: 使用 Focal Loss
from monai.losses import FocalLoss

loss_func = FocalLoss(
    gamma=2.0,
    weight=[0.1, 1.0, 1.0, 1.0],
)

# 方法 3: 调整采样比例
RandCropByPosNegLabeld(
    keys=["image", "label"],
    pos=2, neg=1,  # 67% 前景，33% 背景
)
```

---

### 6.3 如何提高小目标分割精度？

```python
# 方法 1: 增加前景采样比例
RandCropByPosNegLabeld(
    pos=3, neg=1,  # 75% 前景
)

# 方法 2: 使用 Deep Supervision
model = DynUNet(
    ...,
    deep_supervision=True,
    deep_supr_num=2,
)

# 方法 3: 使用 Tversky Loss（关注假阴性）
from monai.losses import TverskyLoss

loss_func = TverskyLoss(
    alpha=0.3,  # 假阳性权重低
    beta=0.7,   # 假阴性权重高
)

# 方法 4: 后处理保留小连通域
KeepLargestConnectedComponentd(
    applied_labels=[1, 2, 3],
    min_size=100,  # 最小连通域大小
)
```

---

## 七、性能优化

### 7.1 使用 PersistentDataset

```python
from monai.data import PersistentDataset

# 对于大数据集，使用磁盘缓存而不是内存缓存
train_ds = PersistentDataset(
    data=train_files,
    transform=train_transforms,
    cache_dir="./cache",  # 预处理结果缓存到磁盘
    hash_func=lambda x: hashlib.md5(str(x).encode()).hexdigest(),
)
```

---

### 7.2 使用 DataLoader 优化

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True,      # 加速 GPU 传输
    persistent_workers=True,  # 保持 worker 进程
    prefetch_factor=2,    # 预取批次数
)
```

---

### 7.3 使用混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    images = batch["image"].cuda()
    labels = batch["label"].cuda()
    
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = loss_function(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

**文档生成完成！**
