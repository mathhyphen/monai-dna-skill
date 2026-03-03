<!-- PAGE_ID: monai-troubleshooting -->
# MONAI 避坑指南 (Troubleshooting DNA)

## 1. 维度陷阱 (The Dimension Pitfall)
- **Problem**: MONAI 默认 (C, H, W, D)，但某些旧代码或自定义 LoadImage 可能是 (H, W, D, C)。
- **Solution**: 始终使用 `EnsureChannelFirstd(keys=["image"])` ([monai/transforms/utility/dictionary.py:50](url))。

## 2. 显存爆炸 (OOM - Out of Memory)
- **Problem**: 3D 卷积极其耗显存，尤其是 SwinUNETR。
- **Solution**: 
    - 强制使用 `SlidingWindowInferer` 进行推理。
    - 训练时使用 `CacheDataset(cache_rate=0.5)` 而非 `1.0` 来平衡内存。
    - 启用 `AMP` (Automatic Mixed Precision)。

## 3. 标签值错误
- **Problem**: 经过 `Spacingd` 之后，原本的 [0, 1, 2] 标签可能变成了 [0, 0.4, 0.8, 1]。
- **Solution**: 标签变换必须指定 `mode="nearest"`。

## 4. 损失函数不匹配
- **Problem**: `DiceLoss` 默认不处理 Background。
- **Solution**: 如果 Background 很重要，设置 `include_background=True`；通常分割任务推荐 `DiceCELoss`。
