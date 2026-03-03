<!-- PAGE_ID: monai-visualize-dna -->
# MONAI 可视化与导出模式

## 1. 3D 切片预览 (Slice Visualization)
使用 `monai.visualize.utils.matshow_3d` 快速查看 3D 体积。
```python
from monai.visualize import matshow_3d
# 绘制中间切片
matshow_3d(image[0, 0], frames_per_row=5, show=True)
```

## 2. TensorBoard / MLFlow 集成
优先使用 `PlotSetter` 或 `StatsHandler` 自动记录。
```python
from monai.handlers import StatsHandler
# 在 SupervisedTrainer 中添加，自动记录 Loss 和 Metric
trainer.add_handler(StatsHandler(name="train_log", output_transform=lambda x: x))
```

## 3. 动态 GIF 生成
针对生成模型，建议导出动画展示去噪过程。
```python
# 核心逻辑：沿时间轴 stack 后使用 imageio 保存
import imageio
images = [unnormalize(img) for img in trajectory]
imageio.mimsave("denoising.gif", images, duration=0.1)
```
