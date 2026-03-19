<!-- PAGE_ID: monai-official-radar -->
# MONAI & Generative AI 官方雷达

## 0. 使用优先级
- **医学影像项目骨架 / API 风格 / transforms / inferers / engines**: 优先参考 MONAI 文档与 Project-MONAI tutorials。
- **生成模型训练框架与组件选型**: 优先参考 Project-MONAI GenerativeModels。
- **Rectified Flow / Flow Matching 核心生成逻辑**: 优先参考 lucidrains 的实现模式。
- **集成策略**: 用 MONAI 承担医学影像项目外壳，用标准生成仓库定义生成目标、时间步、采样与损失逻辑。

## 1. MONAI 官方文档入口
- **API Reference (Latest)**: https://docs.monai.io/en/latest/api.html
- **Transforms API**: https://docs.monai.io/en/latest/transforms.html
- **Networks API**: https://docs.monai.io/en/latest/networks.html
- **Losses API**: https://docs.monai.io/en/latest/losses.html
- **Metrics API**: https://docs.monai.io/en/latest/metrics.html

## 2. 官方教程与示例
- **Project-MONAI Tutorials**: https://github.com/Project-MONAI/tutorials
- **2D Segmentation Tutorial**: https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py
- **3D Segmentation (Spleen)**: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/spleen_segmentation_3d.ipynb
- **Generative Models (VAE/Diffusion)**: https://github.com/Project-MONAI/GenerativeModels

## 3. 核心论文参考
- **MONAI Paper**: https://arxiv.org/abs/2211.02701
- **Rectified Flow (Original)**: https://arxiv.org/abs/2209.03003
- **Swin UNETR**: https://arxiv.org/abs/2201.01266

## 4. Lucidrains (Rectified Flow)
- **Repo**: https://gitlab.com/lucidrains/rectified-flow-pytorch
- **Patterns**: Check `rectified_flow_pytorch/rectified_flow.py` for SOTA implementation.
