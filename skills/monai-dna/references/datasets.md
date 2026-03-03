<!-- PAGE_ID: monai-datasets -->
# 医学影像常用数据集配置 (Dataset DNA)

## 1. BraTS (脑肿瘤分割)
- **Modality**: 4 通道 (T1, T1ce, T2, FLAIR)
- **Labels**: 0: Background, 1: Necrotic/Non-enhancing Tumor, 2: Peritumoral Edema, 3: Enhancing Tumor.
- **Regions**: 
    - TC (Tumor Core): Labels 1 + 3
    - WT (Whole Tumor): Labels 1 + 2 + 3
    - ET (Enhancing Tumor): Label 3

## 2. MSD (Medical Segmentation Decathlon)
- 涵盖 10 个任务（Liver, Brain, Hippocampus, Lung, etc.）。
- **Tip**: 始终通过 `dataset.json` 读取 `modality` 和 `labels` 映射，不要硬编码。
