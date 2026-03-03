<!-- PAGE_ID: monai-transforms-complete -->
# MONAI 数据变换完整指南

> **目的**: 让 Claude Code 掌握 MONAI 所有数据变换的使用方法
> **生成时间**: 2026-03-03
> **源仓库**: https://github.com/Project-MONAI/MONAI

---

## 一、Transform 核心概念

### 1.1 两种模式：Array vs Dictionary

MONAI 的 Transform 分为两种模式：

| 模式 | 命名规则 | 输入类型 | 使用场景 |
|------|---------|---------|---------|
| **Array** | 无后缀 | `np.ndarray` / `torch.Tensor` | 单张图像处理 |
| **Dictionary** | `d` 后缀 | `Dict[str, Any]` | 多模态数据（image + label） |

**示例对比**：

```python
# Array 模式 - 处理单张图像
from monai.transforms import LoadImage, NormalizeIntensity

loader = LoadImage(image_only=True)
img = loader("image.nii.gz")  # 返回 tensor
normalizer = NormalizeIntensity()
img_normalized = normalizer(img)

# Dictionary 模式 - 处理图像+标签
from monai.transforms import LoadImaged, NormalizeIntensityd

loader = LoadImaged(keys=["image", "label"])
data_dict = loader({"image": "image.nii.gz", "label": "label.nii.gz"})
# data_dict = {"image": tensor, "label": tensor, ...}

normalizer = NormalizeIntensityd(keys=["image"])
data_dict = normalizer(data_dict)
# data_dict["image"] 已归一化，data_dict["label"] 不变
```

**源码引用**：
- [monai/transforms/transform.py:50-80](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/transform.py#L50-L80)

---

### 1.2 Transform 基类继承关系

```
Transform (基类)
├── MapTransform (字典模式基类)
│   ├── LoadImaged
│   ├── NormalizeIntensityd
│   └── RandCropByPosNegLabeld
├── RandomizableTransform (随机变换基类)
│   ├── RandFlip
│   ├── RandRotate
│   └── RandZoom
└── Compose (组合器)
```

**核心方法**：

```python
class Transform:
    def __call__(self, data):
        """Apply transform to data"""
        raise NotImplementedError

class MapTransform(Transform):
    def __init__(self, keys: KeysCollection):
        self.keys = keys
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d

class RandomizableTransform(Transform):
    def randomize(self, data) -> None:
        """Generate random factors"""
        self._do_transform = self.R.random() < self.prob
```

**源码引用**：
- [monai/transforms/transform.py:30-150](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/transform.py#L30-L150)

---

## 二、I/O 变换（输入输出）

### 2.1 LoadImage / LoadImaged

**功能**：加载医学图像文件（NIfTI, DICOM, PNG, JPEG 等）

**关键参数**：

```python
LoadImage(
    reader=None,              # 可选："ITKReader", "NibabelReader", "PILReader"
    image_only=True,          # True: 只返回图像; False: 返回 (image, metadata)
    dtype=np.float32,         # 输出数据类型
    ensure_channel_first=True # 确保维度顺序为 (C, H, W, D)
)
```

**完整示例**：

```python
from monai.transforms import LoadImage, LoadImaged

# 1. 加载单张 NIfTI 图像
loader = LoadImage(image_only=True)
img = loader("brain.nii.gz")
print(img.shape)  # (1, 256, 256, 128) - (C, H, W, D)

# 2. 加载图像并获取元数据
loader_with_meta = LoadImage(image_only=False)
img, meta = loader_with_meta("brain.nii.gz")
print(meta["filename_or_obj"])  # "brain.nii.gz"
print(meta["affine"])  # 4x4 仿射矩阵

# 3. 字典模式：加载图像和标签
loader_dict = LoadImaged(keys=["image", "label"])
data_dict = loader_dict({
    "image": "patient001_image.nii.gz",
    "label": "patient001_label.nii.gz"
})
print(data_dict["image"].shape)  # (1, H, W, D)
print(data_dict["label"].shape)  # (1, H, W, D)

# 4. 加载 DICOM 系列
from monai.data import ITKReader
dicom_loader = LoadImage(reader=ITKReader(), image_only=True)
img = dicom_loader("/path/to/dicom/folder")
```

**源码引用**：
- [monai/transforms/io/array.py:20-100](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/io/array.py#L20-L100)
- [monai/transforms/io/dictionary.py:20-80](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/io/dictionary.py#L20-L80)

---

### 2.2 SaveImage / SaveImaged

**功能**：保存处理后的图像到文件

**完整示例**：

```python
from monai.transforms import SaveImage, SaveImaged

# 1. 保存单张图像
saver = SaveImage(
    output_dir="./output",
    output_postfix="processed",
    output_ext=".nii.gz",
    separate_folder=False
)
saver(processed_img)

# 2. 字典模式：批量保存
saver_dict = SaveImaged(
    keys=["pred"],
    output_dir="./predictions",
    output_postfix="seg",
    resample=True  # 恢复原始分辨率
)
saver_dict({"pred": pred_tensor, "image": original_image})
```

**源码引用**：
- [monai/transforms/io/array.py:150-250](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/io/array.py#L150-L250)

---

## 三、空间变换（Spatial Transforms）

### 3.1 Spacingd - 重采样

**功能**：调整体素间距（分辨率）

**完整示例**：

```python
from monai.transforms import Spacingd

# 将所有图像重采样到 1mm isotropic
resample = Spacingd(
    keys=["image", "label"],
    pixdim=(1.0, 1.0, 1.0),  # 目标间距 (x, y, z) in mm
    mode=("bilinear", "nearest"),  # image用双线性，label用最近邻
    padding_mode="border"
)

data_dict = resample(data_dict)
# 现在体素间距为 (1.0, 1.0, 1.0) mm
```

**关键参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `pixdim` | 目标间距 (x, y, z) | 必填 |
| `mode` | 插值方式 | `("bilinear", "nearest")` |
| `padding_mode` | 边界填充方式 | `"border"` |
| `align_corners` | 角对齐 | `False` |

**插值模式选择**：
- **图像**: `"bilinear"` (2D) / `"trilinear"` (3D)
- **标签**: `"nearest"` (避免产生中间值)
- **高精度**: `"bicubic"` (更平滑但更慢)

**源码引用**：
- [monai/transforms/spatial/array.py:500-650](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L500-L650)

---

### 3.2 Orientationd - 方向调整

**功能**：调整图像方向（RAS, LPS 等）

```python
from monai.transforms import Orientationd

# 统一到 RAS (Right-Anterior-Superior) 方向
orient = Orientationd(
    keys=["image", "label"],
    axcodes="RAS"  # 也可以是 "LPS"
)

data_dict = orient(data_dict)
```

**源码引用**：
- [monai/transforms/spatial/array.py:300-400](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L300-L400)

---

### 3.3 Resized - 尺寸调整

**功能**：调整图像尺寸到指定大小

```python
from monai.transforms import Resized

# 调整到固定大小
resize = Resized(
    keys=["image", "label"],
    spatial_size=(128, 128, 64),  # (H, W, D)
    mode=("area", "nearest")  # 缩小用 area，放大用 bilinear
)

data_dict = resize(data_dict)
print(data_dict["image"].shape)  # (1, 128, 128, 64)
```

**源码引用**：
- [monai/transforms/spatial/array.py:700-800](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L700-L800)

---

### 3.4 RandFlipd - 随机翻转

**功能**：沿指定轴随机翻转

```python
from monai.transforms import RandFlipd

flip = RandFlipd(
    keys=["image", "label"],
    prob=0.5,  # 50% 概率翻转
    spatial_axis=[0, 1, 2]  # 可以在任意轴翻转
)

data_dict = flip(data_dict)
```

**源码引用**：
- [monai/transforms/spatial/array.py:100-200](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L100-L200)

---

### 3.5 RandRotated - 随机旋转

**功能**：随机旋转图像

```python
from monai.transforms import RandRotated

rotate = RandRotated(
    keys=["image", "label"],
    range_x=[-0.5, 0.5],  # x轴旋转范围（弧度）
    range_y=[-0.5, 0.5],
    range_z=[-0.5, 0.5],
    prob=0.5,
    mode=("bilinear", "nearest"),
    padding_mode="zeros"
)

data_dict = rotate(data_dict)
```

**源码引用**：
- [monai/transforms/spatial/array.py:200-300](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L200-L300)

---

### 3.6 RandZoomd - 随机缩放

```python
from monai.transforms import RandZoomd

zoom = RandZoomd(
    keys=["image", "label"],
    prob=0.5,
    min_zoom=0.9,
    max_zoom=1.1,
    mode=("trilinear", "nearest"),
    padding_mode="constant"
)

data_dict = zoom(data_dict)
```

**源码引用**：
- [monai/transforms/spatial/array.py:400-500](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L400-L500)

---

### 3.7 RandAffined - 随机仿射变换

**功能**：组合旋转、平移、缩放、剪切

```python
from monai.transforms import RandAffined

affine = RandAffined(
    keys=["image", "label"],
    rotate_range=[0.2, 0.2, 0.2],  # 旋转范围
    translate_range=[10, 10, 10],  # 平移范围（像素）
    scale_range=[0.1, 0.1, 0.1],   # 缩放范围
    shear_range=[0.1, 0.1],        # 剪切范围
    prob=0.5,
    mode=("bilinear", "nearest"),
    padding_mode="zeros"
)

data_dict = affine(data_dict)
```

**源码引用**：
- [monai/transforms/spatial/array.py:800-1000](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/spatial/array.py#L800-L1000)

---

## 四、裁剪变换（Crop Transforms）

### 4.1 CenterSpatialCropd - 中心裁剪

```python
from monai.transforms import CenterSpatialCropd

crop = CenterSpatialCropd(
    keys=["image", "label"],
    roi_size=(128, 128, 64)  # 裁剪到 (H, W, D)
)

data_dict = crop(data_dict)
```

---

### 4.2 RandSpatialCropd - 随机裁剪

```python
from monai.transforms import RandSpatialCropd

random_crop = RandSpatialCropd(
    keys=["image", "label"],
    roi_size=(128, 128, 64),
    random_size=False  # True: 裁剪尺寸随机; False: 固定尺寸
)

data_dict = random_crop(data_dict)
```

---

### 4.3 RandCropByPosNegLabeld - 基于标签的采样裁剪

**功能**：根据正负样本比例裁剪（分割任务核心）

```python
from monai.transforms import RandCropByPosNegLabeld

# 最常用的裁剪方式：平衡前景/背景
crop_by_label = RandCropByPosNegLabeld(
    keys=["image", "label"],
    label_key="label",  # 指定标签的 key
    spatial_size=(96, 96, 96),  # 裁剪大小
    pos=1,  # 正样本权重
    neg=1,  # 负样本权重
    num_samples=4,  # 每个图像采样 4 个 patch
    image_key="image",  # 可选：用于过滤全黑 patch
    image_threshold=0.0  # 过滤阈值
)

# 应用后返回的是 list of dicts
samples = crop_by_label(data_dict)
for sample in samples:
    print(sample["image"].shape)  # (1, 96, 96, 96)
    print(sample["label"].shape)  # (1, 96, 96, 96)
```

**关键参数**：
- `pos=1, neg=1`: 50% 前景，50% 背景
- `pos=2, neg=1`: 67% 前景，33% 背景
- `num_samples=4`: 每个 volume 生成 4 个训练样本

**源码引用**：
- [monai/transforms/crop/array.py:500-700](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/crop/array.py#L500-L700)

---

### 4.4 RandCropByLabelClassesd - 按类别裁剪

```python
from monai.transforms import RandCropByLabelClassesd

crop_by_class = RandCropByLabelClassesd(
    keys=["image", "label"],
    label_key="label",
    spatial_size=(96, 96, 96),
    ratios=[1, 1, 2],  # 每个类别的采样比例 [背景, 类别1, 类别2]
    num_classes=3,
    num_samples=4
)

samples = crop_by_class(data_dict)
```

---

## 五、强度变换（Intensity Transforms）

### 5.1 NormalizeIntensityd - 强度归一化

```python
from monai.transforms import NormalizeIntensityd

# 方法1: 基于均值和标准差
normalize = NormalizeIntensityd(
    keys=["image"],
    subtrahend="mean",  # 减去均值
    divisor="std",      # 除以标准差
    nonzero=True,       # 只使用非零值计算
    channel_wise=True   # 每个通道独立计算
)

# 方法2: 基于固定值
normalize_fixed = NormalizeIntensityd(
    keys=["image"],
    subtrahend=0.0,
    divisor=255.0
)

data_dict = normalize(data_dict)
```

---

### 5.2 ScaleIntensityRanged - 范围映射

**功能**：将强度范围映射到 [0, 1]

```python
from monai.transforms import ScaleIntensityRanged

scale = ScaleIntensityRanged(
    keys=["image"],
    a_min=-1000,  # 输入最小值（如 CT 的 -1000 HU）
    a_max=1000,   # 输入最大值
    b_min=0.0,    # 输出最小值
    b_max=1.0,    # 输出最大值
    clip=True     # 超出范围的值截断
)

data_dict = scale(data_dict)
# 现在 image 的值在 [0, 1] 范围内
```

**CT 常用范围**：
- 软组织窗: `a_min=-150, a_max=250`
- 骨窗: `a_min=-1000, a_max=1000`
- 肺窗: `a_min=-1500, a_max=400`

**源码引用**：
- [monai/transforms/intensity/array.py:200-350](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/transforms/intensity/array.py#L200-L350)

---

### 5.3 AdjustContrastd - 对比度调整

```python
from monai.transforms import AdjustContrastd

contrast = AdjustContrastd(
    keys=["image"],
    gamma=1.5  # gamma > 1 增加对比度
)

data_dict = contrast(data_dict)
```

---

### 5.4 RandGaussianNoised - 高斯噪声

```python
from monai.transforms import RandGaussianNoised

noise = RandGaussianNoised(
    keys=["image"],
    prob=0.5,
    mean=0.0,
    std=0.1  # 噪声标准差
)

data_dict = noise(data_dict)
```

---

### 5.5 RandGaussianSmoothd - 高斯平滑

```python
from monai.transforms import RandGaussianSmoothd

smooth = RandGaussianSmoothd(
    keys=["image"],
    sigma_x=(0.5, 1.5),  # sigma 范围
    sigma_y=(0.5, 1.5),
    sigma_z=(0.5, 1.5),
    prob=0.5
)

data_dict = smooth(data_dict)
```

---

### 5.6 RandScaleIntensityd - 随机缩放强度

```python
from monai.transforms import RandScaleIntensityd

scale_random = RandScaleIntensityd(
    keys=["image"],
    factors=(-0.2, 0.2),  # 缩放因子范围
    prob=0.5
)

data_dict = scale_random(data_dict)
```

---

### 5.7 RandShiftIntensityd - 随机偏移强度

```python
from monai.transforms import RandShiftIntensityd

shift = RandShiftIntensityd(
    keys=["image"],
    offsets=(-50, 50),  # 偏移范围
    prob=0.5
)

data_dict = shift(data_dict)
```

---

### 5.8 RandBiasFieldd - 偏置场模拟

**功能**：模拟 MRI 偏置场伪影

```python
from monai.transforms import RandBiasFieldd

bias = RandBiasFieldd(
    keys=["image"],
    prob=0.5,
    coeff_range=(0.0, 0.5),
    degree=3
)

data_dict = bias(data_dict)
```

---

## 六、后处理变换（Post-Processing Transforms）

### 6.1 Activationsd - 激活函数

```python
from monai.transforms import Activationsd

activation = Activationsd(
    keys=["pred"],
    sigmoid=True,  # 二分类用 sigmoid
    softmax=False  # 多分类用 softmax
)

data_dict = activation(data_dict)
```

---

### 6.2 AsDiscreted - 转换为离散标签

```python
from monai.transforms import AsDiscreted

# 二分类：阈值化
discrete_binary = AsDiscreted(
    keys=["pred"],
    threshold=0.5  # > 0.5 -> 1, else -> 0
)

# 多分类：argmax
discrete_multi = AsDiscreted(
    keys=["pred"],
    argmax=True,  # 取最大值的索引
    to_onehot=None
)

# 多分类：one-hot 编码
discrete_onehot = AsDiscreted(
    keys=["pred", "label"],
    argmax=[True, False],  # pred 做 argmax
    to_onehot=[3, 3],  # 都转为 3-class one-hot
    num_classes=3
)

data_dict = discrete_onehot(data_dict)
# data_dict["pred"].shape = (3, H, W, D)
# data_dict["label"].shape = (3, H, W, D)
```

---

### 6.3 KeepLargestConnectedComponentd - 保留最大连通域

```python
from monai.transforms import KeepLargestConnectedComponentd

keep_largest = KeepLargestConnectedComponentd(
    keys=["pred"],
    applied_labels=[1, 2, 3],  # 对这些类别应用
    independent=True  # 每个类别独立处理
)

data_dict = keep_largest(data_dict)
```

---

### 6.4 FillHolesd - 填充空洞

```python
from monai.transforms import FillHolesd

fill = FillHolesd(
    keys=["pred"],
    applied_labels=[1, 2, 3]
)

data_dict = fill(data_dict)
```

---

## 七、组合变换（Compose）

### 7.1 完整的分割任务预处理流水线

```python
from monai.transforms import Compose, LoadImaged, AddChanneld, Spacingd, Orientationd
from monai.transforms import ScaleIntensityRanged, RandCropByPosNegLabeld, RandFlipd, RandRotated

# 训练数据预处理
train_transforms = Compose([
    # 1. 加载数据
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),  # 添加通道维度
    
    # 2. 空间预处理
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    
    # 3. 强度预处理
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=1000,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    
    # 4. 数据增强（基于标签采样）
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=1, neg=1,
        num_samples=4
    ),
    
    # 5. 随机增强
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
    RandRotated(
        keys=["image", "label"],
        range_x=[-0.3, 0.3],
        range_y=[-0.3, 0.3],
        range_z=[-0.3, 0.3],
        prob=0.5,
        mode=("bilinear", "nearest")
    ),
])

# 验证数据预处理
val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
])
```

---

### 7.2 推理时的后处理流水线

```python
from monai.transforms import Compose, Activationsd, AsDiscreted, KeepLargestConnectedComponentd

post_transforms = Compose([
    Activationsd(keys=["pred"], sigmoid=True),
    AsDiscreted(keys=["pred"], threshold=0.5),
    KeepLargestConnectedComponentd(keys=["pred"], applied_labels=[1]),
])
```

---

## 八、自定义 Transform

### 8.1 自定义 Array Transform

```python
from monai.transforms import Transform
import torch

class MyNormalization(Transform):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std

# 使用
normalize = MyNormalization(mean=0.5, std=0.5)
normalized_img = normalize(img)
```

---

### 8.2 自定义 Dictionary Transform

```python
from monai.transforms import MapTransform
from typing import Dict, Any, Hashable

class MyCustomd(MapTransform):
    def __init__(self, keys, multiplier: float = 1.0):
        super().__init__(keys)
        self.multiplier = multiplier
    
    def __call__(self, data: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = d[key] * self.multiplier
        return d

# 使用
custom_transform = MyCustomd(keys=["image"], multiplier=2.0)
data_dict = custom_transform(data_dict)
```

---

### 8.3 自定义随机 Transform

```python
from monai.transforms import RandomizableTransform

class MyRandomTransform(RandomizableTransform):
    def __init__(self, keys, prob: float = 0.5, intensity_range: tuple = (0.0, 1.0)):
        super().__init__(prob)
        self.keys = keys
        self.intensity_range = intensity_range
        self._intensity = None
    
    def randomize(self, data) -> None:
        super().randomize(None)
        if self._do_transform:
            self._intensity = self.R.uniform(*self.intensity_range)
    
    def __call__(self, data):
        self.randomize(data)
        if not self._do_transform:
            return data
        
        d = dict(data)
        for key in self.keys:
            d[key] = d[key] * self._intensity
        return d
```

---

## 九、Transform 最佳实践

### 9.1 性能优化

```python
# ✅ 推荐：使用 CacheDataset 缓存预处理结果
from monai.data import CacheDataset

train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,  # 缓存所有数据
    num_workers=4
)

# ✅ 推荐：对于大数据集，使用 PersistentDataset
from monai.data import PersistentDataset

train_ds = PersistentDataset(
    data=train_files,
    transform=train_transforms,
    cache_dir="./cache"  # 持久化到磁盘
)
```

---

### 9.2 常见错误避免

```python
# ❌ 错误：忘记添加通道维度
transforms = Compose([
    LoadImaged(keys=["image"]),  # shape: (H, W, D)
    NormalizeIntensityd(keys=["image"]),  # 错误！期望 (C, H, W, D)
])

# ✅ 正确：
transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),  # shape: (1, H, W, D)
    NormalizeIntensityd(keys=["image"]),
])

# ❌ 错误：标签使用了错误的插值方式
resample = Spacingd(
    keys=["image", "label"],
    pixdim=(1.0, 1.0, 1.0),
    mode="bilinear"  # 错误！标签会引入中间值
)

# ✅ 正确：
resample = Spacingd(
    keys=["image", "label"],
    pixdim=(1.0, 1.0, 1.0),
    mode=("bilinear", "nearest")  # 图像双线性，标签最近邻
)
```

---

## 十、所有 Transform 快速参考表

| 类别 | Transform 名称 | 功能 |
|------|---------------|------|
| **I/O** | `LoadImaged`, `SaveImaged` | 加载/保存图像 |
| **空间** | `Spacingd`, `Orientationd`, `Resized` | 重采样、方向、尺寸 |
| **空间** | `RandFlipd`, `RandRotated`, `RandZoomd` | 随机翻转、旋转、缩放 |
| **空间** | `RandAffined`, `RandGridDistortiond` | 仿射、网格畸变 |
| **裁剪** | `CenterSpatialCropd`, `RandSpatialCropd` | 中心裁剪、随机裁剪 |
| **裁剪** | `RandCropByPosNegLabeld` | 基于标签采样 |
| **强度** | `NormalizeIntensityd`, `ScaleIntensityRanged` | 归一化、范围映射 |
| **强度** | `RandGaussianNoised`, `RandGaussianSmoothd` | 噪声、平滑 |
| **强度** | `RandScaleIntensityd`, `RandShiftIntensityd` | 随机缩放、偏移 |
| **强度** | `RandBiasFieldd`, `RandCoarseDropoutd` | 偏置场、粗粒度 dropout |
| **后处理** | `Activationsd`, `AsDiscreted` | 激活、离散化 |
| **后处理** | `KeepLargestConnectedComponentd`, `FillHolesd` | 连通域、填洞 |

---

**文档生成完成！**
