<!-- PAGE_ID: monai-networks-complete -->
# MONAI 网络架构完整指南

> **目的**: 让 Claude Code 掌握 MONAI 所有网络架构的使用方法
> **生成时间**: 2026-03-03
> **源仓库**: https://github.com/Project-MONAI/MONAI

---

## 一、核心设计原则

### 1.1 工厂模式 (Factory Pattern)

MONAI 使用工厂模式动态创建层，而不是直接实例化 `nn.Conv2d` 或 `nn.Conv3d`：

```python
from monai.networks.layers import Conv, Norm, Act, Pool

# ❌ 错误：硬编码维度
conv = nn.Conv2d(in_channels, out_channels, kernel_size=3)

# ✅ 正确：使用工厂模式，自动适配 2D/3D
spatial_dims = 3  # 可以是 2 或 3
conv_type = Conv[Conv.CONV, spatial_dims]
conv = conv_type(in_channels, out_channels, kernel_size=3)

# ✅ 正确：常用层工厂
norm_type = Norm[Norm.BATCH, spatial_dims]  # BatchNorm
act_type = Act[Act.RELU]  # ReLU
pool_type = Pool[Pool.MAX, spatial_dims]  # MaxPool
```

**所有可用的工厂类型**：

| 工厂 | 类型 | 说明 |
|------|------|------|
| `Conv` | CONV, TRANSPOSECONV | 卷积、转置卷积 |
| `Norm` | BATCH, INSTANCE, GROUP, LAYER | 各种归一化 |
| `Act` | RELU, LEAKYRELU, PRELU, ELU, GELU, SELU | 激活函数 |
| `Pool` | MAX, ADAPTIVEMAX, AVG, ADAPTIVEAVG | 池化层 |
| `Dropout` | DROPOUT, DROPPATH | Dropout |

**源码引用**：
- [monai/networks/layers/factories.py:20-200](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/networks/layers/factories.py#L20-L200)

---

### 1.2 2D/3D 兼容性设计

所有网络都应该接受 `spatial_dims` 参数：

```python
from monai.networks.nets import UNet

# 2D UNet
model_2d = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2)
)

# 3D UNet - 只需要改 spatial_dims
model_3d = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2)
)
```

---

### 1.3 维度顺序约定

MONAI 使用 **Channel-First** 格式：

```python
# 2D: (Batch, Channel, Height, Width)
# 3D: (Batch, Channel, Depth, Height, Width)

# 示例
x_2d = torch.randn(2, 1, 256, 256)  # 2 张 256x256 的单通道图像
x_3d = torch.randn(2, 1, 64, 64, 64)  # 2 个 64x64x64 的单通道体积

output_2d = model_2d(x_2d)  # (2, 2, 256, 256)
output_3d = model_3d(x_3d)  # (2, 2, 64, 64, 64)
```

---

## 二、UNet 系列

### 2.1 UNet - 标准 UNet

**特点**：最通用的 UNet 实现，支持残差连接、Deep Supervision

```python
from monai.networks.nets import UNet

model = UNet(
    spatial_dims=3,              # 2 或 3
    in_channels=1,               # 输入通道数
    out_channels=2,              # 输出通道数（分割类别数）
    channels=(16, 32, 64, 128, 256),  # 每层的通道数
    strides=(2, 2, 2, 2),        # 下采样步长
    kernel_size=3,               # 卷积核大小
    up_kernel_size=3,            # 上采样卷积核大小
    num_res_units=2,             # 残差单元数量（0 表示无残差）
    norm="batch",                # 归一化方式："batch", "instance", "group"
    dropout=0.1,                 # Dropout 概率
    bias=True,                   # 是否使用 bias
)

# 前向传播
x = torch.randn(2, 1, 128, 128, 128)
output = model(x)
print(output.shape)  # (2, 2, 128, 128, 128)
```

**关键参数说明**：

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `channels` | 编码器每层的通道数 | `(16, 32, 64, 128, 256)` |
| `strides` | 下采样步长 | `(2, 2, 2, 2)` |
| `num_res_units` | 残差单元数量 | `2` (提高性能) |
| `dropout` | Dropout 概率 | `0.1-0.2` |
| `norm` | 归一化方式 | `"batch"` (小 batch) 或 `"instance"` (大 batch) |

**源码引用**：
- [monai/networks/nets/unet.py:50-300](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/networks/nets/unet.py#L50-L300)

---

### 2.2 BasicUNet - 轻量化 UNet

**特点**：参数更少，适合快速实验

```python
from monai.networks.nets import BasicUNet

model = BasicUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    features=(32, 32, 64, 128, 256, 32),  # 6 层特征
    dropout=0.1,
)

x = torch.randn(2, 1, 64, 64, 64)
output = model(x)
print(output.shape)  # (2, 2, 64, 64, 64)
```

**与 UNet 的区别**：
- 更简单的架构
- 固定的特征图数量
- 适合资源受限的环境

---

### 2.3 DynUNet - 动态 UNet

**特点**：模仿 nnU-Net 的启发式设计，自动配置网络结构

```python
from monai.networks.nets import DynUNet

# 定义每一层的配置
kernel_size = [
    (3, 3, 3),  # 第 0 层
    (3, 3, 3),  # 第 1 层
    (3, 3, 3),  # 第 2 层
    (3, 3, 3),  # 第 3 层
]
strides = [
    (1, 1, 1),  # 第 0 层（不下采样）
    (2, 2, 2),  # 第 1 层
    (2, 2, 2),  # 第 2 层
    (2, 2, 2),  # 第 3 层
]
upsample_kernel_size = [
    (2, 2, 2),  # 第 1 层上采样
    (2, 2, 2),  # 第 2 层上采样
    (2, 2, 2),  # 第 3 层上采样
]

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    kernel_size=kernel_size,
    strides=strides,
    upsample_kernel_size=upsample_kernel_size,
    norm_name="instance",
    deep_supervision=True,  # 启用 Deep Supervision
    deep_supr_num=2,  # Deep Supervision 的层数
)

# 前向传播（返回多层输出）
x = torch.randn(2, 1, 128, 128, 128)
outputs = model(x)
# outputs 是一个 list，包含多层输出
print(outputs[0].shape)  # (2, 2, 128, 128, 128) - 最终输出
```

**Deep Supervision**：
- 在多个层级输出预测
- 加速训练收敛
- 提高分割精度

**源码引用**：
- [monai/networks/nets/dynunet.py:50-400](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/networks/nets/dynunet.py#L50-L400)

---

### 2.4 UNETR - 基于 Transformer 的 UNet

**特点**：结合 ViT 和 UNet，适合大范围上下文建模

```python
from monai.networks.nets import UNETR

model = UNETR(
    in_channels=1,
    out_channels=2,
    img_size=(96, 96, 96),  # 必须指定输入尺寸
    feature_size=16,        # Transformer 特征维度
    hidden_size=768,        # Transformer 隐藏层大小
    mlp_dim=3072,           # MLP 维度
    num_heads=12,           # 注意力头数量
    pos_embed="conv",       # 位置编码方式："conv", "perceptron"
    norm_name="instance",
    conv_block=True,        # 使用 ConvBlock
    res_block=True,         # 使用 ResBlock
    dropout_rate=0.0,
)

x = torch.randn(2, 1, 96, 96, 96)
output = model(x)
print(output.shape)  # (2, 2, 96, 96, 96)
```

**关键参数**：
- `img_size`: 必须与实际输入尺寸一致
- `hidden_size`: 越大越好，但显存占用越高
- `num_heads`: 通常是 hidden_size / 64

**源码引用**：
- [monai/networks/nets/unetr.py:50-350](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/networks/nets/unetr.py#L50-L350)

---

### 2.5 SwinUNETR - Swin Transformer UNet

**特点**：基于 Swin Transformer，SOTA 性能

```python
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=2,
    feature_size=48,        # 特征维度
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,   # 节省显存的梯度检查点
    use_v2=True,            # 使用 Swin Transformer V2
)

x = torch.randn(2, 1, 96, 96, 96)
output = model(x)
print(output.shape)  # (2, 2, 96, 96, 96)

# 加载预训练权重
model.load_from_ckpt(
    "swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    map_location="cpu"
)
```

**优势**：
- 层次化特征表示
- 移动窗口注意力
- 线性时间复杂度

**源码引用**：
- [monai/networks/nets/swin_unetr.py:50-500](https://github.com/Project-MONAI/MONAI/blob/894068a91d5a5f4409897d9808eda765be07da99/monai/networks/nets/swin_unetr.py#L50-L500)

---

### 2.6 AttentionUnet - 注意力 UNet

**特点**：在 skip connection 中加入注意力机制

```python
from monai.networks.nets import AttentionUnet

model = AttentionUnet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    dropout=0.1,
)

x = torch.randn(2, 1, 64, 64, 64)
output = model(x)
print(output.shape)  # (2, 2, 64, 64, 64)
```

---

### 2.7 VNet - V-Net

**特点**：专为 3D 医学图像分割设计

```python
from monai.networks.nets import VNet

model = VNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    dropout_prob=0.5,
    dropout_dim=3,
)

x = torch.randn(2, 1, 128, 128, 128)
output = model(x)
print(output.shape)  # (2, 2, 128, 128, 128)
```

---

### 2.8 SegResNet - 分割 ResNet

**特点**：基于 ResNet 的分割网络，轻量高效

```python
from monai.networks.nets import SegResNet

model = SegResNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_filters=32,
    dropout_prob=0.2,
    norm="batch",
    use_conv_final=True,
)

x = torch.randn(2, 1, 96, 96, 96)
output = model(x)
print(output.shape)  # (2, 2, 96, 96, 96)
```

---

## 三、分类网络（Backbones）

### 3.1 ResNet 系列

```python
from monai.networks.nets import ResNet

# ResNet-18
model = ResNet(
    spatial_dims=3,
    block_type="basic",  # "basic" (ResNet-18/34) 或 "bottleneck" (ResNet-50/101/152)
    layers=[2, 2, 2, 2],  # ResNet-18
    block_inplanes=[64, 128, 256, 512],
    in_channels=1,
    num_classes=2,
    conv1_t_size=7,
    conv1_t_stride=(2, 2, 2),
    shortcut_type="B",
)

# ResNet-50
model_50 = ResNet(
    spatial_dims=3,
    block_type="bottleneck",
    layers=[3, 4, 6, 3],  # ResNet-50
    block_inplanes=[64, 128, 256, 512],
    in_channels=1,
    num_classes=2,
)

x = torch.randn(2, 1, 64, 64, 64)
output = model(x)
print(output.shape)  # (2, 2)
```

**预训练权重**：

```python
from monai.networks.nets import resnet18, resnet50

# MedicalNet 预训练权重
model = resnet18(spatial_dims=3, in_channels=1, num_classes=2)
model.load_state_dict(
    torch.load("resnet_18_23dataset.pth")
)
```

---

### 3.2 DenseNet 系列

```python
from monai.networks.nets import DenseNet

model = DenseNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    init_features=64,
    growth_rate=32,
    block_config=(6, 12, 24, 16),  # DenseNet-121
    bn_size=4,
    dropout_prob=0.0,
)

x = torch.randn(2, 1, 64, 64, 64)
output = model(x)
print(output.shape)  # (2, 2)
```

---

### 3.3 EfficientNet 系列

```python
from monai.networks.nets import EfficientNetBN

model = EfficientNetBN(
    model_name="efficientnet-b0",  # b0 到 b7
    spatial_dims=3,
    in_channels=1,
    num_classes=2,
    pretrained=True,  # 使用 ImageNet 预训练
    progress=True,
)

x = torch.randn(2, 1, 224, 224, 224)
output = model(x)
print(output.shape)  # (2, 2)
```

---

### 3.4 ViT - Vision Transformer

```python
from monai.networks.nets import ViT

model = ViT(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    hidden_size=768,
    mlp_dim=3072,
    num_layers=12,
    num_heads=12,
    pos_embed="conv",
    classification=True,
    num_classes=2,
    dropout_rate=0.0,
)

x = torch.randn(2, 1, 96, 96, 96)
output = model(x)
print(output.shape)  # (2, 2)
```

---

## 四、生成模型

### 4.1 AutoEncoder - 自编码器

```python
from monai.networks.nets import AutoEncoder

model = AutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    norm="batch",
    dropout=0.1,
)

x = torch.randn(2, 1, 64, 64, 64)
reconstruction = model(x)
print(reconstruction.shape)  # (2, 1, 64, 64, 64)
```

---

### 4.2 VariationalAutoEncoder - 变分自编码器

```python
from monai.networks.nets import VariationalAutoEncoder

model = VariationalAutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2),
    latent_size=128,  # 潜在空间维度
    norm="batch",
)

x = torch.randn(2, 1, 64, 64, 64)
reconstruction, z_mu, z_sigma = model(x)
print(reconstruction.shape)  # (2, 1, 64, 64, 64)
print(z_mu.shape)  # (2, 128)
print(z_sigma.shape)  # (2, 128)

# 重参数化技巧在 forward 中自动完成
```

---

### 4.3 DiffusionModelUNet - 扩散模型 UNet

```python
from monai.networks.nets import DiffusionModelUNet

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_res_blocks=2,
    channels=(16, 32, 64, 128, 256),
    attention_levels=(False, False, True, True, True),
    norm_num_groups=16,
    num_head_channels=8,
)

# 扩散模型需要时间步嵌入
x = torch.randn(2, 1, 64, 64, 64)
timesteps = torch.randint(0, 1000, (2,))
noise_pred = model(x, timesteps)
print(noise_pred.shape)  # (2, 1, 64, 64, 64)
```

**完整训练流程**：

```python
from monai.networks.schedulers import DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
)

# 训练
for epoch in range(100):
    for batch in dataloader:
        images = batch["image"]
        
        # 1. 采样时间步
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (images.shape[0],)
        )
        
        # 2. 采样噪声
        noise = torch.randn_like(images)
        
        # 3. 加噪
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # 4. 预测噪声
        noise_pred = model(noisy_images, timesteps)
        
        # 5. 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        # 6. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**推理（去噪）**：

```python
# 从纯噪声开始
noise = torch.randn(1, 1, 64, 64, 64)

# 逐步去噪
for t in reversed(range(scheduler.num_train_timesteps)):
    timesteps = torch.tensor([t])
    noise_pred = model(noise, timesteps)
    noise = scheduler.step(noise_pred, t, noise).prev_sample

# noise 现在是生成的图像
generated_image = noise
```

---

### 4.4 扩散模型调度器（Schedulers）

```python
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

# DDPM - 标准扩散
ddpm_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    beta_start=0.0001,
    beta_end=0.02,
)

# DDIM - 快速采样
ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    clip_sample=True,
)

# DDIM 只需要 50 步就能生成高质量图像
ddim_scheduler.set_timesteps(50)  # 从 1000 步压缩到 50 步
```

---

## 五、损失函数

### 5.1 DiceLoss - Dice 损失

```python
from monai.losses import DiceLoss

loss_func = DiceLoss(
    to_onehot_y=True,  # 将标签转为 one-hot
    softmax=True,      # 对预测应用 softmax
    squared_pred=True, # 使用平方形式的 Dice
    include_background=True,  # 是否包含背景
    reduction="mean",
)

# 使用
pred = torch.randn(2, 3, 64, 64, 64)  # 3-class 分割
label = torch.randint(0, 3, (2, 1, 64, 64, 64))  # 标签

loss = loss_func(pred, label)
print(loss.item())
```

---

### 5.2 DiceCELoss - Dice + CrossEntropy

```python
from monai.losses import DiceCELoss

loss_func = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    lambda_dice=1.0,  # Dice 权重
    lambda_ce=1.0,    # CE 权重
)

# 推荐用于分割任务
loss = loss_func(pred, label)
```

---

### 5.3 FocalLoss - Focal 损失

```python
from monai.losses import FocalLoss

loss_func = FocalLoss(
    gamma=2.0,  # 关注难分类样本
    to_onehot_y=True,
    include_background=True,
)

loss = loss_func(pred, label)
```

---

### 5.4 TverskyLoss - Tversky 损失

```python
from monai.losses import TverskyLoss

loss_func = TverskyLoss(
    alpha=0.3,  # 假阳性权重
    beta=0.7,   # 假阴性权重
    to_onehot_y=True,
    softmax=True,
)

# alpha=0.3, beta=0.7 更关注假阴性（提高召回率）
# alpha=0.7, beta=0.3 更关注假阳性（提高精确率）
```

---

### 5.5 DiffusionLoss - 扩散损失

```python
from monai.losses import DiffusionLoss

loss_func = DiffusionLoss()

# 用于扩散模型训练
noise_pred = model(noisy_images, timesteps)
noise_target = noise
loss = loss_func(noise_pred, noise_target)
```

---

## 六、自定义网络

### 6.1 使用 Convolution 块

```python
from monai.networks.blocks import Convolution

# 包含 Conv + Norm + Act 的复合块
conv_block = Convolution(
    spatial_dims=3,
    in_channels=64,
    out_channels=128,
    strides=2,  # 下采样
    kernel_size=3,
    norm="batch",
    act="relu",
    dropout=0.1,
)

x = torch.randn(2, 64, 64, 64, 64)
output = conv_block(x)
print(output.shape)  # (2, 128, 32, 32, 32)
```

---

### 6.2 使用 ResidualUnit 块

```python
from monai.networks.blocks import ResidualUnit

res_block = ResidualUnit(
    spatial_dims=3,
    in_channels=64,
    out_channels=64,
    strides=1,
    kernel_size=3,
    norm="batch",
    act="relu",
    dropout=0.1,
)

x = torch.randn(2, 64, 64, 64, 64)
output = res_block(x)
print(output.shape)  # (2, 64, 64, 64, 64)
```

---

### 6.3 完整的自定义网络示例

```python
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers import Conv, Norm, Act

class MyCustomSegNet(nn.Module):
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=2):
        super().__init__()
        
        # 编码器
        self.enc1 = Convolution(spatial_dims, in_channels, 32, strides=2, norm="batch", act="relu")
        self.enc2 = Convolution(spatial_dims, 32, 64, strides=2, norm="batch", act="relu")
        self.enc3 = Convolution(spatial_dims, 64, 128, strides=2, norm="batch", act="relu")
        
        # 中间层
        self.middle = ResidualUnit(spatial_dims, 128, 128, strides=1, norm="batch", act="relu")
        
        # 解码器
        self.dec3 = Convolution(spatial_dims, 128, 64, strides=2, is_transposed=True, norm="batch", act="relu")
        self.dec2 = Convolution(spatial_dims, 64, 32, strides=2, is_transposed=True, norm="batch", act="relu")
        self.dec1 = Convolution(spatial_dims, 32, out_channels, strides=2, is_transposed=True, norm=None, act=None)
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # 中间
        middle = self.middle(e3)
        
        # 解码
        d3 = self.dec3(middle)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        return d1

# 使用
model = MyCustomSegNet(spatial_dims=3, in_channels=1, out_channels=2)
x = torch.randn(2, 1, 64, 64, 64)
output = model(x)
print(output.shape)  # (2, 2, 64, 64, 64)
```

---

## 七、网络选择指南

| 任务 | 推荐网络 | 原因 |
|------|---------|------|
| **2D 分割** | UNet, AttentionUnet | 轻量高效 |
| **3D 分割** | UNet, DynUNet, SwinUNETR | SOTA 性能 |
| **小数据集** | BasicUNet, SegResNet | 参数少，不易过拟合 |
| **大数据集** | SwinUNETR, UNETR | 需要大范围上下文 |
| **分类** | ResNet, DenseNet, EfficientNet | 经典 backbone |
| **生成** | DiffusionModelUNet, VAE | 生成任务专用 |
| **资源受限** | BasicUNet, SegResNet | 内存友好 |

---

## 八、常见问题

### 8.1 如何选择通道数？

```python
# 小数据集（<100 cases）
channels = (16, 32, 64, 128, 256)

# 中等数据集（100-1000 cases）
channels = (32, 64, 128, 256, 512)

# 大数据集（>1000 cases）
channels = (64, 128, 256, 512, 1024)
```

---

### 8.2 如何处理显存不足？

```python
# 方法 1: 减小 batch size
batch_size = 1

# 方法 2: 减小输入尺寸
spatial_size = (64, 64, 64)  # 而不是 (128, 128, 128)

# 方法 3: 使用梯度检查点（SwinUNETR）
model = SwinUNETR(..., use_checkpoint=True)

# 方法 4: 减小通道数
channels = (16, 32, 64, 128, 256)  # 而不是 (32, 64, 128, 256, 512)
```

---

### 8.3 如何加载预训练权重？

```python
# MedicalNet 预训练 ResNet
from monai.networks.nets import resnet18

model = resnet18(spatial_dims=3, in_channels=1, num_classes=2)
checkpoint = torch.load("resnet_18_23dataset.pth")
model.load_state_dict(checkpoint["state_dict"])

# SwinUNETR 预训练
model = SwinUNETR(...)
model.load_from_ckpt("swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt")
```

---

**文档生成完成！**
