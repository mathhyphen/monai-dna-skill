<!-- PAGE_ID: monai-generation-complete -->
# MONAI 生成模型完整指南

> **目的**: 让 Claude Code 掌握 MONAI 所有生成模型的使用方法
> **生成时间**: 2026-03-03
> **源仓库**: https://github.com/Project-MONAI/MONAI

---

## 一、生成模型概述

MONAI 提供了完整的生成模型工具链，包括：

| 模型类型 | 应用场景 | 核心类 |
|---------|---------|--------|
| **AutoEncoder** | 降维、特征学习 | `AutoEncoder` |
| **VAE** | 数据生成、潜在空间插值 | `VariationalAutoEncoder` |
| **Diffusion** | 高质量图像生成 | `DiffusionModelUNet` |
| **GAN** | 图像翻译、超分辨率 | 自定义（MONAI 提供基础组件） |

---

## 二、AutoEncoder - 自编码器

### 2.1 基础用法

```python
from monai.networks.nets import AutoEncoder
import torch

# 定义模型
model = AutoEncoder(
    spatial_dims=3,              # 3D 图像
    in_channels=1,               # 单通道输入
    out_channels=1,              # 单通道输出
    channels=(16, 32, 64, 128, 256),  # 编码器通道数
    strides=(2, 2, 2, 2, 2),     # 每层下采样 2 倍
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,             # 残差单元数量
    norm="batch",
    dropout=0.1,
)

# 前向传播
x = torch.randn(2, 1, 64, 64, 64)
reconstruction = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {reconstruction.shape}")
# Input shape: torch.Size([2, 1, 64, 64, 64])
# Output shape: torch.Size([2, 1, 64, 64, 64])
```

---

### 2.2 完整训练示例

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import AutoEncoder
from monai.data import CacheDataset
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd

# 1. 准备数据
transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
])

train_ds = CacheDataset(
    data=train_files,
    transform=transforms,
    cache_rate=1.0
)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

# 2. 定义模型
model = AutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).cuda()

# 3. 定义损失和优化器
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        images = batch["image"].cuda()
        
        # 前向传播
        reconstructions = model(images)
        
        # 计算损失
        loss = loss_func(reconstructions, images)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存检查点
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"autoencoder_epoch_{epoch+1}.pth")
```

---

### 2.3 提取编码器用于下游任务

```python
from monai.networks.nets import AutoEncoder

# 加载预训练的 AutoEncoder
model = AutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2, 2),
)
model.load_state_dict(torch.load("autoencoder_epoch_100.pth"))

# 提取编码器
encoder = model.encoder

# 用于特征提取
x = torch.randn(2, 1, 64, 64, 64)
features = encoder(x)
print(f"Feature shape: {features.shape}")  # (2, 128, 4, 4, 4)

# 展平用于分类
features_flat = features.view(features.size(0), -1)
print(f"Flattened shape: {features_flat.shape}")  # (2, 8192)
```

---

## 三、VariationalAutoEncoder - 变分自编码器

### 3.1 基础用法

```python
from monai.networks.nets import VariationalAutoEncoder
import torch

model = VariationalAutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2, 2),
    latent_size=128,  # 潜在空间维度
    num_res_units=2,
    norm="batch",
)

# 前向传播
x = torch.randn(2, 1, 64, 64, 64)
reconstruction, z_mu, z_sigma = model(x)

print(f"Input shape: {x.shape}")
print(f"Reconstruction shape: {reconstruction.shape}")
print(f"Latent mean shape: {z_mu.shape}")  # (2, 128)
print(f"Latent sigma shape: {z_sigma.shape}")  # (2, 128)
```

---

### 3.2 VAE 损失函数

```python
import torch
import torch.nn.functional as F

def vae_loss(reconstruction, x, z_mu, z_sigma, beta=1.0):
    """
    VAE 损失 = 重建损失 + beta * KL 散度
    
    Args:
        reconstruction: 重建的图像
        x: 原始图像
        z_mu: 潜在空间均值
        z_sigma: 潜在空间标准差
        beta: KL 散度权重（beta-VAE）
    """
    # 1. 重建损失（MSE 或 BCE）
    recon_loss = F.mse_loss(reconstruction, x, reduction="sum")
    
    # 2. KL 散度
    # KL(N(mu, sigma) || N(0, 1))
    kl_loss = -0.5 * torch.sum(1 + torch.log(z_sigma ** 2) - z_mu ** 2 - z_sigma ** 2)
    
    # 3. 总损失
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

# 使用
reconstruction, z_mu, z_sigma = model(x)
total_loss, recon_loss, kl_loss = vae_loss(reconstruction, x, z_mu, z_sigma, beta=0.01)

print(f"Total loss: {total_loss.item():.2f}")
print(f"Reconstruction loss: {recon_loss.item():.2f}")
print(f"KL loss: {kl_loss.item():.2f}")
```

---

### 3.3 完整 VAE 训练示例

```python
import torch
import torch.nn.functional as F
from monai.networks.nets import VariationalAutoEncoder
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd

# 1. 数据准备
transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
])

train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# 2. 模型定义
model = VariationalAutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    latent_size=64,
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 3. 训练循环
def vae_loss(reconstruction, x, z_mu, z_sigma, beta=0.01):
    recon_loss = F.mse_loss(reconstruction, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + torch.log(z_sigma ** 2) - z_mu ** 2 - z_sigma ** 2)
    return recon_loss + beta * kl_loss

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        images = batch["image"].cuda()
        
        # 前向传播
        reconstruction, z_mu, z_sigma = model(images)
        
        # 计算损失
        loss = vae_loss(reconstruction, images, z_mu, z_sigma)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

---

### 3.4 潜在空间插值

```python
# 从两个图像编码到潜在空间
image1 = train_ds[0]["image"].unsqueeze(0).cuda()
image2 = train_ds[1]["image"].unsqueeze(0).cuda()

_, z_mu1, _ = model(image1)
_, z_mu2, _ = model(image2)

# 线性插值
num_steps = 10
interpolations = []
for i in range(num_steps):
    alpha = i / (num_steps - 1)
    z_interp = (1 - alpha) * z_mu1 + alpha * z_mu2
    
    # 解码
    with torch.no_grad():
        generated = model.decode(z_interp)
    interpolations.append(generated.cpu())

# 保存插值结果
import nibabel as nib
for i, img in enumerate(interpolations):
    nib.save(nib.Nifti1Image(img[0, 0].numpy(), np.eye(4)), f"interpolation_{i}.nii.gz")
```

---

## 四、Diffusion Model - 扩散模型

### 4.1 核心概念

扩散模型分为两个过程：

1. **前向过程（加噪）**：逐步向图像添加噪声
2. **反向过程（去噪）**：从噪声逐步恢复图像

```python
# 前向过程示意
image → add_noise(t=0) → noisy_image_1 → add_noise(t=1) → ... → pure_noise(t=T)

# 反向过程示意
pure_noise → denoise(t=T) → less_noisy → ... → denoise(t=0) → generated_image
```

---

### 4.2 基础组件

#### 4.2.1 DiffusionModelUNet

```python
from monai.networks.nets import DiffusionModelUNet
import torch

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_res_blocks=2,
    channels=(32, 64, 128, 256),
    attention_levels=(False, False, True, True),  # 在深层使用注意力
    norm_num_groups=16,
    num_head_channels=8,
)

# 前向传播（需要时间步）
x = torch.randn(2, 1, 64, 64, 64)
timesteps = torch.randint(0, 1000, (2,))
noise_pred = model(x, timesteps)

print(f"Input shape: {x.shape}")
print(f"Noise prediction shape: {noise_pred.shape}")
```

---

#### 4.2.2 DDPMScheduler - DDPM 调度器

```python
from monai.networks.schedulers import DDPMScheduler

scheduler = DDPMScheduler(
    num_train_timesteps=1000,  # 训练步数
    schedule="linear_beta",    # 噪声调度方式
    beta_start=0.0001,         # 起始 beta
    beta_end=0.02,             # 结束 beta
    clip_sample=True,          # 截断样本
)

# 关键方法
print(f"Num timesteps: {scheduler.num_train_timesteps}")

# 1. 添加噪声
image = torch.randn(2, 1, 64, 64, 64)
noise = torch.randn_like(image)
timesteps = torch.tensor([100, 500])
noisy_image = scheduler.add_noise(image, noise, timesteps)

# 2. 单步去噪
noise_pred = model(noisy_image, timesteps)
prev_sample = scheduler.step(noise_pred, timesteps[0], noisy_image[0]).prev_sample
```

---

#### 4.2.3 DDIMScheduler - DDIM 调度器（快速采样）

```python
from monai.networks.schedulers import DDIMScheduler

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    clip_sample=True,
)

# 设置推理步数（从 1000 压缩到 50）
scheduler.set_timesteps(50)
print(f"Inference timesteps: {scheduler.timesteps}")  # [980, 960, 940, ..., 20, 0]

# 推理时只需要 50 步，而不是 1000 步
```

---

### 4.3 完整训练流程

```python
import torch
import torch.nn.functional as F
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd

# 1. 数据准备
transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),  # 归一化到 [-1, 1]
])

train_ds = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)

# 2. 模型和调度器
model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    num_res_blocks=2,
    channels=(32, 64, 128, 256),
    attention_levels=(False, False, True, True),
).cuda()

scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 3. 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        images = batch["image"].cuda()
        
        # 1. 采样时间步
        timesteps = torch.randint(
            0, scheduler.num_train_timesteps, (images.shape[0],)
        ).cuda()
        
        # 2. 采样噪声
        noise = torch.randn_like(images)
        
        # 3. 添加噪声
        noisy_images = scheduler.add_noise(images, noise, timesteps)
        
        # 4. 预测噪声
        noise_pred = model(noisy_images, timesteps)
        
        # 5. 计算损失（MSE between predicted and real noise）
        loss = F.mse_loss(noise_pred, noise)
        
        # 6. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 保存检查点
    if (epoch + 1) % 10 == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }, f"diffusion_epoch_{epoch+1}.pth")
```

---

### 4.4 推理（生成图像）

#### 4.4.1 DDPM 采样（慢，高质量）

```python
from monai.networks.schedulers import DDPMScheduler

# 加载模型
model.load_state_dict(torch.load("diffusion_epoch_100.pth")["model"])
model.eval()

scheduler = DDPMScheduler(num_train_timesteps=1000)

# 从纯噪声开始
noise = torch.randn(1, 1, 64, 64, 64).cuda()

# 逐步去噪（1000 步）
with torch.no_grad():
    for t in reversed(range(scheduler.num_train_timesteps)):
        timesteps = torch.tensor([t]).cuda()
        
        # 预测噪声
        noise_pred = model(noise, timesteps)
        
        # 去噪一步
        noise = scheduler.step(noise_pred, t, noise).prev_sample

# 保存生成的图像
generated_image = noise
nib.save(nib.Nifti1Image(generated_image[0, 0].cpu().numpy(), np.eye(4)), "generated_ddpm.nii.gz")
```

---

#### 4.4.2 DDIM 采样（快，质量略降）

```python
from monai.networks.schedulers import DDIMScheduler

# 加载模型
model.load_state_dict(torch.load("diffusion_epoch_100.pth")["model"])
model.eval()

scheduler = DDIMScheduler(num_train_timesteps=1000)
scheduler.set_timesteps(50)  # 只用 50 步！

# 从纯噪声开始
noise = torch.randn(1, 1, 64, 64, 64).cuda()

# 快速去噪（50 步）
with torch.no_grad():
    for t in scheduler.timesteps:
        timesteps = torch.tensor([t]).cuda()
        
        # 预测噪声
        noise_pred = model(noise, timesteps)
        
        # 去噪一步
        noise = scheduler.step(noise_pred, t, noise).prev_sample

# 保存生成的图像
generated_image = noise
nib.save(nib.Nifti1Image(generated_image[0, 0].cpu().numpy(), np.eye(4)), "generated_ddim.nii.gz")
```

---

### 4.5 条件生成

```python
# 条件扩散模型：基于标签生成
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.unet = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=2,
            channels=(32, 64, 128, 256),
            attention_levels=(False, False, True, True),
        )
        
        # 标签嵌入
        self.label_embedding = nn.Embedding(num_classes, 256)
    
    def forward(self, x, timesteps, labels):
        # 嵌入标签
        label_emb = self.label_embedding(labels)  # (B, 256)
        
        # 传递给 UNet（需要修改 UNet 以接受条件）
        # 这里简化示例
        return self.unet(x, timesteps)

# 训练
labels = torch.randint(0, 10, (2,))
noise_pred = model(noisy_images, timesteps, labels)
```

---

## 五、Latent Diffusion（潜在扩散）

### 5.1 为什么用 Latent Diffusion？

3D 医学图像太大（如 256×256×256），直接做扩散太慢。

**解决方案**：
1. 先用 VAE 压缩到低维潜在空间（如 32×32×32）
2. 在潜在空间做扩散
3. 生成后再用 VAE 解码回原始空间

---

### 5.2 完整流程

```python
# 1. 训练 VAE
vae = VariationalAutoEncoder(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2, 2),
    latent_size=64,
)

# 2. 用 VAE 编码所有图像到潜在空间
latents = []
for batch in train_loader:
    images = batch["image"].cuda()
    _, z_mu, _ = vae(images)
    latents.append(z_mu.detach())

# 3. 在潜在空间训练扩散模型
latent_diffusion = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=64,  # latent_size
    out_channels=64,
    num_res_blocks=2,
    channels=(64, 128, 256),
    attention_levels=(False, True, True),
)

# 训练潜在扩散
for epoch in range(num_epochs):
    for latent_batch in latent_loader:
        # 在潜在空间做扩散
        timesteps = torch.randint(0, 1000, (latent_batch.shape[0],))
        noise = torch.randn_like(latent_batch)
        noisy_latents = scheduler.add_noise(latent_batch, noise, timesteps)
        noise_pred = latent_diffusion(noisy_latents, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. 生成流程
# (1) 在潜在空间生成
latent_noise = torch.randn(1, 64, 8, 8, 8)
for t in reversed(range(1000)):
    noise_pred = latent_diffusion(latent_noise, torch.tensor([t]))
    latent_noise = scheduler.step(noise_pred, t, latent_noise).prev_sample

# (2) 解码回原始空间
generated_image = vae.decode(latent_noise)
```

---

## 六、生成模型最佳实践

### 6.1 数据预处理

```python
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd

# ✅ 推荐：归一化到 [-1, 1]
transforms = Compose([
    LoadImaged(keys=["image"]),
    AddChanneld(keys=["image"]),
    ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
])

# ❌ 避免：归一化到 [0, 1]（扩散模型需要对称范围）
# ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0)
```

---

### 6.2 训练技巧

```python
# 1. 使用 AdamW 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 2. 使用学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

# 3. 使用 EMA（指数移动平均）
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

# 在训练循环中
for batch in train_loader:
    loss = ...
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ema.update()  # 更新 EMA

# 推理时使用 EMA 参数
with ema.average_parameters():
    generated = model(noise, timesteps)
```

---

### 6.3 评估指标

```python
from monai.metrics import FIDMetric

# Fréchet Inception Distance（FID）
fid_metric = FIDMetric()

# 计算真实图像和生成图像的 FID
real_features = extract_features(real_images)
fake_features = extract_features(generated_images)

fid_score = fid_metric(real_features, fake_features)
print(f"FID Score: {fid_score:.2f}")  # 越低越好
```

---

## 七、常见问题

### 7.1 如何选择扩散步数？

```python
# 训练：1000 步（标准）
num_train_timesteps = 1000

# 推理：
# - DDPM: 1000 步（慢，高质量）
# - DDIM: 50-100 步（快，质量略降）
# - DPM-Solver: 10-20 步（最快，质量好）
```

---

### 7.2 如何处理显存不足？

```python
# 方法 1: 减小 batch size
batch_size = 1

# 方法 2: 减小图像尺寸
spatial_size = (32, 32, 32)  # 而不是 (64, 64, 64)

# 方法 3: 使用 Latent Diffusion
# 先用 VAE 压缩，再在低维空间扩散

# 方法 4: 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    noise_pred = model(noisy_images, timesteps)
    loss = F.mse_loss(noise_pred, noise)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### 7.3 如何提高生成质量？

```python
# 1. 增加训练时间
num_epochs = 500  # 而不是 100

# 2. 增加模型容量
channels = (64, 128, 256, 512)  # 而不是 (32, 64, 128, 256)

# 3. 使用注意力机制
attention_levels = (False, True, True, True)  # 在更多层使用注意力

# 4. 使用预训练 VAE（Latent Diffusion）

# 5. 使用 Classifier-Free Guidance
guidance_scale = 7.5  # 增强条件控制
```

---

## 八、生成模型对比

| 模型 | 生成质量 | 训练速度 | 推理速度 | 显存占用 |
|------|---------|---------|---------|---------|
| **AutoEncoder** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **VAE** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Diffusion** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
| **Latent Diffusion** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

**文档生成完成！**
