<!-- PAGE_ID: monai-troubleshooting-gotchas -->
# MONAI DNA: Claude Gotchas

This document captures the failure points Claude Code most commonly encounters when implementing MONAI medical imaging projects. Unlike a FAQ, each entry starts from Claude's mistake, not from the correct pattern.

---

## Transform & Data Pipeline Gotchas

### Gotcha 1: Channel-First Dimension Ordering Confusion

- **What Claude does**: Writes transforms or model code assuming `(H, W, D)` or `(H, W, D, C)` shape order, or mixes channel-first and channel-last conventions mid-pipeline. For example, using numpy-style indexing `[:, :, 0]` on a MONAI tensor that is actually `(C, H, W, D)`.
- **Why it happens**: Medical imaging files (NIfTI, DICOM) are sometimes stored in `(H, W, D, C)` order on disk. MONAI's `LoadImage` with `ensure_channel_first=True` converts to `(C, H, W, D)`, but Claude often reaches for standard Python/numpy indexing habits that assume channels-last.
- **How to recognize**: Shape mismatch errors at the first `model(x)` call. Error messages like `RuntimeError: Expected 5D input tensor with 5D (N, C, D, H, W) or 4D (N, C, H, W)`. The error only appears at the model, not at the transform.
- **How to fix**: Always use `AddChanneld` or `EnsureChannelFirstd` immediately after `LoadImaged` to make channel-first explicit. Verify shapes with `print(data_dict["image"].shape)` immediately after the load transform. For 3D: MONAI expects `(B, C, D, H, W)`, never `(B, D, H, W, C)`.

---

### Gotcha 2: Label Interpolation Mode (Bilinear for Segmentation Labels)

- **What Claude does**: When resampling or rotating images and labels together, Claude passes a single interpolation mode (usually `"bilinear"` or `"area"`) to `mode=` and applies it to both image and label keys.
- **Why it happens**: Claude applies its natural bias toward smooth interpolation everywhere — it sees "image" and thinks "smooth resampling is better." But segmentation labels are categorical integers; interpolation introduces non-integer values that corrupt the label.
- **How to recognize**: After `Spacingd` or `Resized`, the label tensor contains values like `0.0`, `0.47`, `1.47`, `2.47` instead of `0, 1, 2`. Loss functions complain about float labels. Training loss becomes `NaN` after a few epochs.
- **How to fix**: Always use a tuple for `mode` in spatial transforms when keys contain both image and label: `mode=("bilinear", "nearest")` or `mode=("area", "nearest")`. The first element is for the image key, the second for the label key. If you used a single string, replace it: `mode="nearest"` for label-only transforms.

---

### Gotcha 3: Mixing Array-Mode and Dictionary-Mode Transforms

- **What Claude does**: Chains `LoadImage` (array-mode, returns a tensor) directly into a dictionary-mode transform like `Spacingd`, or mixes `NormalizeIntensity` with `NormalizeIntensityd` in the same compose pipeline.
- **Why it happens**: Claude reads MONAI docs and sees both forms exist but doesn't track which mode each transform operates in. It assumes both forms are interchangeable and just "do the same thing."
- **How to recognize**: `TypeError: 'Tensor' object has no attribute 'keys'` or `'numpy.ndarray' object has no attribute 'keys'` at the dictionary transform. Or the reverse: `AttributeError: 'dict' object has no attribute 'shape'` at an array transform.
- **How to fix**: Keep the pipeline consistent. For single-image pipelines, use array-mode transforms (`LoadImage`, `NormalizeIntensity`, `Spacing`). For multimodal or multi-key pipelines, use dictionary-mode transforms (`LoadImaged`, `NormalizeIntensityd`, `Spacingd`). Never mix them. The tell: array transforms accept/return raw tensors; dictionary transforms accept/return `Dict[str, Any]`.

---

### Gotcha 4: `spatial_dims` Inconsistency Across Transforms, Models, and Inferers

- **What Claude does**: Sets `spatial_dims=2` on the model but `spatial_dims=3` on the inferer, or uses `Spacingd` without specifying `spatial_dims` (defaulting to 2 or 3 inconsistently across transforms). Or defines a 3D model but passes 2D-shaped tensors.
- **Why it happens**: Claude plans the data pipeline and model separately and forgets to audit that all components agree on dimensionality. Each component defaults differently or has different spatial dim interpretations.
- **How to recognize**: `RuntimeError: size mismatch for encoder` or `ValueError: unexpecting 2D signal` inside a Conv3d. The model might be `SwinUNETR` which is 3D-only but the inferer or transforms were configured for 2D.
- **How to fix**: Pick `spatial_dims=2` or `spatial_dims=3` once at the start and propagate it to every component: transforms (`Spacingd`, `Orientationd`, `RandFlipd`), model (`UNet(spatial_dims=3, ...)`), inferer (`SlidingWindowInferer(roi_size=...)` — no explicit spatial_dims but must match), and postprocessing. Audit all three places before running.

---

## Model & Network Gotchas

### Gotcha 5: SwinUNETR `img_size` Mismatch With Actual Input

- **What Claude does**: Creates `SwinUNETR(img_size=(96, 96, 96), ...)` but the actual patch size after cropping or the input volume is a different shape, e.g., `(128, 128, 128)` or `(96, 96, 80)`.
- **Why it happens**: SwinUNETR requires `img_size` to exactly match the input tensor spatial dimensions because the patch embedding layer is hardcoded to that size. Claude sets `img_size` to a "reasonable" default without auditing it against the actual data pipeline output.
- **How to recognize**: `RuntimeError: The shape of the input 0 does not match the expected shape for SwinUNETR` or `ValueError: window_size must be less than or equal to the input size` during the first forward pass. The error mentions window size or patch embedding.
- **How to fix**: Set `img_size` in `SwinUNETR` to exactly match the spatial output size of your preprocessing pipeline. If using `RandCropByPosNegLabeld` with `spatial_size=(96, 96, 96)`, then `SwinUNETR(img_size=(96, 96, 96), ...)` is correct. If you change the crop size, you must change `img_size` too.

---

### Gotcha 6: Missing AMP (Automatic Mixed Precision) for 3D Training

- **What Claude does**: Writes a standard full-precision (`torch.float32`) training loop for 3D segmentation or diffusion models, or enables AMP only on the forward pass but not the backward pass.
- **Why it happens**: Claude sees 3D convolutions are memory-heavy and tries to fix it by reducing batch size or image size, not realizing that AMP typically gives 2-3x memory reduction with zero code complexity increase and negligible quality loss.
- **How to recognize**: `CUDA out of memory` errors on a single 3D volume with batch_size=1. Training crashes immediately at the first backward pass, not at forward. The model is a 3D UNet or SwinUNETR with channels like `(32, 64, 128, 256)`.
- **How to fix**: Always use AMP for 3D training:

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for batch in dataloader:
    with autocast():
        preds = model(images)
        loss = loss_func(preds, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

Note: `autocast` must wrap both forward and loss computation, and `scaler.scale(loss).backward()` must be used instead of `loss.backward()`.

---

### Gotcha 7: Metric Shape Mismatch (One-Hot, Sigmoid, Softmax Confusion)

- **What Claude does**: Computes a metric (Dice, IoU) between a model's raw logits output and a label tensor, without applying the same `softmax`/`sigmoid` discretization to both prediction and label. Or applies `softmax` to the label by mistake.
- **Why it happens**: MONAI losses (DiceLoss, DiceCELoss) internally call `softmax` or `sigmoid` depending on settings, but metrics do not. Claude assumes "if the loss works, the metric should work too" and skips the postprocessing step for metrics.
- **How to recognize**: Metric reports `0.0` or `1.0` for every sample. The predicted tensor has values like `[[0.2, 0.7], [0.1, 0.9]]` while the label is one-hot `[[0,1], [0,1]]`. Shape mismatch or type mismatch errors in the metric computation.
- **How to fix**: Apply the same discretization to both before the metric:

```python
# For softmax multi-class
pred = torch.argmax(output, dim=1)  # (B, H, W, D)
label = label.squeeze(1)           # (B, H, W, D)
# For sigmoid binary
pred = (torch.sigmoid(output) > 0.5).float()
label = label.float()
metric(pred, label)
```

Or use MONAI's postprocessing transforms before the metric: `AsDiscreted(keys=["pred"], argmax=True)` for multi-class, `AsDiscreted(keys=["pred"], threshold=0.5)` for binary.

---

## Generative Model Gotchas (Diffusion/VAE/Rectified Flow)

### Gotcha 8: Diffusion Objective Mismatch (Epsilon vs Velocity vs Flow)

- **What Claude does**: Implements a diffusion training loop that predicts epsilon (noise) but uses a loss function that assumes velocity prediction, or vice versa. Or copies code from a DDPM tutorial and applies it to a model that was trained to predict flow.
- **Why it happens**: The three main diffusion/flow objectives look similar mathematically but produce different gradients. `epsilon` prediction: `loss = ||epsilon_pred - epsilon||`. `velocity` prediction: `loss = ||v_pred - (x_0 - x_t)||`. `flow` prediction: `loss = ||v_pred - (x_1 - x_0)||`. Claude pastes loss code from different tutorials without auditing which objective the model actually predicts.
- **How to recognize**: The loss starts at a reasonable value (e.g., 0.5) but never decreases below 0.3 across 100 epochs. Generated samples look like pure noise or a blurred uniform blob. The model predicts a quantity that doesn't match the scheduler's `add_noise` or `step` method.
- **How to fix**: Check what the model outputs. `DiffusionModelUNet` in MONAI predicts noise (epsilon). If using a rectified flow model from lucidrains, it predicts flow `x_1 - x_0`. If using a velocity-prediction model, it predicts `v = x_0 - x_t`. Match your loss exactly:

```python
# For epsilon prediction (MONAI DDPM)
noise_pred = model(noisy_images, timesteps)
loss = F.mse_loss(noise_pred, noise)  # noise is the epsilon target

# For flow prediction (lucidrains rectified flow)
flow_pred = model(noised, t)
loss = F.mse_loss(flow_pred, data - noise)  # data is x_0, noise is x_1
```

---

### Gotcha 9: VAE Latent Space Dimension Confusion

- **What Claude does**: Instantiates a VAE with `latent_size=128` and then tries to pass the model's `z_mu` output (which has shape `(B, 128)`) into a decoder that expects a different spatial shape, or encodes to a latent vector and attempts to decode it without reordering the channel/spatial dimensions.
- **Why it happens**: MONAI's `VariationalAutoEncoder` encodes to a 1D latent vector `(B, latent_size)` with no spatial dimensions, but the encoder's intermediate spatial representation (before the flattening to 1D) has a different number of channels. Claude confuses `latent_size` (the bottleneck dimension) with the latent spatial size.
- **How to recognize**: `RuntimeError: size mismatch` when calling `vae.decode(z_mu)`. The latent vector has shape `(B, 128)` but the decoder expects `(B, 128, D, H, W)` spatial latent format. Or the reconstructed image has severe artifacts because the latent was not sampled properly (forgot the reparameterization trick: `z = z_mu + z_sigma * epsilon`).
- **How to fix**: For MONAI VAE, the forward returns `(reconstruction, z_mu, z_sigma)`. Always use the reparameterized sample for generation:

```python
recon, z_mu, z_sigma = model(x)
# Training: use reparameterized sample
z = z_mu + z_sigma * torch.randn_like(z_mu)
generated = model.decode(z)
# Note: model.decode takes (B, channels, D, H, W), not (B, latent_size)
```

If you need a spatial latent VAE (common in latent diffusion), you must use a custom VAE or check the reference repository — MONAI's VAE produces a 1D bottleneck.

---

### Gotcha 10: Rectified Flow Interpolation Formula (`noise.lerp(data, t)`)

- **What Claude does**: Implements rectified flow training as `noised = t * data + (1 - t) * noise` (the standard DDPM-style linear interpolation) instead of `noised = noise.lerp(data, t)`, or swaps the order of arguments so that `t=0` gives the data instead of the noise.
- **Why it happens**: The lucidrains rectified flow convention is `noised = noise.lerp(data, t)`, which means at `t=0` you get `noise` and at `t=1` you get `data`. Claude's mathematical instinct says "interpolate from noise to data" and writes `(1-t)*noise + t*data`, which is actually the same formula but with reversed time semantics — causing the model to receive inverted timestep conditioning.
- **How to recognize**: Generated samples converge to a blurred average of the training data rather than sharp samples. Loss decreases but samples look like `t=0.5` interpolations of random noise and data. The model never learns to produce clean samples at `t=0`.
- **How to fix**: Use the exact lucidrains convention:

```python
# CORRECT lucidrains convention
t = torch.rand(b, 1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1, 1)
noised = noise.lerp(data, t)        # t=0 -> noise, t=1 -> data
flow_target = data - noise           # flow is data minus noise
loss = F.mse_loss(model(noised, t), flow_target)
```

Verify: at `t=0` the input should be pure noise; at `t=1` it should be the original data.

---

### Gotcha 11: Scheduler `step` Argument Order

- **What Claude does**: Calls `scheduler.step(noise_pred, t, sample)` with arguments in the wrong order — passing `t` as the first argument instead of the second, or passing `sample` as the second argument instead of the third. Uses `DDPMScheduler.step` signature on a `DDIMScheduler` or vice versa.
- **Why it happens**: MONAI schedulers have different `step()` signatures that look similar but differ in argument order and count. `DDPMScheduler.step(noise_pred, t, sample)` vs `DDIMScheduler.step(noise_pred, t, sample)`. The argument names in the docs can appear ambiguous in context.
- **How to recognize**: `TypeError: step() missing 1 required positional argument` or `TypeError: step() takes 3 positional arguments but 4 were given`. Or no error but samples never improve because the scheduler step is silently misaligned.
- **How to fix**: Check the exact signature of the scheduler's `step` method. For `DDPMScheduler`, the call is: `scheduler.step(noise_pred, t, sample)` where `t` is an int timestep and `sample` is the current sample. For `DDIMScheduler`, it is: `scheduler.step(noise_pred, t, sample)` — same order but `t` is a tensor. For DDIM always set timesteps first with `scheduler.set_timesteps(N)`. When iterating with `for t in reversed(range(num_timesteps))` on DDPM, convert `t` to a tensor: `t = torch.tensor([t], device=device)`.

---

## Dataset & Memory Gotchas

### Gotcha 12: `CacheDataset` Misuse — `cache_rate=1.0` on Large 3D Volumes

- **What Claude does**: Sets `CacheDataset(cache_rate=1.0)` for a dataset of 500 3D brain MRIs at `(1, 256, 256, 128)` float32 each, expecting "maximum speed." The process crashes at startup with `OOM` or `MemoryError`, or causes system-wide slowdown.
- **Why it happens**: Claude thinks "cache everything = fastest training" without calculating memory requirements. 500 volumes × 256×256×128 × 4 bytes × 1.0 cache rate = ~80 GB of RAM. This exceeds available memory. The bias toward maximal caching for speed is correct for small 2D datasets but catastrophically wrong for large 3D volumes.
- **How to recognize**: Python process killed at startup with no GPU activity. `CacheDataset` initialization takes more than 30 seconds then crashes. System becomes unresponsive. Error: `RuntimeError: Unable to allocate array` or `OOM killed`.
- **How to fix**: Calculate memory first. For large 3D volumes, use `cache_rate=0.5` or lower, or switch to `PersistentDataset` which caches to disk instead of RAM. A safer pattern for large 3D data:

```python
# For large 3D volumes: use cache_rate < 1.0 or PersistentDataset
train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=0.5,       # Only cache 50% of the dataset
    num_workers=4,
)

# Alternative: PersistentDataset (caches preprocessed data to disk)
from monai.data import PersistentDataset
train_ds = PersistentDataset(
    data=train_files,
    transform=train_transforms,
    cache_dir="./persistent_cache",
)
```

Target no more than 20-30 GB of RAM usage for the cache on a typical workstation.

---

### Gotcha 13: Epoch-Based vs Step-Based Scheduler Updates

- **What Claude does**: Updates the learning rate scheduler inside the batch loop (step-based) but passes `epoch_length=len(dataloader)` to a `MonaiEpochWiseRLRScheduler`, or updates it once per epoch but uses a step-based scheduler.
- **Why it happens**: MONAI has both epoch-level and step-level LR schedulers. `MonaiEpochWiseRLRScheduler` updates once per epoch. `MonaiAdamWarmupScheduler` updates per step with warmup. Claude picks a scheduler by name without checking which update domain it uses.
- **How to recognize**: Learning rate stays constant throughout training despite a scheduler being present. Or the LR reaches the minimum in the first 10 steps and stays there for the remaining 990 epochs. Training loss decreases but the model underperforms expectations.
- **How to fix**: Match the scheduler to the update frequency. For step-based: use `MonaiAdamWarmupScheduler` or call `scheduler.step()` inside the batch loop. For epoch-based: call `scheduler.step(epoch)` once after each epoch. Do not mix update frequencies.

```python
# Step-based warmup scheduler
scheduler = MonaiAdamWarmupScheduler(warmup_epochs=5, min_lr=1e-6)
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = ...
        optimizer.step()
        scheduler.step(step)  # Updates every step

# Epoch-based scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step(epoch)  # Updates every epoch
```
