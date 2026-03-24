"""Estimate GPU memory required for a MONAI model configuration.

Use this script when:
- Planning a new training run and want to check if the model fits in GPU memory
- Choosing batch_size given model architecture and image size
- Investigating out-of-memory (OOM) errors
- Comparing memory footprint of different model types (UNet vs SwinUNETR vs DiffusionModelUNet)
- Setting up multi-GPU training and needing to estimate per-GPU memory

Claude should run this before submitting a training job to prevent wasted time
on runs that fail due to OOM on a typical 24GB GPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryEstimate:
    """Memory estimate breakdown."""

    model_params_mb: float
    activations_mb: float
    gradients_mb: float
    optimizer_state_mb: float
    total_training_mb: float
    total_inference_mb: float
    per_sample_inference_mb: float
    oom_risk_24gb: bool
    oom_risk_16gb: bool
    oom_risk_8gb: bool
    warnings: list[str]


MODEL_MEMORY_FOOTPRINTS: dict[str, float] = {
    # Estimated MB per million parameters (activation memory dominates)
    # This is approximate and varies by architecture
    "UNet": 4.0,
    "BasicUNet": 3.0,
    "DynUNet": 5.0,
    "AttentionUNet": 4.5,
    "SwinUNETR": 8.0,
    "UNETR": 7.0,
    "VNet": 4.5,
    "SegResNet": 4.0,
    "ResNet": 3.5,
    "DenseNet": 4.5,
    "EfficientNetBN": 5.0,
    "AutoEncoder": 3.0,
    "VariationalAutoEncoder": 4.0,
    "DiffusionModelUNet": 6.0,
    "SwinUNETR_diffusion": 9.0,
}


def estimate_unet_params(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...],
    strides: tuple[int, ...],
    num_res_units: int = 2,
    kernel_size: int = 3,
) -> int:
    """Estimate number of parameters in a UNet-style model.

    This is a rough analytical estimate based on channel counts and layer count.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        in_channels: Input channels.
        out_channels: Output channels.
        channels: Tuple of channel counts per level.
        strides: Tuple of stride values per level.
        num_res_units: Number of residual units per block.
        kernel_size: Convolution kernel size.

    Returns:
        Estimated parameter count.
    """
    k = kernel_size ** spatial_dims
    total_params = 0

    # Encoder
    prev_ch = in_channels
    for ch, stride in zip(channels, strides):
        # Down block: conv + norm + act + res units
        down_params = ch * prev_ch * k + ch  # first conv
        for _ in range(num_res_units):
            down_params += ch * ch * k + ch  # res unit convs
        prev_ch = ch

    # Bottleneck
    bottleneck_params = channels[-1] * channels[-1] * k + channels[-1]

    # Decoder
    for ch, stride in zip(reversed(channels[:-1]), reversed(strides[:-1])):
        # Up block: upsample + conv + skip conv + res units
        up_params = ch * channels[-1] * k + ch  # upsample conv
        for _ in range(num_res_units):
            up_params += ch * ch * k + ch
        channels = channels[:-1]

    # Final conv
    final_params = out_channels * channels[0] * k + out_channels if channels else out_channels * prev_ch * k + out_channels

    return max(total_params, 1_000_000)  # floor at 1M to avoid zero


def estimate_diffusion_unet_params(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...],
    attention_levels: tuple[bool, ...],
    num_res_blocks: int = 2,
) -> int:
    """Estimate DiffusionModelUNet parameter count.

    DiffusionModelUNet is significantly larger than standard UNet due to
    time embedding and attention layers.

    Args:
        spatial_dims: Number of spatial dimensions.
        in_channels: Input channels.
        out_channels: Output channels.
        channels: Channel progression.
        attention_levels: Which levels have attention.
        num_res_blocks: Number of residual blocks per level.

    Returns:
        Estimated parameter count.
    """
    # Diffusion UNet typically has 50-100M params for typical medical imaging configs
    # Rough scaling: base UNet * attention_multiplier * res_block_multiplier
    base = estimate_unet_params(spatial_dims, in_channels, out_channels, channels, (2,) * len(channels))

    attention_multiplier = 1.0 + 0.1 * sum(attention_levels)
    res_multiplier = 1.0 + 0.15 * (num_res_blocks - 1)

    return int(base * attention_multiplier * res_multiplier)


def estimate_model_memory_mb(
    model_type: str,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    channels: tuple[int, ...],
    image_size: tuple[int, ...],
    batch_size: int = 1,
    **model_kwargs,
) -> MemoryEstimate:
    """Estimate GPU memory required for a model configuration.

    Memory breakdown:
    - Model parameters: ~4 bytes/param (FP32)
    - Activations: depends on spatial size and batch size
    - Gradients: same as model parameters (during training)
    - Optimizer state (Adam): ~8 bytes/param (2 moments)

    Args:
        model_type: Type of model (UNet, SwinUNETR, DiffusionModelUNet, etc.).
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        channels: Channel progression tuple.
        image_size: Input image size.
        batch_size: Batch size for activation memory estimation.
        **model_kwargs: Additional model-specific parameters.

    Returns:
        MemoryEstimate with detailed breakdown.
    """
    # Estimate parameter count
    if "Diffusion" in model_type or "diffusion" in model_type.lower():
        num_params = estimate_diffusion_unet_params(
            spatial_dims, in_channels, out_channels,
            channels, model_kwargs.get("attention_levels", (False,) * len(channels)),
            model_kwargs.get("num_res_blocks", 2),
        )
    elif model_type == "UNet":
        num_params = estimate_unet_params(
            spatial_dims, in_channels, out_channels,
            channels, model_kwargs.get("strides", (2,) * (len(channels) - 1)),
        )
    else:
        # Fallback: use heuristic based on channels and spatial dims
        spatial_elements = int(np.prod(image_size))
        num_params = int(
            sum(channels) * channels[0] * spatial_elements * 0.001
            + in_channels * channels[0] * spatial_elements * 0.01
        )
        num_params = max(num_params, 500_000)

    bytes_per_param = 4.0  # FP32
    model_params_mb = num_params * bytes_per_param / (1024 ** 2)

    # Estimate activation memory
    # Activation memory ≈ batch_size * channels * spatial_size * dtype_bytes
    # UNets have roughly 4-8x activation overhead over param count per sample
    if "Diffusion" in model_type:
        activation_multiplier = 16.0  # diffusion models have large activations
    elif "Swin" in model_type or "swin" in model_type.lower():
        activation_multiplier = 12.0
    elif model_type == "UNet":
        activation_multiplier = 8.0
    else:
        activation_multiplier = 8.0

    activation_mb = model_params_mb * activation_multiplier

    # Activation scales with batch size
    activations_mb = activation_mb * batch_size

    # Gradients (training only)
    gradients_mb = model_params_mb * bytes_per_param / (1024 ** 2)

    # Optimizer state (Adam): 2 moments per parameter
    optimizer_state_mb = model_params_mb * 2.0

    # Training total
    total_training_mb = model_params_mb + activations_mb + gradients_mb + optimizer_state_mb

    # Inference total (no gradients, no optimizer state)
    total_inference_mb = model_params_mb + activation_mb  # per-batch
    per_sample_inference_mb = model_params_mb + activation_mb / batch_size if batch_size > 1 else total_inference_mb

    # OOM risk assessment
    oom_risk_24gb = total_training_mb > 20_000  # 24GB GPU, leave headroom
    oom_risk_16gb = total_training_mb > 13_000   # 16GB GPU
    oom_risk_8gb = total_training_mb > 6_500     # 8GB GPU

    warnings = []
    if oom_risk_24gb:
        warnings.append(
            f"HIGH OOM RISK: Estimated {total_training_mb:.0f}MB exceeds typical 24GB GPU. "
            "Reduce batch_size, image_size, or channels."
        )
    elif oom_risk_16gb:
        warnings.append(
            f"MEDIUM OOM RISK: {total_training_mb:.0f}MB may be tight on 16GB GPU. "
            "Consider reducing batch_size."
        )
    else:
        warnings.append(
            f"Memory estimate: {total_training_mb:.0f}MB training / {total_inference_mb:.0f}MB inference. "
            "Should fit on 16GB+ GPU."
        )

    # Check for very small images which might indicate errors
    if all(s < 32 for s in image_size):
        warnings.append("WARN: image_size seems very small. Double-check configuration.")

    return MemoryEstimate(
        model_params_mb=model_params_mb,
        activations_mb=activations_mb,
        gradients_mb=gradients_mb,
        optimizer_state_mb=optimizer_state_mb,
        total_training_mb=total_training_mb,
        total_inference_mb=total_inference_mb,
        per_sample_inference_mb=per_sample_inference_mb,
        oom_risk_24gb=oom_risk_24gb,
        oom_risk_16gb=oom_risk_16gb,
        oom_risk_8gb=oom_risk_8gb,
        warnings=warnings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate GPU memory required for a MONAI model configuration."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        help="Model type (UNet, SwinUNETR, DiffusionModelUNet, etc.)",
    )
    parser.add_argument(
        "--spatial-dims",
        type=int,
        default=3,
        choices=[2, 3],
        help="Spatial dimensionality",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=1,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=2,
        help="Number of output channels",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=(32, 64, 128, 256),
        help="Channel progression, e.g. --channels 32 64 128 256",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=(64, 64, 64),
        help="Image size as H W or H W D",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size",
    )
    parser.add_argument(
        "--attention-levels",
        type=str,
        default=None,
        help="Comma-separated attention levels, e.g. 'False,False,True,True'",
    )
    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=2,
        help="Number of residual blocks (for DiffusionModelUNet)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    monai = _check_monai()
    if monai is None:
        logger.warning("MONAI not installed — memory estimate uses analytical model, not actual weights")

    if args.attention_levels:
        attention_levels = tuple(
            al.strip().lower() == "true"
            for al in args.attention_levels.split(",")
        )
    else:
        attention_levels = tuple(False for _ in args.channels)

    estimate = estimate_model_memory_mb(
        model_type=args.model_type,
        spatial_dims=args.spatial_dims,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        channels=tuple(args.channels),
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        attention_levels=attention_levels,
        num_res_blocks=args.num_res_blocks,
    )

    print()
    print(f"=== Memory Estimate for {args.model_type} ===")
    print(f"  spatial_dims={args.spatial_dims}, channels={args.channels}")
    print(f"  image_size={args.image_size}, batch_size={args.batch_size}")
    print()
    print(f"  Model parameters (FP32):     {estimate.model_params_mb:8.1f} MB")
    print(f"  Activations (per batch):     {estimate.activations_mb:8.1f} MB")
    print(f"  Gradients (training):        {estimate.gradients_mb:8.1f} MB")
    print(f"  Optimizer state (Adam):       {estimate.optimizer_state_mb:8.1f} MB")
    print(f"  ─────────────────────────────────────────")
    print(f"  Total training memory:        {estimate.total_training_mb:8.1f} MB")
    print(f"  Total inference memory:      {estimate.total_inference_mb:8.1f} MB")
    print(f"  Per-sample inference:         {estimate.per_sample_inference_mb:8.1f} MB")
    print()
    print(f"  OOM Risk (8GB):   {'YES' if estimate.oom_risk_8gb else 'no'}")
    print(f"  OOM Risk (16GB): {'YES' if estimate.oom_risk_16gb else 'no'}")
    print(f"  OOM Risk (24GB): {'YES' if estimate.oom_risk_24gb else 'no'}")
    print()

    for warn in estimate.warnings:
        print(f"  {warn}")

    print()
    overall_pass = not estimate.oom_risk_24gb
    print("PASS — likely fits on 24GB GPU" if overall_pass else "FAIL — likely OOM on 24GB GPU")
    sys.exit(0 if overall_pass else 1)


def _check_monai():
    """Check if MONAI is installed."""
    try:
        import monai
        return monai
    except ImportError:
        return None


if __name__ == "__main__":
    main()
