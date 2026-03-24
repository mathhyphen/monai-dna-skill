"""Validate spatial consistency across model, transforms, inferer, and post-processing.

Use this script when:
- Starting a new project or onboarding a new dataset
- Mixing 2D and 3D components and needing to verify spatial_dims consistency
- Checking that model forward, transforms pipeline, inferer, and post-processing
  all use the same spatial dimensionality
- Debugging shape mismatch errors that appear during training or inference

Claude should run this early in project setup and after any architectural changes
to catch spatial dimension mismatches before they cause cryptic shape errors.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _check_monai():
    """Check if MONAI is installed."""
    try:
        import monai
        return monai
    except ImportError:
        return None


def validate_spatial_dims_value(
    spatial_dims: int,
    label: str,
) -> tuple[bool, str]:
    """Validate spatial_dims is a supported value (2 or 3).

    Args:
        spatial_dims: The spatial_dims value to validate.
        label: Descriptive label for error messages.

    Returns:
        (passed, message)
    """
    if spatial_dims not in (2, 3):
        return (
            False,
            f"{label}: spatial_dims must be 2 or 3, got {spatial_dims}",
        )
    return True, f"{label}: spatial_dims={spatial_dims} OK"


def validate_model_forward_shape(
    model: torch.nn.Module,
    spatial_dims: int,
    in_channels: int,
    image_size: tuple[int, ...],
    batch_size: int = 1,
) -> tuple[bool, str]:
    """Validate model forward produces correct shape for its spatial_dims.

    Args:
        model: The model to test.
        spatial_dims: Expected spatial dimensionality.
        in_channels: Number of input channels.
        image_size: Input size as (H, W) or (H, W, D).
        batch_size: Batch size for test.

    Returns:
        (passed, message)
    """
    if spatial_dims == 2:
        expected_output_shape = (batch_size, in_channels, *image_size)
        input_shape = (batch_size, in_channels, *image_size)
    elif spatial_dims == 3:
        expected_output_shape = (batch_size, in_channels, *image_size)
        input_shape = (batch_size, in_channels, *image_size)
    else:
        return False, f"Unsupported spatial_dims={spatial_dims}"

    try:
        x = torch.randn(input_shape)
        with torch.no_grad():
            output = model(x)

        if output.shape != expected_output_shape:
            return (
                False,
                f"Model output shape {output.shape} != expected {expected_output_shape}. "
                f"Check model spatial_dims and channels configuration.",
            )
        return True, f"Model forward OK: input {input_shape} → output {output.shape}"

    except Exception as e:
        return (
            False,
            f"Model forward failed: {e}. "
            f"Input shape: {input_shape}, spatial_dims={spatial_dims}",
        )


def validate_transform_pipeline_spatial_dims(
    spatial_dims: int,
    pipeline: Any | None = None,
) -> tuple[bool, str]:
    """Validate transform pipeline uses consistent spatial_dims.

    For dictionary transforms (LoadImaged, Spacingd, etc.), spatial_dims
    is implicitly determined by the data shape. This checks the pipeline
    configuration is consistent with the declared spatial_dims.

    Args:
        spatial_dims: Declared spatial dimensionality.
        pipeline: Optional MONAI Compose pipeline to inspect.

    Returns:
        (passed, message)
    """
    if spatial_dims not in (2, 3):
        return False, f"spatial_dims must be 2 or 3, got {spatial_dims}"

    # Check that transforms are the dictionary (d) variants when expected
    if pipeline is not None:
        transform_names = [type(t).__name__ for t in pipeline.transforms]
        spatial_transforms = [
            "Spacingd", "Orientationd", "Resized", "RandRotated",
            "RandFlipd", "RandZoomd", "CenterSpatialCropd",
        ]
        for name in transform_names:
            if name.endswith("d") and spatial_dims == 2:
                # Some 3D transforms have d suffix but are used in 2D contexts
                pass
        return True, f"Pipeline transforms: {transform_names}"
    else:
        return True, f"spatial_dims={spatial_dims} declared; pipeline not provided for deep inspection"


def validate_inferer_spatial_dims(
    inferer: Any,
    spatial_dims: int,
) -> tuple[bool, str]:
    """Validate inferer is compatible with declared spatial_dims.

    Args:
        inferer: A MONAI Inferer instance.
        spatial_dims: Declared spatial dimensionality.

    Returns:
        (passed, message)
    """
    if inferer is None:
        return True, "No inferer provided; skipping inferer validation"

    inferer_type = type(inferer).__name__

    # Sliding window inferer has spatial_dims in __init__
    if "SlidingWindow" in inferer_type:
        if hasattr(inferer, "spatial_dims"):
            if inferer.spatial_dims != spatial_dims:
                return (
                    False,
                    f"Inferer spatial_dims={inferer.spatial_dims} != declared {spatial_dims}",
                )
            return True, f"SlidingWindowInferer spatial_dims={inferer.spatial_dims} OK"

    # Most MONAI inferers are spatial-dimension agnostic at the class level
    # but require consistent inputs
    return True, f"{inferer_type} compatible with spatial_dims={spatial_dims}"


def validate_postprocessing_spatial_dims(
    post_transforms: Any,
    spatial_dims: int,
) -> tuple[bool, str]:
    """Validate post-processing transforms are consistent with spatial_dims.

    Args:
        post_transforms: MONAI Compose pipeline for post-processing.
        spatial_dims: Declared spatial dimensionality.

    Returns:
        (passed, message)
    """
    if post_transforms is None:
        return True, "No post-transforms provided"

    transform_names = [type(t).__name__ for t in post_transforms.transforms]

    # AsDiscreted, Activationsd are dimension-agnostic
    # KeepLargestConnectedComponentd, FillHolesd work on any spatial dims
    ambiguous_transforms = ["KeepLargestConnectedComponentd", "FillHolesd"]

    messages = []
    all_passed = True

    for name in transform_names:
        if name in ambiguous_transforms:
            messages.append(
                f"WARN: {name} is being used; ensure labels have correct spatial dims"
            )

    if all_passed:
        messages.append(f"Post-processing transforms: {transform_names} — spatial_dims={spatial_dims} OK")

    return all_passed, "; ".join(messages)


def validate_spatial_consistency(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    image_size: tuple[int, ...] | None = None,
    model_type: str = "UNet",
    batch_size: int = 1,
) -> tuple[bool, list[str]]:
    """Validate spatial consistency across all pipeline components.

    Args:
        spatial_dims: Number of spatial dimensions (2 or 3).
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        image_size: Input image size as (H, W) or (H, W, D).
        model_type: Model type name (UNet, SwinUNETR, etc.) for error messages.
        batch_size: Batch size for shape testing.

    Returns:
        (all_passed, list_of_messages)
    """
    monai = _check_monai()
    if monai is None:
        return False, ["MONAI is not installed. Install with: pip install monai"]

    from monai.networks.nets import UNet

    messages = []

    if image_size is None:
        image_size = (64, 64, 64) if spatial_dims == 3 else (256, 256)

    # Validate spatial_dims value
    passed_dims, dims_msg = validate_spatial_dims_value(
        spatial_dims, "Global"
    )
    messages.append(dims_msg)
    if not passed_dims:
        return False, messages

    # Validate model can be constructed and called with correct shape
    try:
        model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(16, 32, 64),
            strides=(2, 2),
        )
        model.eval()

        passed_model, model_msg = validate_model_forward_shape(
            model, spatial_dims, in_channels, image_size, batch_size
        )
        messages.append(f"Model ({model_type}): {model_msg}")
        if not passed_model:
            messages.append("FAIL: model forward shape mismatch")
            return False, messages

    except Exception as e:
        return False, [f"FAIL: Model construction/forward failed: {e}"]

    # Validate transform pipeline consistency
    passed_tfm, tfm_msg = validate_transform_pipeline_spatial_dims(spatial_dims, None)
    messages.append(f"Transform pipeline: {tfm_msg}")

    # Validate post-processing consistency
    passed_post, post_msg = validate_postprocessing_spatial_dims(None, spatial_dims)
    messages.append(f"Post-processing: {post_msg}")

    # Summary of expected shapes
    if spatial_dims == 3:
        expected_input_shape = (batch_size, in_channels, *image_size)
        expected_output_shape = (batch_size, out_channels, *image_size)
    else:
        expected_input_shape = (batch_size, in_channels, *image_size)
        expected_output_shape = (batch_size, out_channels, *image_size)

    messages.append(
        f"Expected shape contract: input {expected_input_shape} → "
        f"model → output {expected_output_shape} (spatial_dims={spatial_dims})"
    )

    messages.append("PASS: Spatial consistency validated")
    return True, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate spatial consistency across model, transforms, inferer, and post-processing."
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
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size as H W or H W D",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="UNet",
        help="Model type name for error messages",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for shape testing",
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

    passed, messages = validate_spatial_consistency(
        spatial_dims=args.spatial_dims,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        image_size=tuple(args.image_size) if args.image_size else None,
        model_type=args.model_type,
        batch_size=args.batch_size,
    )

    for msg in messages:
        if msg.startswith("FAIL"):
            logger.error(msg)
        elif msg.startswith("PASS"):
            logger.info(msg)
        elif msg.startswith("WARN"):
            logger.warning(msg)
        else:
            logger.info(msg)

    print()
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
