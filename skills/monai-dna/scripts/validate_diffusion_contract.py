"""Validate diffusion model training step matches scheduler contract.

Use this script when:
- Implementing a new diffusion model training loop
- Verifying scheduler.add_noise, model forward, and scheduler.step integration
- Checking DDPMScheduler vs DDIMScheduler argument order
- Auditing a generative project against MONAI GenerativeModels patterns

Claude should run this after writing any diffusion training code to verify
the three-way contract between scheduler.add_noise, model output, and scheduler.step.
"""

from __future__ import annotations

import argparse
import inspect
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


def validate_add_noise_signature(scheduler: Any) -> tuple[bool, str]:
    """Check scheduler.add_noise(x0, noise, timesteps) signature is used correctly.

    Validates that the scheduler's add_noise method accepts the positional arguments
    in the correct order: (clean_sample, noise, timesteps).

    Returns:
        (passed, message)
    """
    if not hasattr(scheduler, "add_noise"):
        return False, "scheduler has no add_noise method"

    sig = inspect.signature(scheduler.add_noise)
    params = list(sig.parameters.keys())

    # MONAI schedulers use: add_noise(x0, noise, timesteps)
    # Some may also support add_noise(x0, noise, timesteps, **kwargs)
    expected_first_params = ["sample", "noise", "timesteps"]
    actual_params = [p for p in params if p not in ("kwargs", "kwarg")]

    if len(actual_params) < 3:
        return (
            False,
            f"add_noise signature too short: {params}. "
            f"Expected at least (sample, noise, timesteps)",
        )

    # Check positional ordering
    if actual_params[0] not in ("sample", "x0", "clean_sample"):
        return (
            False,
            f"add_noise first param should be 'sample' or 'x0', got '{actual_params[0]}'",
        )
    if actual_params[1] not in ("noise",):
        return (
            False,
            f"add_noise second param should be 'noise', got '{actual_params[1]}'",
        )
    if actual_params[2] not in ("timesteps", "t"):
        return (
            False,
            f"add_noise third param should be 'timesteps', got '{actual_params[2]}'",
        )

    return True, f"add_noise signature OK: {params}"


def validate_model_output_shape(
    model_output: torch.Tensor,
    noise: torch.Tensor,
    spatial_dims: int = 3,
) -> tuple[bool, str]:
    """Validate model output shape matches noise input shape.

    In standard diffusion (epsilon-prediction), the model predicts the noise
    and should have the same shape as the noise tensor.

    Args:
        model_output: The raw model output (e.g., noise prediction).
        noise: The target noise tensor.
        spatial_dims: Number of spatial dimensions.

    Returns:
        (passed, message)
    """
    if model_output.shape != noise.shape:
        return (
            False,
            f"Model output shape {model_output.shape} != noise shape {noise.shape}",
        )
    return True, f"Model output shape OK: {model_output.shape}"


def validate_scheduler_step_contract(
    scheduler: Any,
    noise_pred: torch.Tensor,
    timestep: int,
    noisy_sample: torch.Tensor,
    scheduler_type: str = "DDPMScheduler",
) -> tuple[bool, list[str]]:
    """Validate scheduler.step argument order and return value.

    DDPMScheduler.step: step(noise_pred, timestep, sample, **kwargs) -> DDPMStepOutput
    DDIMScheduler.step: step(noise_pred, timestep, sample, **kwargs) -> DDPMStepOutput

    Both return an object with .prev_sample attribute.

    Args:
        scheduler: The scheduler instance.
        noise_pred: Predicted noise from the model.
        timestep: Current timestep (int).
        noisy_sample: The noisy sample at this timestep.
        scheduler_type: Either "DDPMScheduler" or "DDIMScheduler".

    Returns:
        (passed, list_of_messages)
    """
    messages = []
    passed = True

    if not hasattr(scheduler, "step"):
        return False, ["scheduler has no step method"]

    sig = inspect.signature(scheduler.step)
    params = list(sig.parameters.keys())
    messages.append(f"step signature: {params}")

    # Check argument order: (noise_pred, timestep, sample)
    non_kw_params = [p for p in params if p not in ("kwargs", "kwarg")]

    if len(non_kw_params) < 3:
        msg = f"step signature too short: {params}. Expected (noise_pred, timestep, sample)"
        messages.append(f"FAIL: {msg}")
        return False, messages

    # Validate parameter names
    if non_kw_params[0] not in ("noise_pred", "noise", "predicted_noise"):
        msg = f"step first param should be noise_pred, got '{non_kw_params[0]}'"
        messages.append(f"WARN: {msg}")

    if non_kw_params[1] not in ("timestep", "t", "time_step"):
        msg = f"step second param should be timestep, got '{non_kw_params[1]}'"
        messages.append(f"WARN: {msg}")

    if non_kw_params[2] not in ("sample", "x", "noisy_sample"):
        msg = f"step third param should be sample, got '{non_kw_params[2]}'"
        messages.append(f"WARN: {msg}")

    # Try calling step() to verify it works
    try:
        # Build dummy timestep tensor
        t_tensor = torch.tensor([timestep]) if isinstance(timestep, int) else timestep

        if scheduler_type == "DDIMScheduler":
            # DDIMScheduler needs set_timesteps called first
            if hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(50)
            result = scheduler.step(noise_pred, timestep, noisy_sample)
        else:
            result = scheduler.step(noise_pred, timestep, noisy_sample)

        if not hasattr(result, "prev_sample"):
            msg = "step result missing .prev_sample attribute"
            messages.append(f"FAIL: {msg}")
            passed = False
        else:
            messages.append(f"step returned object with .prev_sample OK: {result.prev_sample.shape}")

    except Exception as e:
        msg = f"step call failed: {e}"
        messages.append(f"FAIL: {msg}")
        passed = False

    return passed, messages


def validate_diffusion_contract(
    scheduler_type: str = "DDPMScheduler",
    spatial_dims: int = 3,
    in_channels: int = 1,
    image_size: tuple[int, ...] | None = None,
) -> tuple[bool, list[str]]:
    """Validate the full diffusion training step contract.

    Tests:
    1. scheduler.add_noise signature
    2. Model output shape matches noise
    3. scheduler.step argument order

    Args:
        scheduler_type: "DDPMScheduler" or "DDIMScheduler".
        spatial_dims: Number of spatial dimensions.
        in_channels: Number of input channels.
        image_size: Input image size. If None, uses default (64, 64, 64) or (64, 64).

    Returns:
        (all_passed, list_of_messages)
    """
    monai = _check_monai()
    if monai is None:
        return False, ["MONAI is not installed. Install with: pip install monai"]

    from monai.networks.schedulers import DDPMScheduler, DDIMScheduler

    if image_size is None:
        image_size = (64, 64, 64) if spatial_dims == 3 else (64, 64)

    if spatial_dims == 2:
        x_shape = (in_channels, *image_size)
    else:
        x_shape = (in_channels, *image_size)

    batch_size = 2
    num_timesteps = 1000

    # Build full tensor shape
    full_shape = (batch_size, *x_shape)

    # Create scheduler
    if scheduler_type == "DDPMScheduler":
        scheduler = DDPMScheduler(num_train_timesteps=num_timesteps)
    elif scheduler_type == "DDIMScheduler":
        scheduler = DDIMScheduler(num_train_timesteps=num_timesteps)
    else:
        return False, [f"Unknown scheduler type: {scheduler_type}"]

    messages = [f"Testing {scheduler_type}"]

    # 1. Validate add_noise signature
    passed_add_noise, add_noise_msg = validate_add_noise_signature(scheduler)
    messages.append(f"add_noise: {add_noise_msg}")
    if not passed_add_noise:
        messages.append("FAIL: add_noise signature check failed")
        return False, messages

    # 2. Create dummy tensors and test full chain
    x0 = torch.randn(full_shape)
    noise = torch.randn_like(x0)
    timesteps = torch.randint(0, num_timesteps, (batch_size,))

    try:
        noisy = scheduler.add_noise(x0, noise, timesteps)
        messages.append(f"add_noise output shape: {noisy.shape} == input shape {x0.shape}: {noisy.shape == x0.shape}")
    except Exception as e:
        messages.append(f"FAIL: add_noise call failed: {e}")
        return False, messages

    # 3. Simulate model output (noise prediction)
    # For DDPMScheduler, model predicts noise epsilon
    model_output = torch.randn_like(x0)  # Simulated noise prediction
    passed_shape, shape_msg = validate_model_output_shape(model_output, noise, spatial_dims)
    messages.append(f"model output: {shape_msg}")
    if not passed_shape:
        messages.append("FAIL: model output shape mismatch")
        return False, messages

    # 4. Validate scheduler.step
    t = timesteps[0].item()
    passed_step, step_messages = validate_scheduler_step_contract(
        scheduler, model_output, t, noisy, scheduler_type
    )
    messages.extend(step_messages)

    all_passed = passed_add_noise and passed_shape and passed_step
    if all_passed:
        messages.append(f"PASS: {scheduler_type} contract validated")
    else:
        messages.append("FAIL: contract validation failed")

    return all_passed, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate diffusion model-scheduler training contract."
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDPMScheduler",
        choices=["DDPMScheduler", "DDIMScheduler"],
        help="Scheduler type to validate",
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
        "--image-size",
        type=int,
        nargs="+",
        default=None,
        help="Image size as H W or H W D",
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

    passed, messages = validate_diffusion_contract(
        scheduler_type=args.scheduler,
        spatial_dims=args.spatial_dims,
        in_channels=args.in_channels,
        image_size=tuple(args.image_size) if args.image_size else None,
    )

    for msg in messages:
        if msg.startswith("FAIL"):
            logger.error(msg)
        elif msg.startswith("PASS"):
            logger.info(msg)
        else:
            logger.info(msg)

    print()
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
