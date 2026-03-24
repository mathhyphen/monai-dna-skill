"""Orchestrate all MONAI DNA validation scripts against a project directory.

Use this script when:
- Running a full pre-flight check on a MONAI generative or segmentation project
- Validating an entire workspace after pulling changes
- Onboarding a new project and wanting to verify all components are consistent

This script runs all five validation scripts in sequence:
1. validate_transform_pipeline
2. validate_diffusion_contract
3. validate_rectified_flow
4. validate_spatial_consistency
5. memory_estimator

Each script's PASS/FAIL result is reported individually.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent


class CheckResult(NamedTuple):
    """Result of a single validation check."""

    name: str
    passed: bool
    output: str


def run_script(script_name: str, extra_args: list[str] | None = None) -> CheckResult:
    """Run a validation script and capture its output.

    Args:
        script_name: Name of the script file (e.g., "validate_transform_pipeline.py").
        extra_args: Additional command-line arguments to pass.

    Returns:
        CheckResult with name, passed status, and captured output.
    """
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        return CheckResult(
            name=script_name,
            passed=False,
            output=f"Script not found: {script_path}",
        )

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        passed = result.returncode == 0
        output = result.stdout or ""
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        return CheckResult(name=script_name, passed=passed, output=output)

    except subprocess.TimeoutExpired:
        return CheckResult(
            name=script_name,
            passed=False,
            output="Script timed out after 120 seconds",
        )
    except Exception as e:
        return CheckResult(
            name=script_name,
            passed=False,
            output=f"Failed to run script: {e}",
        )


def run_all_checks(
    spatial_dims: int = 3,
    in_channels: int = 1,
    out_channels: int = 2,
    image_size: tuple[int, ...] | None = None,
    channels: tuple[int, ...] = (32, 64, 128, 256),
    batch_size: int = 2,
    scheduler_type: str = "DDPMScheduler",
    verbose: bool = False,
) -> list[CheckResult]:
    """Run all validation scripts in sequence.

    Args:
        spatial_dims: Spatial dimensionality for tests.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        image_size: Image size for shape tests.
        channels: Channel progression for model tests.
        batch_size: Batch size for memory tests.
        scheduler_type: Scheduler type for diffusion tests.
        verbose: Enable verbose output.

    Returns:
        List of CheckResult for each script.
    """
    if image_size is None:
        image_size = (64, 64, 64) if spatial_dims == 3 else (64, 64)

    size_arg = [str(s) for s in image_size]
    channels_arg = [str(c) for c in channels]

    base_args = [
        "--spatial-dims", str(spatial_dims),
        "--in-channels", str(in_channels),
        "--image-size", *size_arg,
    ]
    if verbose:
        base_args.append("--verbose")

    logger.info("Running validate_transform_pipeline...")
    transform_result = run_script(
        "validate_transform_pipeline.py",
        base_args
        + ["--normalize-range", "-1000", "1000"],
    )

    logger.info("Running validate_diffusion_contract...")
    diffusion_result = run_script(
        "validate_diffusion_contract.py",
        [
            "--scheduler", scheduler_type,
            "--spatial-dims", str(spatial_dims),
            "--in-channels", str(in_channels),
        ] + (["--verbose"] if verbose else []),
    )

    logger.info("Running validate_rectified_flow...")
    rectified_result = run_script(
        "validate_rectified_flow.py",
        [
            "--spatial-dims", str(spatial_dims),
            "--in-channels", str(in_channels),
        ] + (["--verbose"] if verbose else []),
    )

    logger.info("Running validate_spatial_consistency...")
    spatial_result = run_script(
        "validate_spatial_consistency.py",
        [
            "--spatial-dims", str(spatial_dims),
            "--in-channels", str(in_channels),
            "--out-channels", str(out_channels),
            "--image-size", *size_arg,
            "--batch-size", str(batch_size),
        ] + (["--verbose"] if verbose else []),
    )

    logger.info("Running memory_estimator...")
    memory_result = run_script(
        "memory_estimator.py",
        [
            "--model-type", "UNet",
            "--spatial-dims", str(spatial_dims),
            "--in-channels", str(in_channels),
            "--out-channels", str(out_channels),
            "--channels", *channels_arg,
            "--image-size", *size_arg,
            "--batch-size", str(batch_size),
        ],
    )

    return [
        transform_result,
        diffusion_result,
        rectified_result,
        spatial_result,
        memory_result,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all MONAI DNA validation scripts against a project."
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
        "--channels",
        type=int,
        nargs="+",
        default=(32, 64, 128, 256),
        help="Channel progression",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for memory test",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDPMScheduler",
        choices=["DDPMScheduler", "DDIMScheduler"],
        help="Diffusion scheduler type",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    image_size = tuple(args.image_size) if args.image_size else None

    results = run_all_checks(
        spatial_dims=args.spatial_dims,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        image_size=image_size,
        channels=tuple(args.channels),
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        verbose=args.verbose,
    )

    print()
    print("=" * 60)
    print("MONAI DNA — All Checks Report")
    print("=" * 60)

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        symbol = "[PASS]" if result.passed else "[FAIL]"
        print()
        print(f"{symbol} {result.name}")
        # Show last line of output (usually PASS/FAIL message)
        lines = result.output.strip().split("\n")
        relevant_lines = [l for l in lines if l.strip() and not l.startswith("DEBUG:")]
        for line in relevant_lines[-5:]:
            print(f"    {line}")

        if not result.passed:
            all_passed = False

    print()
    print("=" * 60)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED — review output above")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
