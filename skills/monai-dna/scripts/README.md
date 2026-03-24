# MONAI DNA Validation Scripts

Executable Python scripts that Claude can run during MONAI DNA skill usage to validate
pipelines, contracts, and configurations before training or inference.

## When to Invoke Each Script

### `validate_transform_pipeline.py`

**When:** Building or debugging a MONAI data preprocessing pipeline.

**Why:** Catches shape mismatches in the LoadImage → AddChannel → Spacing → Normalization chain before training starts.

**Typical invocation:**
```bash
python scripts/validate_transform_pipeline.py --spatial-dims 3 --in-channels 1 --image-size 64 64 64 --pixdim 1.0 1.0 1.0 --normalize-range -1000 1000
```

**What it checks:**
- LoadImaged adds channel dimension correctly
- Spacingd produces expected spatial shape
- Normalization maps to correct output range
- Pipeline shape contract: `(C, H, W, D)` or `(C, H, W)`

---

### `validate_diffusion_contract.py`

**When:** Implementing or auditing a diffusion model training loop.

**Why:** The three-way contract between `scheduler.add_noise`, model output, and `scheduler.step` breaks silently if arguments are in the wrong order or have wrong shapes.

**Typical invocation:**
```bash
python scripts/validate_diffusion_contract.py --scheduler DDPMScheduler --spatial-dims 3 --in-channels 1 --image-size 64 64 64
```

**What it checks:**
- `scheduler.add_noise(sample, noise, timesteps)` signature is used correctly
- Model output shape matches noise tensor shape
- `scheduler.step(noise_pred, timestep, sample)` argument order and return value (`.prev_sample`)
- Supports both `DDPMScheduler` and `DDIMScheduler`

---

### `validate_rectified_flow.py`

**When:** Implementing or auditing a rectified flow / flow matching generative model.

**Why:** Rectified flow has subtle implementation choices (lerp direction, velocity parameterization, sampling direction) that differ from standard diffusion. Wrong choices cause incorrect gradients or divergent sampling.

**Typical invocation:**
```bash
python scripts/validate_rectified_flow.py --spatial-dims 3 --in-channels 1 --image-size 64 64 64
```

**What it checks:**
- **(a)** Interpolation uses `noise.lerp(data, t)` (mathematically equivalent to `(1-t)*noise + t*data` but semantically explicit)
- **(b)** Model output represents velocity `v = x1 - x0`, not noise epsilon or clean sample
- **(c)** Sampling direction is `t=0` (noise) → `t=1` (data)

---

### `validate_spatial_consistency.py`

**When:** Starting a new project, onboarding a new dataset, or mixing 2D/3D components.

**Why:** `spatial_dims` mismatches between model, transforms, inferer, and post-processing produce cryptic shape errors deep in training. This catches them at the surface.

**Typical invocation:**
```bash
python scripts/validate_spatial_consistency.py --spatial-dims 3 --in-channels 1 --out-channels 2 --image-size 64 64 64
```

**What it checks:**
- `spatial_dims` value is valid (2 or 3)
- Model forward produces correct output shape for declared `spatial_dims`
- Transform pipeline uses consistent dictionary transforms
- Post-processing transforms are compatible with declared `spatial_dims`
- Shape contract: input → model → output

---

### `memory_estimator.py`

**When:** Planning a training run, choosing batch size, or investigating OOM errors.

**Why:** GPU memory errors waste hours of compute. This gives an analytical estimate before launching training.

**Typical invocation:**
```bash
python scripts/memory_estimator.py --model-type UNet --spatial-dims 3 --channels 32 64 128 256 --image-size 64 64 64 --batch-size 2
```

**What it reports:**
- Model parameters memory (FP32)
- Activation memory per batch
- Gradient memory (training)
- Optimizer state memory (Adam, ~2x params)
- Total training and inference memory
- OOM risk flag for 8GB / 16GB / 24GB GPUs

**Flagged as FAIL (OOM risk) if estimated training memory exceeds ~20GB on a 24GB GPU.**

---

### `run_all_checks.py`

**When:** Running a full pre-flight check on a project workspace, validating after pulling changes, or onboarding a new project.

**Why:** Orchestrates all five scripts above in sequence and reports each result.

**Typical invocation:**
```bash
python scripts/run_all_checks.py --spatial-dims 3 --in-channels 1 --out-channels 2 --image-size 64 64 64 --channels 32 64 128 256 --batch-size 2
```

## All Scripts Share These Conventions

| Convention | Details |
|------------|---------|
| **Entry point** | `if __name__ == "__main__"` with `argparse` |
| **Output** | Prints `PASS` or `FAIL` to stdout; exit code 0 = pass |
| **Logging** | Uses `logging` module (not `print`) for errors and diagnostics |
| **MONAI missing** | Gracefully degrades with clear message if MONAI is not installed |
| **Type hints** | All public functions have type hints |
| **Docstrings** | Each module and function has a docstring explaining purpose |

## Integration with Claude

Claude should invoke these scripts:

1. **Before writing a training loop** — run `validate_transform_pipeline` and `validate_spatial_consistency` to confirm the data pipeline is sound.

2. **After writing a generative training loop** — run `validate_diffusion_contract` or `validate_rectified_flow` to verify the scheduler-model contract.

3. **Before launching a training job** — run `memory_estimator` to check if the configuration fits in GPU memory.

4. **After any architectural change** — run `run_all_checks` for a full project health report.

All scripts are designed to be runnable from the repo root:
```bash
python skills/monai-dna/scripts/<script_name>.py [args...]
```
