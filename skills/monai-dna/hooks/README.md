# MONAI DNA On-Demand Guardrail Hooks

These PreToolUse hooks implement Thariq's "On Demand Hooks" principle from https://x.com/trq212/status/2033949937936085378:

> Skills can include hooks that are only activated when the skill is called, and last for the duration of the session. Use this for more opinionated hooks that you don't want to run all the time, but are extremely useful sometimes.

## Activation

These hooks are **session-duration only** and are activated when the `monai-dna` skill is invoked. They are not global — they do not run for other skills or general Claude Code sessions.

To activate these hooks for a session, ensure the `monai-dna` skill is loaded. The hooks are automatically scoped to this skill and will not interfere with other projects or skills.

## Guardrails

| # | Name | Action | Purpose |
|---|------|--------|---------|
| 1 | `destructive_data_guard` | Block | Blocks `rm -rf` on paths containing medical imaging data directories |
| 2 | `generic_pytorch_replacement_guard` | Block | Blocks replacing MONAI factories with raw `torch.nn` layers |
| 3 | `diffusion_objective_mismatch_guard` | Block | Blocks changing diffusion prediction objective without scheduler update |
| 4 | `spatial_dims_inconsistency_guard` | Block | Blocks training when spatial_dims is inconsistent across codebase |
| 5 | `unsafe_checkpoint_overwrite_guard` | Block | Blocks overwriting `best_model.pt` without a backup |
| 6 | `label_interpolation_guard` | Block | Blocks bilinear/bicubic interpolation on label keys |
| 7 | `amp_absence_warning_for_3d` | Warn | Warns when 3D training lacks AMP (torch.cuda.amp) |

## Hook Schema

Each hook follows the Claude Code PreToolUse hook schema:

```json
{
  "name": "hook_name",
  "description": "Why this guard exists",
  "matcher": {
    "tool_name": "ToolName",
    "params_match": {
      "param": "regex_pattern"
    }
  },
  "action": "block | warn",
  "message": "Explanation shown to Claude when triggered"
}
```

## Hook Details

### 1. destructive_data_guard (Block)

**Trigger:** `Bash` commands matching `rm -rf` on paths with `data`, `images`, `labels`, `dataset`, `nii`, `nifti`, `dcm`.

**Why:** Medical imaging data is extremely high-value and often irreplaceable. Recursive deletion of data directories can destroy weeks of preprocessing work.

**Recovery:** Use a specific non-recursive path, or move to trash instead of permanent deletion.

---

### 2. generic_pytorch_replacement_guard (Block)

**Trigger:** `Edit` operations introducing raw `torch.nn.Conv2d/Conv3d`, `torch.nn.BatchNorm`, `torch.nn.MaxPool2d/3d`, etc.

**Why:** MONAI's `Conv`, `Norm`, `Pool`, `Act` factories automatically handle `spatial_dims=2` vs `spatial_dims=3`. Hardcoding `Conv2d` instead of `Conv[Conv.CONV, spatial_dims]` silently breaks 3D pipelines.

**Exception:** Allowed if an explicit comment states why the MONAI factory is unsuitable for the specific case.

---

### 3. diffusion_objective_mismatch_guard (Block)

**Trigger:** `Edit` operations that set `predict='flow'`, `predict='noise'`, `predict='epsilon'`, or `predict='clean'`.

**Why:** The prediction objective is tightly coupled with the scheduler's step function. Switching from noise prediction to flow prediction (common in rectified flow) without updating the scheduler produces a model whose forward pass does not match its sampling loop.

---

### 4. spatial_dims_inconsistency_guard (Block)

**Trigger:** `Bash` commands running training scripts (`python train`, `python run`, `python main`).

**Why:** This hook scans Python files for `spatial_dims=` values and flags inconsistencies between model init, transforms, and inferer before training starts. A mismatch (e.g., model=3D, transforms=2D) causes silent shape crashes after hours of training.

---

### 5. unsafe_checkpoint_overwrite_guard (Block)

**Trigger:** `Edit` or `Write` operations that produce a `torch.save` call targeting `best_model.pt`.

**Why:** Overwriting the best checkpoint without a backup risks permanent loss if the save is corrupted or interrupted. Medical imaging training runs can take days.

**Recovery:** Save to `best_model.pt.tmp`, verify with `torch.load`, then atomically rename.

---

### 6. label_interpolation_guard (Block)

**Trigger:** `Edit` operations on spatial transforms (`Spacingd`, `Orientationd`, `Resized`, `RandRotated`, `RandZoomd`, `RandAffined`) that assign `bilinear`, `bicubic`, or `area` mode to label keys.

**Why:** Labels contain discrete integer class indices. Interpolation creates non-integer values that corrupt segmentation masks, crashing `AsDiscreted` or producing invalid metrics.

**Correct pattern:** `mode=("bilinear", "nearest")` — image gets bilinear, label gets nearest.

---

### 7. amp_absence_warning_for_3d (Warn)

**Trigger:** `Bash` commands running 3D training scripts where no `torch.cuda.amp.autocast` or `GradScaler` is detected.

**Why:** 3D volumes are memory-intensive. AMP typically provides 40-50% memory savings, enabling 2x larger batch sizes on Ampere/Ada GPUs. This is a warning-only hook to educate without blocking.

## Integration with Claude Code

These hooks are loaded as part of the `monai-dna` skill definition. When the skill is active, Claude Code evaluates the `pre_tool_use` hooks before executing any tool call. If a matcher fires:

- **`block`**: The tool call is aborted and the message is shown to Claude, which can then correct the issue or explain why the block should be bypassed.
- **`warn`**: The message is shown as a warning but the tool call proceeds.

## Extending Hooks

To add a new hook:

1. Add a new entry to `guardrails.json` under `pre_tool_use`.
2. Follow the schema: `name`, `description`, `matcher` (tool_name + params_match), `action`, `message`.
3. Test that the matcher fires on the intended command/Edit and does not fire on similar-but-allowed commands.

## Programmatic Registration

For advanced use, `activate_hooks.py` provides a Python interface to register hooks programmatically (useful for testing or custom activation logic).
