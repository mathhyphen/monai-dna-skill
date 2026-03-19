---
name: monai-dna
description: "Expert guidance for MONAI-based medical imaging development and advanced generative workflows. Use this skill when Codex needs to: (1) build or refactor MONAI transforms and dictionary pipelines, (2) design 2D or 3D networks such as UNet, SwinUNETR, and ResNet, (3) implement segmentation training or inference workflows, (4) build generative models such as AutoEncoder, VAE, diffusion, or rectified flow, or (5) troubleshoot tensor shapes, spatial dimensions, caching, or memory issues in MONAI code."
---

# MONAI DNA

Use this skill to keep MONAI code native, reusable, and validation-first.

## Workflow

1. Check [references/official-links.md](references/official-links.md) when you need to verify official MONAI APIs, tutorials, or papers.
2. Read only the references needed for the task:
   - [references/transforms.md](references/transforms.md) for data pipelines and dictionary transforms.
   - [references/networks.md](references/networks.md) for 2D or 3D model architecture patterns.
   - [references/generation.md](references/generation.md) for AutoEncoder, VAE, diffusion, and generative training loops.
   - [references/rectified-flow.md](references/rectified-flow.md) for flow matching or rectified flow implementations.
   - [references/segmentation.md](references/segmentation.md) for end-to-end segmentation workflows.
   - [references/datasets.md](references/datasets.md) when the task uses public datasets such as BraTS or MSD.
   - [references/troubleshooting.md](references/troubleshooting.md) for shape, interpolation, or memory pitfalls.
3. Before non-trivial implementation, write a concise technical spec that captures:
   - spatial dimensionality (`2D` or `3D`)
   - expected input and output tensor shapes
   - MONAI APIs, layers, transforms, or engines to reuse
   - existing workspace helpers or modules that should be imported instead of rewritten
   - a minimal validation plan
4. If the scope is already clear and the user asked for direct implementation, keep the spec short, surface key assumptions, and proceed.
5. After implementation, verify tensor flow with a `dummy_input` or similarly small shape test and run the repository's existing tests or linters when available.

## Coding Rules

- Search the current workspace before adding helpers. Reuse existing utilities instead of duplicating logic from MONAI or the project.
- Prefer MONAI factories such as `Act`, `Norm`, `Conv`, and `Pool` over hardcoded torch layers when dimension-safe abstractions exist.
- Prefer dictionary-mode transforms (`LoadImaged`, `Spacingd`, `RandFlipd`, and similar) for multimodal image and label pipelines.
- Keep functions compact. If implementation size grows, split configuration, model building, and validation into smaller units.
- Use explicit type annotations on public functions and helpers.
- Reach for `CacheDataset` or `PersistentDataset` when large 3D volumes would otherwise make iteration slow.
- For rectified flow, keep interpolation and target-velocity logic explicit and keep tensor reshaping easy to audit.

## Validation Checklist

- Confirm `spatial_dims` and keep it consistent across transforms, models, and inferers.
- Add a `dummy_input` or shape test that proves the tensor path end to end.
- Verify interpolation choices for labels versus images when resampling.
- Prefer lint-clean, type-annotated code that can pass the repository's configured checks.

## Source of Truth

If these references are insufficient, use available web or documentation tools to verify behavior against official MONAI sources linked from [references/official-links.md](references/official-links.md). Prefer primary sources over third-party summaries.
