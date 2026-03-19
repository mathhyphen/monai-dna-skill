---
name: monai-dna
description: "Expert guidance for building medical deep learning projects in a strict MONAI-native style. Use this skill when Codex needs to: (1) structure a medical imaging codebase, training loop, or inference pipeline to match MONAI conventions, (2) build or refactor MONAI transforms and dictionary pipelines, (3) design 2D or 3D networks such as UNet, SwinUNETR, and ResNet, (4) implement segmentation, reconstruction, diffusion, VAE, or rectified flow workflows with generation logic aligned to the referenced MONAI GenerativeModels or lucidrains-style repositories, or (5) troubleshoot tensor shapes, spatial dimensions, caching, interpolation, or memory issues in MONAI code."
---

# MONAI DNA

Use this skill to keep medical AI code aligned with MONAI's project style, API choices, and validation habits. For generative work, keep the project shell MONAI-native and keep the generation logic faithful to the referenced generation repositories instead of inventing a fresh pattern.

## Priority Order

1. Treat MONAI APIs, tutorials, and project structure as the default source of truth for medical imaging code organization.
2. For generative tasks, treat the referenced generation repositories as the source of truth for diffusion, VAE, rectified flow, or flow-matching logic.
3. Reconcile the two by keeping data, transforms, loaders, engines, inferers, metrics, and deployment flow MONAI-native while preserving the core generation objective, scheduler, sampling, and loss logic from the reference generation implementation.
4. Do not replace an existing MONAI or reference-repo pattern with generic PyTorch code unless there is a concrete reason and you state it.

## Workflow

1. Check [references/official-links.md](references/official-links.md) to identify the authoritative MONAI or generation source before you design the solution.
2. Read only the references needed for the task:
   - [references/transforms.md](references/transforms.md) for data pipelines and dictionary transforms.
   - [references/networks.md](references/networks.md) for 2D or 3D model architecture patterns.
   - [references/generation.md](references/generation.md) for MONAI-style AutoEncoder, VAE, diffusion, and generative training loops.
   - [references/rectified-flow.md](references/rectified-flow.md) for lucidrains-style flow matching or rectified flow implementations.
   - [references/segmentation.md](references/segmentation.md) for end-to-end segmentation workflows.
   - [references/datasets.md](references/datasets.md) when the task uses public datasets such as BraTS or MSD.
   - [references/troubleshooting.md](references/troubleshooting.md) for shape, interpolation, or memory pitfalls.
3. Before non-trivial implementation, write a concise technical spec that captures:
   - spatial dimensionality (`2D` or `3D`)
   - expected input and output tensor shapes
   - the MONAI modules, APIs, and project patterns to reuse
   - the exact generation reference logic to preserve if the task is generative
   - existing workspace helpers or modules that should be imported instead of rewritten
   - what must stay MONAI-native and what must stay reference-faithful
   - a minimal validation plan
4. If the scope is already clear and the user asked for direct implementation, keep the spec short, surface key assumptions, and proceed.
5. After implementation, verify tensor flow with a `dummy_input` or similarly small shape test and run the repository's existing tests or linters when available.

## Coding Rules

- Search the current workspace before adding helpers. Reuse existing utilities instead of duplicating logic from MONAI or the project.
- Match MONAI naming, file organization, transform style, and engine usage whenever a MONAI-native pattern exists.
- Prefer MONAI factories such as `Act`, `Norm`, `Conv`, and `Pool` over hardcoded torch layers when dimension-safe abstractions exist.
- Prefer dictionary-mode transforms (`LoadImaged`, `Spacingd`, `RandFlipd`, and similar) for multimodal image and label pipelines.
- For generative projects, preserve the reference repository's objective, scheduler contract, timestep handling, noise or flow parameterization, and sampling loop.
- When blending MONAI with a reference generation repo, adapt interfaces around the core logic instead of rewriting the core logic to fit a generic template.
- Keep functions compact. If implementation size grows, split configuration, model building, and validation into smaller units.
- Use explicit type annotations on public functions and helpers.
- Reach for `CacheDataset` or `PersistentDataset` when large 3D volumes would otherwise make iteration slow.
- For rectified flow, keep interpolation, time conditioning, target-velocity logic, and tensor reshaping explicit and easy to audit.

## Generative Project Contract

- Use [references/generation.md](references/generation.md) when the project should follow MONAI GenerativeModels structure and APIs.
- Use [references/rectified-flow.md](references/rectified-flow.md) when the project should follow lucidrains-style rectified flow or flow-matching logic.
- Keep MONAI responsible for medical-imaging-facing concerns such as transforms, datasets, inferers, spatial metadata, sliding-window inference, caching, and evaluation.
- Keep the reference generation repository responsible for the modeling contract: what the network predicts, how noise or flow is formed, how losses are computed, and how sampling is integrated.
- If the user asks for a hybrid design, say explicitly which parts come from MONAI and which parts come from the generation reference.

## Validation Checklist

- Confirm `spatial_dims` and keep it consistent across transforms, models, and inferers.
- Add a `dummy_input` or shape test that proves the tensor path end to end.
- For generative models, verify one training-step contract and one sampling-step contract, not just the model forward shape.
- Verify interpolation choices for labels versus images when resampling.
- Verify that the implemented objective matches the intended reference, such as reconstruction target, epsilon prediction, velocity prediction, or clean-sample prediction.
- Prefer lint-clean, type-annotated code that can pass the repository's configured checks.

## Source of Truth

If these references are insufficient, verify behavior against the primary sources linked from [references/official-links.md](references/official-links.md). Prefer MONAI docs, MONAI tutorials, Project-MONAI GenerativeModels, and the linked lucidrains implementation over third-party summaries.
