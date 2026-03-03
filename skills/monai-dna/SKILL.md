---
name: monai-dna
description: Expert guidance for medical imaging AI development using the MONAI framework and advanced generative models. Use this skill when the user wants to: (1) Build generative models (Diffusion, VAE, Rectified Flow), (2) Design 2D/3D networks (UNet, SwinUNETR, ResNet), (3) Configure data transforms and pipelines, (4) Implement semantic segmentation workflows, or (5) Implement Flow Matching / Rectified Flow algorithms.
---

# MONAI DNA Skill

This skill provides high-density, expert-level coding patterns and best practices for the MONAI framework.

## How to use this skill

When working on MONAI or Rectified Flow tasks, follow this order of reference:
1. **Quick Links**: Check [references/official-links.md](references/official-links.md) for official documentation and latest tutorials.
2. **Context Check**: If using public datasets (BraTS, MSD), read [references/datasets.md](references/datasets.md).
3. **Implementation Patterns**: Reference the specialized documents for coding:
    - **Data Transforms**: [references/transforms.md](references/transforms.md)
    - **Network Architectures**: [references/networks.md](references/networks.md)
    - **Generative Models**: [references/generation.md](references/generation.md)
    - **Rectified Flow**: [references/rectified-flow.md](references/rectified-flow.md)
    - **Segmentation**: [references/segmentation.md](references/segmentation.md)
    - **Visualization**: [references/visualize.md](references/visualize.md)
4. **Safety Check**: Always read [references/troubleshooting.md](references/troubleshooting.md) before finalizing any 3D imaging code.

## Mandatory Validation Steps
Whenever you generate a network or a full pipeline:
1. **Shape Check**: Provide a small `dummy_input` test script to verify that the output shape matches expectations.
2. **2D/3D Confirmation**: Explicitly state if the implementation is 2D or 3D.

## Advanced Search with MCP
If the provided references are insufficient, use the `google_search` or `fetch` MCP tools to query the official MONAI documentation site using the links in `official-links.md`.

## Core Principles to Follow

1. **Factory Pattern First**: Always prefer MONAI factories (`Act`, `Norm`, `Conv`) over hardcoded torch layers to maintain 2D/3D compatibility.
2. **Strict Type Annotations**: Use explicit type hints for all function signatures as per MONAI standards.
3. **Dictionary Mode**: Prefer Dictionary-based transforms (`LoadImaged`, etc.) for complex pipelines involving multiple modalities or labels.
4. **Performance**: Use `CacheDataset` or `PersistentDataset` for large 3D volumes.
5. **Flow Matching**: For Rectified Flow, use linear interpolation (`lerp`) and predict velocity vectors. Prefer `einops` for dimension manipulation.
