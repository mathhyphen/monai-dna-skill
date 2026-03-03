---
name: monai-dna
description: Expert guidance for medical imaging AI development using the MONAI framework and advanced generative models. Use this skill when the user wants to: (1) Build generative models (Diffusion, VAE, Rectified Flow), (2) Design 2D/3D networks (UNet, SwinUNETR, ResNet), (3) Configure data transforms and pipelines, (4) Implement semantic segmentation workflows, or (5) Implement Flow Matching / Rectified Flow algorithms.
---

# MONAI DNA Skill

This skill provides high-density, expert-level coding patterns and best practices for the MONAI framework.

## How to use this skill

When working on MONAI or Rectified Flow tasks, follow this strict **Precision-First** workflow:

1. **Quick Links**: Check [references/official-links.md](references/official-links.md) for official documentation.
2. **Context Check**: If using public datasets (BraTS, MSD), read [references/datasets.md](references/datasets.md).
3. **Spec-Before-Code (MANDATORY)**: Before writing any implementation, generate a **Technical Specification** for user approval. It must include:
    - Input/Output Tensor Shapes (e.g., `(B, 1, 96, 96, 96)`).
    - Specific MONAI APIs to be used (no re-implementing existing logic).
    - A list of existing utility functions to reuse (check current workspace).
4. **Reference Implementation Patterns**: Use these for coding:
    - **Data Transforms**: [references/transforms.md](references/transforms.md)
    - **Network Architectures**: [references/networks.md](references/networks.md)
    - **Generative Models**: [references/generation.md](references/generation.md)
    - **Rectified Flow**: [references/rectified-flow.md](references/rectified-flow.md)
    - **Segmentation**: [references/segmentation.md](references/segmentation.md)
5. **Post-Implementation Validation**: Always include a `dummy_input` test to verify tensor flow.

## Core Principles for Precision Coding

### 1. Strict DRY (Don't Repeat Yourself)
- **Zero Redundancy**: Never duplicate logic found in MONAI or the current workspace.
- **Search-First**: Before writing any helper function, `/grep` for existing symbols. If found, import them.
- **Complexity Budget**: Favor concise, readable code. If a function exceeds 50 lines, refactor into modular components.

### 2. Native API Fidelity
- **Factory-Only**: Use `Act`, `Norm`, `Conv` factories. Hardcoded `nn.Conv2d` is forbidden for 3D tasks.
- **Standard Dictionary Mode**: Stick to `d` suffixed transforms for multimodal workflows.

### 3. Progressive Refinement
- **Atomic Commits**: Suggest small, focused changes over massive code dumps.
- **Refactor on Edit**: When modifying a file, proactively remove dead code or outdated comments.

## Mandatory Validation Steps
1. **Shape Check**: Provide a small `dummy_input` test script.
2. **2D/3D Confirmation**: Explicitly state the spatial dimensionality.
3. **Linter-Ready**: Code must pass `ruff` and `mypy` style checks (type annotations required).

## Advanced Search with MCP
If the provided references are insufficient, use the `google_search` or `fetch` MCP tools to query the official MONAI documentation site using the links in `official-links.md`.

## Core Principles to Follow

1. **Factory Pattern First**: Always prefer MONAI factories (`Act`, `Norm`, `Conv`) over hardcoded torch layers to maintain 2D/3D compatibility.
2. **Strict Type Annotations**: Use explicit type hints for all function signatures as per MONAI standards.
3. **Dictionary Mode**: Prefer Dictionary-based transforms (`LoadImaged`, etc.) for complex pipelines involving multiple modalities or labels.
4. **Performance**: Use `CacheDataset` or `PersistentDataset` for large 3D volumes.
5. **Flow Matching**: For Rectified Flow, use linear interpolation (`lerp`) and predict velocity vectors. Prefer `einops` for dimension manipulation.
