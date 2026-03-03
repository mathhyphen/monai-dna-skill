# MONAI Project Guidelines

## 🧬 DNA References
Project core patterns are stored in .claude/references/ and skills/monai-dna/references/.
Always refer to these documents to maintain native MONAI coding style:
- Transforms: .claude/references/transforms.md
- Networks: .claude/references/networks.md
- Generative: .claude/references/generation.md
- Segmentation: .claude/references/segmentation.md
- Rectified Flow: .claude/references/rectified-flow.md

## 🛠 Commands
- Test: `pytest tests/`
- Format: `black . && ruff check --fix`
