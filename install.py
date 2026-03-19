import argparse
import os
import shutil
import urllib.request
from pathlib import Path


REPO_URL = "https://raw.githubusercontent.com/mathhyphen/monai-dna-skill/main"
FILES = [
    "skills/monai-dna/references/transforms.md",
    "skills/monai-dna/references/networks.md",
    "skills/monai-dna/references/generation.md",
    "skills/monai-dna/references/rectified-flow.md",
    "skills/monai-dna/references/segmentation.md",
    "skills/monai-dna/references/official-links.md",
    "skills/monai-dna/references/datasets.md",
    "skills/monai-dna/references/visualize.md",
    "skills/monai-dna/references/troubleshooting.md",
    "skills/monai-dna/SKILL.md",
]


def install_claude() -> None:
    print("Starting MONAI DNA installation for Claude/OpenClaw...")
    os.makedirs(".claude/references", exist_ok=True)

    for file_path in FILES:
        dest = file_path.replace("skills/monai-dna/", ".claude/")
        print(f"  - Downloading {os.path.basename(file_path)}...")
        try:
            urllib.request.urlretrieve(f"{REPO_URL}/{file_path}", dest)
        except Exception as exc:
            print(f"  - Failed to download {file_path}: {exc}")

    if os.path.exists(".claude/SKILL.md"):
        shutil.copy(".claude/SKILL.md", ".clauderules")
        os.remove(".claude/SKILL.md")

    claude_md_content = """# MONAI Project Guidelines

## DNA References
- Transforms: .claude/references/transforms.md
- Networks: .claude/references/networks.md
- Generative: .claude/references/generation.md
- Segmentation: .claude/references/segmentation.md
- Rectified Flow: .claude/references/rectified-flow.md
"""
    with open("CLAUDE.md", "w", encoding="utf-8") as handle:
        handle.write(claude_md_content)

    print("\nInstallation complete. Claude/OpenClaw files were written locally.")


def install_codex() -> None:
    print("Starting MONAI DNA installation for Codex...")
    source_dir = Path(__file__).resolve().parent / "skills" / "monai-dna"
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    target_dir = codex_home / "skills" / "monai-dna"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    print(f"\nInstallation complete. Skill copied to: {target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the MONAI DNA skill.")
    parser.add_argument(
        "target",
        choices=("codex", "claude"),
        nargs="?",
        default="codex",
        help="Install for Codex or Claude/OpenClaw.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.target == "claude":
        install_claude()
    else:
        install_codex()
