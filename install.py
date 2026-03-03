import os
import shutil
import urllib.request
from pathlib import Path

def install():
    repo_url = "https://raw.githubusercontent.com/mathhyphen/monai-dna-skill/main"
    files = [
        "skills/monai-dna/references/transforms.md",
        "skills/monai-dna/references/networks.md",
        "skills/monai-dna/references/generation.md",
        "skills/monai-dna/references/rectified-flow.md",
        "skills/monai-dna/references/segmentation.md",
        "skills/monai-dna/references/official-links.md",
        "skills/monai-dna/references/datasets.md",
        "skills/monai-dna/references/visualize.md",
        "skills/monai-dna/references/troubleshooting.md",
        "skills/monai-dna/SKILL.md"
    ]

    print("🚀 Starting MONAI DNA installation...")

    # 1. Create directory structure
    os.makedirs(".claude/references", exist_ok=True)

    # 2. Download files
    for f in files:
        dest = f.replace("skills/monai-dna/", ".claude/")
        print(f"  - Downloading {os.path.basename(f)}...")
        try:
            urllib.request.urlretrieve(f"{repo_url}/{f}", dest)
        except Exception as e:
            print(f"  ❌ Failed to download {f}: {e}")

    # 3. Create .clauderules (from SKILL.md)
    if os.path.exists(".claude/SKILL.md"):
        shutil.copy(".claude/SKILL.md", ".clauderules")
        os.remove(".claude/SKILL.md")

    # 4. Create CLAUDE.md
    claude_md_content = """# MONAI Project Guidelines\n\n## 🧬 DNA References\n- Transforms: .claude/references/transforms.md\n- Networks: .claude/references/networks.md\n- Generative: .claude/references/generation.md\n- Segmentation: .claude/references/segmentation.md\n- Rectified Flow: .claude/references/rectified-flow.md\n"""
    
    with open("CLAUDE.md", "w", encoding="utf-8") as f:
        f.write(claude_md_content)

    print("\n✅ Installation complete! Your project is now MONAI-aware.")

if __name__ == "__main__":
    install()