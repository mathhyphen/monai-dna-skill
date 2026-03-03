# MONAI DNA: Expert Skill for Medical AI Development

[English Version Below](#english-version)

## 简介 (Introduction)

**MONAI DNA** 是一个专为 OpenClaw 和 Claude Code 优化的专家级 AI 技能。它将医学影像 AI 框架 **MONAI** 的核心精髓与前沿的 **Rectified Flow (流匹配)** 生成算法相结合，为 AI 助手注入了深层的编码模式、最佳实践和自动避坑指南。

通过此技能，AI 助手不再只是简单地调用 API，而是能够像资深架构师一样，编写出符合原生风格、高性能且可自我验证的医学 AI 代码。

## 核心特性

- 🧬 **高密度 DNA 注入**：涵盖数据变换 (Transforms)、网络架构 (Networks)、生成模型 (VAE/Diffusion) 和语义分割。
- 🌊 **前沿算法支持**：融合了 `lucidrains` 风格的 Rectified Flow 实现模式。
- 📡 **官方雷达**：内置官方 API、教程和论文的直达索引。
- 🛡️ **自动避坑**：针对 3D 维度陷阱、显存优化 (OOM) 和标签插值错误提供专门的知识库。
- ✅ **强制验证**：要求 AI 在输出代码后自动生成 `shape validation` 测试脚本。

## 目录结构

```
monai-dna/
├── SKILL.md                 # 核心调度逻辑与验证准则
└── references/
    ├── transforms.md        # 数据变换 DNA (Array/Dict 模式)
    ├── networks.md          # 2D/3D 工厂模式与网络架构
    ├── generation.md        # VAE 与 Diffusion 训练模板
    ├── rectified-flow.md    # 流匹配算法实战
    ├── segmentation.md      # 端到端分割工作流
    ├── official-links.md    # 官方文档与教程雷达
    ├── datasets.md          # BraTS, MSD 等数据集配置
    ├── visualize.md         # 3D 可视化与 GIF 导出
    └── troubleshooting.md   # 避坑指南
```

## 安装方法

1. 将 `monai-dna` 文件夹复制到你的 OpenClaw 技能目录：
   `~/.openclaw/skills/` (或 OpenClaw 安装路径下的 `node_modules/openclaw/skills/`)
2. 重启 OpenClaw Gateway。
3. 在 Claude Code 或 OpenClaw 中即可直接使用。

---

<a name="english-version"></a>
# MONAI DNA: Expert Skill for Medical AI Development (English)

## Overview

**MONAI DNA** is an expert-level AI skill optimized for OpenClaw and Claude Code. It infuses the core essence of the **MONAI** medical imaging framework and cutting-edge **Rectified Flow (Flow Matching)** algorithms into your AI assistant.

This skill transforms your AI from a simple API caller into a senior architect capable of writing native-style, high-performance, and self-validating medical AI code.

## Key Features

- 🧬 **High-Density DNA**: Covers Transforms, Networks, Generative Models (VAE/Diffusion), and Segmentation.
- 🌊 **Advanced Algorithms**: Integrated `lucidrains`-style Rectified Flow patterns.
- 📡 **Official Radar**: Direct indexing for official APIs, tutorials, and papers.
- 🛡️ **Troubleshooting**: Specialized knowledge base for 3D dimension pitfalls, OOM optimization, and label interpolation.
- ✅ **Mandatory Validation**: Enforces the AI to generate `shape validation` test scripts alongside the code.

## Installation

1. Copy the `monai-dna` folder to your OpenClaw skills directory.
2. Restart the OpenClaw Gateway.
3. Start coding with your enhanced AI assistant!
