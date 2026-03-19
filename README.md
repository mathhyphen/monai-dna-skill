# MONAI DNA: 面向医学 AI 开发的专家技能

[English Version Below](#english-version)

## 简介

**MONAI DNA** 是一个面向 AI 编程助手的专家技能，用来指导其编写医学深度学习项目，尤其是医学影像项目与生成式项目。

这个 skill 的核心目标不是让 AI “大概会用 MONAI”，而是让它在写项目时尽量遵循两套源头约束：

- **项目外壳遵循 MONAI 原生风格**：包括目录组织、数据管线、transforms、networks、training loop、inferers、metrics 和常见工程模式。
- **生成逻辑遵循参考生成仓库**：尤其是 VAE、Diffusion、Rectified Flow、Flow Matching 等任务中的目标定义、scheduler、时间步处理、采样流程和损失设计。

换句话说，这个 skill 不是为了生成“能跑就行”的医学 AI 代码，而是为了让 AI 写出**更接近 MONAI 原生项目风格、同时对生成模型实现更忠实**的代码。

## 适用场景

当你希望 AI 助手执行以下任务时，可以使用这个 skill：

- 搭建 MONAI 风格的医学影像项目骨架
- 编写或重构 2D / 3D 数据 transforms 和 dictionary pipelines
- 设计 UNet、SwinUNETR、ResNet 等网络结构
- 编写医学图像分割、重建、生成、推理与评估代码
- 实现或改造 VAE、Diffusion、Rectified Flow、Flow Matching
- 排查 shape、spatial dims、插值、缓存、OOM 等 MONAI 常见问题

## 核心原则

### 1. 严格对齐 MONAI 风格

- 优先复用 MONAI 原生 API，而不是手写一套通用 PyTorch 模板
- 优先使用 dictionary transforms 处理图像与标签
- 优先使用 MONAI 的 factories、datasets、inferers、metrics、losses 和工程习惯

### 2. 生成逻辑忠于参考实现

- 生成模型项目不仅要“长得像 MONAI”
- 更重要的是核心生成逻辑要忠于参考仓库
- 包括预测目标、噪声构造、flow 定义、scheduler、sampling loop、loss contract 等

### 3. 先规格，后实现

在编写非平凡实现之前，AI 应先明确：

- 输入输出 tensor shape
- `spatial_dims`
- 需要复用的 MONAI 模块
- 需要保持一致的生成逻辑
- 最小验证方式

### 4. 强制最小验证

产出代码后，至少应补上：

- `dummy_input` shape 检查
- 一次最小训练步验证
- 一次最小采样或推理验证

## 仓库结构

```text
monai-dna-skill/
├── AGENTS.md
├── CLAUDE.md
├── install.py
├── README.md
└── skills/
    └── monai-dna/
        ├── SKILL.md
        ├── agents/
        │   └── openai.yaml
        └── references/
            ├── datasets.md
            ├── generation.md
            ├── networks.md
            ├── official-links.md
            ├── rectified-flow.md
            ├── segmentation.md
            ├── transforms.md
            ├── troubleshooting.md
            └── visualize.md
```

## 参考资料说明

- `transforms.md`
  MONAI 数据预处理、后处理与 dictionary transform 模式
- `networks.md`
  MONAI 2D / 3D 网络与工厂模式
- `generation.md`
  MONAI GenerativeModels 风格的 VAE / Diffusion 参考
- `rectified-flow.md`
  lucidrains 风格的 Rectified Flow / Flow Matching 核心模式
- `segmentation.md`
  医学影像分割工作流参考
- `official-links.md`
  MONAI、Project-MONAI GenerativeModels 与相关论文入口
- `troubleshooting.md`
  常见 shape、OOM、插值和维度问题

## 安装与使用

### 在 Codex 中使用

方式 1：直接复制 skill 目录

将 `skills/monai-dna/` 复制到：

```text
$CODEX_HOME/skills/monai-dna/
```

如果没有设置 `CODEX_HOME`，通常可使用：

```text
~/.codex/skills/monai-dna/
```

方式 2：使用安装脚本

```bash
python install.py codex
```

### 在 Claude Code / OpenClaw 中使用

```bash
python install.py claude
```

或手动将 `monai-dna` 复制到 OpenClaw skill 目录后重启对应环境。

## 这个 skill 特别适合什么

如果你希望 AI 在写下面这些项目时别“自由发挥过头”，这个 skill 会很有用：

- 3D 脑肿瘤分割
- 医学图像重建
- 3D Diffusion 生成
- Latent Diffusion 医学影像项目
- Rectified Flow / Flow Matching 医学生成项目
- 基于 MONAI 的研究原型或工程化训练脚本

---

<a name="english-version"></a>

# MONAI DNA: Expert Skill for Medical AI Development

## Overview

**MONAI DNA** is an expert skill for AI coding assistants that build medical deep learning projects, especially medical imaging and generative modeling projects.

The goal is not merely to make the assistant "use MONAI somehow." The goal is to make it follow two strong constraints:

- **Keep the project shell MONAI-native**: project layout, transforms, networks, training and inference flows, inferers, metrics, and common engineering patterns should stay close to MONAI style.
- **Keep the generative logic faithful to the reference repositories**: especially for VAE, Diffusion, Rectified Flow, and Flow Matching objectives, schedulers, timestep handling, sampling loops, and loss design.

This skill is meant to produce code that is not only functional, but also **structurally aligned with MONAI and implementation-faithful for generative modeling**.

## Good Fits

Use this skill when the AI assistant needs to:

- scaffold a MONAI-style medical imaging project
- build or refactor 2D / 3D transforms and dictionary pipelines
- design UNet, SwinUNETR, ResNet, or related architectures
- implement segmentation, reconstruction, generation, inference, and evaluation workflows
- implement or adapt VAE, Diffusion, Rectified Flow, or Flow Matching
- troubleshoot shape, interpolation, caching, dimensionality, or memory issues

## Core Principles

### 1. Stay MONAI-native

- Prefer MONAI APIs over ad hoc generic PyTorch rewrites
- Prefer dictionary transforms for image and label workflows
- Reuse MONAI factories, datasets, inferers, losses, metrics, and project patterns

### 2. Preserve reference generative logic

- A generative project should not merely "look like MONAI"
- Its core generative logic should remain faithful to the reference implementation
- This includes objectives, scheduler contracts, noise or flow construction, sampling loops, and loss definitions

### 3. Write specs before non-trivial code

Before implementation, the assistant should clarify:

- input and output tensor shapes
- `spatial_dims`
- MONAI modules to reuse
- generative logic that must remain unchanged
- minimal validation steps

### 4. Enforce minimal validation

Generated code should include at least:

- a `dummy_input` shape check
- one minimal training-step validation
- one minimal sampling or inference validation

## Repository Layout

```text
monai-dna-skill/
├── AGENTS.md
├── CLAUDE.md
├── install.py
├── README.md
└── skills/
    └── monai-dna/
        ├── SKILL.md
        ├── agents/
        │   └── openai.yaml
        └── references/
            ├── datasets.md
            ├── generation.md
            ├── networks.md
            ├── official-links.md
            ├── rectified-flow.md
            ├── segmentation.md
            ├── transforms.md
            ├── troubleshooting.md
            └── visualize.md
```

## Installation

### Use in Codex

Option 1: copy the skill directory manually

```text
$CODEX_HOME/skills/monai-dna/
```

If `CODEX_HOME` is not set, the common location is:

```text
~/.codex/skills/monai-dna/
```

Option 2: use the installer

```bash
python install.py codex
```

### Use in Claude Code / OpenClaw

```bash
python install.py claude
```

Or copy the `monai-dna` skill folder into the OpenClaw skill directory manually.

## Best Use Cases

This skill is especially useful when you want the AI assistant to stay disciplined while building:

- 3D brain tumor segmentation projects
- medical image reconstruction systems
- 3D diffusion generation projects
- latent diffusion pipelines for medical imaging
- rectified flow or flow matching projects
- research prototypes and production-leaning MONAI training code