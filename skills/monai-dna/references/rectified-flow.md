<!-- PAGE_ID: rectified-flow-dna -->
# Rectified Flow & Flow Matching: Lucidrains Patterns

## 1. 核心概念 (Core Logic)
Rectified Flow 通过直线插值将噪声 $x_0$ 转变为数据 $x_1$。
- **Linear Interpolation**: `noised = noise.lerp(data, t)` 其中 $t \in [0, 1]$。
- **Target Flow**: 模型预测的是从噪声到数据的“速度”向量，即 $v = x_1 - x_0$ ([rectified_flow.py:350](https://gitlab.com/lucidrains/rectified-flow-pytorch/-/blob/main/rectified_flow_pytorch/rectified_flow.py#L350))。

## 2. 代码实现模式 (Coding Patterns)
`lucidrains` 风格的 PyTorch 代码特点：
- **大量使用 einops**: `rearrange`, `repeat`, `reduce` 进行维度变换。
- **Conditioning**: 时间步通过 `SinusoidalPosEmb` 注入。
- **Predict Objective**: 支持预测 `flow` (速度), `noise` (噪声), 或 `clean` (原始数据)。

```python
# 预测 Flow 的核心逻辑
padded_times = t.reshape(-1, *((1,) * (noised.ndim - 1)))
if predict == 'flow':
    flow = model_output
elif predict == 'noise':
    flow = (noised - noise) / padded_times.clamp_min(eps)
```

## 3. 采样与 ODE
Rectified Flow 在推理时通过求解常微分方程 (ODE) 进行：
- **Solver**: 默认使用 `torchdiffeq.odeint`，支持 `midpoint`, `rk4` 等方法。
- **Fast Sampling**: 使用 `DDIMScheduler` 可以将 1000 步采样压缩至 50 步。

## 4. 损失函数与优化 (Advanced Losses)
| 损失函数 | 适用场景 | 说明 |
|----------|----------|---------|
| `LPIPSLoss` | 感知质量 | 使用 VGG 特征确保生成图像在视觉上接近真实数据 |
| `PseudoHuberLoss` | 鲁棒性 | 相比 MSE，对离散值更鲁棒 |
| `ConsistencyLoss` | 蒸馏 | 用于训练 Consistency Models，实现单步生成 |

## 5. 结合 MONAI 的实战建议
- **3D 扩散**: 建议使用 MONAI 的 `SwinUNETR` 作为 `RectifiedFlow` 的 backbone 处理器。
- **数据流**: 使用 MONAI 的 `Dictionary Transforms` 进行归一化后，再喂入 `RectifiedFlow`。
