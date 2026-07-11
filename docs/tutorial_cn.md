# Seismic Benchmark 教程

`seismic-benchmark-code` 仓库的实战指南。

## 目录

- [第 1 章：简介](#chapter-1-introduction)
- [第 2 章：项目结构与核心概念](#chapter-2-project-structure-and-core-concepts)
- [第 3 章：快速开始](#chapter-3-quick-start)
- [第 4 章：完整端到端示例 — 数据与预处理](#chapter-4-complete-end-to-end-example-data-and-preprocessing)
- [第 5 章：扩展到其他任务](#chapter-5-extending-to-other-tasks)
- [第 6 章：自定义与扩展库](#chapter-6-customizing-and-extending-the-library)
- [第 7 章：故障排查与快速参考](#chapter-7-troubleshooting-and-quick-reference)

---

<a id="chapter-1-introduction"></a>
## 第 1 章：简介

<a id="1-1-what-this-repository-is"></a>
### 1.1 本仓库是什么

`seismic-benchmark-code` 是一个面向勘探地球物理（地震）数据处理的 PyTorch 基准测试模板。它支持 SEG-Y、NPY 或 MAT 格式数据体上的插值、去噪与有监督恢复等任务。代码库围绕**注册表 + 工厂模式**构建：模型、数据集、损失与指标均作为插件注册，并通过 YAML 配置文件选择，因此训练脚本保持与组件无关。

<a id="1-2-who-this-tutorial-is-for"></a>
### 1.2 本教程适用读者

本教程同时面向：

- **初学者**：希望在不阅读每个源文件的情况下运行首个地震去噪实验。
- **有经验的使用者**：需要 CLI 命令、YAML 字段和注册表模式的快速参考。

你不需要深厚的地震学背景即可跟随本示例，但具备一些深度学习概念会更有帮助。

<a id="1-3-prerequisites"></a>
### 1.3 前置知识

开始之前，你应熟悉以下内容：

- 基础 PyTorch（`nn.Module`、`DataLoader`、训练循环）。
- YAML 语法与命令行标志。
- NumPy 数组形状与索引。

可选但有帮助的：

- 对叠前地震炮集（SEG-Y、FFID、道头）有一定了解。**FFID**（野外记录号，Field Record ID）是 SEG-Y 道头中用于标识一个炮集的值。
- 用于训练的 CUDA GPU（CPU 训练可行但较慢）。

<a id="1-4-dependencies"></a>
### 1.4 依赖

目前尚无统一的 `requirements.txt` 或 `pyproject.toml`。请手动安装以下包：

- `torch`
- `numpy`
- `matplotlib`
- `pyyaml`
- `segyio`
- `scipy`

可通过以下命令安装：

```bash
pip install torch numpy matplotlib pyyaml segyio scipy
```

---

<a id="chapter-2-project-structure-and-core-concepts"></a>
## 第 2 章：项目结构与核心概念

<a id="2-1-directory-overview"></a>
### 2.1 目录概览

仓库按职责划分为若干自包含目录，每个目录只承担单一职责。

| 目录 | 用途 |
|------|------|
| `tools/` | 数据工具：I/O（`array_io.py`、`segy_read.py`）、预处理（`preprocessing.py`）和分块（`patching.py`）。 |
| `model/` | 神经网络定义和 `MODEL_REGISTRY`。任务级子包负责注册各自的模型；共享注册原语位于 `model/registry.py`；任务特定模型文件位于 `model/<task>/`。 |
| `utils/` | 训练基础设施：数据集、损失、指标、可视化、日志、优化器/调度器构建器、训练/评估循环和检查点 I/O。 |
| `configs/` | 每个实验对应一个 YAML 文件。超参数绝不硬编码在源码中。 |
| `scripts/` | 训练（`train_*.py`）和推理（`inference_*.py`）的 CLI 入口，以及 Bash 启动脚本（`*.sh`）。 |
| `results/` | 实验输出：检查点、日志、CSV 和 PNG。该目录被 Git 忽略。 |
| `memory/` | 项目记忆：设计决策、更新日志、技术记录和研究参考。 |

<a id="2-2-registry-factory-pattern"></a>
### 2.2 注册表 + 工厂模式

项目采用**注册表 + 工厂模式**，以便在不修改训练脚本的情况下添加新组件。

每类组件都有自己的注册表和装饰器：

| 组件 | 注册表文件 | 装饰器 | 工厂函数 |
|------|-----------|--------|---------|
| 模型 | `model/registry.py` | `@register_model("name")` | `build_model(cfg)` |
| 数据集 | `utils/datasets.py` | `@register_dataset("name")` | `build_dataset(cfg)` |
| 损失 | `utils/losses.py` | `@register_loss("name")` | `build_loss(cfg)` |
| 指标 | `utils/metrics.py` | `@register_metric("name")` | `build_metrics(cfg)` |

YAML 中每个可插拔块都采用相同格式：

```yaml
component:
  type: registered_name
  params:
    key: value
```

例如，UNet 模型声明为：

```yaml
model:
  type: unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4
```

注册自定义模型的伪代码：

```python
# model/random_noise_suppression/my_net.py
import torch.nn as nn
from model.registry import register_model

@register_model("my_net")
class MyNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
```

```python
# model/random_noise_suppression/__init__.py
from . import unet      # noqa: F401
from . import my_net    # noqa: F401
```

```yaml
# config snippet
model:
  type: my_net
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
```

注意：模型注册需要在 `model/<task>/__init__.py` 中添加 `from . import <file>  # noqa: F401`，以便装饰器在导入时执行。顶层 `model/__init__.py` 只暴露注册原语（`MODEL_REGISTRY`、`register_model`、`build_model`）和占位模型，它**不会**导入每一个具体模型文件。任务特定模型只有在导入对应的任务子包时才会注册，例如 `from model.random_noise_suppression import build_model`。

<a id="2-3-component-agnostic-training-scripts"></a>
### 2.3 与组件无关的训练脚本

`scripts/train.py` 刻意保持与组件无关。它只解析 CLI 参数、加载 YAML 配置并连接工厂函数。它绝不直接导入具体模型、数据集、损失或指标。

任务特定脚本（如 `scripts/random_noise_suppression/train_denoise_unet.py`）遵循相同模式：解析 CLI、加载配置、构建组件并运行任务特定流程。所有具体行为由 YAML 配置驱动。

---

<a id="chapter-3-quick-start"></a>
## 第 3 章：快速开始

本节展示如何通过几个命令训练并评估一个随机噪声压制模型。示例使用 SEG C3 45 炮合成数据集。

<a id="3-1-data-placement"></a>
### 3.1 数据放置

默认配置指向：

```
data/SEG_45Shot_shots1-9.sgy
```

相对于仓库根目录。本教程假设 SEG-Y 文件位于 `data/SEG_45Shot_shots1-9.sgy`。如果你的文件在其他位置，请在训练前更新 `configs/random_noise_suppression/denoise_unet.yaml`（或其他任何配置）中的 `data.segy.path` 值。

<a id="3-2-train-a-model-in-one-command"></a>
### 3.2 单命令训练模型

```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

脚本读取 YAML 配置，从注册表构建模型/损失/数据集/指标，并运行训练循环。

<a id="3-3-run-inference-in-one-command"></a>
### 3.3 单命令运行推理

训练完成后，使用验证集最佳检查点运行推理：

```bash
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5
```

你可以通过命令行覆盖噪声类型、SNR、批次大小、设备等推理设置，无需修改 YAML 文件。

> **警告：** 默认配置中的 `inference.checkpoint` 可能指向过期或用户特定的路径。请显式传入 `--checkpoint`，或修改配置使 `inference.checkpoint` 指向你自己训练产生的 `results/<exp>/checkpoints/best.pt`。缺失的检查点路径或指向其他目录的路径会抛出 `FileNotFoundError`，或静默加载错误权重。

> **注意：** 上述推理命令只展示了最常用的标志。`--device` 和 `--batch-size` 也可以覆盖；完整标志列表见第 4.5 节。

<a id="3-4-expected-output-tree"></a>
### 3.4 预期输出目录树

训练完成后，实验目录包含：

```
results/random_noise/random_noise_unet_base/
├── checkpoints/
│   ├── epoch_0020.pt
│   ├── epoch_0040.pt
│   ├── ...
│   ├── epoch_0200.pt
│   └── best.pt
├── logs/
│   ├── train_log.txt
│   ├── loss_history.csv
│   ├── metrics_history.csv
│   ├── loss_curve.png
│   └── metrics_curve.png
├── visualizations/
│   └── epoch_*.png
└── config_used.yaml
```

检查点每隔 `ckpt_interval` 轮保存一次，直到 `epochs` 轮。使用默认配置（`epochs: 200`，`ckpt_interval: 20`）时，最终的周期性检查点是 `epoch_0200.pt`。

推理完成后，输出目录包含：

```
results/random_noise/random_noise_unet_base/inference/
├── inference.log
├── metrics_summary.json
├── metrics_per_shot.csv
├── visualizations/
│   └── shot_*.png
└── npy/                    # 仅在传入 --save-npy 时生成
    ├── input_shots.npy
    ├── pred_shots.npy
    └── target_shots.npy
```

<a id="3-5-note-on-execution"></a>
### 3.5 执行说明

本教程不会自动运行训练与推理脚本。请把上述命令复制到终端并手动执行。训练时间取决于 GPU、批次大小和轮数，可能从几分钟到几小时不等。

---

<a id="chapter-4-complete-end-to-end-example-data-and-preprocessing"></a>
## 第 4 章：完整端到端示例 — 数据与预处理

本章逐步讲解随机噪声压制示例中的数据与预处理阶段。这里使用的 YAML 配置是 `configs/random_noise_suppression/denoise_unet.yaml`。

<a id="4-1-data-format-and-loading"></a>
### 4.1 数据格式与加载

#### SEG-Y 读取

仓库通过 `tools/segy_read.py` 读取 SEG-Y 文件。用于规则炮集的函数是：

```python
read_regular_shots(path, traces_per_shot, time_downsample=1)
```

它返回形状为 `(n_shots, n_traces, n_time)` 的 NumPy 数组，以及一个道头字典。通过检查每个炮集切片是否共享同一个 `FieldRecord`（FFID）道头值来验证规则性。

> **什么是炮集？** 炮集是由接收点从一个震源（一炮）记录到的地震道组成的二维图像。**地震道**是一个接收点的记录。**FFID**（Field Record ID）是 SEG-Y 道头中标识一个炮集的值。

对于本教程使用的 SEG C3 45 炮数据体：

- `n_shots = 9`
- `traces_per_shot = 201`
- `dt = 0.008` 秒（8 毫秒采样间隔）

加载后的数组形状为 `(9, 201, n_time)`。配置中的 `dt` 值是训练脚本用于球面扩散校正和频率估计的值；它应与源数据的时间采样间隔一致。一些旧文档可能为同一数据体列出不同的 `dt`，因此请始终使用配置中与你文件匹配的值。

#### 在 YAML 中切换为 NPY 或 MAT

配置中的 `data` 块一次只支持一种格式。默认使用 SEG-Y：

```yaml
data:
  segy:
    path: data/SEG_45Shot_shots1-9.sgy  # 如果文件在其他位置，请更新路径
    traces_per_shot: 201
    time_downsample: 1
```

若要改用 NPY 或 MAT 文件，请取消对应块的注释并注释掉 SEG-Y 块：

```yaml
data:
  # npy:
  #   path: /path/to/SEG_45Shot_shots1-9.npy
  # mat:
  #   path: /path/to/SEG_45Shot_shots1-9.mat
  #   key: shots
```

三个加载器都返回形状为 `(n_shots, n_traces, n_time)` 的 `float32` 数据体。MAT 文件通过 `scipy.io.loadmat` 加载；如果文件中没有配置的 `key`，加载器会抛出 `KeyError`，并列出可用的变量名。

`key` 只对 MAT 文件必填，用于告诉加载器哪个 MATLAB 变量包含数据体。如果省略，加载器会回退到找到的第一个数组变量（可能并非你想要的）。要查看可用变量名，请运行：

```bash
python -c "import scipy.io; print(list(scipy.io.loadmat('data/SEG_45Shot_shots1-9.mat').keys()))"
```

找到值为 `(n_shots, n_traces, n_time)` 形状的变量，并将其用作 `key`。

#### 形状约定

在本仓库中，地震数据体统一存储为：

```
(n_shots, n_traces, n_time)
```

其中：

- `n_shots` 是炮集（或 FFID）数量。
- `n_traces` 是每炮的接收道数（一道即一个接收点记录）。
- `n_time` 是时间采样点数。

当二维卷积模型对从单个炮集中提取的分块进行操作时，分块形状为 `(1, patch_trace, patch_time)`，其中前导 `1` 是通道维度，`(patch_trace, patch_time)` 是分块的空间范围。

<a id="4-2-preprocessing-pipeline"></a>
### 4.2 预处理流程

`configs/random_noise_suppression/denoise_unet.yaml` 中的 `preprocess` 块定义了训练和推理前应用于原始数据体的变换。两个阶段必须使用相同的值。

```yaml
preprocess:
  dt: 0.008
  t0: 0.0
  spherical_power: 0
  noise_kind: gaussian
  snr_db: 5.0
  normalize_mode: max_abs
  normalize_scope: shot
  patch_time: 256
  patch_trace: 128
  patch_overlap: 0.5
  max_shots: null
  skip: ["spherical_divergence_correction"]
```

#### 球面扩散校正

`spherical_divergence_correction(shots, dt, t0, power)` 将每个采样点乘以 `(t + t0) ** power`，其中 `t` 为时间轴。它用于补偿球面扩散引起的振幅衰减。在本示例中，`spherical_power: 0` 且该步骤列在 `skip` 中，因此校正被禁用。

#### 归一化

`normalize(shots, mode, per)` 将数据缩放到模型友好的范围。示例配置使用：

```yaml
normalize_mode: max_abs
normalize_scope: shot
```

这通过将每个炮集除以其内部最大绝对值，将每个炮集映射到 `[-1, 1]`。其他支持的模式：`minmax`（映射到 `[0, 1]`）和 `mean_std`（零均值、单位方差）。其他作用域为 `trace` 和 `global`。

`normalize_mode` 必须与 `metrics` 块中 SSIM 和 PSNR 使用的 `data_range` 设置保持一致。对于 `max_abs`，SSIM 使用 `data_range: 2.0`，PSNR 使用 `data_range: 1.0`。对于 `minmax`，两者都使用 `data_range: 1.0`。

> **初学者提示：为什么 SSIM 和 PSNR 使用不同的 `data_range` 值**
>
> 使用 `max_abs` 时，归一化后的数据体范围为 `[-1, 1]`，因此全范围为 `max - min = 1 - (-1) = 2.0`。SSIM 期望 `data_range` 为全范围，因此设为 `2.0`。
>
> 而 PSNR 是按峰值信号振幅定义的。对于 `[-1, 1]` 数据，峰值绝对振幅为 `1.0`，因此 PSNR 使用 `data_range: 1.0`。
>
> 如果切换到 `minmax` 归一化（`[0, 1]`），全范围与峰值振幅都是 `1.0`，因此 SSIM 和 PSNR 都使用 `data_range: 1.0`。请始终让这些值与所选的 `normalize_mode` 保持一致。

#### 合成噪声注入

随机噪声压制使用对干净数据体加噪后的合成版本进行训练。加噪步骤使用：

```python
add_noise(shots, kind="gaussian"|"poisson", snr_db=5.0, rng=None)
```

SNR 以分贝定义为：

```
SNR_dB = 10 * log10(var_signal / var_noise)
```

示例配置使用 `noise_kind: gaussian` 和 `snr_db: 5.0`。较小的值会产生更强的噪声。你可以在推理时通过 `--noise-kind` 和 `--snr-db` 覆盖这些值。

#### 分块

归一化和加噪之后，每个炮集被切分成重叠的二维分块，供 UNet 使用。仓库使用 `tools/patching.py`：

```python
patches, info = patchify_uniform(data, patch_size=(trace, time), overlap=0.0, output_ndim=3|4)
reconstructed = unpatchify_uniform(patches, info)
```

在本示例中：

- `patch_size = (128, 256)`（道，时间）
- `patch_overlap = 0.5`
- `output_ndim = 4`，因此分块以 `(P, 1, 128, 256)` 返回，可直接用于 `nn.Conv2d` 层。

`info` 是一个小型元数据对象，记录原始数组形状、分块网格布局和重叠信息，以便 `unpatchify_uniform` 重建原始形状。重叠区域通过取平均（`sum / count`）融合，从而减少全炮推理时的边缘伪影。

#### 按炮集划分

数据集使用 `data.shot_split` 按炮集级别划分：

```yaml
shot_split:
  train: 7
  val: 1
  test: 1
```

9 个炮集按 FFID 顺序划分：前 7 炮用于训练，第 8 炮用于验证，第 9 炮用于测试。这避免了将同一炮集的分块同时放入训练集和测试集所造成的数据泄漏。

> **注意：** 文件名 `SEG_45Shot_shots1-9.sgy` 指原始 45 炮调查；这里使用的子集包含第 1 至第 9 炮，因此 `n_shots = 9`。

如果省略 `shot_split`，代码会回退到按分块随机划分。

<a id="4-3-yaml-config-walkthrough"></a>
### 4.3 YAML 配置详解

随机噪声压制示例使用的完整配置是 `configs/random_noise_suppression/denoise_unet.yaml`。每个顶层块直接对应流程中的一个阶段。影响预处理的值在训练与推理时必须完全一致。

#### `experiment` 块

```yaml
experiment:
  name: random_noise_unet_base
  output_dir: results/random_noise
  seed: 42
  device: cuda
```

- `name` — 最终实验目录为 `output_dir / name`。
- `output_dir` — 可以是相对于仓库根目录的路径（例如 `results/random_noise`），也可以是绝对路径（例如 `/data/experiments`）。
- `seed` — 噪声注入、炮集选择和数据加载使用的全局随机种子。
- `device` — 训练设备；在 `torchrun` 下运行时会由 `LOCAL_RANK` 覆盖。

#### `data` 块

```yaml
data:
  segy:
    path: data/SEG_45Shot_shots1-9.sgy  # 如果文件在其他位置，请更新路径
    traces_per_shot: 201
    time_downsample: 1
  # npy:
  #   path: ...
  # mat:
  #   path: ...
  #   key: shots
  shot_split:
    train: 7
    val: 1
    test: 1
  # test_ratio: 0.1
  loader:
    batch_size: 192
    num_workers: 4
    pin_memory: true
```

一次只能启用一种格式块（`segy`、`npy` 或 `mat`）。所有加载器都返回形状为 `(n_shots, n_traces, n_time)` 的 `float32` 数据体。

`shot_split` 控制按炮集（FFID）级别的训练/验证/测试划分。当存在时，`test_ratio` 被忽略。划分是顺序的：前 7 个唯一 FFID 用于训练，接下来用于验证，最后用于测试。这避免了重叠分块导致的数据泄漏。如果省略 `shot_split`，代码会回退到按分块随机划分。

`loader` 设置训练 `DataLoader` 参数。推理可以使用自己的 `inference.batch_size` 来减少内存，而无需修改此块。

#### `preprocess` 块

```yaml
preprocess:
  dt: 0.008
  t0: 0.0
  spherical_power: 0
  noise_kind: gaussian
  snr_db: 5.0
  normalize_mode: max_abs
  normalize_scope: shot
  patch_time: 256
  patch_trace: 128
  patch_overlap: 0.5
  max_shots: null
  skip: ["spherical_divergence_correction"]
```

- `dt` — 时间采样间隔（秒），用于球面扩散校正和 FB-FRE 频率估计。
- `t0` — 增益的参考时间偏移，`gain = (t + t0) ** power`。
- `spherical_power` — 球面扩散校正的幂次。设为 `0` 并将该步骤加入 `skip` 即可禁用。
- `noise_kind` — 随机噪声压制的合成噪声类型：`gaussian` 或 `poisson`。
- `snr_db` — 注入噪声的目标 SNR（dB）。值越小，噪声越强。
- `normalize_mode` — `max_abs` 映射到 `[-1, 1]`，`minmax` 映射到 `[0, 1]`，`mean_std` 映射为零均值和单位方差。
- `normalize_scope` — 统计量按 `shot`（炮集）、`trace`（道）还是 `global`（全局）计算。
- `patch_time` / `patch_trace` — 沿时间和道轴的分块大小。
- `patch_overlap` — 推理时重叠分块的重叠比例。`0.0` 表示无重叠。
- `max_shots` — 用于快速冒烟测试的可选限制。`null` 表示使用所有炮集。
- `skip` — 要跳过的预处理步骤名列表。`skip` 是 YAML 字符串列表，不是布尔值。示例：
  - `["spherical_divergence_correction"]` — 只跳过球面扩散校正。
  - `["spherical_divergence_correction", "normalize", "add_noise"]` — 跳过这三个步骤。
  - `[]` — 运行所有预处理步骤。

#### `model` 块

```yaml
model:
  type: unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4
```

`type` 必须是 `MODEL_REGISTRY` 中注册的名称。文件 `model/random_noise_suppression/__init__.py` 导入任务模型，使装饰器执行。`params` 直接传给模型构造函数。

对于随机噪声压制任务，已注册模型包括 `unet`、`dncnn`、`res_unet` 和 `atten_unet`。你只需修改 `type`（必要时再修改模型特定的 `params`）即可切换模型。SCRN 配置与 Shell 脚本已存在（`denoise_SCRN.yaml`、`train_denoise_SCRN.sh`、`inference_denoise_SCRN.sh`），但 `model/random_noise_suppression/` 中目前没有对应实现，因此选择 `type: scrn` 会抛出“模型未注册”错误。

#### `loss`、`optim` 和 `scheduler` 块

```yaml
loss:
  type: mse
  params:
    reduction: mean

optim:
  type: adamw
  params:
    lr: 1.0e-4
    weight_decay: 1.0e-5

scheduler:
  type: cosine
  params:
    min_lr: 1.0e-6
```

- `loss` — 在 `LOSS_REGISTRY` 中注册。去噪通常选择 `mse` 且 `reduction: mean`。
- `optim` — 在优化器构建器中注册。此处 `adamw` 使用 `lr=1e-4` 和 `weight_decay=1e-5`。
- `scheduler` — 从优化器初始学习率下降到 `min_lr` 的余弦退火。

#### `metrics` 块

```yaml
metrics:
  - name: snr
    params: { reduction: per_sample }
  - name: psnr
    params: { data_range: 1.0, reduction: per_sample }
  - name: ssim
    params: { data_range: 2.0, window_size: 11, sigma: 1.5 }
  - name: mae
    params: {}
  - name: mse
    params: {}
  - name: rmse
    params: { reduction: per_sample }
```

这些指标在训练与推理过程中计算。`rmse`、`snr` 和 `psnr` 支持 `reduction: per_sample`（逐炮分数的均值）或 `reduction: global`。

> **重要：** `data_range` 必须与所选的 `normalize_mode` 匹配。
>
> 使用 `max_abs` 时，归一化数据体范围为 `[-1, 1]`。SSIM 期望峰峰值范围，因此 `data_range: 2.0`；PSNR 使用峰值绝对振幅，因此 `data_range: 1.0`。
>
> 使用 `minmax` 时，数据体范围为 `[0, 1]`，因此 SSIM 和 PSNR 都使用 `data_range: 1.0`。
>
> 如果你切换 `normalize_mode`，请同步更新这两个值。

#### `train` 和 `log` 块

```yaml
train:
  epochs: 200
  grad_clip: 1.0
  log_step: false
  log_interval: 10
  eval_interval: 1
  ckpt_interval: 20
  vis_interval: 5
  resume: null

log:
  log_dir: logs
  plot_interval: 5
```

- `epochs` — 训练总轮数。
- `grad_clip` — 梯度裁剪值。
- `log_step` — 若为 `true`，记录每一步训练；若为 `false`，每轮记录一次摘要（由 `log_interval` 控制）。
- `log_interval` — 当 `log_step` 为 `false` 时，控制每轮日志和曲线图在轮内刷新频率。每轮仍会写入一次摘要。
- `eval_interval` — 多久运行一次验证评估。
- `ckpt_interval` — 保存周期性检查点（`epoch_*.pt`）的频率。
- `vis_interval` — 保存随机验证可视化的频率。
- `resume` — 检查点路径占位符；当前 `train_denoise_unet.py` 脚本不会从 CLI 解析它。
- `log_dir` — `output_dir / name` 下存放文本和 CSV 日志的子目录。
- `plot_interval` — 重绘 `loss_curve.png` 和 `metrics_curve.png` 的频率。设为 `0` 禁用。

#### `inference` 块

```yaml
inference:
  data:
    segy:
      path: data/SEG_45Shot_shots1-9.sgy  # 如果文件在其他位置，请更新路径
      traces_per_shot: 201
      time_downsample: 1
  shot_split:
    train: 7
    val: 1
    test: 1
  checkpoint: results/random_noise/random_noise_unet_base/checkpoints/best.pt
  output_dir: results/random_noise/random_noise_unet_base/inference
  n_viz_shots: 5
  device: cuda:1
  batch_size: 48
  binned_metrics:
    enabled: true
    eb_wse:
      enabled: true
      bins: [[5, 20], [20, 40], [40, 70], [70, 100]]
      smooth_sigma: 1.0
    fb_fre:
      enabled: true
      rel_threshold: 0.001
      band_ratios: [0.20, 0.30, 0.30, 0.20]
      band_names: ["low", "mid", "high", "very_high"]
      taper_width: 0.0
```

> **警告：** 提交配置中的默认 `inference.checkpoint` 可能是绝对路径或已过期路径。请显式传入 `--checkpoint`，或更新配置使其指向你自己训练产生的检查点（`results/<exp>/checkpoints/best.pt`）。

分箱诊断（EB-WSE 和 FB-FRE）是可选的。理解主要去噪任务并不需要它们，初学者在学习流程时可以设置 `binned_metrics.enabled: false` 跳过。

- `inference.data` — 可选的推理专用数据源。如果省略，则使用训练 `data` 块。
- `inference.shot_split` — 必须与训练划分一致，以便一致地选择测试炮集。
- `checkpoint` — 要加载的模型检查点路径。通常是 `results/<exp>/checkpoints/best.pt`。
- `output_dir` — 推理输出目录。
- `n_viz_shots` — 要可视化的随机测试炮集数量。
- `device` — 推理设备，例如 `cuda:0` 或 `cpu`。
- `batch_size` — 推理批次大小，独立于 `data.loader.batch_size`。
- `binned_metrics` — EB-WSE 和 FB-FRE 诊断。`enabled` 标志控制整个子系统开关；`eb_wse.enabled` 和 `fb_fre.enabled` 分别控制两个指标。
  - `eb_wse.smooth_sigma` — 对能量图分箱前应用的高斯平滑 sigma。
  - `fb_fre.rel_threshold` — 将频率保留在有效频带内的相对功率阈值。
  - `fb_fre.taper_width` — 频带边缘处的余弦锥化宽度（Hz）。

<a id="4-4-training-in-detail"></a>
### 4.4 训练详解

#### 单 GPU 命令

```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

脚本读取 YAML 文件，从注册表构建模型/损失/数据集/指标，并运行训练循环。无需编辑脚本本身。

#### 输出目录结构

训练产物写入 `results/<experiment.output_dir>/<experiment.name>/`。对于默认配置，即：

```
results/random_noise/random_noise_unet_base/
├── checkpoints/
│   ├── epoch_0020.pt
│   ├── epoch_0040.pt
│   ├── ...
│   ├── epoch_0200.pt
│   └── best.pt
├── logs/
│   ├── train_log.txt
│   ├── loss_history.csv
│   ├── metrics_history.csv
│   ├── loss_curve.png
│   └── metrics_curve.png
├── visualizations/
│   └── epoch_*.png
└── config_used.yaml
```

- `checkpoints/` — 周期性检查点（`epoch_*.pt`）和验证损失最低的 `best.pt`。检查点每隔 `ckpt_interval` 轮保存一次，直到 `epochs` 轮；使用默认值会生成 `epoch_0020.pt`、`epoch_0040.pt`、...、`epoch_0200.pt`。
- `logs/` — 人类可读的日志、CSV 历史记录以及自动刷新的曲线图。
- `visualizations/` — 每隔 `vis_interval` 轮保存的随机验证样本。
- `config_used.yaml` — 解析后配置的副本，用于可复现。

#### 日志文件与曲线图

`logs/train_log.txt` 包含每轮带时间戳的一行摘要。`logs/loss_history.csv` 的列为 `epoch, lr, train, val`，`logs/metrics_history.csv` 的列为 `epoch, train_<metric>, val_<metric>`。`TrainingLogger` 在恢复时会重新加载已有 CSV，因此曲线在重启后保持连续。

#### 从检查点恢复

当前 `train_denoise_unet.py` 脚本不支持通过 CLI 恢复。脚本只解析 `--config`，不解析 `--resume` 标志，也不会从之前的检查点恢复优化器或学习率调度器状态。如果你需要恢复训练，必须编辑脚本，在轮循环前调用 `utils.train_utils` 中的 `load_checkpoint(...)`，并手动恢复优化器/调度器状态。

#### 多 GPU 命令

多 GPU 训练使用 `torchrun`，每个 GPU 一个进程：

> **注意：** 多 GPU 训练是可选的。如果你只有单 GPU，请使用第 4.4 节的单 GPU 命令。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

当 `WORLD_SIZE > 1` 时，脚本会自动：

- 使用 `DistributedDataParallel` 包装模型。
- 为训练加载器使用 `DistributedSampler`。
- 仅由 rank 0 执行检查点保存、日志记录和可视化。
- 对进程间的训练损失进行 all-reduce。

分布式模式下会忽略 `experiment.device`，进程使用 `cuda:LOCAL_RANK`。

> **注意：** 不要通过本教程自动运行训练脚本。请将上述命令复制到终端并手动执行。训练时间取决于 GPU、批次大小和轮数。

<a id="4-5-inference-and-evaluation"></a>
### 4.5 推理与评估

#### 带显式覆盖的推理命令

```bash
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5 \
  --device cuda:0 \
  --batch-size 48
```

所有覆盖项都是可选的。如果省略，脚本会回退到 `inference` 块或 `preprocess` 块中的值。可用的 CLI 参数包括：

- `--checkpoint` — `.pt` 检查点路径（如果 `inference.checkpoint` 未设置则为必填）。
- `--output-dir` — 推理输出目录。
- `--n-viz-shots` — 要可视化的随机炮集数量。
- `--seed` — 炮集选择和噪声注入的随机种子。
- `--device` — 推理设备。
- `--batch-size` — 推理批次大小。
- `--save-npy` — 保存 `input_shots.npy`、`pred_shots.npy` 和 `target_shots.npy`。
- `--noise-kind` — 覆盖 `preprocess.noise_kind`。
- `--snr-db` — 覆盖 `preprocess.snr_db`。

#### 推理流程分步说明

推理脚本对测试炮集执行以下步骤：

1. **加载原始数据体**：使用 `tools.array_io.load_volume`，按文件扩展名分派到 SEG-Y、NPY 或 MAT 读取器。
2. **选择测试划分**：使用 `inference.shot_split`。采用与训练相同的顺序 FFID 排序，从而一致地选择留出测试炮集。
3. **应用与训练相同的预处理**：球面扩散校正（如果未跳过）、归一化和合成噪声注入。噪声注入使用命令行覆盖后的 `noise_kind` 和 `snr_db`。
4. **分块**：将含噪测试炮集分块为重叠的 `(1, patch_trace, patch_time)` 分块。
5. **模型前向**：在指定设备上按批次运行模型。
6. **合并**：使用重叠平均（`sum / count`）对输出进行合并，重建完整炮集。
7. **应用逆变换**：将预测、含噪输入和干净目标恢复到原始振幅域。指标在逆变换前于归一化域计算，因此报告的值与训练一致。
8. **计算指标并保存可视化**。

#### 指标分组

`metrics_summary.json` 包含三组指标：

- `noisy` — 含噪输入与干净目标对比。这是基线。
- `denoised` — 模型预测与干净目标对比。
- `delta` — 从含噪输入到去噪输出的每个指标变化（`denoised_metric - noisy_metric`），表示模型在该指标上改善（或恶化）的程度。正值通常表示改善。

`metrics_per_shot.csv` 包含按炮集计算的相同指标，列名以 `noisy_`、`denoised_` 和 `delta_` 为前缀。

#### EB-WSE 与 FB-FRE

分箱诊断由 `inference.binned_metrics.enabled` 启用。

- **EB-WSE（能量分箱弱信号评估）**在参考能量百分位分箱内计算归一化误差（NE）和 SNR。默认分箱为 `[5, 20]`、`[20, 40]`、`[40, 70]`、`[70, 100]`，分别对应极弱、弱、中等、强信号区域。输出键形如 `eb_wse_very_weak_5_20_ne` 和 `eb_wse_very_weak_5_20_snr`。

- **FB-FRE（频率分箱保真度与恢复评估）**从参考频谱估计有效频带，按 `band_ratios` 拆分为自适应低/中/高/甚高频带，并逐频带计算 NE 和 SNR。输出键形如 `fb_fre_low_ne`、`fb_fre_low_snr`、`fb_fre_low_energy_ratio` 和 `fb_fre_low_frequency_range_hz`。

两个指标都在归一化域计算，并以均值形式写入 `metrics_summary.json` 的 `noisy` 和 `denoised` 组。它们的差值也报告在 `delta` 组中。

#### 输出文件

推理完成后，输出目录包含：

```
results/random_noise/random_noise_unet_base/inference/
├── inference.log
├── metrics_summary.json
├── metrics_per_shot.csv
├── visualizations/
│   └── shot_*.png
└── npy/                    # 仅在传入 --save-npy 时生成
    ├── input_shots.npy
    ├── pred_shots.npy
    └── target_shots.npy
```

- `inference.log` — 推理运行的标准输出和标准错误。
- `metrics_summary.json` — `noisy`、`denoised` 和 `delta` 组的标量指标与分箱指标均值。
- `metrics_per_shot.csv` — 逐炮标量指标。
- `visualizations/shot_*.png` — 每个可视化炮集的输入、预测、目标和残差并排面板。
- `npy/` — 传入 `--save-npy` 时保存的可选 NumPy 数组。

<a id="4-6-batch-sweeps"></a>
### 4.6 批量扫描

为进行系统性基准测试，仓库提供了 Shell 启动脚本，可在噪声类型、SNR 级别和随机种子上进行扫描，而无需手动编辑 YAML。

> **注意：** 这些扫描脚本是便捷辅助工具。运行前请阅读 `.sh` 文件，了解其定义的变量（例如 `NPROC_PER_NODE`、`TORCHRUN_EXTRA`、`STOP_ON_ERROR`），并确认生成的实验名称与路径符合预期。

#### `train_denoise_unet.sh`

`scripts/random_noise_suppression/train_denoise_unet.sh` 为每种噪声类型、SNR 和种子的组合重写临时基础配置，然后启动 `torchrun`。

脚本顶部的可编辑块为：

```bash
CUDA_VISIBLE_DEVICES="0,1" # 物理 GPU 编号，逗号分隔。
NPROC_PER_NODE=2           # 应与可见 GPU 数量一致。
N=3                        # 每个 (noise_kind, snr) 组合的运行次数。
START_SEED=42              # 第一个种子；后续运行使用 START_SEED+1, ...
NOISE_KIND_LIST=("gaussian" "poisson")
SNR_LIST=(-5 0 5)          # 合成噪声 SNR 值（dB）。
TORCHRUN_EXTRA=""          # 可选 torchrun 参数，例如 "--standalone"。
```

默认扫描运行：

- 2 种噪声类型 × 3 个 SNR 级别 × 3 个种子 = 18 次训练。

每次运行都会获得唯一实验名称，例如 `random_noise_unet_base_gaussian_snr5_seed42`，从而避免输出冲突。

#### `inference_denoise_unet.sh`

`scripts/random_noise_suppression/inference_denoise_unet.sh` 对推理进行同样的扫描。它循环遍历噪声类型、SNR 和种子，为每个训练好的检查点运行推理脚本，然后聚合各种子的结果。

默认配置块为：

```bash
DEVICE="cuda:0"
NOISE_KIND_LIST=("gaussian" "poisson")
SNR_LIST=(-5 0 5)
N=3                         # 运行次数 / 种子数
START_SEED=42               # 种子：START_SEED, START_SEED+1, ...
N_VIZ_SHOTS=5
SAVE_NPY=0                  # 1 = 保存 .npy 输出，0 = 跳过
CHECKPOINT_NAME="best.pt"   # 例如 "best.pt" 或 "epoch_0049.pt"
```

每次推理循环结束后，脚本聚合所有种子的 `metrics_summary.json`，并将均值/标准差汇总写入：

```
results/random_noise/random_noise_unet_base_<noise_kind>_snr<tag>_seed_stats/metrics_summary_mean_std.json
```

`<tag>` 对 `-5` dB 为 `neg5`，对 `5` dB 为 `5`，与训练脚本命名约定一致。

#### `run_all_random_noise_models.sh`

`scripts/random_noise_suppression/run_all_random_noise_models.sh` 依次为四个模型系列运行完整的训练与推理扫描：

```bash
MODEL_LIST=("unet" "dncnn" "res_unet" "atten_unet")
```

对于每个模型，它会查找 `scripts/random_noise_suppression/train_denoise_${model}.sh` 和 `scripts/random_noise_suppression/inference_denoise_${model}.sh`，先运行训练扫描，再运行推理扫描。脚本将所有日志写入 `scripts/random_noise_suppression/run_all_random_noise_models.log`。如果 `STOP_ON_ERROR=1`，任一阶段失败时脚本会立即退出。

这是跨架构生成完整基准测试的便捷方式，但由于每个阶段顺序运行，耗时较长。

#### 手动 SNR 推理小循环

如果你希望手动运行一个小的推理扫描，最小化 bash 循环如下：

```bash
CHECKPOINT=results/random_noise/random_noise_unet_base/checkpoints/best.pt
CONFIG=configs/random_noise_suppression/denoise_unet.yaml
for snr in -5 0 5; do
  python scripts/random_noise_suppression/inference_denoise_unet.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "results/random_noise/random_noise_unet_base/inference_snr${snr}" \
    --noise-kind gaussian \
    --snr-db "${snr}" \
    --n-viz-shots 5 \
    --device cuda:0
done
```

每个 SNR 级别写入独立输出目录，便于在不编辑 Shell 扫描脚本的情况下快速对比。

---

<a id="chapter-5-extending-to-other-tasks"></a>
## 第 5 章：扩展到其他任务

仓库包含四个任务族。它们都使用相同的注册表 + 工厂模式骨架，但训练对的生成方式不同。

<a id="5-1-task-comparison"></a>
### 5.1 任务对比

| 任务 | 输入数据 | 入口脚本 | 配置目录 | 主要区别 |
|------|----------|---------|---------|---------|
| `random_noise_suppression` | 干净数据体 + 合成噪声 | `scripts/random_noise_suppression/train_denoise_*.py`、`inference_denoise_*.py` | `configs/random_noise_suppression/` | 使用 `add_noise` 注入噪声；指标将去噪输出与干净目标对比。 |
| `ground_roll_attenuation` | 成对含噪 / 噪声标签数据体 | `scripts/ground_roll_attenuation/train_denoise_*.py`、`batch_evaluate.py` | `configs/ground_roll_attenuation/` | 无合成噪声注入；模型预测加性噪声标签；`data` 块使用 `segy_pair`（或 `npy_pair` / `mat_pair`）。 |
| `multiples_attenuation` | 成对含噪 / 噪声标签数据体 | `scripts/multiples_attenuation/train_denoise_*.py`、`batch_evaluate.py` | `configs/multiples_attenuation/` | 与面波衰减结构相同；任务特定的数据与语义。 |
| `interpolation` | 单数据体 + 道掩码 | `scripts/interpolation/train_interpolation_unet.py`、`inference_interpolation.py` | `configs/interpolation/` | `mask_traces` 模拟缺失道；模型重建完整炮集。 |

<a id="5-2-random-noise-suppression"></a>
### 5.2 `random_noise_suppression`

这是第 4 章介绍的任务。

<a id="5-3-ground-roll-attenuation"></a>
### 5.3 `ground_roll_attenuation`

面波衰减使用成对数据体训练：一个含噪输入数据体和一个对应的噪声标签数据体（即加性噪声分量）。模型学习预测噪声图；去噪估计为 `noisy_input - predicted_noise`。

训练 U-Net 基线：

> **警告：** 默认 `configs/ground_roll_attenuation/denoise_unet.yaml` 包含仓库未包含数据的绝对路径。在运行下方命令前，将 `input_path` 和 `target_path` 替换为你自己的成对数据体路径。

```bash
python scripts/ground_roll_attenuation/train_denoise_unet.py \
  --config configs/ground_roll_attenuation/denoise_unet.yaml
```

训练完成后，在实验目录树上运行批量评估器。`batch_evaluate.py` 需要 `openpyxl`；如果尚未安装，请先安装：

```bash
pip install openpyxl
```

```bash
python scripts/ground_roll_attenuation/batch_evaluate.py \
  --root_dir results/ground_roll_attenuation \
  --output results/ground_roll_attenuation/batch_evaluation.xlsx \
  --device cuda:0 \
  --batch_size 8
```

`batch_evaluate.py` 扫描每个实验目录，加载 `checkpoints/best.pt`，在留出的 `test_set/` 上运行推理，并写入一个每个噪声级别一个工作表的 Excel 工作簿。工作簿对比原始输入指标（含噪 vs 参考）和去噪指标（模型输出 vs 参考）。

面波配置使用 `data.segy_pair` 块（NPY/MAT 变体为 `npy_pair` / `mat_pair`）：

```yaml
data:
  segy_pair:
    input_path: /path/to/noisy.sgy
    target_path: /path/to/noise_label.sgy
    traces_per_shot: 201
    time_downsample: 1
```

两个数据体加载后必须具有相同形状。

<a id="5-4-multiples-attenuation"></a>
### 5.4 `multiples_attenuation`

多次波衰减遵循与面波衰减相同的成对数据体设置；只有数据和噪声标签的物理含义不同。

训练 U-Net 基线：

> **警告：** 默认 `configs/multiples_attenuation/denoise_unet.yaml` 包含仓库未包含数据的绝对路径。在运行下方命令前，将 `input_path` 和 `target_path` 替换为你自己的成对数据体路径。

```bash
python scripts/multiples_attenuation/train_denoise_unet.py \
  --config configs/multiples_attenuation/denoise_unet.yaml
```

运行批量评估器。`batch_evaluate.py` 需要 `openpyxl`：

```bash
pip install openpyxl
```

```bash
python scripts/multiples_attenuation/batch_evaluate.py \
  --root_dir results/multiples_attenuation \
  --output results/multiples_attenuation/batch_evaluation.xlsx \
  --device cuda:0 \
  --batch_size 8
```

<a id="5-5-interpolation"></a>
### 5.5 `interpolation`

插值训练模型重建缺失道。加载单个数据体，沿道轴进行掩码，模型学习恢复原始道。

训练 U-Net 基线，均匀缺失 50% 的道：

> **警告：** 默认 `configs/interpolation/interpolation_unet.yaml` 包含仓库未包含 SEG-Y 数据体的绝对路径。在运行下方命令前，将 `data.segy.path`（或当前启用的格式块）更新为你自己的数据体路径。

```bash
python scripts/interpolation/train_interpolation_unet.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --mask-mode uniform \
  --mask-ratio 0.5
```

训练脚本将掩码参数附加到实验名称后，因此上述运行的输出目录变为 `results/interp_unet_base_uniform_miss50/`。

使用掩码检查点运行推理：

```bash
python scripts/interpolation/inference_interpolation.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --checkpoint results/interp_unet_base_uniform_miss50/checkpoints/epoch_0049.pt \
  --output-dir results/interp_unet_base_uniform_miss50/inference \
  --n-viz-shots 5 \
  --device cuda:0
```

插值专用 YAML 字段：

- `preprocess.mask_mode`（或 CLI `--mask-mode`）：`uniform`、`random` 或 `continuous`。
- `preprocess.mask_ratio`（或 CLI `--mask-ratio`）：要掩码的道比例，范围 `(0, 1)`。
- `preprocess.uniform_stride`：仅在 `mask_mode` 为 `uniform` 时使用；保留每第 `uniform_stride` 道。例如，`uniform_stride: 2` 会移除每隔一道。
- `preprocess.spherical_power`：插值中通常启用（例如 `1.2`），在掩码和归一化前补偿球面扩散。

插值任务是四个任务中唯一使用 `mask_traces` 而非 `add_noise` 或成对噪声标签的任务。其他 YAML 块（`model`、`loss`、`metrics`、`optim`、`scheduler`、`train`、`log`）遵循与第 4 章相同的注册表模式。

---

<a id="chapter-6-customizing-and-extending-the-library"></a>
## 第 6 章：自定义与扩展库

所有可插拔组件都通过第 2 章介绍的注册表 + 工厂模式添加。本章展示每种组件类型的具体步骤。

<a id="6-1-adding-a-new-model"></a>
### 6.1 添加新模型

模型位于 `model/<task>/` 中，并通过 `MODEL_REGISTRY` 注册。

步骤：

1. 创建 `model/<task>/my_model.py`。
2. 继承自 `nn.Module`。
3. 使用 `@register_model("my_model")` 装饰类。
4. 在 `model/<task>/__init__.py` 中添加 `from . import my_model  # noqa: F401`，以便装饰器在导入时执行。
5. 在 YAML 配置中引用新模型。

最小示例：

```python
# model/random_noise_suppression/my_model.py
import torch.nn as nn
from ..registry import register_model


@register_model("my_model")
class MyModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
```

```python
# model/random_noise_suppression/__init__.py
from . import unet       # noqa: F401
from . import my_model     # noqa: F401
```

```yaml
# 配置片段
model:
  type: my_model
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
```

相同模式适用于 `model/ground_roll_attenuation/`、`model/multiples_attenuation/` 和 `model/interpolation/`。注意，顶层 `model/__init__.py` 不会导入每个任务；每个任务子包有自己的注册表视图，因此脚本从对应任务导入 `build_model`（例如 `from model.ground_roll_attenuation import build_model`）。

<a id="6-2-adding-a-new-loss"></a>
### 6.2 添加新损失

损失在 `utils/losses.py` 中注册。

步骤：

1. 继承自 `BaseLoss`。
2. 实现 `forward(self, pred, target=None, **extras)`。
3. 使用 `@register_loss("my_loss")` 装饰。

`extras` 字典由训练循环传入，可携带可选掩码或权重。

示例：

```python
# utils/losses.py
@register_loss("my_loss")
class MyLoss(BaseLoss):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target=None, **extras):
        if target is None:
            raise ValueError("MyLoss requires target.")
        return self.weight * (pred - target).abs().mean()
```

YAML：

```yaml
loss:
  type: my_loss
  params:
    weight: 1.0
```

<a id="6-3-adding-a-new-metric"></a>
### 6.3 添加新指标

指标在 `utils/metrics.py` 中注册。

步骤：

1. 继承自 `BaseMetric`。
2. 实现 `__call__(self, pred, target)`，返回 Python `float`。
3. 适当地设置 `higher_is_better`。
4. 使用 `@register_metric("my_metric")` 装饰。

归约模式：

- `reduction="per_sample"`（默认）：对前导批次维度中的每个样本独立计算指标，然后对批次取平均。这与地震领域常见的“报告平均逐炮 SNR 或 PSNR”惯例一致。
- `reduction="global"`：先对所有元素进行池化，再应用非线性操作（例如 `sqrt` 或 `log10`）。这保留了教科书中的恒等式，例如 `RMSE == sqrt(MSE)` 和 `PSNR == 10*log10(peak^2 / MSE)`。

示例：

```python
# utils/metrics.py
@register_metric("my_metric")
class MyMetric(BaseMetric):
    """Mean absolute error (example custom metric)."""

    higher_is_better = False

    def __call__(self, pred, target):
        pred, target = _prepare(pred, target, "MyMetric")
        return float((pred - target).abs().mean())
```

YAML：

```yaml
metrics:
  - name: my_metric
    params: {}
```

<a id="6-4-adding-a-new-dataset"></a>
### 6.4 添加新数据集

数据集类位于 `utils/datasets.py` 中，继承自 `BaseArrayDataset`。

需要重写的部分：

- `_build_index()`：扫描 `self.root`，并用 `Path` 对象填充 `self._index`。
- `_load_sample(path)`：返回 `(input_tensor, target_tensor_or_none)`。两者都应为 CPU 张量。

数据集期望标准形状为 `(n_shots, n_traces, n_time)` 的地震数据体。

示例：

```python
# utils/datasets.py
@register_dataset("my_dataset")
class MyDataset(BaseArrayDataset):
    """Dataset backed by custom NumPy files."""

    def _build_index(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self._index = sorted(self.root.rglob("*.npy"))
        if not self._index:
            raise FileNotFoundError(f"No .npy files found under {self.root}")

    def _load_sample(self, path):
        arr = np.load(path)
        x = torch.from_numpy(np.asarray(arr)).float()
        return x, None
```

YAML：

```yaml
data:
  type: my_dataset
  params:
    root: /path/to/data
  loader:
    batch_size: 8
    num_workers: 4
    pin_memory: true
```

<a id="6-5-adding-a-new-preprocessing-step"></a>
### 6.5 添加新预处理步骤

新的预处理函数应作为纯 NumPy 操作添加到 `tools/preprocessing.py` 中，作用于 `(n_shots, n_traces, n_time)` 或 `(n_traces, n_time)`。

步骤：

1. 实现该函数，如果步骤是可选的，则将其添加到任务特定训练脚本的 `skip` 机制中。
2. 将配置字段添加到 YAML `preprocess` 块。
3. 从任务特定脚本的预处理函数中调用（例如 `scripts/interpolation/train_interpolation_unet.py` 中的 `_preprocess_shots`）。

示例：

```python
# tools/preprocessing.py
def scale_amplitude(shots, scale=1.0):
    """Scale amplitudes by a constant factor."""
    return shots * scale, {"scale": scale}
```

然后在相关训练脚本中：

```python
if "scale_amplitude" not in skip:
    shots, _ = scale_amplitude(shots, scale=float(prep.get("amplitude_scale", 1.0)))
```

YAML：

```yaml
preprocess:
  amplitude_scale: 1.0
  skip: []
```

添加改变振幅尺度的步骤时，请记得同步更新 `normalize_mode` 和指标 `data_range` 值。例如，如果你将振幅缩放使峰值范围变为 `[-2, 2]`，则 SSIM 应设 `data_range: 4.0`，PSNR 应设 `data_range: 2.0`。

---

<a id="chapter-7-troubleshooting-and-quick-reference"></a>
## 第 7 章：故障排查与快速参考

<a id="7-1-troubleshooting-and-faq"></a>
### 7.1 故障排查与常见问题

#### 检查点未找到 / 路径问题

- 确认检查点文件存在。对于随机噪声示例，默认路径为 `results/random_noise/<experiment.name>/checkpoints/best.pt`。
- 对于 `ground_roll_attenuation` 和 `multiples_attenuation`，`batch_evaluate.py` 要求扫描的每个实验目录都包含 `checkpoints/best.pt` 和 `test_set/` 目录。
- 如果用 `--mask-mode` / `--mask-ratio` 训练插值，实验名称会自动添加后缀（例如 `interp_unet_base_uniform_miss50`），因此检查点路径会相应变化。

#### 未安装 `segyio` 或 SEG-Y 路径错误

- 安装依赖：`pip install segyio`。
- 确认文件存在：`ls /path/to/volume.sgy`。
- 检查配置中的 `traces_per_shot` 和 `time_downsample` 是否与实际文件几何一致。
- 对于成对任务，含噪输入和噪声标签必须具有相同的道数、采样点数和 FFID 顺序。

#### 显存不足

- 减小 `data.loader.batch_size` 或 `inference.batch_size`。
- 减小 `preprocess.patch_trace` 或 `preprocess.patch_time`。
- 对于显存受限的 GPU，使用更小的模型（`base_channels`、`depth`）。

#### 可视化过慢

- 减少 `inference.n_viz_shots`，渲染更少的炮集面板。
- 降低图像 DPI，或如果只需要指标则关闭可视化。

#### SSIM / PSNR 的 `data_range` 与 `normalize_mode` 不匹配

- `max_abs` 归一化到 `[-1, 1]`：SSIM 需要 `data_range: 2.0`，PSNR 需要 `data_range: 1.0`。
- `minmax` 归一化到 `[0, 1]`：SSIM 和 PSNR 都使用 `data_range: 1.0`。
- `mean_std` 无界；请根据实际目标数据体设置范围，或保留在指标参数中。

#### 训练与推理之间的 `shot_split` 不一致

- `inference.shot_split` 块必须与训练时的 `data.shot_split` 一致，以便选择相同的测试炮集。
- 如果训练时使用了分块级划分（`test_ratio` 而非 `shot_split`），则推理时不要添加 `inference.shot_split`。
- 划分基于顺序 FFID，而非任意炮集索引。

#### 模型未注册：缺少导入或装饰器拼写错误

- 检查注册表内容：`from model.<task> import MODEL_REGISTRY; print(sorted(MODEL_REGISTRY))`。
- 确认 `model/<task>/__init__.py` 包含 `from . import my_model  # noqa: F401`。
- 确认装饰器名称与 YAML 配置中的 `model.type` 完全一致（区分大小写）。
- 确认脚本从正确的任务子包导入 `build_model`（例如 `from model.ground_roll_attenuation import build_model`）。

<a id="7-2-quick-reference-cards"></a>
### 7.2 速查卡

#### CLI 命令速查表

> **注意：** `ground_roll_attenuation` 和 `multiples_attenuation` 的 `batch_evaluate.py` 需要 `openpyxl`。请先安装：`pip install openpyxl`。

##### `random_noise_suppression`

训练：

```bash
python scripts/random_noise_suppression/train_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml
```

推理：

```bash
python scripts/random_noise_suppression/inference_denoise_unet.py \
  --config configs/random_noise_suppression/denoise_unet.yaml \
  --checkpoint results/random_noise/random_noise_unet_base/checkpoints/best.pt \
  --output-dir results/random_noise/random_noise_unet_base/inference \
  --noise-kind gaussian \
  --snr-db 5 \
  --n-viz-shots 5 \
  --device cuda:0
```

##### `ground_roll_attenuation`

训练：

```bash
python scripts/ground_roll_attenuation/train_denoise_unet.py \
  --config configs/ground_roll_attenuation/denoise_unet.yaml
```

批量评估（需要 `openpyxl`）：

```bash
pip install openpyxl
python scripts/ground_roll_attenuation/batch_evaluate.py \
  --root_dir results/ground_roll_attenuation \
  --output results/ground_roll_attenuation/batch_evaluation.xlsx \
  --device cuda:0 \
  --batch_size 8
```

##### `multiples_attenuation`

训练：

```bash
python scripts/multiples_attenuation/train_denoise_unet.py \
  --config configs/multiples_attenuation/denoise_unet.yaml
```

批量评估（需要 `openpyxl`）：

```bash
pip install openpyxl
python scripts/multiples_attenuation/batch_evaluate.py \
  --root_dir results/multiples_attenuation \
  --output results/multiples_attenuation/batch_evaluation.xlsx \
  --device cuda:0 \
  --batch_size 8
```

##### `interpolation`

训练：

```bash
python scripts/interpolation/train_interpolation_unet.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --mask-mode uniform \
  --mask-ratio 0.5
```

推理：

```bash
python scripts/interpolation/inference_interpolation.py \
  --config configs/interpolation/interpolation_unet.yaml \
  --checkpoint results/interp_unet_base_uniform_miss50/checkpoints/epoch_0049.pt \
  --output-dir results/interp_unet_base_uniform_miss50/inference \
  --n-viz-shots 5 \
  --device cuda:0
```

#### YAML 顶层键与常用字段

| 键 | 常用字段 |
|----|---------|
| `experiment` | `name`、`output_dir`、`seed`、`device` |
| `data` | 数据源（`segy`、`npy`、`mat`、`segy_pair`、`npy_pair`、`mat_pair`）、`shot_split`、`loader`（`batch_size`、`num_workers`、`pin_memory`） |
| `preprocess` | `dt`、`t0`、`spherical_power`、`normalize_mode`、`normalize_scope`、`patch_time`、`patch_trace`、`patch_overlap`、`max_shots`、`skip`、`noise_kind`、`snr_db`（随机噪声）、`mask_mode`（仅插值）、`mask_ratio`（仅插值）、`uniform_stride`（仅插值）、`clip_percentile`（仅插值） |
| `model` | `type`、`params` |
| `loss` | `type`、`params` |
| `metrics` | `{name, params}` 列表 |
| `optim` | `type`、`params` |
| `scheduler` | `type`、`params` |
| `train` | `epochs`、`grad_clip`、`log_step`、`log_interval`、`eval_interval`、`ckpt_interval`、`vis_interval`、`resume` |
| `log` | `log_dir`、`plot_interval` |
| `inference` | `data`、`shot_split`、`checkpoint`、`output_dir`、`n_viz_shots`、`device`、`batch_size`、`save_npy`、`binned_metrics` |

#### 注册表装饰器 / 工厂 / 基类速查表

| 类型 | 装饰器 | 基类 | 工厂 | 注册表 |
|------|--------|------|------|--------|
| 模型 | `@register_model("name")` | `nn.Module` | `build_model(cfg)` | `MODEL_REGISTRY` |
| 损失 | `@register_loss("name")` | `BaseLoss` | `build_loss(cfg)` | `LOSS_REGISTRY` |
| 指标 | `@register_metric("name")` | `BaseMetric` | `build_metrics(cfg_list)` | `METRIC_REGISTRY` |
| 数据集 | `@register_dataset("name")` | `BaseArrayDataset` | `build_dataset(cfg)`（由训练脚本内部调用） | `DATASET_REGISTRY` |

#### 指标参数速查表

| 指标 | 参数 | 说明 |
|------|------|------|
| `mse` | — | 所有元素的全局均值。 |
| `mae` | — | 所有元素的全局均值。 |
| `rmse` | `reduction: per_sample \| global` | `global` 保持 `RMSE == sqrt(MSE)`。 |
| `snr` | `reduction`、`eps`、`min_signal_energy` | 信噪比（dB）。使用 `min_signal_energy` 避免能量接近零的炮集除零。 |
| `psnr` | `data_range`、`reduction`、`eps` | 参考信号的峰值振幅。 |
| `ssim` | `data_range`、`window_size`、`sigma`、`k1`、`k2` | 参考信号的峰峰值范围。 |
