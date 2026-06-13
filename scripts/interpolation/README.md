# scripts/interpolation/ — Interpolation Training & Inference

Three independent pipelines for seismic trace interpolation, each with its own
data representation, model interface, and training script.

---

## Pipeline Overview

| Pipeline | Script | Data Format | Model Input | Model Output | Registry |
|----------|--------|-------------|-------------|-------------|----------|
| **U-Net** (CNN baseline) | `train_interpolation_unet.py` | 2D patch `(B,1,H,W)` | `model(x)` | `(B,1,H,W)` | `unet`, `res_unet`, `atten_unet`, `dncnn`, `unet_plusplus` |
| **Gated Transformer** (token) | `train_interpolation_transformer.py` | 1D token `(B,L,D)` + coords + mask | `model(x, coords, time_bounds, mask)` | `(B,L,D)` | `gated_transformer_v9`, `gated_transformer_v9_encdec`, `gated_transformer_v11`, `gated_transformer_v11_encdec` |
| **Patch Transformer** (ViT / trace-token) | `train_interpolation_patch_transformer.py` | 2D patch `(B,C,H,W)` | `model(x)` | `(B,1,H,W)` | `trace_token_transformer_interpolator`, `hf_vit_interpolator` |

---

## Pipeline 1: U-Net (CNN Baseline)

### Data Flow

```
SEG-Y → load_volume → spherical_divergence → normalize → mask_traces
       → patchify_uniform(2D) → (P, 1, H, W) patches
       → UNet forward → (P, 1, H, W) → MSE loss
```

### Key Characteristics

- **Patch shape**: `(B, 1, patch_trace, patch_time)` — single-channel input (masked seismic only)
- **Model interface**: `forward(self, x)` — standard single-tensor CNN interface
- **No mask channel**: The model must infer missing traces from spatial context alone
- **Masking**: Default `continuous` mode applied **after** patchification on target patches, then reassigned as input
- **Visualization**: Random validation patch (2D)

### Models (registered under `model.interpolation`)

| Registry Name | File | Description |
|---------------|------|-------------|
| `unet` | `model/interpolation/unet.py` | Standard 2D U-Net, configurable depth/channels |
| `res_unet` | `model/interpolation/res_unet.py` | U-Net with residual blocks |
| `atten_unet` | `model/interpolation/atten_unet.py` | U-Net with attention gates on skip connections |
| `dncnn` | `model/interpolation/dncnn.py` | DnCNN (residual learning, `pred = x - net(x)`) |
| `unet_plusplus` | `model/interpolation/unet_plusplus.py` | U-Net++ with dense skip connections |

### Config

```yaml
# configs/interpolation/interpolation_unet.yaml
model:
  type: unet
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4

preprocess:
  patch_time: 256
  patch_trace: 128
  patch_overlap: 0.5
  # No include_mask_channel field (C=1 always)
```

### Training

```bash
# Single GPU
python scripts/interpolation/train_interpolation_unet.py \
  --config configs/interpolation/interpolation_unet.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/interpolation/train_interpolation_unet.py \
  --config configs/interpolation/interpolation_unet.yaml

# Multi-seed sweep
bash scripts/interpolation/train_interpolation_unet.sh
```

### CLI Overrides

```
--mask-mode {uniform,random,continuous}   # default: continuous
--mask-ratio FLOAT                         # default: 0.2
--continuous-missing-traces INT            # fixed count for continuous mode
```

### Inference

```bash
python scripts/interpolation/inference_interpolation.py \
  --config configs/interpolation/interpolation_unet.yaml
```

---

## Pipeline 2: Gated Transformer (Token-Based)

### Data Flow

```
SEG-Y → load_volume → spherical_divergence → normalize → mask_traces
       → extract coords from headers (sx, sy, rx, ry)
       → trace_time_chunk(1D) → (n_shots, L, chunk_length) tokens
       → InterpolationTokenDataset → (B, L, D) + coords + time_bounds + mask
       → model(x, coords, time_bounds, mask) → (B, L, D)
       → MSE loss (all tokens)
```

### Key Characteristics

- **Token shape**: `(B, L, chunk_length)` — each token is a time-segment of one trace, `L = n_traces × n_chunks`
- **Model interface**: `forward(self, x, coords=None, time_bounds=None, mask=None)` — multi-argument
- **Coordinate encoding**: 6D coordinates (sx, sy, rx, ry, time_start, time_end) with sincos or RoPE
- **Encoder-decoder**: Encoder processes only observed tokens; decoder reconstructions all tokens via cross-attention
- **Token-level mask**: `1.0` = observed, `0.0` = missing (derived from all-zeros check)
- **Visualization**: Full shot reconstructed via `trace_time_unchunk`

### Models

| Registry Name | File | Description |
|---------------|------|-------------|
| `gated_transformer_v9` | `model/interpolation/gated_transformer_v9.py` | Self-attention encoder + self-attention decoder, Qwen3-style gated attention |
| `gated_transformer_v9_encdec` | `model/interpolation/gated_transformer_v9_encdec.py` | Self-attn encoder + cross-attn decoder |
| `gated_transformer_v11` | `model/interpolation/gated_transformer_v11.py` | V9 + optional data embedding (`mlp`/`conv`/`trace_conv_attn`) |
| `gated_transformer_v11_encdec` | `model/interpolation/gated_transformer_v11_encdec.py` | V11 + cross-attn decoder |

### Config

```yaml
# configs/interpolation/interpolation_transformer.yaml
model:
  type: gated_transformer_v11_encdec
  params:
    input_dim: 256            # must match chunk_length
    d_model: 512
    n_heads: 8
    num_encoder_layers: 4
    num_decoder_layers: 4
    d_ff: 2048
    dropout: 0.1
    norm_type: rms
    use_rope: true
    use_coord_encoding: true
    coord_dim: 6              # 4 spatial + 2 time
    encode_observed_only: true

preprocess:
  chunk_length: 256           # time samples per token
  overlap_ratio: 0.0          # no overlap between tokens
```

### Training

```bash
# Single GPU
python scripts/interpolation/train_interpolation_transformer.py \
  --config configs/interpolation/interpolation_transformer.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/interpolation/train_interpolation_transformer.py \
  --config configs/interpolation/interpolation_transformer.yaml

# Multi-seed sweep
bash scripts/interpolation/train_interpolation_transformer.sh
```

### Inference

```bash
# Single run
python scripts/interpolation/inference_interpolation_transformer.py \
  --config configs/interpolation/interpolation_transformer.yaml

# Using shell wrapper (supports env-var overrides)
bash scripts/interpolation/inference_interpolation_transformer.sh
```

### Train + Inference Loop

```bash
# Full automated loop: multi-experiment, multi-seed train → inference → extra generalization test
bash scripts/interpolation/train_infer_loop_transformer.sh

# Custom experiments
EXPERIMENTS=("random:0.3" "continuous:30tr") \
  bash scripts/interpolation/train_infer_loop_transformer.sh
```

### Shell Script Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU selection |
| `NPROC_PER_NODE` | `1` | GPUs per node (for DDP) |
| `N_SEEDS` | `3` | Number of random seeds to sweep |
| `START_SEED` | `42` | Starting seed value |
| `MASTER_PORT` | `auto` | DDP master port |
| `TORCHRUN_EXTRA` | `""` | Extra `torchrun` arguments |
| `MASK_MODE` | `continuous` | Training mask mode |
| `MASK_RATIO` | `0.1` | Training mask ratio |

For the loop script (`train_infer_loop_transformer.sh`), see the configuration block at the top of the script for all available variables (including `EXPERIMENTS`, `N_SEEDS`, `DRY_RUN`, etc.).

---

## Pipeline 3: Patch Transformer / ViT

### Data Flow

```
SEG-Y → load_volume → spherical_divergence → normalize → mask_traces
       → patchify_uniform(2D) → (P, 1, H, W) patches
       → [optional] include_mask_channel → (P, 2, H, W) with mask channel
       → model(x) → (P, 1, H, W)
       → MSE loss
```

### Key Characteristics

- **Patch shape**: `(B, C, H, W)` — `C=1` when `include_mask_channel=false`, `C=2` when `true`
- **Model interface**: `forward(self, x)` — standard single-tensor interface, same as U-Net pipeline
- **Mask channel**: When enabled, channel 1 = masked seismic data, channel 2 = binary mask (`1` = missing, `0` = observed)
- **Training loop**: Reuses `train_one_epoch` / `evaluate` from `utils.train_utils` (single-arg forward)
- **Visualization**: Random validation patch (2D)

### Mask Channel (`include_mask_channel`)

When `include_mask_channel: true` in config:
1. After patchify and optional `patch_normalize`, a binary mask is constructed:
   ```python
   mask_channel = (input_patches[:, :1, :, :] == 0).astype(np.float32)
   input_patches = np.concatenate([input_patches, mask_channel], axis=1)
   ```
2. Input goes from `(P, 1, H, W)` to `(P, 2, H, W)`
3. Target remains `(P, 1, H, W)` (clean seismic only)
4. Zero-detection is safe because `mask_traces` sets missing traces to exactly 0, and `max_abs` normalization + `patch_normalize` both preserve zeros

### Models

| Registry Name | File | Description |
|---------------|------|-------------|
| `trace_token_transformer_interpolator` | `model/interpolation/trace_token_transformer_interpolator.py` | Each trace as a token; sincos PE + Transformer encoder-decoder + learned query reconstruction |
| `hf_vit_interpolator` | `model/interpolation/hf_vit_interpolator.py` | HuggingFace ViT encoder + learnable unpatchify head; supports pretrained weights |

#### `trace_token_transformer_interpolator`

Architecture: `(B,C,H,W)` → trace tokenization → `Linear(C×W, embed_dim)` → sincos PE → `TransformerEncoder(depth=6)` → `TransformerDecoder(depth=4)` with learned query → `Linear(embed_dim, W)` → `(B,1,H,W)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_channels` | 2 | Input channels (1 or 2) |
| `trace_length` | 256 | Time samples per trace (must match `patch_time`) |
| `embed_dim` | 256 | Token embedding dimension |
| `encoder_depth` | 6 | Transformer encoder layers |
| `decoder_depth` | 4 | Transformer decoder layers |
| `num_heads` | 8 | Attention heads |
| `dropout` | 0.0 | Dropout probability |
| `pos_encoding` | `sincos` | Position encoding type |
| `coord_dim` | 1 | Placeholder for future 5D coordinate extension |

#### `hf_vit_interpolator`

Architecture: `(B,C,H,W)` → `Conv2d(C,3,1)` → ViT encoder → remove CLS → reshape to 2D feature map → `Conv2d(hidden, patch², 1)` → `pixel_shuffle` → `(B,1,H,W)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `in_channels` | 2 | Input channels (1 or 2) |
| `model_name` | `google/vit-base-patch16-224-in21k` | HF model ID **or** local directory path |
| `pretrained` | `true` | Loading mode (see below) |
| `patch_size` | 16 | ViT patch size; H and W must be divisible by this |
| `freeze_encoder` | `false` | Freeze ViT encoder during training |

**Three loading modes** (controlled by `pretrained` and `model_name`):

| Mode | `pretrained` | `model_name` | Behaviour |
|------|-------------|-------------|-----------|
| **HuggingFace Hub** | `true` | `"google/vit-base-patch16-224-in21k"` | Downloads pretrained weights from HF Hub (requires network on first run; cached thereafter) |
| **Local weights** | `true` | `"/data/models/my_vit"` | Loads from a local directory. Any path starting with `/`, `./`, `../`, or `~` is auto-detected as local. Uses `local_files_only=True`. |
| **Random init** | `false` | *(ignored)* | Builds a small `ViTConfig` (192 hidden, 4 layers, 3 heads) for offline CPU testing |

Position embeddings are **automatically interpolated** when the input patch grid
differs from the pretrained image grid (e.g. loading a 224×224 ViT and feeding
64×64 seismic patches).

**Requirements**: `pip install transformers`. If not installed, a clear error
message is shown at instantiation time.

**Local weights example**:

```bash
# Download once from HF Hub to a local directory
huggingface-cli download google/vit-base-patch16-224-in21k \
  --local-dir /data/models/vit-base-patch16-224

# Then use in config
model:
  type: hf_vit_interpolator
  params:
    model_name: /data/models/vit-base-patch16-224
    pretrained: true
```

**`loading_mode()` introspection**:

```python
m = HFViTInterpolator(model_name="google/vit-base-...", pretrained=True)
print(m.loading_mode())  # "hub"

m = HFViTInterpolator(model_name="/data/my_vit", pretrained=True)
print(m.loading_mode())  # "local"

m = HFViTInterpolator(pretrained=False)
print(m.loading_mode())  # "random"
```

### Configs

#### `configs/interpolation/interpolation_trace_transformer.yaml`

```yaml
model:
  type: trace_token_transformer_interpolator
  params:
    in_channels: 2
    trace_length: 64           # must match preprocess.patch_time
    embed_dim: 256
    encoder_depth: 6
    decoder_depth: 4
    num_heads: 8
    dropout: 0.0
    pos_encoding: sincos
    coord_dim: 1

preprocess:
  include_mask_channel: true   # C=1 → C=2
  patch_time: 64
  patch_trace: 32              # 32 tokens per patch; keep small for Transformer
  patch_overlap: 0.5
  patch_normalize: true

optim:
  type: adamw
  params:
    lr: 1.0e-4
    weight_decay: 1.0e-4
```

#### `configs/interpolation/interpolation_hf_vit.yaml`

```yaml
model:
  type: hf_vit_interpolator
  params:
    in_channels: 2
    model_name: google/vit-base-patch16-224-in21k
    pretrained: true
    patch_size: 16             # 64 must be divisible by 16
    freeze_encoder: false

preprocess:
  include_mask_channel: true
  patch_time: 64               # must be divisible by patch_size (16)
  patch_trace: 64              # must be divisible by patch_size (16)
  patch_overlap: 0.5
  patch_normalize: true

optim:
  type: adamw
  params:
    lr: 1.0e-5                 # lower lr for pretrained encoder
    weight_decay: 1.0e-4
```

### Training

```bash
# Trace Token Transformer
python scripts/interpolation/train_interpolation_patch_transformer.py \
  --config configs/interpolation/interpolation_trace_transformer.yaml

# HF ViT (requires internet for pretrained weights on first run)
python scripts/interpolation/train_interpolation_patch_transformer.py \
  --config configs/interpolation/interpolation_hf_vit.yaml

# DDP multi-GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  scripts/interpolation/train_interpolation_patch_transformer.py \
  --config configs/interpolation/interpolation_trace_transformer.yaml

# Multi-seed sweep via shell script
PIPELINE=patch \
  bash scripts/interpolation/train_interpolation_transformer.sh
# For HF ViT:
PIPELINE=patch PATCH_CONFIG=configs/interpolation/interpolation_hf_vit.yaml \
  bash scripts/interpolation/train_interpolation_transformer.sh
```

### Inference

```bash
# Single run (reuses U-Net inference data flow — patchify → model(x) → unpatchify)
python scripts/interpolation/inference_interpolation.py \
  --config configs/interpolation/interpolation_trace_transformer.yaml

# Using shell wrapper
PIPELINE=patch CONFIG=configs/interpolation/interpolation_trace_transformer.yaml \
  bash scripts/interpolation/inference_interpolation_transformer.sh
```

### Train + Inference Loop

```bash
# Full automated loop
PIPELINE=patch \
  bash scripts/interpolation/train_infer_loop_transformer.sh

# HF ViT with custom experiments
PIPELINE=patch BASE_CONFIG=configs/interpolation/interpolation_hf_vit.yaml \
  EXPERIMENTS=("random:0.3" "continuous:20tr") \
  bash scripts/interpolation/train_infer_loop_transformer.sh
```

### CLI Overrides

Same as U-Net pipeline:
```
--mask-mode {uniform,random,continuous}
--mask-ratio FLOAT
--continuous-missing-traces INT
```

### Quick Shape Test

```bash
python3 -c "
import torch
from model.interpolation import build_model

# trace_token_transformer: (2,2,32,64) -> (2,1,32,64)
m = build_model({'type':'trace_token_transformer_interpolator',
    'params':{'in_channels':2,'trace_length':64,'embed_dim':256,
              'encoder_depth':6,'decoder_depth':4,'num_heads':8}})
print(m(torch.randn(2,2,32,64)).shape)

# hf_vit (offline): (2,2,64,64) -> (2,1,64,64)
m2 = build_model({'type':'hf_vit_interpolator',
    'params':{'in_channels':2,'pretrained':False,'patch_size':16}})
print(m2(torch.randn(2,2,64,64)).shape)
"
```

---

## Choosing a Pipeline

| Consideration | U-Net | Gated Transformer | Patch Transformer |
|---------------|-------|-------------------|-------------------|
| **Input format** | 2D patch `(B,1,H,W)` | 1D token `(B,L,D)` + coords | 2D patch `(B,C,H,W)` |
| **Coordinate info** | Not used | 6D coordinates via headers | Via mask channel (C=2) |
| **Model complexity** | Low (CNN) | High (Transformer + RoPE) | Medium-High (Transformer/ViT) |
| **Batch size** | Large (192) | Small (4) | Medium (16-32) |
| **Spatial context** | Local (conv receptive field) | Global (self-attention over all tokens) | Global (self-attention over trace tokens) |
| **Pretrained weights** | No | No | Yes (ViT via HuggingFace) |
| **Dependencies** | PyTorch only | PyTorch only | PyTorch + transformers |
| **Best for** | Fast baseline, small datasets | Large shots, needs coordinate cues | Transfer learning, medium datasets |

---

## Shared Infrastructure

All three pipelines share:

- **Preprocessing**: `tools/preprocessing.py` — `normalize`, `spherical_divergence_correction`, `mask_traces`
- **Data loading**: `tools/array_io.py` — `load_volume` (`.sgy` / `.npy` / `.mat`)
- **Patching** (Pipelines 1 & 3): `tools/patching.py` — `patchify_uniform`
- **Tokenization** (Pipeline 2): `tools/patching.py` — `trace_time_chunk` / `trace_time_unchunk`
- **Loss / Metrics**: `utils/losses.py`, `utils/metrics.py` — registry-based
- **Training loop**: `utils/train_utils.py` — `train_one_epoch`, `evaluate`, DDP helpers
- **Logging**: `utils/logger.py` — `TrainingLogger` (CSV + PNG curves)
- **Visualization**: `utils/visualization.py` — `plot_sample`, `visualize_random_sample`
- **Checkpointing**: `utils/train_utils.py` — `save_checkpoint`, `maybe_save_best_checkpoint`
- **Config**: `utils/train_utils.py` — `load_config`, YAML-based

---

## Adding a New Model to Pipeline 3

1. Create `model/interpolation/my_model.py` with `@register_model("my_model")` on your `nn.Module`
2. Ensure `forward(self, x)` accepts `(B, C, H, W)` and returns `(B, 1, H, W)`
3. Add `from . import my_model  # noqa: F401` to `model/interpolation/__init__.py`
4. Create `configs/interpolation/interpolation_my_model.yaml` referencing `type: my_model`
5. Train with:
   ```bash
   python scripts/interpolation/train_interpolation_patch_transformer.py \
     --config configs/interpolation/interpolation_my_model.yaml
   ```
