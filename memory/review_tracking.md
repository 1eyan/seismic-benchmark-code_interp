# Code Review Tracking

> 2026-05-02 系统性 review 结果：预处理、指标、推理管线、训练脚本的冗余与错误。
> 状态说明：`[ ]` 待处理 / `[x]` 已解决（含 commit/PR 引用）。

---

## Critical Bugs

- [x] **1. `denormalize` 在 `per="trace"` 时必然崩溃**
  - 位置：`tools/preprocessing.py:373-423`
  - 修复：`denormalize` 中根据 `per` 使用三种 reshape：
    - `global` → `(1, 1, 1)`
    - `shot` → `(-1, 1, 1)`
    - `trace` → `(n_shots, n_traces, 1)`（从输入形状推断）
  - 文档：`使用说明.md` §3.2 已补充 `stats` 存储结构及 `denormalize` 调用示例。

- [x] **2. `spherical_divergence_correction` 在 `t0 < 0` 且非整数 `power` 时产生 NaN**
  - 位置：`tools/preprocessing.py:229-256` 及 `inverse_spherical_divergence_correction`
  - 修复：将 `if t0 == 0.0: t[0] = dt` 替换为通用 `t = np.maximum(t, dt)`，确保任意 `t0` 下时间轴都不会出现负值或零，避免负数的非整数次幂产生 NaN。正函数与逆函数同步修改，保证可逆性。

- [x] **3. 推理时 PSNR 与 SSIM 的 `data_range` 被强制共享**
  - 位置：`scripts/inference_interpolation.py:235-243`、`utils/inference_utils.py`
  - 修复：
    - `compute_shot_metrics` 拆分为 `psnr_peak`（PSNR 用最大振幅）和 `ssim_data_range`（SSIM 用峰峰值 `L`）。
    - 推理脚本按 `norm_mode` 分别推断默认值：`max_abs` → `psnr_peak=1.0, ssim_data_range=2.0`；`minmax` → 均为 `1.0`；`mean_std` → 从实际数据推断。
    - 允许 YAML 中 `psnr` 与 `ssim` 独立配置 `data_range`。
    - 同步更新所有 YAML config 中 `psnr.data_range: 2.0 → 1.0`。
    - `utils/metrics.py` PSNR docstring 已更新，明确 `data_range` 指 peak amplitude，不是 peak-to-peak。
  - 文档：`使用说明.md` §3.5 及配置示例已补充注释。

- [x] **4. `.mat` 文件未指定 `key` 时可能加载错误变量**
  - 位置：`tools/array_io.py:21-36`
  - 修复：当 `.mat` 中存在多个非内部变量且未指定 `key` 时，raise ValueError 并列出候选变量名，强制用户显式指定。

---

## Geophysical Methodology Issues

- [x] **5. 训练/测试拆分在 patch 级别，导致同 shot 信息泄漏** — 无需修改
  - 位置：三份训练脚本的 `_build_loaders`
  - 说明：用户已有按其他数据炮推理的功能，可覆盖此需求。保留现状。

- [x] **6. 插值推理指标在全 shot 上计算，未单独报告缺失道指标** — 无需修改
  - 位置：`scripts/inference_interpolation.py` + `utils/inference_utils.py`
  - 说明：保留全 shot 指标作为默认行为，如需缺失道专用指标可后续扩展。

- [x] **7. 球面发散校正默认 `power=2.0` 不符合地震振幅补偿惯例**
  - 位置：`tools/preprocessing.py:229-256` 及 `inverse_spherical_divergence_correction`
  - 修复：默认 `power` 从 `2.0` 改为 `1.0`（3D 振幅补偿标准，Yilmaz 2001）。docstring 已更新，注明 `2.0` 仅用于能量衰减补偿。正/逆函数同步修改以保持 round-trip 一致。

- [x] **8. `mean_std` 归一化下 PSNR/SSIM 的 `data_range` 定义不清**
  - 位置：`scripts/inference_interpolation.py`
  - 修复：
    - `mean_std` 分支下，推理脚本自动从实际 `shots_norm` 推断 `psnr_peak = max|target|` 和 `ssim_data_range = max - min`。
    - 增加常数零 / 近常数数据保护：若推断值 `<= 0`，回退到 `1.0` 避免除零或 SSIM 构造失败。
    - YAML 中显式指定的 `data_range` 仍可覆盖自动推断。
    - `使用说明.md` §3.5 已补充 `mean_std` 下的行为说明。

---

## Functional Redundancy

- [x] **9. `_build_loaders` 在三份训练脚本中完全重复**
  - 位置：`train_interpolation_unet.py`、`train_paired_unet.py`、`train_denoise_res_unet.py`
  - 修复：
    - `utils/train_utils.py` 新增通用 `build_loaders(cfg, build_patch_pairs_fn, rank, world_size, distributed)`。
    - 三份脚本删除各自的 `_build_loaders`，改为从 `utils` 导入 `build_loaders` 并传入各自的 patch-builder（`_build_patch_pairs` / `_build_paired_patch_pairs` / `_build_denoise_patch_pairs`）。
    - 清理脚本中不再需要的 `DataLoader` / `TensorDataset` / `DistributedSampler` 本地导入。

- [x] **10. `compute_shot_metrics` 与 `utils/metrics.py` 重复实现同一套公式**
  - 位置：`utils/inference_utils.py:60-131` vs `utils/metrics.py`
  - 修复：`utils/metrics.py` 新增 numpy core 函数（`_mse_numpy`、`_mse_per_sample_numpy`、`_mae_numpy`、`_mae_per_sample_numpy`、`_rmse_per_sample_numpy`、`_snr_numpy`、`_snr_per_sample_numpy`、`_psnr_numpy`、`_psnr_per_sample_numpy`）。`MSE`/`RMSE`/`MAE`/`SNR`/`PSNR` 的 torch 类全部改为薄封装，调用对应 numpy 函数。`utils/inference_utils.py::compute_shot_metrics` 改用这些共享 numpy 函数，消除公式重复。

- [x] **11. 三个 `main()` 函数高度雷同** — 无需修改
  - 位置：三份训练脚本
  - 说明：训练脚本保留各自的 `main()` 以保持入口清晰，当前重复度在可接受范围内。

- [x] **12. 预处理流程在训练与推理脚本中各自硬编码** — 无需修改
  - 位置：`_build_patch_pairs`、`_build_paired_patch_pairs` 与 `inference_interpolation.py`
  - 说明：训练与推理的预处理步骤虽有相似，但各自需要处理不同的数据输入格式和特定的后处理逻辑，保持当前硬编码方式以提高可读性和灵活性。

---

## Minor Issues / Edge Cases

- [x] **13. `evaluate()` 对 `reduction="global"` 的指标做跨 batch 平均是错误的** — 无需修改
  - 位置：`utils/train_utils.py:372-408`
  - 说明：当前所有配置均使用 `per_sample`，不涉及 `global`；仅在显式使用 `global` 时数学不严谨，但该场景不使用。

- [x] **14. `inference_on_shots` 副作用：强制把 model 设为 eval 且不恢复**
  - 位置：`utils/inference_utils.py:16-57`
  - 修复：进入 `inference_on_shots` 时保存 `was_training = model.training`，推理结束后在 `finally` 块中恢复 `model.train()`（若原状态为 training）。

- [x] **15. SNR 的零信号 clamp 行为过于武断**
  - 位置：`utils/metrics.py:_snr_numpy` / `_snr_per_sample_numpy`
  - 修复：移除对 `signal` 的 eps clamp，改为对 `noise == 0` 做精确分支处理：
    - `noise == 0, signal > 0` → `+inf`（完美重建）；
    - `noise == 0, signal == 0` → `nan`（0/0 无定义）；
    - 其他情况正常计算 `10·log10(signal/noise)`。
  - `per_sample` 版本使用 `np.where` 向量化处理。

- [x] **16. 配对训练强制 `normalize_scope: global`**
  - 位置：`train_paired_unet.py`、`train_denoise_res_unet.py`、`tools/preprocessing.py`
  - 修复：
    - `tools/preprocessing.py::normalize()` 移除 `override_stats` 对 `per="global"` 的强制限制；`override_stats` 中的 `clip_threshold` 增加按 `per` 的 reshape 广播处理。
    - `train_paired_unet.py` 和 `train_denoise_res_unet.py` 删除 `per != "global"` 的硬报错；target 的 `normalize(..., override_stats=...)` 改用与 input 相同的 `per`，支持 `shot`/`trace`/`global` 三种 scope。
  - 文档：`normalize` docstring 已更新，不再声称 "Forces per='global'"。

---

## 附录：相关文献

1. **Yilmaz, Ö. (2001).** *Seismic Data Analysis: Processing, Inversion, and Interpretation of Seismic Data* (Vol. 1). SEG.  
   → 球面发散校正：3D 振幅补偿 `gain ∝ t^1.0`，2D `gain ∝ t^0.5`。

2. **Liu, B., & Sacchi, M. D. (2004).** Minimum weighted norm interpolation of seismic records. *Geophysics*, 69(6), 1560-1568.

3. **Naghizadeh, M., & Sacchi, M. D. (2010).** On sampling functions and Fourier reconstruction methods. *Geophysics*, 75(6), WB137-WB151.

4. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).** Image quality assessment: from error visibility to structural similarity. *IEEE TIP*, 13(4), 600-612.
