# SD Accelerator

一个高性能的 Stable Diffusion 推理加速框架，通过混合精度优化、torch.compile 编译和自适应步数调度等技术，显著提升图像生成速度和降低显存占用。

## 特性

- 🚀 **混合精度优化**：支持 UNet、VAE、Text Encoder 分别设置不同的精度（fp32/fp16/bf16/int8），灵活平衡速度与质量
- ⚡ **torch.compile 优化**：利用 PyTorch 2.0+ 的编译优化，提升推理性能
- 🎯 **自适应步数调度**：根据生成过程中的不确定性动态调整推理步数，在保证质量的同时减少不必要的计算
- 🔧 **多种采样器支持**：支持 DDPM、DPM Solver++、DDIM 等主流采样器
- 📊 **性能基准测试**：内置 benchmark 工具，方便评估不同配置的性能表现
- ⚙️ **配置驱动**：使用 YAML 配置文件管理所有参数，易于实验和部署

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+ (支持 CUDA)
- CUDA 11.8+ (推荐)

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `torch` - PyTorch 深度学习框架
- `diffusers` - Hugging Face Diffusers 库
- `transformers` - Hugging Face Transformers 库
- `pillow` - 图像处理
- `pyyaml` - 配置文件解析

## 快速开始

### 1. 生成图像

使用配置文件生成图像：

```bash
python scripts/run_generate.py \
    --cfg configs/sd15_mix_bf16_int8_ddpm50.yaml \
    --prompt "a beautiful landscape with mountains and lakes"
```

### 2. 性能基准测试

运行性能基准测试：

```bash
python scripts/run_benchmark.py
```

或者指定配置文件：

```python
from scripts.run_benchmark import benchmark

prompts = ["a photo of a cat", "a futuristic city at sunset", "a watercolor landscape"]
benchmark("configs/sd15_mix_bf16_int8_ddpm50.yaml", prompts, iters=20, warmup=3)
```

## 配置文件说明

项目使用 YAML 配置文件管理所有参数。配置文件主要包含以下部分：

### 模型配置

```yaml
model_id: "stable-diffusion-v1-5/stable-diffusion-v1-5"
device: "cuda"
unet_dtype: "bf16"          # UNet 精度: "fp32" | "fp16" | "bf16"
vae_dtype: "bf16"           # VAE 精度: "fp32" | "fp16" | "bf16"
text_encoder_dtype: "int8"  # Text Encoder 精度: "fp32" | "fp16" | "int8"
seed: 42
```

### 编译优化

```yaml
compile:
  enabled: true
  mode: "reduce-overhead"   # "reduce-overhead" | "max-autotune"
  fullgraph: false
  dynamic: false
  compile_vae: false        # 是否编译 VAE（可能不稳定）
```

### 采样器配置

```yaml
sampler:
  name: "dpm_solver++"      # "ddpm" | "dpm_solver++" | "ddim"
  steps: 10
```

### 自适应步数调度

```yaml
adaptive:
  enabled: true
  warmup_steps: 5           # 预热步数
  min_steps: 5              # 最小步数
  max_steps: 20             # 最大步数
  uncertainty:
    method: "pred_var"      # 不确定性估计方法
    threshold_low: 0.10     # 低不确定性阈值
    threshold_high: 0.25    # 高不确定性阈值
```

### 生成参数

```yaml
generation:
  height: 512
  width: 512
  guidance_scale: 7.5
  num_images_per_prompt: 1
```

### 输出配置

```yaml
io:
  out_dir: "outputs/sd15_mix_bf16_int8_ddpm50"
  save_latents: false
  save_grid: true
```

## 预设配置

项目提供了多个预设配置文件，位于 `configs/` 目录：

- `base_sd15.yaml` - 基础配置（包含自适应步数调度）
- `sd15_fp32_ddpm50.yaml` - FP32 精度，DDPM 采样器，50 步
- `sd15_mix_bf16_int8_ddpm50.yaml` - 混合精度（UNet/VAE: bf16, Text Encoder: int8），DDPM 采样器，50 步
- `sd15_mix_bf16_int8_compile_ddpm50.yaml` - 混合精度 + torch.compile 优化
- `sd15_mix_bf16_int8_dpm10.yaml` - 混合精度，DPM Solver++ 采样器，10 步

## 项目结构

```
sd_accl/
├── configs/              # 配置文件目录
│   ├── base_sd15.yaml
│   ├── sd15_fp32_ddpm50.yaml
│   └── ...
├── scripts/              # 脚本目录
│   ├── run_generate.py   # 图像生成脚本
│   └── run_benchmark.py  # 性能基准测试脚本
├── sd_accel/             # 核心代码
│   ├── adaptive/         # 自适应步数调度
│   │   ├── step_scheduler.py
│   │   └── uncertainty.py
│   ├── core/             # 核心功能
│   │   ├── pipeline_factory.py  # Pipeline 构建工厂
│   │   ├── compile_utils.py     # torch.compile 工具
│   │   ├── optimizers.py        # 混合精度优化
│   │   ├── attention_utils.py   # Attention 优化
│   │   └── seed.py              # 随机种子设置
│   ├── eval/             # 评估工具
│   │   └── metrics.py
│   └── utils/            # 工具函数
│       └── gpu_stats.py  # GPU 统计
├── outputs/              # 输出目录
├── requirements.txt       # 依赖列表
└── README.md            # 本文件
```

## 核心功能

### 混合精度优化

项目支持对 Stable Diffusion 的不同组件分别设置精度：

- **UNet**：通常使用 bf16 或 fp16，在保持质量的同时显著提升速度
- **VAE**：可以使用 bf16 或 fp16，对最终图像质量影响较小
- **Text Encoder**：可以使用 int8 量化，进一步降低显存占用

### torch.compile 优化

利用 PyTorch 2.0+ 的 `torch.compile` 功能，可以显著提升推理速度：

- `reduce-overhead`：快速编译，适合开发调试
- `max-autotune`：深度优化，编译时间较长但性能更好

### 自适应步数调度

通过分析生成过程中的不确定性，动态调整推理步数：

1. 使用少量步数（warmup_steps）进行预热
2. 计算不确定性分数
3. 根据阈值决定最终使用的步数（min_steps 到 max_steps 之间）

这样可以避免对简单图像使用过多步数，对复杂图像使用过少步数。

## 性能优化建议

1. **混合精度配置**：
   - 推荐：UNet/VAE 使用 bf16，Text Encoder 使用 int8
   - 如果显存充足，可以尝试 fp16
   - 如果追求最高质量，可以使用 fp32

2. **torch.compile**：
   - 首次运行会进行编译，需要额外时间
   - 建议在开发时使用 `reduce-overhead`，部署时使用 `max-autotune`
   - 如果遇到问题，可以设置 `fullgraph=False` 和 `dynamic=False`

3. **采样器选择**：
   - DPM Solver++ 通常比 DDPM 更快，可以用更少步数达到相似质量
   - 10-20 步通常已经足够生成高质量图像

4. **自适应步数**：
   - 适合批量生成场景，可以根据图像复杂度自动调整
   - 对于单次生成，可以关闭以使用固定步数

## 输出说明

生成的图像会保存在配置文件中指定的 `out_dir` 目录，文件名格式为：
```
img_{index:03d}_steps{steps}.png
```

例如：`img_000_steps10.png` 表示第一张图像，使用了 10 步推理。


## 开发

### 添加新的优化策略

1. 在 `sd_accel/core/` 中添加新的优化函数
2. 在 `pipeline_factory.py` 中集成新优化
3. 在配置文件中添加相应配置项

### 扩展采样器

在 `pipeline_factory.py` 的 `build_pipeline` 函数中添加新的采样器支持。


## 致谢

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)

