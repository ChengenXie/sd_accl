# sd_accel/core/compile_utils.py
from typing import Dict, Any
import torch
from diffusers import StableDiffusionPipeline

def maybe_compile(pipe: StableDiffusionPipeline, cfg: Dict[str, Any]) -> StableDiffusionPipeline:
    if not cfg.get("enabled", False):
        return pipe

    mode = cfg.get("mode", "reduce-overhead")
    fullgraph = cfg.get("fullgraph", False)

    # 常见做法：compile UNet（收益最大），VAE/文本编码器可选
    pipe.unet = torch.compile(pipe.unet, mode=mode, fullgraph=fullgraph)
    return pipe
