# sd_accel/core/compile_utils.py
from typing import Dict, Any
import torch
from diffusers import StableDiffusionPipeline

def torch_compile(pipe: StableDiffusionPipeline, cfg: Dict[str, Any]) -> StableDiffusionPipeline:
    if not cfg.get("enabled", False):
        return pipe

    mode = cfg.get("mode", "max-autotune")
    fullgraph = bool(cfg.get("fullgraph", False))  # 不稳定时用 False
    dynamic = bool(cfg.get("dynamic", False))      # 固定分辨率/固定batch False

    pipe.unet = torch.compile(pipe.unet, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    if cfg.get("compile_vae", False):
        pipe.vae = torch.compile(pipe.vae, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
    return pipe
