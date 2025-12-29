# sd_accel/core/pipeline_factory.py
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)
from transformers import CLIPTextModel

from sd_accel.core.compile_utils import torch_compile
from sd_accel.core.optimizers import apply_mixed_precision
from sd_accel.adaptive.step_scheduler import AdaptiveStepController

def _dtype_from_str(s: str) -> torch.dtype:
    m = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if s not in m:
        raise ValueError(f"Unknown dtype: {s}, expected one of {list(m.keys())}")
    return m[s]

@dataclass
class PipelineBundle:
    pipe: StableDiffusionPipeline
    adaptive_controller: Optional[AdaptiveStepController] = None

def build_pipeline(cfg: Dict[str, Any]) -> PipelineBundle:
    """
    根据配置构建并优化 Stable Diffusion pipeline。
    
    Args:
        cfg: 配置字典，包含 model_id, device, dtype, compile, attention, adaptive 等配置
        
    Returns:
        PipelineBundle: 包含优化后的 pipeline 和可选的 adaptive controller
    """
    model_id = cfg.get("model_id", "stabilityai/stable-diffusion-2-1")
    device = cfg.get("device", "cuda")
    # dtype_str = cfg.get("dtype", "bf16")
    
    # # 映射 dtype 字符串到 torch dtype
    # dtype_map = {
    #     "fp32": torch.float32,
    #     "fp16": torch.float16,
    #     "bf16": torch.bfloat16,
    # }
    # torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # 加载 pipeline
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     model_id,
    #     torch_dtype=torch_dtype,
    # )
    # pipe = pipe.to(device)
    
    unet_dtype_str = cfg.get("unet_dtype", "bf16")
    vae_dtype_str = cfg.get("vae_dtype", "fp16")
    text_dtype_str = cfg.get("text_encoder_dtype", "fp16")  # "fp16" or "int8" or "fp32"

    unet_dtype = _dtype_from_str(unet_dtype_str)
    vae_dtype = _dtype_from_str(vae_dtype_str)
    text_dtype = None if text_dtype_str == "int8" else _dtype_from_str(text_dtype_str)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  
        safety_checker=None,         
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    if text_dtype_str == "int8":
        text_encoder_8bit = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            load_in_8bit=True,
            device_map={"": 0} if device == "cuda" else {"": "cpu"},
        )
        pipe.text_encoder = text_encoder_8bit
    
    # 设置 scheduler
    sampler_cfg = cfg.get("sampler", {})
    sampler_name = sampler_cfg.get("name", "dpm_solver++")
    if sampler_name == "dpm_solver++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif sampler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 应用优化
    # Mixed precision 优化
    apply_mixed_precision(pipe, unet_dtype, vae_dtype, text_dtype)
    
    
    # torch.compile
    compile_cfg = cfg.get("compile", {})
    pipe = torch_compile(pipe, compile_cfg)
    
    # 创建 adaptive controller
    adaptive_controller = None
    adaptive_cfg = cfg.get("adaptive", {})
    if adaptive_cfg.get("enabled", False):
        adaptive_controller = AdaptiveStepController.from_config(adaptive_cfg)
    
    return PipelineBundle(pipe=pipe, adaptive_controller=adaptive_controller)

