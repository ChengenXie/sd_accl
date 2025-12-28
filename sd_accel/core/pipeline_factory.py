# sd_accel/core/pipeline_factory.py
from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from sd_accel.core.attention_utils import enable_xformers
from sd_accel.core.compile_utils import maybe_compile
from sd_accel.core.optimizers import apply_mixed_precision
from sd_accel.adaptive.step_scheduler import AdaptiveStepController

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
    dtype_str = cfg.get("dtype", "bf16")
    
    # 映射 dtype 字符串到 torch dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    # 加载 pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)
    
    # 设置 scheduler（如果需要）
    sampler_cfg = cfg.get("sampler", {})
    sampler_name = sampler_cfg.get("name", "dpm_solver++")
    if sampler_name == "dpm_solver++":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # 应用优化
    # 1. Mixed precision 优化
    apply_mixed_precision(pipe, dtype_str)
    
    # 2. xFormers attention（如果启用）
    attention_cfg = cfg.get("attention", {})
    if attention_cfg.get("xformers", False):
        enable_xformers(pipe)
    
    # 3. torch.compile（如果启用）
    compile_cfg = cfg.get("compile", {})
    pipe = maybe_compile(pipe, compile_cfg)
    
    # 4. 创建 adaptive controller（如果启用）
    adaptive_controller = None
    adaptive_cfg = cfg.get("adaptive", {})
    if adaptive_cfg.get("enabled", False):
        adaptive_controller = AdaptiveStepController.from_config(adaptive_cfg)
    
    return PipelineBundle(pipe=pipe, adaptive_controller=adaptive_controller)

