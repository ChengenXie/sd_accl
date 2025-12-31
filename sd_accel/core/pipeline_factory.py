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
from sd_accel.core.fx_optimizer import FXUNetOptimizer


def _dtype_from_str(s: str) -> torch.dtype:
    m = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if s not in m:
        raise ValueError(f"Unknown dtype: {s}, expected one of {list(m.keys())}")
    return m[s]

def build_pipeline_with_custom_scheduler(cfg: Dict[str, Any], pipe: StableDiffusionPipeline) :
    
    sampler_cfg = cfg.get("sampler", {})
    sampler_name = sampler_cfg.get("name", "dpm_solver++")
    
    if sampler_name == "dpm_solver++_custom":
        from sd_accel.schedulers.dpm_solver_pp import DPMSolverPP, DPMSolverPPConfig
        
        scheduler_config = DPMSolverPPConfig(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            solver_order=sampler_cfg.get("solver_order", 2),
            prediction_type=sampler_cfg.get("prediction_type", "epsilon"),
            algorithm_type="dpmsolver++",
            solver_type=sampler_cfg.get("solver_type", "midpoint"),
            lower_order_final=True,
        )
        pipe.scheduler = DPMSolverPP(scheduler_config)
    return PipelineBundle(pipe=pipe)



def apply_fx_optimization(pipe, fx_config: Dict[str, Any]):
    """
    Args:
        pipe: StableDiffusionPipeline
        fx_config: FX 优化配置
    """
    if not fx_config.get("enabled", False):
        return pipe
    
    
    fx_optimizer = FXUNetOptimizer(
        enable_fusion=fx_config.get("enable_fusion", True),
        enable_constant_folding=fx_config.get("enable_constant_folding", True),
        enable_dce=fx_config.get("enable_dce", True),
        cache_size=fx_config.get("cache_size", 16)
    )
    
    pipe.unet = fx_optimizer.optimize_unet(pipe.unet)
    
    pipe._fx_optimizer = fx_optimizer
    
    
    return pipe

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
    sampler_name = sampler_cfg.get("name", "dpm_solver++_custom")

    if sampler_name == "dpm_solver++_custom":
        # 使用自定义实现的 DPM-Solver++
        from sd_accel.schedulers.dpm_solver_pp import DPMSolverPP, DPMSolverPPConfig
    
        scheduler_config = DPMSolverPPConfig(
            num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
            beta_start=pipe.scheduler.config.beta_start,
            beta_end=pipe.scheduler.config.beta_end,
            beta_schedule=pipe.scheduler.config.beta_schedule,
            solver_order=sampler_cfg.get("solver_order", 2),
            prediction_type=getattr(pipe.scheduler.config, "prediction_type", "epsilon"),
            algorithm_type="dpmsolver++",
            solver_type=sampler_cfg.get("solver_type", "midpoint"),
            lower_order_final=sampler_cfg.get("lower_order_final", True),
            thresholding=sampler_cfg.get("thresholding", False),
            dynamic_thresholding_ratio=sampler_cfg.get("dynamic_thresholding_ratio", 0.995),
            sample_max_value=sampler_cfg.get("sample_max_value", 1.0),
        )
        pipe = build_pipeline_with_custom_scheduler(cfg, pipe)
    
    elif sampler_name == "dpm_solver++":
    # 使用 diffusers 的 DPM-Solver++
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = build_pipeline_with_custom_scheduler(cfg, pipe)
    
    elif sampler_name == "ddpm": 
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe = build_pipeline_with_custom_scheduler(cfg, pipe)
    
    # Mixed precision 优化
    apply_mixed_precision(pipe, unet_dtype, vae_dtype, text_dtype)

    # FX 优化（在 torch.compile 之前）
    fx_cfg = cfg.get("fx_optimization", {})
    if fx_cfg.get("enabled", False):
        pipe = apply_fx_optimization(pipe, fx_cfg)

    # torch.compile
    compile_cfg = cfg.get("compile", {})
    if compile_cfg.get("enabled", False):
        pipe = torch_compile(pipe, compile_cfg)

    
    # 创建 adaptive controller
    adaptive_controller = None
    adaptive_cfg = cfg.get("adaptive", {})
    if adaptive_cfg.get("enabled", False):
        adaptive_controller = AdaptiveStepController.from_config(adaptive_cfg)
    
    return PipelineBundle(pipe=pipe, adaptive_controller=adaptive_controller)

