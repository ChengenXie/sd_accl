# sd_accel/adaptive/uncertainty.py
from dataclasses import dataclass
from typing import Any, Dict
import torch

@dataclass
class UncertaintyResult:
    score: float
    details: dict

@torch.no_grad()
def uncertainty_pred_variance(pred_eps: torch.Tensor) -> UncertaintyResult:
    """
    pred_eps: [B, C, H, W] UNet 预测噪声/残差（或v-pred），用方差当作不确定性proxy。
    """
    # 一个简单可用的proxy：按样本算空间方差，再取batch平均
    per_sample = pred_eps.float().var(dim=(1,2,3), unbiased=False)  # [B]
    score = per_sample.mean().item()
    return UncertaintyResult(score=score, details={"per_sample_var": per_sample.cpu()})

@torch.no_grad()
def estimate_uncertainty_from_pipeline(pipe, prompt: str, warmup_steps: int, cfg: Dict[str, Any]) -> float:
    """
    从 pipeline 快速估计不确定性：执行一步采样，获取 UNet 的预测噪声，计算方差作为不确定性。
    
    Args:
        pipe: StableDiffusionPipeline 实例
        prompt: 文本提示
        warmup_steps: 预热步数（用于设置 scheduler）
        cfg: 配置字典，包含 generation 等配置
        
    Returns:
        float: 不确定性分数
    """
    gen_cfg = cfg.get("generation", {})
    guidance_scale = float(gen_cfg.get("guidance_scale", 7.5))
    height = int(gen_cfg.get("height", 512))
    width = int(gen_cfg.get("width", 512))
    num_images_per_prompt = int(gen_cfg.get("num_images_per_prompt", 1))
    
    # 准备文本 embeddings
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_inputs.input_ids.to(pipe.device))[0]
    
    # 准备 unconditional embeddings (negative prompt)
    uncond_tokens = [""] * num_images_per_prompt
    uncond_inputs = pipe.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids.to(pipe.device))[0]
    
    # 合并 conditional 和 unconditional embeddings (用于 classifier-free guidance)
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # 生成初始 latent (随机噪声)
    latents_shape = (num_images_per_prompt, pipe.unet.config.in_channels, height // 8, width // 8)
    latents = torch.randn(
        latents_shape,
        dtype=pipe.unet.dtype,
        device=pipe.device,
        generator=torch.Generator(device=pipe.device).manual_seed(cfg.get("seed", 42)),
    )
    # 根据 scheduler 类型设置初始噪声尺度
    if hasattr(pipe.scheduler, "init_noise_sigma"):
        latents = latents * pipe.scheduler.init_noise_sigma
    elif hasattr(pipe.scheduler, "sigma_max"):
        # 某些 scheduler 使用 sigma_max
        latents = latents * pipe.scheduler.sigma_max
    
    # 设置 scheduler 的步数
    pipe.scheduler.set_timesteps(warmup_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    
    # 选择第一个 timestep 来评估不确定性（通常包含更多信息）
    t = timesteps[0:1].expand(num_images_per_prompt)
    
    # 扩展 latents 用于 classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t[0])
    
    # 调用 UNet 获取预测噪声
    with torch.autocast(device_type="cuda", dtype=pipe.unet.dtype):
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample
    
    # 分离 conditional 和 unconditional 预测
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
    # 使用 conditional 预测（noise_pred_text）来计算不确定性
    # 这是模型对去噪过程的预测，方差反映了不确定性
    pred_eps = noise_pred_text  # [B, C, H, W]
    
    # 计算不确定性分数
    uncertainty_result = uncertainty_pred_variance(pred_eps)
    
    return uncertainty_result.score
