# sd_accel/core/optimizers.py
import torch
from diffusers import StableDiffusionPipeline

def apply_mixed_precision(pipe: StableDiffusionPipeline, unet_dtype: torch.dtype, vae_dtype: torch.dtype, text_dtype) -> None:
    if unet_dtype is not None and unet_dtype != torch.float32:
        pipe.unet.to(device=pipe.device, dtype=unet_dtype)
    
    if vae_dtype is not None and vae_dtype != torch.float32:
        pipe.vae.to(device=pipe.device, dtype=vae_dtype)
    
    if text_dtype is not None and getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to(device=pipe.device, dtype=text_dtype)
  
