# sd_accel/core/optimizers.py
from diffusers import StableDiffusionPipeline

def apply_mixed_precision(pipe: StableDiffusionPipeline, dtype_str: str) -> None:
    # diffusers里多数权重dtype已经由 from_pretrained(torch_dtype=...) 决定
    # 你可以在这里放一些额外设定，比如关闭某些不必要模块、VAE slicing等
    pipe.enable_vae_slicing()     # 降显存（可能稍慢）
    pipe.enable_attention_slicing()  # 降显存（可能稍慢）
