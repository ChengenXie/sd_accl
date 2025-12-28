# sd_accel/core/attention_utils.py
from diffusers import StableDiffusionPipeline

def enable_xformers(pipe: StableDiffusionPipeline) -> None:
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        raise RuntimeError(f"Failed to enable xFormers: {e}")
