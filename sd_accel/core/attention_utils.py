# sd_accel/core/attention_utils.py
import warnings
from diffusers import StableDiffusionPipeline

def enable_xformers(pipe: StableDiffusionPipeline) -> None:
    """
    尝试启用 xFormers 内存高效的注意力机制。
    如果 xFormers 不可用，会记录警告并继续使用默认的注意力机制。
    """
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        warnings.warn(
            f"Failed to enable xFormers: {e}. "
            "Falling back to default attention mechanism. "
            "To use xFormers, install it with: pip install xformers "
            "or refer to https://github.com/facebookresearch/xformers for installation instructions.",
            UserWarning
        )
