# sd_accel/utils/gpu_stats.py
import torch

def reset_peak_memory(device="cuda"):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)

def peak_memory_gb(device="cuda") -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device=device) / (1024**3)
