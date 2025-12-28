# sd_accel/adaptive/uncertainty.py
from dataclasses import dataclass
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
