# sd_accel/adaptive/step_scheduler.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AdaptiveConfig:
    warmup_steps: int
    min_steps: int
    max_steps: int
    method: str
    thr_low: float
    thr_high: float

class AdaptiveStepController:
    def __init__(self, cfg: AdaptiveConfig):
        self.cfg = cfg

    @staticmethod
    def from_config(adaptive_cfg: Dict[str, Any]) -> "AdaptiveStepController":
        ucfg = adaptive_cfg.get("uncertainty", {})
        cfg = AdaptiveConfig(
            warmup_steps=int(adaptive_cfg.get("warmup_steps", 5)),
            min_steps=int(adaptive_cfg.get("min_steps", 5)),
            max_steps=int(adaptive_cfg.get("max_steps", 20)),
            method=str(ucfg.get("method", "pred_var")),
            thr_low=float(ucfg.get("threshold_low", 0.10)),
            thr_high=float(ucfg.get("threshold_high", 0.25)),
        )
        return AdaptiveStepController(cfg)

    def decide_total_steps(self, uncertainty_score: float) -> int:
        """
        简单三段式映射：低→min_steps，中→(min+max)/2，高→max_steps
        你后面可以替换成更平滑的映射函数。
        """
        if uncertainty_score <= self.cfg.thr_low:
            return self.cfg.min_steps
        if uncertainty_score >= self.cfg.thr_high:
            return self.cfg.max_steps
        return int((self.cfg.min_steps + self.cfg.max_steps) / 2)
