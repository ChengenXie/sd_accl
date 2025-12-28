# sd_accel/eval/metrics.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalResult:
    fid: Optional[float] = None
    clip_score: Optional[float] = None

def compute_fid(fake_dir: str, real_dir: str) -> float:
    # TODO: 接 torch-fidelity / clean-fid / 自己实现
    raise NotImplementedError

def compute_clip_score(image_dir: str, prompts_file: str) -> float:
    # TODO: 用 open_clip 或 transformers CLIPModel 做图文相似度
    raise NotImplementedError
