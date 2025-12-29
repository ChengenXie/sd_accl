import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time, yaml
import torch
from PIL import Image

from sd_accel.core.pipeline_factory import build_pipeline
from sd_accel.utils.gpu_stats import reset_peak_memory, peak_memory_gb
from sd_accel.core.seed import seed_everything

def save_image(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def main(cfg_path: str, prompt: str):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))

    bundle = build_pipeline(cfg)
    pipe = bundle.pipe
    adaptive = bundle.adaptive_controller

    gen_cfg = cfg.get("generation", {})
    sampler_cfg = cfg.get("sampler", {})
    out_dir = cfg.get("io", {}).get("out_dir", "outputs")

    # 决策步数：adaptive启用则先warmup估计再决定；否则用配置固定steps
    if adaptive is None:
        total_steps = int(sampler_cfg.get("steps", 10))
    else:
        total_steps = run_adaptive_decision(pipe, adaptive, prompt, cfg)

    reset_peak_memory(cfg.get("device", "cuda"))
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=pipe.unet.dtype):
            out = pipe(
                prompt=prompt,
                num_inference_steps=total_steps,
                guidance_scale=float(gen_cfg.get("guidance_scale", 7.5)),
                height=int(gen_cfg.get("height", 512)),
                width=int(gen_cfg.get("width", 512)),
                num_images_per_prompt=int(gen_cfg.get("num_images_per_prompt", 1)),
            )

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    latency_s = t1 - t0
    mem_gb = peak_memory_gb(cfg.get("device", "cuda"))

    for i, img in enumerate(out.images):
        save_image(img, os.path.join(out_dir, f"img_{i:03d}_steps{total_steps}.png"))

    print(f"[DONE] steps={total_steps} latency={latency_s*1000:.1f}ms peak_mem={mem_gb:.2f}GB")

@torch.no_grad()
def run_adaptive_decision(pipe, adaptive, prompt: str, cfg):
    warmup_steps = adaptive.cfg.warmup_steps
    proxy_score = estimate_uncertainty_quick(pipe, prompt, warmup_steps, cfg)
    total_steps = adaptive.decide_total_steps(proxy_score)
    print(f"[ADAPT] warmup_steps={warmup_steps} uncertainty={proxy_score:.4f} -> total_steps={total_steps}")
    return total_steps

def estimate_uncertainty_quick(pipe, prompt: str, warmup_steps: int, cfg) -> float:
   
    _ = pipe(
        prompt=prompt,
        num_inference_steps=warmup_steps,
        guidance_scale=float(cfg.get("generation", {}).get("guidance_scale", 7.5)),
        height=int(cfg.get("generation", {}).get("height", 512)),
        width=int(cfg.get("generation", {}).get("width", 512)),
        output_type="latent",
    )
    return 0.18  # TODO: 替换成真实的不确定性估计

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()
    main(args.cfg, args.prompt)
