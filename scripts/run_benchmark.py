# scripts/run_benchmark.py
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time, yaml
import torch
from sd_accel.core.pipeline_factory import build_pipeline
from sd_accel.utils.gpu_stats import reset_peak_memory, peak_memory_gb
from sd_accel.core.seed import seed_everything

def benchmark(cfg_path: str, prompts: list[str], iters: int = 20, warmup: int = 3):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))
    bundle = build_pipeline(cfg)
    pipe = bundle.pipe

    gen_cfg = cfg.get("generation", {})
    steps = int(cfg.get("sampler", {}).get("steps", 10))

    # warmup
    for _ in range(warmup):
        _ = pipe(prompts[0], num_inference_steps=steps, output_type="latent")
    torch.cuda.synchronize()

    times = []
    reset_peak_memory(cfg.get("device", "cuda"))
    for i in range(iters):
        prompt = prompts[i % len(prompts)]
        t0 = time.perf_counter()
        _ = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=float(gen_cfg.get("guidance_scale", 7.5)),
            height=int(gen_cfg.get("height", 512)),
            width=int(gen_cfg.get("width", 512)),
            output_type="latent",
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mem = peak_memory_gb(cfg.get("device", "cuda"))
    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times)//2]
    print(f"avg={avg*1000:.1f}ms p50={p50*1000:.1f}ms throughput={1/avg:.2f} img/s peak_mem={mem:.2f}GB")

if __name__ == "__main__":
    prompts = ["a photo of a cat", "a futuristic city at sunset", "a watercolor landscape"]
    benchmark("configs/ours_adaptive.yaml", prompts)
