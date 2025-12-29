from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import os
import glob

@dataclass
class EvalResult:
    fid: Optional[float] = None
    clip_score: Optional[float] = None


# -----------------------
# Helpers
# -----------------------
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

def _list_images(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    paths = []
    for ext in _IMG_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(image_dir, f"*{ext.upper()}")))

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir} (exts={_IMG_EXTS})")
    return paths


def _read_prompts(prompts_file: str) -> List[str]:
    if not os.path.isfile(prompts_file):
        raise FileNotFoundError(f"prompts_file not found: {prompts_file}")

    with open(prompts_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    prompts = [p for p in lines if p]
    if not prompts:
        raise ValueError(f"prompts_file is empty: {prompts_file}")
    return prompts


def _align_prompts_to_images(img_paths: List[str], prompts: List[str]) -> List[str]:
    """
    支持两种常见格式：
    1) prompts.txt 多行：行数 == 图片数，按排序后的图片顺序一一对应
    2) prompts.txt 单行：认为是固定 prompt，自动广播到所有图片
    """
    if len(prompts) == 1:
        return [prompts[0]] * len(img_paths)

    if len(prompts) != len(img_paths):
        raise ValueError(
            "prompts 与图片数量不一致。\n"
            f"  num_prompts={len(prompts)}\n"
            f"  num_images={len(img_paths)}\n"
            "如果你是固定 prompt，请让 prompts_file 只写一行；\n"
            "如果是多 prompt，请确保行数与图片数一致，并且图片按文件名排序后与 prompts 顺序一致。"
        )
    return prompts


# -----------------------
# Metrics
# -----------------------
def compute_fid(fake_dir: str, real_dir: str) -> float:
    """
    FID(fake vs real) using clean-fid.

    说明：
    - fake_dir: 生成图片目录（不要是 grid 拼图）
    - real_dir: 真实图片目录（建议同类别/同分布，比如真实猫图集）
    """
    try:
        from cleanfid import fid as cleanfid_fid
    except Exception as e:
        raise ImportError(
            "未安装 clean-fid。请运行：pip install clean-fid"
        ) from e

    # 让它先检查目录里是否确实有图片，避免算到一半才报错
    _ = _list_images(fake_dir)
    _ = _list_images(real_dir)

    # clean-fid 的 compute_fid 会自己做必要的预处理
    score = cleanfid_fid.compute_fid(
        fake_dir,
        real_dir,
        device="cuda"  # 没有 CUDA 的话 clean-fid 内部也可能回退，但这里先明确
    )
    return float(score)


def compute_clip_score(
    image_dir: str,
    prompts_file: str,
    model_name: str = "ViT-g-14",
    pretrained: str = "laion2b_s34b_b88k",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> float:
    """
    CLIP 图文相似度（cosine similarity）均值。
    - image_dir: 图片目录
    - prompts_file: 文本文件（1 行：固定 prompt；多行：与图片一一对应）
    - model_name/pretrained: open_clip 模型配置
    """
    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception as e:
        raise ImportError(
            "缺少依赖。请运行：pip install open_clip_torch pillow torch"
        ) from e

    img_paths = _list_images(image_dir)
    prompts = _read_prompts(prompts_file)
    prompts = _align_prompts_to_images(img_paths, prompts)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device).eval()

    # 逐 batch 计算，避免一次性塞太多导致显存炸
    sims: List[float] = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]

            images = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                images.append(preprocess(img))
            image_tensor = torch.stack(images, dim=0).to(device)

            text_tokens = tokenizer(batch_prompts).to(device)

            image_feat = model.encode_image(image_tensor)
            text_feat = model.encode_text(text_tokens)

            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

            # cosine similarity per sample: diag(image_feat @ text_feat.T)
            sim_mat = image_feat @ text_feat.T
            sim_diag = sim_mat.diag().detach().float().cpu().tolist()
            sims.extend(sim_diag)

    if not sims:
        raise RuntimeError("No CLIP similarities computed (unexpected).")

    return float(sum(sims) / len(sims))


# -----------------------
# Optional: convenience runner
# -----------------------
def evaluate(
    fake_dir: Optional[str] = None,
    real_dir: Optional[str] = None,
    image_dir: Optional[str] = None,
    prompts_file: Optional[str] = None,
) -> EvalResult:
    """
    一个方便用的封装：想算哪个就传哪个参数。
    """
    res = EvalResult()

    if fake_dir is not None and real_dir is not None:
        res.fid = compute_fid(fake_dir=fake_dir, real_dir=real_dir)

    if image_dir is not None and prompts_file is not None:
        res.clip_score = compute_clip_score(image_dir=image_dir, prompts_file=prompts_file)

    return res
