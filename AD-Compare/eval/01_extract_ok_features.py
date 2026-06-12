"""Task 1 — 抽 OK 池 (15380 张) 视觉特征。

外网不可用，且仓库内未提供 SigLIP/CLIP 本地权重；改用与 Qwen3-VL ViT 同源的轻量
``raw`` 策略（灰度 → 64×64 → flatten → L2-normalize）。

理由：
1. OK 池均为 384×384 灰度晶圆纹理图，简单的像素相似度足以反映 "样式 / 区域分布" 相似度；
2. 不依赖外部模型下载，30 秒内即可抽完；
3. 可作为 stage3 model.visual ViT embedding 不可用时的稳定 fallback。

输出:
    meta/eval_infos/silicon_instance_grounding/ok_features.npy   shape=(15380, 4096) float32
    meta/eval_infos/silicon_instance_grounding/ok_paths.txt      每行一个 OK 绝对路径
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))  # 让项目根目录可导入
from eval.utils import EVAL_OUT, list_ok_paths


def embed_raw(path: Path, size: int = 64) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).flatten()
    arr -= float(arr.mean())
    n = float(np.linalg.norm(arr))
    if n > 1e-8:
        arr = arr / n
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    ok_paths = list_ok_paths()
    if args.limit:
        ok_paths = ok_paths[: args.limit]
    print(f"[task1] OK pool: {len(ok_paths)} images, embed_dim={args.size * args.size}")

    feats = np.zeros((len(ok_paths), args.size * args.size), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(ok_paths):
        feats[i] = embed_raw(p, args.size)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(ok_paths)}  elapsed={time.time() - t0:.1f}s")

    out_npy = EVAL_OUT / "ok_features.npy"
    out_txt = EVAL_OUT / "ok_paths.txt"
    np.save(out_npy, feats)
    out_txt.write_text("\n".join(str(p) for p in ok_paths) + "\n")
    print(f"[task1] saved -> {out_npy}  shape={feats.shape}  size={out_npy.stat().st_size / 1e6:.1f}MB")
    print(f"[task1] saved -> {out_txt}")
    print(f"[task1] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
