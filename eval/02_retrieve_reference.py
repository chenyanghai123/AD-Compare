"""为每张 NG 检索 top-1 OK reference + 整合 GT bbox。

输出:
    {EVAL_OUT}/pair_map.json
    {EVAL_OUT}/ng_index.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from eval.utils import (
    EVAL_OUT,
    TRAIN_CSV,
    VAL_CSV,
    list_ng_paths,
    load_split_set,
    ng_to_tzjson,
    parse_tzjson,
)


def embed_raw(path: Path, size: int = 64) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32).flatten()
    arr -= float(arr.mean())
    n = float(np.linalg.norm(arr))
    if n > 1e-8:
        arr = arr / n
    return arr


def main() -> None:
    ok_features_path = EVAL_OUT / "ok_features.npy"
    ok_paths_txt = EVAL_OUT / "ok_paths.txt"
    if not ok_features_path.exists() or not ok_paths_txt.exists():
        raise SystemExit(f"missing OK features: run 01_extract_ok_features.py first")

    ok_features = np.load(ok_features_path)               # (15380, 4096)
    ok_paths = [Path(p) for p in ok_paths_txt.read_text().splitlines() if p.strip()]
    print(f"[task2] OK features: {ok_features.shape}, paths: {len(ok_paths)}")

    ng_paths = list_ng_paths()
    train_set = load_split_set(TRAIN_CSV)
    val_set = load_split_set(VAL_CSV)
    print(
        f"[task2] NG: {len(ng_paths)}  (train={len(train_set)}, val={len(val_set)}, "
        f"covered={sum(p.name in train_set or p.name in val_set for p in ng_paths)})"
    )

    # 编码 NG
    ng_feats = np.zeros((len(ng_paths), ok_features.shape[1]), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(ng_paths):
        ng_feats[i] = embed_raw(p)
    print(f"[task2] NG embedded in {time.time() - t0:.1f}s")

    # 检索 top-1
    sim = ng_feats @ ok_features.T                        # (663, 15380)
    best_idx = sim.argmax(axis=1)
    best_sim = sim[np.arange(len(ng_paths)), best_idx]

    pair_map: dict[str, dict] = {}
    ng_index: dict[str, dict] = {}
    n_with_gt = 0
    n_total_box = 0
    label_count: dict[str, int] = {}
    for i, ng_p in enumerate(ng_paths):
        ok_p = ok_paths[int(best_idx[i])]
        gt = parse_tzjson(ng_to_tzjson(ng_p))
        for g in gt:
            label_count[g["label"]] = label_count.get(g["label"], 0) + 1
        if gt:
            n_with_gt += 1
            n_total_box += len(gt)
        with Image.open(ng_p) as im:
            ng_w, ng_h = im.size
        split = "train" if ng_p.name in train_set else ("val" if ng_p.name in val_set else "unknown")
        pair_map[str(ng_p)] = {
            "ok_path": str(ok_p),
            "similarity": float(best_sim[i]),
            "ng_size": [ng_w, ng_h],
            "split": split,
        }
        ng_index[str(ng_p)] = {
            "ok_path": str(ok_p),
            "similarity": float(best_sim[i]),
            "ng_size": [ng_w, ng_h],
            "split": split,
            "gt": [{"bbox": g["bbox"], "label": g["label"]} for g in gt],
        }

    out_pair = EVAL_OUT / "pair_map.json"
    out_index = EVAL_OUT / "ng_index.json"
    out_pair.write_text(json.dumps(pair_map, ensure_ascii=False, indent=2))
    out_index.write_text(json.dumps(ng_index, ensure_ascii=False, indent=2))
    print(f"[task2] -> {out_pair}")
    print(f"[task2] -> {out_index}")
    print(
        f"[task2] NG with GT: {n_with_gt}/{len(ng_paths)}, total GT bboxes: {n_total_box}, "
        f"avg per NG: {n_total_box / max(len(ng_paths), 1):.2f}"
    )
    print(f"[task2] label distribution: {label_count}")
    print(f"[task2] similarity stats: min={best_sim.min():.3f}  max={best_sim.max():.3f}  mean={best_sim.mean():.3f}")

    print("\n[task2] sanity check (5 random pairs):")
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(ng_paths), size=min(5, len(ng_paths)), replace=False)
    for k, i in enumerate(sample_idx):
        ng_p = ng_paths[int(i)]
        ok_p = ok_paths[int(best_idx[int(i)])]
        gt = ng_index[str(ng_p)]["gt"]
        print(f"  [{k}] sim={best_sim[int(i)]:.3f}  ng={ng_p.name}  ok={ok_p.name}  n_gt={len(gt)}")


if __name__ == "__main__":
    main()
