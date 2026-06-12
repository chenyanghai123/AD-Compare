"""
AD-Compare 推理测试脚本

用法:
    CUDA_VISIBLE_DEVICES=0 python tools/infer.py \
        --model_path ./checkpoints/stage3_multitask_sft_merged \
        --data_path ./data/stage3_real.json \
        --num_per_task 2 --max_new_tokens 256
"""
import os
import sys
import json
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLForConditionalGeneration,
)
from ad_compare.dataset_ad_compare import load_processor


def open_image(path: str, max_side: int = 448) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image missing: {path}")
    img = Image.open(p).convert("RGB")
    if max_side > 0 and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    return img


def build_messages(question: str, num_images: int):
    """把 Q 文本中的 <image> 占位符转换为 chat-template 的 image content block。"""
    parts = question.split("<image>")
    contents = []
    img_idx = 0
    for i, seg in enumerate(parts):
        if i > 0 and img_idx < num_images:
            contents.append({"type": "image"})
            img_idx += 1
        if seg:
            contents.append({"type": "text", "text": seg})
    while img_idx < num_images:
        contents.insert(0, {"type": "image"})
        img_idx += 1
    return [{"role": "user", "content": contents}]


def run_one(model, processor, item, max_new_tokens: int, max_image_side: int):
    images = [open_image(p, max_image_side) for p in item["images"]]
    question = item["messages"][0]["content"]
    answer_gt = item["messages"][1]["content"]
    msgs = build_messages(question, len(images))

    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    )
    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, in_len:]
    pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return question, answer_gt, pred, dt, len(gen_ids)


def main():
    ap = argparse.ArgumentParser(description="AD-Compare inference test")
    ap.add_argument("--model_path", default="./checkpoints/stage3_multitask_sft_merged")
    ap.add_argument("--data_path", default="./data/stage3_real.json")
    ap.add_argument("--num_per_task", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_image_side", type=int, default=448)
    ap.add_argument("--attn_implementation", default="sdpa")
    args = ap.parse_args()

    print("=" * 68)
    print(f"AD-Compare Qwen3-VL-8B Inference Test")
    print(f"  model: {args.model_path}")
    print(f"  data : {args.data_path}")
    print(f"  num_per_task: {args.num_per_task}, max_new_tokens: {args.max_new_tokens}")
    print("=" * 68)

    config = AdCompareQwen3VLConfig.from_pretrained(args.model_path)
    model = AdCompareQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    processor = load_processor(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"[ok] model loaded, dtype={next(model.parameters()).dtype}, device={model.device}")

    data = json.load(open(args.data_path))
    by_task = {}
    for item in data:
        paths = item.get("images") or item.get("image") or []
        if isinstance(paths, str):
            paths = [paths]
        if not paths or not all(os.path.exists(p) for p in paths):
            continue
        t = item.get("task_type", "?")
        by_task.setdefault(t, []).append(item)

    print(f"[data] usable task counts:", {t: len(v) for t, v in by_task.items()})
    print()

    total = 0
    for t, items in by_task.items():
        print("=" * 68)
        print(f"[task] {t}   ({min(args.num_per_task, len(items))} samples)")
        print("=" * 68)
        for i, item in enumerate(items[: args.num_per_task]):
            try:
                q, gt, pred, dt, n_tok = run_one(
                    model, processor, item, args.max_new_tokens, args.max_image_side
                )
            except Exception as e:
                print(f"  [{t} #{i}]  FAILED: {e}")
                continue
            print(f"  ---- [{t} #{i}]  gen_tokens={n_tok}  time={dt:.1f}s ----")
            print(f"  Q : {q[:300]}")
            print(f"  GT: {gt[:300]}")
            print(f"  PD: {pred[:600]}")
            print()
            total += 1

    print(f"[done] inferred {total} samples.")


if __name__ == "__main__":
    main()
