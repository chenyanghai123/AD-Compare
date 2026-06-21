"""Task 4 — 批量 grounding 推理（663 张 NG × 单卡 A800）。

复用 stage3_multitask_sft_merged 模型 + 训练 grounding prompt + 模态对齐预处理：
- NG: 灰度→RGB（保持原尺寸）
- OK: 灰度→RGB→resize 到 NG 尺寸

输入: meta/eval_infos/silicon_instance_grounding/ng_index.json
输出: meta/eval_infos/silicon_instance_grounding/pred_raw.jsonl  (jsonl, 增量写, 支持续跑)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from eval.utils import (
    DEFAULT_MODEL_PATH,
    EVAL_OUT,
    GROUNDING_PROMPT,
    extract_json_from_text,
    load_aligned_pair,
    smart_resize,
)

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLForConditionalGeneration,
)
from ad_compare.dataset_ad_compare import load_processor


def build_messages(question: str, num_images: int):
    """复刻训练时 build_messages：按 <image> 占位符插入 image 块。"""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="0=全部 663")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--output", type=str, default=str(EVAL_OUT / "pred_raw.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.output)
    done_keys: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.open():
            try:
                rec = json.loads(line)
                done_keys.add(rec["ng_path"])
            except Exception:
                continue
        print(f"[task4] resume: {len(done_keys)} already done, skipping them")

    ng_index = json.loads((EVAL_OUT / "ng_index.json").read_text())
    items = list(ng_index.items())
    if args.limit:
        items = items[: args.limit]
    todo = [(k, v) for k, v in items if k not in done_keys]
    print(f"[task4] total={len(items)}  todo={len(todo)}  model={args.model_path}")

    print("[task4] loading model ...")
    t_load = time.time()
    config = AdCompareQwen3VLConfig.from_pretrained(args.model_path)
    model = AdCompareQwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval().cuda()
    processor = load_processor(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    print(f"[task4] model loaded in {time.time() - t_load:.1f}s, dtype={next(model.parameters()).dtype}, device={model.device}")

    fout = out_path.open("a")
    n_done = 0
    n_empty_pd = 0
    n_failed = 0
    t_start = time.time()
    for k, (ng_path_str, info) in enumerate(todo):
        ng_path = Path(ng_path_str)
        ok_path = Path(info["ok_path"])
        try:
            ok_img, ng_img = load_aligned_pair(ng_path, ok_path)
            images = [ok_img, ng_img]
            msgs = build_messages(GROUNDING_PROMPT, len(images))
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
            inputs = {kk: (vv.to(model.device) if isinstance(vv, torch.Tensor) else vv) for kk, vv in inputs.items()}

            t0 = time.time()
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                )
            dt = time.time() - t0
            in_len = inputs["input_ids"].shape[1]
            gen_ids = out[0, in_len:]
            pred = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            ng_w, ng_h = ng_img.size
            new_h, new_w = smart_resize(ng_h, ng_w)
            pd_list = extract_json_from_text(pred)
            if not pd_list:
                n_empty_pd += 1
            rec = {
                "ng_path": ng_path_str,
                "ok_path": str(ok_path),
                "split": info["split"],
                "ng_size_orig": [ng_w, ng_h],
                "ng_size_resized": [new_w, new_h],
                "gt": info["gt"],
                "pd_raw": pred,
                "pd": pd_list,
                "gen_tokens": int(len(gen_ids)),
                "dt": float(dt),
            }
        except Exception as e:
            n_failed += 1
            rec = {
                "ng_path": ng_path_str,
                "ok_path": str(ok_path),
                "split": info["split"],
                "error": f"{type(e).__name__}: {e}",
            }

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 10 == 0 or k < 3:
            elapsed = time.time() - t_start
            avg = elapsed / n_done
            eta = avg * (len(todo) - n_done)
            print(
                f"[task4] {n_done}/{len(todo)}  empty_pd={n_empty_pd}  fail={n_failed}  "
                f"avg={avg:.1f}s  eta={eta / 60:.1f}min"
            )

    fout.close()
    print(
        f"[task4] done. processed={n_done}  empty_pd={n_empty_pd}  failed={n_failed}  "
        f"total_time={(time.time() - t_start) / 60:.1f}min  -> {out_path}"
    )


if __name__ == "__main__":
    main()
