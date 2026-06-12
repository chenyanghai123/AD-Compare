"""
独立 LoRA merge 工具（cascade merge）

用法：
    python tools/merge_lora.py \
        --base ./checkpoints/init \
        --lora ./outputs/stage1_llm_lora \
        --out  ./checkpoints/stage1_llm_lora_merged
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLForConditionalGeneration,
)
from ad_compare.dataset_ad_compare import load_processor


def merge(base_path: str, lora_path: str, out_path: str):
    base_path = Path(base_path)
    lora_path = Path(lora_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading base: {base_path}")
    config = AdCompareQwen3VLConfig.from_pretrained(base_path)
    base_model = AdCompareQwen3VLForConditionalGeneration.from_pretrained(
        base_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"[2/4] Attaching LoRA adapter: {lora_path}")
    peft_model = PeftModel.from_pretrained(base_model, lora_path)

    print("[3/4] merge_and_unload ...")
    merged = peft_model.merge_and_unload()

    print(f"[4/4] Saving merged model to {out_path}")
    merged.save_pretrained(out_path, safe_serialization=True)

    # processor / chat_template / tokenizer
    try:
        processor = load_processor(str(lora_path))
        processor.save_pretrained(out_path)
        print("    processor saved")
    except Exception as e:
        print(f"    processor not found in lora_path, fallback to base: {e}")
        processor = load_processor(str(base_path))
        processor.save_pretrained(out_path)

    print(f"\n[done] merged checkpoint: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="AD-Compare LoRA merge tool")
    parser.add_argument("--base", required=True, help="基座（上一阶段 merged 或 init ckpt）")
    parser.add_argument("--lora", required=True, help="LoRA adapter 目录")
    parser.add_argument("--out", required=True, help="输出 merged 全模型目录")
    args = parser.parse_args()
    merge(args.base, args.lora, args.out)


if __name__ == "__main__":
    main()
