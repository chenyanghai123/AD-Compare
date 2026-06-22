"""
从官方 Qwen3-VL-8B-Instruct 构造 AD-Compare 初始检查点：
- 复用 Qwen3-VL-8B 的 ViT + LLM 权重
- 随机初始化 Comparison Encoder
- 写入 AdCompareQwen3VLConfig（含 compare_token_size=100）
- 一并保存 Processor / chat_template / tokenizer

用法：
    python -m ad_compare.build_initial_checkpoint \
        --src /path/to/Qwen3-VL-8B-Instruct \
        --dst /path/to/AD-Compare-Qwen3-VL-8B-init
"""
import argparse
import shutil
from pathlib import Path

import torch

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLVisionConfig,
    AdCompareQwen3VLForConditionalGeneration,
    AdCompareQwen3VLProcessor,
)
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLConfig,
    AutoTokenizer,
    AutoImageProcessor,
    AutoVideoProcessor,
)


def build(src: str, dst: str, compare_token_size: int = 100):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    # 1. 加载源 config，转换为 AdCompare config
    print(f"[1/5] 读取源 config: {src}/config.json")
    src_cfg = Qwen3VLConfig.from_pretrained(src)
    new_cfg = AdCompareQwen3VLConfig(
        text_config=src_cfg.text_config.to_dict(),
        vision_config={
            **src_cfg.vision_config.to_dict(),
            "compare_token_size": compare_token_size,
        },
        image_token_id=src_cfg.image_token_id,
        video_token_id=src_cfg.video_token_id,
        vision_start_token_id=src_cfg.vision_start_token_id,
        vision_end_token_id=src_cfg.vision_end_token_id,
        tie_word_embeddings=getattr(src_cfg, "tie_word_embeddings", False),
    )
    new_cfg.architectures = ["AdCompareQwen3VLForConditionalGeneration"]

    # 2. 加载源模型权重
    print(f"[2/5] 加载源模型权重 (bf16): {src}")
    src_model = Qwen3VLForConditionalGeneration.from_pretrained(
        src, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )

    # 3. 构造目标模型骨架并复制权重
    print(f"[3/5] 构造 AdCompare 模型骨架（CE 随机初始化）")
    new_model = AdCompareQwen3VLForConditionalGeneration(new_cfg)
    new_model = new_model.to(torch.bfloat16)

    src_state = src_model.state_dict()
    missing, unexpected = new_model.load_state_dict(src_state, strict=False)
    ce_keys = [k for k in missing if "compare_visual_encoder" in k]
    other_missing = [k for k in missing if "compare_visual_encoder" not in k]
    print(f"    CE missing: {len(ce_keys)}, other missing: {len(other_missing)}, unexpected: {len(unexpected)}")
    if other_missing:
        print(f"    !!! 存在非 CE 的 missing 参数（前 5 个）: {other_missing[:5]}")
    if unexpected:
        print(f"    !!! 存在 unexpected（前 5 个）: {unexpected[:5]}")

    new_model.model.visual.compare_visual_encoder.init_query_embeddings()
    print("    CE.query_embeddings 已 normal_(0, 0.02)")

    # 4. 保存模型 + config
    print(f"[4/5] 保存模型到 {dst}")
    new_model.save_pretrained(dst, safe_serialization=True)

    # 5. 复制 tokenizer / processor 文件并构建 AdCompare Processor
    print(f"[5/5] 复制 tokenizer/processor 配置 + 构建 AdCompare Processor")
    files_to_copy = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
        "generation_config.json",
    ]
    for fname in files_to_copy:
        sf = src / fname
        if sf.exists():
            shutil.copy2(sf, dst / fname)

    image_processor = AutoImageProcessor.from_pretrained(src)
    video_processor = AutoVideoProcessor.from_pretrained(src)
    tokenizer = AutoTokenizer.from_pretrained(src)
    chat_template = None
    ct_path = src / "chat_template.json"
    if ct_path.exists():
        import json
        with open(ct_path) as f:
            chat_template = json.load(f).get("chat_template")
    processor = AdCompareQwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
        compare_token_size=compare_token_size,
    )
    processor.save_pretrained(dst)
    print(f"\n[done] 初始 checkpoint: {dst}")
    print(f"       compare_token_size = {compare_token_size}")
    print(f"       架构 = AdCompareQwen3VLForConditionalGeneration")


def main():
    parser = argparse.ArgumentParser(description="Build AD-Compare initial checkpoint from Qwen3-VL-8B-Instruct")
    parser.add_argument("--src", required=True, help="官方 Qwen3-VL-8B-Instruct 路径")
    parser.add_argument("--dst", required=True, help="目标输出路径")
    parser.add_argument("--compare_token_size", type=int, default=100)
    args = parser.parse_args()
    build(args.src, args.dst, args.compare_token_size)


if __name__ == "__main__":
    main()
