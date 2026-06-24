"""
AD-Compare GRPO 训练入口（Stage 4）

用法：
    # 单卡 dry-run
    CUDA_VISIBLE_DEVICES=0 python tools/train_grpo.py \
        configs/stage4_grpo.yaml --max_steps 3

    # 8 卡分布式
    python -m torch.distributed.run --nproc_per_node=8 \
        tools/train_grpo.py configs/stage4_grpo.yaml

设计：
- 基于 TRL GRPOTrainer，子类化以支持 AD-Compare 的双图输入
- 复用 Stage 3 训练数据，解析 GT bbox/label 用于奖励计算
- 规则奖励：format + count + IoU + classification
"""
import logging
import sys
import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, List

import torch
from transformers import HfArgumentParser

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLForConditionalGeneration,
)
from ad_compare.dataset_ad_compare import load_processor
from ad_compare.dataset_grpo import GRPODataset
from ad_compare.reward_functions import REWARD_FUNCS, DEFAULT_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./checkpoints/stage3_multitask_sft_merged")
    attn_implementation: str = field(default="flash_attention_2")
    mm_projector_lr: float = field(default=0.0)
    vision_tower_lr: float = field(default=0.0)


@dataclass
class DataArguments:
    data_path: str = field(default="data/stage3_data.json")
    image_base_dir: str = field(default="./data/images")
    max_image_side: int = field(default=448)


@dataclass
class LoRAArguments:
    use_lora: bool = field(default=True)
    train_ce: bool = field(default=False)
    freeze_vit: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: Optional[List[str]] = field(default=None)
    lora_exclude_modules: Optional[List[str]] = field(default=None)


@dataclass
class GRPOArguments:
    """GRPO 超参数。"""
    num_generations: int = field(default=8)
    beta: float = field(default=0.04)
    max_new_tokens: int = field(default=512)
    reward_weights: Optional[List[float]] = field(default=None)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    output_dir: str = field(default="./outputs/stage4_grpo")
    learning_rate: float = field(default=1e-6)
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=5)
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=1)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    deepspeed: Optional[str] = field(default=None)
    report_to: str = field(default="none")
    local_rank: int = field(default=-1)
    max_steps: int = field(default=-1)


def llm_linear_modules():
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def vit_linear_modules():
    return ["qkv", "proj", "linear_fc1", "linear_fc2"]


def setup_lora(model, lora_args: LoRAArguments):
    """设置 LoRA + 冻结 CE/ViT。"""
    from peft import LoraConfig, get_peft_model, TaskType

    if lora_args.lora_target_modules is None:
        target_modules = llm_linear_modules()
        exclude_modules = ["compare_visual_encoder", "visual"]
    else:
        target_modules = list(lora_args.lora_target_modules)
        exclude_modules = list(lora_args.lora_exclude_modules or ["compare_visual_encoder", "visual"])

    logger.info(f"LoRA target_modules: {target_modules}")
    logger.info(f"LoRA exclude_modules: {exclude_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_args.lora_rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
    )
    model = get_peft_model(model, peft_config)

    # CE 冻结（GRPO 阶段不训练 CE）
    if not lora_args.train_ce:
        frozen_ce = 0
        for n, p in model.named_parameters():
            if "compare_visual_encoder" in n:
                p.requires_grad = False
                frozen_ce += p.numel()
        logger.info(f"CE frozen: {frozen_ce / 1e6:.2f}M params")

    # ViT 冻结（GRPO 阶段只动 LLM policy）
    if lora_args.freeze_vit:
        frozen_vit = 0
        for n, p in model.named_parameters():
            if "visual" in n and "compare_visual_encoder" not in n:
                p.requires_grad = False
                frozen_vit += p.numel()
        logger.info(f"ViT frozen: {frozen_vit / 1e6:.2f}M params")

    model.print_trainable_parameters()
    return model


def load_yaml_to_args(yaml_path: str, cli_overrides: List[str]):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, GRPOArguments))
    field_owners = {}
    for dc in (ModelArguments, DataArguments, LoRAArguments, GRPOArguments):
        for f in fields(dc):
            field_owners.setdefault(f.name, dc.__name__)

    flat_cli = []
    for k, v in cfg.items():
        if k not in field_owners:
            logger.warning(f"yaml key '{k}' not in any dataclass fields, ignored")
            continue
        if isinstance(v, bool):
            flat_cli.extend([f"--{k}", str(v)])
        elif isinstance(v, (list, tuple)):
            flat_cli.append(f"--{k}")
            flat_cli.extend([str(x) for x in v])
        else:
            flat_cli.extend([f"--{k}", str(v)])
    flat_cli.extend(cli_overrides)

    m, d, l, g = parser.parse_args_into_dataclasses(flat_cli)
    return m, d, l, g


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    yaml_path = sys.argv[1]
    cli_overrides = sys.argv[2:]
    if not yaml_path.endswith((".yaml", ".yml")):
        raise ValueError(f"first arg must be a yaml file, got {yaml_path}")

    m, d, l, g = load_yaml_to_args(yaml_path, cli_overrides)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if g.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.info("=" * 60)
    logger.info("AD-Compare Stage 4 (GRPO)")
    logger.info(f"yaml: {yaml_path}  overrides: {cli_overrides}")
    logger.info(f"model_name_or_path: {m.model_name_or_path}")
    logger.info(f"data_path: {d.data_path}")
    logger.info(f"output_dir: {g.output_dir}")
    logger.info(f"num_generations: {g.num_generations}, beta: {g.beta}")
    logger.info("=" * 60)

    # 1. Load model
    config = AdCompareQwen3VLConfig.from_pretrained(m.model_name_or_path)
    model = AdCompareQwen3VLForConditionalGeneration.from_pretrained(
        m.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=m.attn_implementation,
        low_cpu_mem_usage=True,
    )

    # 2. Processor
    processor = load_processor(m.model_name_or_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # 3. LoRA setup
    if l.use_lora:
        model = setup_lora(model, l)

    # 4. Dataset
    train_ds = GRPODataset(
        data_path=d.data_path,
        image_base_dir=d.image_base_dir,
        max_image_side=d.max_image_side,
    )

    # 5. GRPO Trainer
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        logger.error("TRL not installed. Please run: pip install trl>=0.12.0")
        sys.exit(1)

    # 奖励函数权重
    reward_weights = g.reward_weights or DEFAULT_WEIGHTS
    if len(reward_weights) != len(REWARD_FUNCS):
        logger.warning(f"reward_weights length mismatch, using defaults")
        reward_weights = DEFAULT_WEIGHTS

    # 创建加权奖励函数
    weighted_rewards = []
    for func, weight in zip(REWARD_FUNCS, reward_weights):
        def make_weighted_fn(f, w):
            def weighted_fn(*args, **kwargs):
                scores = f(*args, **kwargs)
                return [s * w for s in scores]
            weighted_fn.__name__ = f"weighted_{f.__name__}"
            return weighted_fn
        weighted_rewards.append(make_weighted_fn(func, weight))

    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=g.output_dir,
        num_train_epochs=g.num_train_epochs,
        per_device_train_batch_size=g.per_device_train_batch_size,
        gradient_accumulation_steps=g.gradient_accumulation_steps,
        learning_rate=g.learning_rate,
        weight_decay=g.weight_decay,
        warmup_ratio=g.warmup_ratio,
        lr_scheduler_type=g.lr_scheduler_type,
        logging_steps=g.logging_steps,
        save_strategy=g.save_strategy,
        save_total_limit=g.save_total_limit,
        bf16=g.bf16,
        gradient_checkpointing=g.gradient_checkpointing,
        report_to=g.report_to,
        num_generations=g.num_generations,
        beta=g.beta,
        max_completion_length=g.max_new_tokens,
        max_steps=g.max_steps if g.max_steps > 0 else -1,
        remove_unused_columns=False,
        temperature=g.temperature,
        top_p=g.top_p,
    )

    # 子类化 GRPOTrainer 以支持多图像输入和 GT annotations
    class AdCompareGRPOTrainer(GRPOTrainer):
        """适配 AD-Compare 双图输入的 GRPO Trainer。"""

        def __init__(self, *args, processor=None, **kwargs):
            self._ad_processor = processor
            super().__init__(*args, **kwargs)

        def _prepare_inputs(self, examples):
            """处理多图像输入：用 AdCompare processor 处理图片。"""
            # 调用父类方法处理文本 prompt
            return super()._prepare_inputs(examples)

        def _get_per_token_rewards(self, model, inputs, completions, **kwargs):
            """重写以传递 gt_annotations 给奖励函数。"""
            # 从 inputs 中提取 gt_annotations
            gt_annotations = inputs.get("gt_annotations", [])
            # 调用父类方法，注入 gt_annotations
            all_rewards = []
            for reward_func in self.reward_funcs:
                try:
                    scores = reward_func(completions, gt_annotations=gt_annotations)
                except Exception as e:
                    logger.warning(f"Reward func {reward_func.__name__} failed: {e}")
                    scores = [0.0] * len(completions)
                all_rewards.append(scores)
            return torch.tensor(all_rewards, device=model.device).T

    trainer = AdCompareGRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=weighted_rewards,
        args=grpo_config,
        train_dataset=train_ds,
        processor=processor,
    )

    logger.info("Start GRPO training...")
    out = trainer.train()
    trainer.save_model()
    trainer.save_state()
    metrics = out.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    processor.save_pretrained(g.output_dir)

    # 6. Merge LoRA
    if l.use_lora:
        merged_path = g.output_dir.rstrip("/") + "_merged"
        if g.local_rank in [-1, 0]:
            logger.info(f"Merge LoRA -> {merged_path}")
            unwrapped = trainer.accelerator.unwrap_model(model) if hasattr(trainer, "accelerator") else model
            if hasattr(unwrapped, "merge_and_unload"):
                merged = unwrapped.merge_and_unload()
                merged.save_pretrained(merged_path, safe_serialization=True)
                processor.save_pretrained(merged_path)
                logger.info(f"Merged checkpoint saved: {merged_path}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    logger.info("Done.")


if __name__ == "__main__":
    main()
