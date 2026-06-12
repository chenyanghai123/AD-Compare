"""
AD-Compare 统一训练入口（合并 Stage 0 / 1 / 2 / 3）

用法：
    # 单卡 dry-run
    CUDA_VISIBLE_DEVICES=0 python tools/train.py \
        configs/stage0_ce_pretrain.yaml --max_steps 3

    # 8 卡分布式
    python -m torch.distributed.run --nproc_per_node=8 \
        tools/train.py configs/stage0_ce_pretrain.yaml

设计：
- 极简 yaml（纯 key:value）→ 解析为 ModelArguments / DataArguments / LoRAArguments / TrainingArguments
- 命令行额外参数（如 --max_steps 3）会覆盖 yaml
- stage 0：CE 全参，无 LoRA
- stage 1/2/3：CE 全参 + LoRA（LLM only / LLM+ViT / LLM+ViT 多任务）
- 训练后自动 cascade merge（仅 LoRA stage），输出 {output_dir}_merged
"""
import os
import sys
import json
import yaml
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, List

import torch
from transformers import TrainingArguments, Trainer, HfArgumentParser

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLForConditionalGeneration,
    AdCompareQwen3VLProcessor,
)
from ad_compare.dataset_ad_compare import (
    AdCompareDataset,
    AdCompareDataCollator,
    load_processor,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Argument Dataclasses
# ============================================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="./checkpoints/init")
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "attention 实现：flash_attention_2 / sdpa / eager"},
    )
    mm_projector_lr: float = field(
        default=0.0,
        metadata={"help": "merger / projector 单独 lr，0 表示沿用 base learning_rate"},
    )
    vision_tower_lr: float = field(
        default=0.0,
        metadata={"help": "ViT 单独 lr（推荐 base lr 的 1/10），0 表示沿用 base learning_rate"},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="data/stage0_real.json")
    image_base_dir: str = field(
        default="./data/images",
        metadata={"help": "图片相对路径的拼接根"},
    )
    max_seq_length: int = field(default=4096)
    max_image_side: int = field(default=448)


@dataclass
class StageLoRAArguments:
    """stage 0 时 use_lora=False，本组将被忽略。"""
    use_lora: bool = field(default=False)
    train_ce: bool = field(default=True, metadata={"help": "stage>=1 时是否手动解冻 CE 全参；stage 0 永远 True"})
    freeze_vit: bool = field(default=True)
    freeze_llm: bool = field(default=True)

    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: Optional[List[str]] = field(default=None)
    lora_exclude_modules: Optional[List[str]] = field(default=None)


@dataclass
class AdCompareTrainingArguments(TrainingArguments):
    stage: int = field(default=0, metadata={"help": "0/1/2/3"})


# ============================================================================
# Stage-aware freeze / LoRA 设置
# ============================================================================
def llm_linear_modules():
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def vit_linear_modules():
    return ["qkv", "proj", "linear_fc1", "linear_fc2"]


def freeze_for_stage0(model):
    """Stage 0：仅训 CE。"""
    for p in model.parameters():
        p.requires_grad = False
    trainable, total = 0, 0
    for n, p in model.named_parameters():
        if "compare_visual_encoder" in n:
            p.requires_grad = True
            trainable += p.numel()
        total += p.numel()
    pct = 100 * trainable / total if total > 0 else 0
    logger.info(f"Stage 0: trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M ({pct:.2f}%)")
    return model


def setup_lora(model, lora_args: StageLoRAArguments, stage: int):
    """Stage 1/2/3：CE 全参 + LoRA。"""
    from peft import LoraConfig, get_peft_model, TaskType

    if lora_args.lora_target_modules is None:
        if stage == 1:
            target_modules = llm_linear_modules()
            exclude_modules = ["visual"]
            logger.info("Stage 1: LoRA on LLM only, exclude visual")
        else:
            target_modules = list(set(llm_linear_modules() + vit_linear_modules()))
            exclude_modules = ["compare_visual_encoder"]
            logger.info(f"Stage {stage}: LoRA on LLM + ViT, exclude compare_visual_encoder")
    else:
        target_modules = list(lora_args.lora_target_modules)
        if lora_args.lora_exclude_modules is not None:
            exclude_modules = list(lora_args.lora_exclude_modules)
        else:
            exclude_modules = ["compare_visual_encoder"] if stage in (2, 3) else ["visual"]

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

    # CE 全参数解冻
    if lora_args.train_ce:
        ce_params = 0
        for n, p in model.named_parameters():
            if "compare_visual_encoder" in n:
                p.requires_grad = True
                ce_params += p.numel()
        logger.info(f"CE full-finetune: {ce_params / 1e6:.2f}M params")
    model.print_trainable_parameters()
    return model


# ============================================================================
# 分组学习率
# ============================================================================
def build_param_groups(model, base_lr: float, weight_decay: float,
                       mm_projector_lr: float = 0.0,
                       vision_tower_lr: float = 0.0):
    """按模块名前缀对 trainable params 分组。"""
    no_decay_keys = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]

    def is_no_decay(name: str) -> bool:
        return any(k in name for k in no_decay_keys)

    def get_group_lr(name: str):
        if "compare_visual_encoder" in name:
            return base_lr, "ce"
        if mm_projector_lr > 0 and ("merger" in name or "projector" in name):
            return mm_projector_lr, "projector"
        if vision_tower_lr > 0 and "visual" in name:
            return vision_tower_lr, "visual"
        return base_lr, "base"

    buckets = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr, tag = get_group_lr(n)
        nd = is_no_decay(n)
        key = (tag, nd, lr)
        buckets.setdefault(key, []).append(p)

    groups = []
    for (tag, nd, lr), params in buckets.items():
        groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": 0.0 if nd else weight_decay,
            "name": f"{tag}_{'no_decay' if nd else 'decay'}_lr{lr}",
        })
        n_params = sum(p.numel() for p in params)
        logger.info(f"  optim group [{tag}, {'no_decay' if nd else 'decay'}, lr={lr}] "
                    f"#tensors={len(params)} #params={n_params/1e6:.2f}M")
    return groups


class GroupedLRTrainer(Trainer):
    """支持 mm_projector_lr / vision_tower_lr 分组学习率的 Trainer 子类。"""

    def __init__(self, *args, mm_projector_lr: float = 0.0,
                 vision_tower_lr: float = 0.0, **kwargs):
        self._mm_projector_lr = mm_projector_lr
        self._vision_tower_lr = vision_tower_lr
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self._mm_projector_lr <= 0 and self._vision_tower_lr <= 0:
            return super().create_optimizer()

        if self.optimizer is None:
            logger.info(
                f"Building grouped optimizer "
                f"(mm_projector_lr={self._mm_projector_lr}, "
                f"vision_tower_lr={self._vision_tower_lr}, "
                f"base_lr={self.args.learning_rate})..."
            )
            param_groups = build_param_groups(
                self.model,
                base_lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                mm_projector_lr=self._mm_projector_lr,
                vision_tower_lr=self._vision_tower_lr,
            )
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr", None)
            optimizer_kwargs.pop("weight_decay", None)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer


# ============================================================================
# YAML loader
# ============================================================================
def load_yaml_to_args(yaml_path: str, cli_overrides: List[str]):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, StageLoRAArguments, AdCompareTrainingArguments)
    )
    field_owners = {}
    for dc in (ModelArguments, DataArguments, StageLoRAArguments, AdCompareTrainingArguments):
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

    m, d, l, t = parser.parse_args_into_dataclasses(flat_cli)
    return m, d, l, t


# ============================================================================
# Main
# ============================================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    yaml_path = sys.argv[1]
    cli_overrides = sys.argv[2:]
    if not yaml_path.endswith((".yaml", ".yml")):
        raise ValueError(f"first arg must be a yaml file, got {yaml_path}")

    m, d, l, t = load_yaml_to_args(yaml_path, cli_overrides)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if t.local_rank in [-1, 0] else logging.WARNING,
    )
    logger.info("=" * 60)
    logger.info(f"AD-Compare Stage {t.stage}  (use_lora={l.use_lora})")
    logger.info(f"yaml: {yaml_path}  overrides: {cli_overrides}")
    logger.info(f"model_name_or_path: {m.model_name_or_path}")
    logger.info(f"data_path: {d.data_path}")
    logger.info(f"output_dir: {t.output_dir}")
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

    # 3. Stage-aware freeze / LoRA
    if t.stage == 0:
        model = freeze_for_stage0(model)
    else:
        if not l.use_lora:
            raise ValueError(f"stage {t.stage} requires use_lora=true in yaml")
        model = setup_lora(model, l, t.stage)

    # 4. Dataset
    train_ds = AdCompareDataset(
        data_path=d.data_path,
        processor=processor,
        image_base_dir=d.image_base_dir,
        max_length=d.max_seq_length,
        max_image_side=d.max_image_side,
    )
    collator = AdCompareDataCollator(processor)

    # 5. Trainer
    trainer = GroupedLRTrainer(
        model=model,
        args=t,
        train_dataset=train_ds,
        data_collator=collator,
        mm_projector_lr=m.mm_projector_lr,
        vision_tower_lr=m.vision_tower_lr,
    )
    logger.info("Start training...")
    out = trainer.train()
    trainer.save_model()
    trainer.save_state()
    metrics = out.metrics
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    processor.save_pretrained(t.output_dir)

    # 6. Cascade merge（仅 LoRA stage）
    if l.use_lora:
        merged_path = t.output_dir.rstrip("/") + "_merged"
        if t.local_rank in [-1, 0]:
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
