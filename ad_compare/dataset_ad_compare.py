"""ad_compare 数据集与 collator"""
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import AutoTokenizer, AutoImageProcessor, AutoVideoProcessor

from .processing_ad_compare import AdCompareQwen3VLProcessor

logger = logging.getLogger(__name__)


def load_processor(model_path: str, compare_token_size: int = 100) -> AdCompareQwen3VLProcessor:
    """显式构造 AdCompareQwen3VLProcessor。

    chat_template 查找顺序：
      1) chat_template.json
      2) chat_template.jinja
      3) tokenizer.chat_template
    """
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    video_processor = AutoVideoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    chat_template = None
    ct_json = Path(model_path) / "chat_template.json"
    ct_jinja = Path(model_path) / "chat_template.jinja"
    if ct_json.exists():
        with open(ct_json) as f:
            chat_template = json.load(f).get("chat_template")
    elif ct_jinja.exists():
        chat_template = ct_jinja.read_text(encoding="utf-8")
    elif getattr(tokenizer, "chat_template", None):
        chat_template = tokenizer.chat_template
    return AdCompareQwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=chat_template,
        compare_token_size=compare_token_size,
    )


class AdCompareDataset(Dataset):
    """读取 stageN_real.json 训练数据（messages+images 格式）。
    通过 processor.apply_chat_template + processor() 转成 input_ids / pixel_values。
    超长样本会跳到下一条（禁止截断 input_ids，避免破坏 image token 对齐）。
    """

    def __init__(
        self,
        data_path: str,
        processor: AdCompareQwen3VLProcessor,
        image_base_dir: str = "training",
        max_length: int = 4096,
        max_image_side: int = 448,
    ):
        self.processor = processor
        self.image_base_dir = Path(image_base_dir)
        self.max_length = max_length
        self.max_image_side = max_image_side
        with open(data_path) as f:
            self.data = json.load(f)
        logger.info(f"[AdCompareDataset] Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def _open_image(self, img_path: str) -> Image.Image:
        p = Path(img_path)
        if not p.is_absolute():
            p = self.image_base_dir / p
        if p.exists():
            img = Image.open(p).convert("RGB")
        else:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.max_image_side > 0 and max(img.size) > self.max_image_side:
            img.thumbnail((self.max_image_side, self.max_image_side), Image.BILINEAR)
        return img

    def _build_user_content(self, text: str, num_images: int) -> list:
        parts = text.split("<image>")
        content = []
        img_idx = 0
        for i, part in enumerate(parts):
            if i > 0 and img_idx < num_images:
                content.append({"type": "image"})
                img_idx += 1
            if part.strip():
                content.append({"type": "text", "text": part})
        return content

    def _mask_non_assistant(self, input_ids, labels):
        tokenizer = self.processor.tokenizer
        marker = tokenizer.encode("assistant\n", add_special_tokens=False)
        ids = input_ids.tolist()
        last = -1
        L = len(marker)
        for i in range(len(ids) - L):
            if ids[i:i + L] == marker:
                last = i + L
        if last > 0:
            labels[:last] = -100
        if tokenizer.pad_token_id is not None:
            labels[input_ids == tokenizer.pad_token_id] = -100
        return labels

    def _build_one(self, idx):
        item = self.data[idx]
        messages = item["messages"]
        image_paths = item.get("images", [])
        images = [self._open_image(p) for p in image_paths]

        user_msg = messages[0]["content"]
        assistant_msg = messages[1]["content"]
        conv = [
            {"role": "user", "content": self._build_user_content(user_msg, len(images))},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_msg}]},
        ]
        text = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text],
            images=images if images else None,
            padding=False,
            return_tensors="pt",
        )
        return inputs

    def __getitem__(self, idx):
        for offset in range(min(len(self.data), 50)):
            real_idx = (idx + offset) % len(self.data)
            inputs = self._build_one(real_idx)
            input_ids = inputs["input_ids"].squeeze(0)
            if input_ids.size(0) <= self.max_length:
                break
        else:
            inputs = self._build_one(idx)
            input_ids = inputs["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels = self._mask_non_assistant(input_ids, labels)
        attention_mask = inputs["attention_mask"].squeeze(0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            result["pixel_values"] = pv.squeeze(0) if pv.dim() > 3 else pv
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result


class AdCompareDataCollator:
    def __init__(self, processor):
        self.pad_id = processor.tokenizer.pad_token_id

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        am = [f["attention_mask"] for f in features]
        max_len = max(x.size(0) for x in input_ids)
        pi, pl, pa = [], [], []
        for ids, lab, mask in zip(input_ids, labels, am):
            pad = max_len - ids.size(0)
            if pad > 0:
                pi.append(torch.cat([ids, torch.full((pad,), self.pad_id, dtype=ids.dtype)]))
                pl.append(torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)]))
                pa.append(torch.cat([mask, torch.zeros(pad, dtype=mask.dtype)]))
            else:
                pi.append(ids); pl.append(lab); pa.append(mask)
        batch = {
            "input_ids": torch.stack(pi),
            "labels": torch.stack(pl),
            "attention_mask": torch.stack(pa),
        }
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        return batch
