"""GRPO 数据集模块
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
_RE_LABEL = re.compile(r'"label"\s*:\s*"([^"]*)"')


def _parse_gt_from_text(text: str) -> Dict[str, Any]:
    """从 assistant response 中解析 GT bbox 和 label。
    返回: {"bboxes": [[x1,y1,x2,y2], ...], "labels": ["...", ...]}
    """
    bboxes = []
    labels = []
    try:
        # 尝试 JSON 解析
        code = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        json_str = code.group(1) if code else text
        # 找到第一个有效的 JSON array
        match = re.search(r"\[\s*(?:\{[^\[\]]*?\}\s*,?\s*)+\]", json_str, re.DOTALL)
        if match:
            obj = json.loads(match.group())
            for it in obj:
                if not isinstance(it, dict):
                    continue
                bb = it.get("bbox_2d") or it.get("bbox")
                if bb and len(bb) == 4:
                    bboxes.append([int(round(float(v))) for v in bb])
                    labels.append(str(it.get("label", "")))
    except Exception:
        pass
    # 正则兜底
    if not bboxes:
        bboxes_raw = _RE_BBOX.findall(text)
        labels_raw = _RE_LABEL.findall(text)
        for i, (a, b, c, d) in enumerate(bboxes_raw):
            bboxes.append([int(a), int(b), int(c), int(d)])
            labels.append(labels_raw[i] if i < len(labels_raw) else "")
    return {"bboxes": bboxes, "labels": labels}


class GRPODataset(Dataset):
    """GRPO 训练数据集。

    数据格式同 SFT: [{"messages": [...], "images": [...]}]
    返回: {"prompt": messages, "images": [PIL.Image, ...], "gt_annotations": {...}}
    """

    def __init__(
        self,
        data_path: str,
        image_base_dir: str = "./data/images",
        max_image_side: int = 448,
    ):
        self.image_base_dir = Path(image_base_dir)
        self.max_image_side = max_image_side
        with open(data_path) as f:
            raw_data = json.load(f)
        self.data = []
        for item in raw_data:
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue
            user_content = messages[0].get("content", "")
            assistant_content = messages[1].get("content", "")
            gt = _parse_gt_from_text(assistant_content)
            self.data.append({
                "user_content": user_content,
                "image_paths": item.get("images", []),
                "gt_annotations": gt,
            })
        logger.info(f"[GRPODataset] Loaded {len(self.data)} samples from {data_path}")

    def _build_prompt_messages(self, text: str, num_images: int, images: List[Image.Image]) -> List[Dict]:
        parts = text.split("<image>")
        contents = []
        img_idx = 0
        for i, part in enumerate(parts):
            if i > 0 and img_idx < num_images:
                contents.append({"type": "image", "image": images[img_idx]})
                img_idx += 1
            if part.strip():
                contents.append({"type": "text", "text": part})
        # 补齐剩余图片
        while img_idx < num_images:
            contents.insert(0, {"type": "image", "image": images[img_idx]})
            img_idx += 1
        return [{"role": "user", "content": contents}]

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images = [self._open_image(p) for p in item["image_paths"]]
        prompt = self._build_prompt_messages(item["user_content"], len(images), images)
        return {
            "prompt": prompt,
            "gt_annotations": item["gt_annotations"],
        }
