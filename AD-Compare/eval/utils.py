"""AD-Compare 评估 pipeline 共享工具。

涵盖：
- 数据集路径配置（通过环境变量或函数参数）
- tzjson 解析（polygon → bbox）
- LLM 输出 JSON 容错抽取
- smart_resize 计算（与 Qwen3-VL Processor 一致）
- 模态对齐预处理

环境变量配置：
    EVAL_DATA_ROOT: 评估数据集根目录（含 images/1, images/ok, Annotations/）
    EVAL_OUT_DIR:   评估输出目录（默认 ./eval_outputs）
    MODEL_PATH:     模型路径（默认 ./checkpoints/stage3_multitask_sft_merged）
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# 路径配置（通过环境变量，用户按需设置）
# ---------------------------------------------------------------------------
EVAL_DATA_ROOT = Path(os.environ.get("EVAL_DATA_ROOT", "./data/eval_dataset"))
NG_DIR = EVAL_DATA_ROOT / "images" / "1"
OK_DIR = EVAL_DATA_ROOT / "images" / "ok"
TRAIN_CSV = EVAL_DATA_ROOT / "Annotations" / "train.csv"
VAL_CSV = EVAL_DATA_ROOT / "Annotations" / "val.csv"
CLASS_LIST = EVAL_DATA_ROOT / "Annotations" / "class_names_list.txt"

EVAL_OUT = Path(os.environ.get("EVAL_OUT_DIR", "./eval_outputs"))
EVAL_OUT.mkdir(parents=True, exist_ok=True)
(EVAL_OUT / "vis").mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH", "./checkpoints/stage3_multitask_sft_merged"
)

# 与训练 grounding 任务一致的 prompt
GROUNDING_PROMPT = (
    "<image><image>Given the normal reference (first), identify and localize defects "
    "in the second image. Format: [{\"bbox_2d\": [x1,y1,x2,y2], \"label\": \"type\"}]"
)


# ---------------------------------------------------------------------------
# NG 列表枚举
# ---------------------------------------------------------------------------
def list_ng_paths() -> List[Path]:
    """返回 NG 的 jpg 绝对路径（按文件名排序）。"""
    return sorted(NG_DIR.glob("*.jpg"))


def list_ok_paths() -> List[Path]:
    """返回 OK 池（png）绝对路径（按文件名排序）。"""
    return sorted(OK_DIR.glob("*.png"))


def ng_to_tzjson(ng_path: Path) -> Path:
    return ng_path.with_suffix(".tzjson")


def load_split_set(csv_path: Path) -> set:
    """返回 csv 中出现的 jpg basename 集合。"""
    out = set()
    if not csv_path.exists():
        return out
    with csv_path.open() as f:
        next(f, None)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel = line.split(",")[0]
            out.add(Path(img_rel).name)
    return out


# ---------------------------------------------------------------------------
# tzjson 解析
# ---------------------------------------------------------------------------
def _polygon_to_bbox(points: Any) -> Optional[Tuple[int, int, int, int]]:
    """tzjson 的 points 形如 [[[x,y],[x,y],...]] 或 [[x,y],...]，统一展平后取外接矩形。"""
    if not points:
        return None
    flat: List[List[float]] = []
    stack = [points]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, list) or not cur:
            continue
        head = cur[0]
        if isinstance(head, (int, float)) and len(cur) >= 2:
            flat.append([float(cur[0]), float(cur[1])])
        else:
            stack.extend(cur)
    if not flat:
        return None
    xs = [p[0] for p in flat]
    ys = [p[1] for p in flat]
    return int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))


def parse_tzjson(path: Path) -> List[Dict[str, Any]]:
    """解析 tzjson 返回 GT 列表 [{bbox:[x1,y1,x2,y2], label, points}]."""
    with path.open() as f:
        raw = json.load(f)

    out: List[Dict[str, Any]] = []
    for shape in raw.get("shapes", []) or []:
        if shape.get("enable") is False:
            continue
        if shape.get("is_ignored") is True:
            continue
        bbox = shape.get("bbox")
        if (not bbox or len(bbox) != 4) and shape.get("points"):
            bbox = _polygon_to_bbox(shape["points"])
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": shape.get("label", ""),
                "points": shape.get("points"),
            }
        )
    return out


def get_image_size_from_tzjson(path: Path) -> Optional[Tuple[int, int]]:
    """读 tzjson 自带的 imageWidth/imageHeight。"""
    try:
        with path.open() as f:
            raw = json.load(f)
        w = raw.get("imageWidth")
        h = raw.get("imageHeight")
        if w and h:
            return int(w), int(h)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Qwen3-VL smart_resize
# ---------------------------------------------------------------------------
SMART_RESIZE_FACTOR = 28
SMART_RESIZE_MIN_PIXELS = 64 * 28 * 28           # 50_176
SMART_RESIZE_MAX_PIXELS = 1280 * 28 * 28         # 1_003_520


def smart_resize(
    height: int,
    width: int,
    factor: int = SMART_RESIZE_FACTOR,
    min_pixels: int = SMART_RESIZE_MIN_PIXELS,
    max_pixels: int = SMART_RESIZE_MAX_PIXELS,
) -> Tuple[int, int]:
    """复刻 Qwen2/Qwen3-VL Processor 的 smart_resize 长宽计算。"""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def scale_bbox(
    bbox: Iterable[float],
    src_size: Tuple[int, int],
    dst_size: Tuple[int, int],
) -> List[float]:
    """把 bbox 从 src_size 坐标系缩到 dst_size 坐标系。size = (W, H)。"""
    sw, sh = src_size
    dw, dh = dst_size
    fx = dw / max(sw, 1)
    fy = dh / max(sh, 1)
    x1, y1, x2, y2 = list(bbox)
    return [x1 * fx, y1 * fy, x2 * fx, y2 * fy]


# ---------------------------------------------------------------------------
# 模态对齐：NG → 灰度→3 通道；OK 灰度→3 通道并 resize 到 NG 尺寸
# ---------------------------------------------------------------------------
def load_aligned_pair(ng_path: Path, ok_path: Path) -> Tuple[Image.Image, Image.Image]:
    """返回 (ok_rgb, ng_rgb)，ok 已 resize 到 ng 尺寸。"""
    ng = Image.open(ng_path).convert("L").convert("RGB")
    ok = Image.open(ok_path).convert("L").convert("RGB")
    if ok.size != ng.size:
        ok = ok.resize(ng.size, Image.BILINEAR)
    return ok, ng


# ---------------------------------------------------------------------------
# LLM 输出 JSON 容错解析
# ---------------------------------------------------------------------------
_RE_JSON_ARRAY = re.compile(r"\[\s*(?:\{[^\[\]]*?\}\s*,?\s*)+\]", re.DOTALL)
_RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
_RE_LABEL = re.compile(r'"label"\s*:\s*"([^"]*)"')


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """从 LLM 输出抽出 [{bbox_2d:[..], label:".."}] 列表。"""
    if not text:
        return []
    code = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    candidates: List[str] = []
    if code:
        candidates.append(code.group(1))
    candidates.extend(_RE_JSON_ARRAY.findall(text))
    for s in candidates:
        try:
            obj = json.loads(s)
        except Exception:
            continue
        out = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            bb = it.get("bbox_2d") or it.get("bbox")
            if not bb or len(bb) != 4:
                continue
            try:
                out.append({"bbox_2d": [int(round(float(v))) for v in bb], "label": str(it.get("label", ""))})
            except Exception:
                continue
        if out:
            return out
    # 正则兜底
    bboxes = _RE_BBOX.findall(text)
    labels = _RE_LABEL.findall(text)
    out = []
    for i, (a, b, c, d) in enumerate(bboxes):
        out.append(
            {
                "bbox_2d": [int(a), int(b), int(c), int(d)],
                "label": labels[i] if i < len(labels) else "",
            }
        )
    return out


# ---------------------------------------------------------------------------
# bbox 几何
# ---------------------------------------------------------------------------
def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def normalize_bbox(bbox: Iterable[float], w: int, h: int) -> List[float]:
    """裁剪到图像边界 + 自动交换 x1>x2/y1>y2。"""
    x1, y1, x2, y2 = list(bbox)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(x1), w))
    x2 = max(0.0, min(float(x2), w))
    y1 = max(0.0, min(float(y1), h))
    y2 = max(0.0, min(float(y2), h))
    return [x1, y1, x2, y2]
