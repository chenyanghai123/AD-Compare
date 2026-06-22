"""抽样可视化 GT vs PD 对比图。

分类可视化：gold（全命中）、partial（部分命中）、miss（全失）、fp（误检过多）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from eval.utils import EVAL_OUT, iou_xyxy

PRED_NORM_SCALE = 1000.0


def pd_to_orig(pd_bbox, orig_w, orig_h):
    fx = orig_w / PRED_NORM_SCALE
    fy = orig_h / PRED_NORM_SCALE
    x1, y1, x2, y2 = pd_bbox
    return [x1 * fx, y1 * fy, x2 * fx, y2 * fy]


def categorize(rec: Dict[str, Any]) -> str:
    n_gt = len(rec["gt"])
    n_pd = len(rec["pd"])
    if n_gt == 0:
        return "no_gt"
    orig_w, orig_h = rec["ng_size_orig"]
    pd_orig = [pd_to_orig(p["bbox_2d"], orig_w, orig_h) for p in rec["pd"]]
    gt_boxes = [g["bbox"] for g in rec["gt"]]
    if not pd_orig:
        return "miss"
    # 每个 GT 的最大 IoU
    per_gt_iou = [max((iou_xyxy(pd, gt) for pd in pd_orig), default=0.0) for gt in gt_boxes]
    n_hit = sum(1 for v in per_gt_iou if v >= 0.5)
    n_pd_low = sum(1 for pd in pd_orig if max((iou_xyxy(pd, gt) for gt in gt_boxes), default=0.0) < 0.3)
    if n_hit == n_gt:
        return "gold"
    if n_hit == 0:
        return "miss"
    return "partial"
    # 注：fp 类型由额外条件挑选（在 main 中处理）


def find_font(size: int = 14) -> ImageFont.ImageFont:
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(cand, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_one(rec: Dict[str, Any], save_path: Path, category: str) -> None:
    ng_path = Path(rec["ng_path"])
    ok_path = Path(rec["ok_path"])
    orig_w, orig_h = rec["ng_size_orig"]

    ng = Image.open(ng_path).convert("L").convert("RGB")
    draw = ImageDraw.Draw(ng)
    font = find_font(max(12, min(orig_w, orig_h) // 30))
    line_w = max(2, min(orig_w, orig_h) // 200)

    # GT (绿)
    for g in rec["gt"]:
        x1, y1, x2, y2 = g["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=line_w)
        draw.text((x1 + 2, max(0, y1 - 16)), f"GT:{g['label']}", fill=(0, 200, 0), font=font)
    # PD (红)
    for p in rec["pd"]:
        bb = pd_to_orig(p["bbox_2d"], orig_w, orig_h)
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=line_w)
        label = p.get("label", "")
        draw.text((x1 + 2, y2 + 2), f"PD:{label}", fill=(220, 30, 30), font=font)

    # OK 缩略图（左上角）
    try:
        ok = Image.open(ok_path).convert("L").convert("RGB")
        thumb_w = max(80, orig_w // 6)
        thumb_h = int(ok.size[1] * (thumb_w / max(ok.size[0], 1)))
        ok = ok.resize((thumb_w, thumb_h), Image.BILINEAR)
        ng.paste(ok, (4, 4))
        draw.rectangle([4, 4, 4 + thumb_w, 4 + thumb_h], outline=(60, 110, 200), width=2)
        draw.text((6, thumb_h + 6), "OK ref", fill=(60, 110, 200), font=font)
    except Exception:
        pass

    # 顶部标题条（黑底白字）
    title = f"[{category}] {ng_path.name}  GT={len(rec['gt'])}  PD={len(rec['pd'])}"
    bbox = draw.textbbox((0, 0), title, font=font)
    th = bbox[3] - bbox[1] + 6
    draw.rectangle([0, 0, orig_w, th], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ng.save(save_path, quality=92)


def main() -> None:
    pred_path = EVAL_OUT / "pred_raw.jsonl"
    if not pred_path.exists():
        raise SystemExit("missing pred_raw.jsonl")

    records: List[Dict[str, Any]] = []
    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "error" in rec:
            continue
        records.append(rec)

    by_cat: Dict[str, List[Dict[str, Any]]] = {"gold": [], "partial": [], "miss": [], "fp": []}
    for rec in records:
        c = categorize(rec)
        if c == "no_gt":
            continue
        if c == "miss" and len(rec["pd"]) > len(rec["gt"]) + 2:
            by_cat["fp"].append(rec)
        elif c == "miss":
            by_cat["miss"].append(rec)
        elif c == "gold":
            by_cat["gold"].append(rec)
        elif c == "partial":
            by_cat["partial"].append(rec)
        # 严重 fp：即使 partial / gold 但 pd_count 远超 gt_count
        if c != "miss" and len(rec["pd"]) > len(rec["gt"]) + 3:
            by_cat["fp"].append(rec)

    print({k: len(v) for k, v in by_cat.items()})

    out_dir = EVAL_OUT / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []
    n_each = 5
    for cat, lst in by_cat.items():
        chosen = lst[:n_each]
        for i, rec in enumerate(chosen):
            save_path = out_dir / f"{cat}_{i:02d}_{Path(rec['ng_path']).stem}.jpg"
            draw_one(rec, save_path, cat)
            summary.append({"category": cat, "image": str(save_path), "ng_path": rec["ng_path"]})
    (EVAL_OUT / "vis_index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[task6] saved {len(summary)} samples to {out_dir}")


if __name__ == "__main__":
    main()
