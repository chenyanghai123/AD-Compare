"""坐标系对齐 + class-agnostic mAP 计算。

坐标系约定：
- GT bbox: NG 原图坐标系
- PD bbox: [0, 1000] 归一化坐标系（Qwen-VL 惯例），评估前缩放回原图

输出: metrics.json, pr_curve.png, iou_hist.png, per_image_metrics.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from eval.utils import EVAL_OUT, iou_xyxy, normalize_bbox

# matplotlib 后端不依赖 X
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRED_NORM_SCALE = 1000.0          # Qwen-VL 训练 grounding bbox 归一化空间


def pd_to_orig(pd_bbox: List[float], orig_w: int, orig_h: int) -> List[float]:
    """[0,1000] -> 原图坐标系。"""
    fx = orig_w / PRED_NORM_SCALE
    fy = orig_h / PRED_NORM_SCALE
    x1, y1, x2, y2 = pd_bbox
    return [x1 * fx, y1 * fy, x2 * fx, y2 * fy]


def match_one(gt_boxes: List[List[float]], pd_boxes: List[List[float]], iou_thr: float):
    """class-agnostic 贪心匹配：按 PD 顺序（VLM 输出无 conf，认为越早越置信），
    每个 PD 匹配 IoU 最大且未被占用的 GT，返回每个 PD 的 (max_iou, matched_or_not)。"""
    matched = [False] * len(gt_boxes)
    out: List[Tuple[float, bool]] = []
    for pd in pd_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched[j]:
                continue
            iou = iou_xyxy(pd, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        hit = best_iou >= iou_thr and best_j >= 0
        if hit:
            matched[best_j] = True
        out.append((best_iou, hit))
    return out, matched


def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """11-point VOC AP (also used for class-agnostic AP)."""
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        p = prec[rec >= t].max() if (rec >= t).any() else 0.0
        ap += p / 11.0
    return float(ap)


def compute_ap_at(records: List[Dict[str, Any]], iou_thr: float) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """class-agnostic AP @ given IoU threshold."""
    all_pd = []  # (score_proxy, hit, iou)
    total_gt = 0
    for r in records:
        gt = r["gt_orig"]
        pd = r["pd_orig"]
        total_gt += len(gt)
        matches, _ = match_one(gt, pd, iou_thr)
        # score proxy: 输出顺序倒序（越靠前越高分）
        n = len(pd)
        for i, (iou, hit) in enumerate(matches):
            score = (n - i) / max(n, 1)  # 0~1
            all_pd.append((score, 1 if hit else 0, iou))
    if not all_pd:
        return 0.0, 0.0, 0.0, 0.0, np.zeros(0), np.zeros(0)
    all_pd.sort(key=lambda x: x[0], reverse=True)
    tp = np.array([x[1] for x in all_pd], dtype=np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = voc_ap(recall, precision)
    final_tp = int(cum_tp[-1])
    final_fp = int(cum_fp[-1])
    p = final_tp / max(final_tp + final_fp, 1)
    r = final_tp / max(total_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return ap, p, r, f1, recall, precision


def main() -> None:
    pred_path = EVAL_OUT / "pred_raw.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"missing {pred_path}; run 03_infer_grounding.py first")

    records: List[Dict[str, Any]] = []
    n_total = 0
    n_no_gt = 0
    n_empty_pd = 0
    n_with_error = 0
    iou_samples: List[float] = []
    pd_count_dist: List[int] = []
    gt_count_dist: List[int] = []
    per_image_rows: List[Dict[str, Any]] = []

    for line in pred_path.open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        n_total += 1
        if "error" in rec:
            n_with_error += 1
            continue
        gt_list = rec.get("gt", [])
        pd_list = rec.get("pd", [])
        orig_w, orig_h = rec["ng_size_orig"]
        gt_orig = [
            normalize_bbox(g["bbox"], orig_w, orig_h)
            for g in gt_list
            if g.get("bbox") and len(g["bbox"]) == 4
        ]
        pd_orig = [
            normalize_bbox(pd_to_orig(p["bbox_2d"], orig_w, orig_h), orig_w, orig_h)
            for p in pd_list
            if p.get("bbox_2d") and len(p["bbox_2d"]) == 4
        ]
        if not gt_orig:
            n_no_gt += 1
        if not pd_orig:
            n_empty_pd += 1
        records.append(
            {
                "ng_path": rec["ng_path"],
                "ok_path": rec.get("ok_path"),
                "split": rec.get("split"),
                "ng_size_orig": rec["ng_size_orig"],
                "ng_size_resized": rec.get("ng_size_resized"),
                "gt_orig": gt_orig,
                "pd_orig": pd_orig,
                "pd_raw": rec.get("pd_raw"),
                "labels_pd": [p.get("label", "") for p in pd_list],
                "labels_gt": [g.get("label", "") for g in gt_list],
            }
        )
        gt_count_dist.append(len(gt_orig))
        pd_count_dist.append(len(pd_orig))
        # 这张图的最大 IoU（计算每张图主指标）
        per_pd_max = []
        for pd in pd_orig:
            best = max((iou_xyxy(pd, gt) for gt in gt_orig), default=0.0)
            per_pd_max.append(best)
        per_gt_max = []
        for gt in gt_orig:
            best = max((iou_xyxy(pd, gt) for pd in pd_orig), default=0.0)
            per_gt_max.append(best)
        iou_samples.extend(per_pd_max)
        per_image_rows.append(
            {
                "ng_path": rec["ng_path"],
                "n_gt": len(gt_orig),
                "n_pd": len(pd_orig),
                "max_iou_per_pd": per_pd_max,
                "max_iou_per_gt": per_gt_max,
                "best_iou": max(per_gt_max) if per_gt_max else 0.0,
            }
        )

    # mAP@0.5
    ap50, p50, r50, f150, rec_curve, prec_curve = compute_ap_at(records, 0.5)
    # mAP@[.5:.95]
    aps = []
    for thr in np.arange(0.5, 1.0, 0.05):
        ap, _, _, _, _, _ = compute_ap_at(records, float(thr))
        aps.append(ap)
    map_50_95 = float(np.mean(aps))

    metrics = {
        "n_total": n_total,
        "n_with_error": n_with_error,
        "n_no_gt": n_no_gt,
        "n_empty_pd": n_empty_pd,
        "empty_pd_ratio": n_empty_pd / max(n_total, 1),
        "n_records_eval": len(records),
        "total_gt_box": int(sum(gt_count_dist)),
        "total_pd_box": int(sum(pd_count_dist)),
        "avg_gt_per_image": float(np.mean(gt_count_dist)) if gt_count_dist else 0.0,
        "avg_pd_per_image": float(np.mean(pd_count_dist)) if pd_count_dist else 0.0,
        "mAP@0.5": ap50,
        "Precision@0.5": p50,
        "Recall@0.5": r50,
        "F1@0.5": f150,
        "mAP@[.5:.95]": map_50_95,
        "ap_per_thr": {f"{thr:.2f}": float(ap) for thr, ap in zip(np.arange(0.5, 1.0, 0.05), aps)},
        "iou_mean_per_pd": float(np.mean(iou_samples)) if iou_samples else 0.0,
        "iou_median_per_pd": float(np.median(iou_samples)) if iou_samples else 0.0,
    }

    out_metrics = EVAL_OUT / "metrics.json"
    out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[task5] metrics -> {out_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # PR curve
    if rec_curve.size:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec_curve, prec_curve, lw=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Class-agnostic PR curve @ IoU=0.5  (AP={ap50:.3f})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(EVAL_OUT / "pr_curve.png", dpi=120)
        plt.close(fig)
        print(f"[task5] pr_curve -> {EVAL_OUT / 'pr_curve.png'}")

    # IoU histogram
    if iou_samples:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(iou_samples, bins=20, color="#4477aa", edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", label="IoU=0.5")
        ax.set_xlabel("max IoU per PD")
        ax.set_ylabel("count")
        ax.set_title("PD-level IoU distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(EVAL_OUT / "iou_hist.png", dpi=120)
        plt.close(fig)
        print(f"[task5] iou_hist -> {EVAL_OUT / 'iou_hist.png'}")

    out_per = EVAL_OUT / "per_image_metrics.jsonl"
    with out_per.open("w") as f:
        for row in per_image_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[task5] per_image_metrics -> {out_per}")


if __name__ == "__main__":
    main()
