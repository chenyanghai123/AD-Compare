"""GRPO 奖励函数模块。

4 个规则奖励函数，加权组合：
- format_reward: 输出是否为合法 JSON array
- count_reward: 预测数量与 GT 数量的差异惩罚
- iou_reward: 贪心匹配后平均 IoU（class-agnostic）
- cls_reward: 匹配 bbox 的 label 精确匹配率
"""
import json
import re
from typing import Any, Dict, List

_RE_JSON_ARRAY = re.compile(r"\[\s*(?:\{[^\[\]]*?\}\s*,?\s*)+\]", re.DOTALL)
_RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
_RE_LABEL = re.compile(r'"label"\s*:\s*"([^"]*)"')


def _parse_predictions(text: str) -> List[Dict[str, Any]]:
    """从 LLM 输出中解析 [{bbox_2d: [...], label: "..."}] 列表。"""
    if not text:
        return []
    # 尝试 code block
    code = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    candidates = []
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
                out.append({
                    "bbox_2d": [int(round(float(v))) for v in bb],
                    "label": str(it.get("label", ""))
                })
            except Exception:
                continue
        if out:
            return out
    # 正则兜底
    bboxes = _RE_BBOX.findall(text)
    labels = _RE_LABEL.findall(text)
    out = []
    for i, (a, b, c, d) in enumerate(bboxes):
        out.append({
            "bbox_2d": [int(a), int(b), int(c), int(d)],
            "label": labels[i] if i < len(labels) else "",
        })
    return out


def _iou_xyxy(a, b) -> float:
    """计算两个 bbox 的 IoU。"""
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


def _greedy_match(gt_boxes, pd_boxes, iou_thr=0.0):
    """贪心匹配：每个 PD 匹配 IoU 最大且未被占用的 GT。
    返回: [(pd_idx, gt_idx, iou), ...]
    """
    matched_gt = [False] * len(gt_boxes)
    matches = []
    for i, pd in enumerate(pd_boxes):
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            iou = _iou_xyxy(pd, gt)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thr and best_j >= 0:
            matched_gt[best_j] = True
            matches.append((i, best_j, best_iou))
    return matches


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """检查输出是否为合法 JSON array。"""
    rewards = []
    for comp in completions:
        preds = _parse_predictions(comp)
        rewards.append(1.0 if preds else 0.0)
    return rewards


def count_reward(completions: List[str], gt_annotations: List[Dict] = None, **kwargs) -> List[float]:
    """预测数量与 GT 数量的差异惩罚: 1 - |n_pd - n_gt| / max(n_gt, 1)。"""
    rewards = []
    for i, comp in enumerate(completions):
        preds = _parse_predictions(comp)
        n_pd = len(preds)
        if gt_annotations and i < len(gt_annotations):
            n_gt = len(gt_annotations[i].get("bboxes", []))
        else:
            n_gt = 0
        if n_gt == 0 and n_pd == 0:
            rewards.append(1.0)
        elif n_gt == 0:
            rewards.append(0.0)
        else:
            penalty = abs(n_pd - n_gt) / max(n_gt, 1)
            rewards.append(max(0.0, 1.0 - penalty))
    return rewards


def iou_reward(completions: List[str], gt_annotations: List[Dict] = None, **kwargs) -> List[float]:
    """贪心匹配后平均 IoU（class-agnostic）。"""
    rewards = []
    for i, comp in enumerate(completions):
        preds = _parse_predictions(comp)
        pd_boxes = [p["bbox_2d"] for p in preds]
        if gt_annotations and i < len(gt_annotations):
            gt_boxes = gt_annotations[i].get("bboxes", [])
        else:
            gt_boxes = []
        if not gt_boxes or not pd_boxes:
            rewards.append(0.0)
            continue
        matches = _greedy_match(gt_boxes, pd_boxes, iou_thr=0.0)
        if matches:
            avg_iou = sum(m[2] for m in matches) / len(matches)
            rewards.append(avg_iou)
        else:
            rewards.append(0.0)
    return rewards


def cls_reward(completions: List[str], gt_annotations: List[Dict] = None, **kwargs) -> List[float]:
    """匹配对的 bbox 中 label 精确匹配的比例。"""
    rewards = []
    for i, comp in enumerate(completions):
        preds = _parse_predictions(comp)
        if gt_annotations and i < len(gt_annotations):
            gt_boxes = gt_annotations[i].get("bboxes", [])
            gt_labels = gt_annotations[i].get("labels", [])
        else:
            gt_boxes = []
            gt_labels = []
        if not gt_boxes or not preds:
            rewards.append(0.0)
            continue
        pd_boxes = [p["bbox_2d"] for p in preds]
        pd_labels = [p["label"] for p in preds]
        matches = _greedy_match(gt_boxes, pd_boxes, iou_thr=0.5)
        if not matches:
            rewards.append(0.0)
            continue
        correct = 0
        for pd_idx, gt_idx, _ in matches:
            if pd_idx < len(pd_labels) and gt_idx < len(gt_labels):
                if pd_labels[pd_idx].lower() == gt_labels[gt_idx].lower():
                    correct += 1
        rewards.append(correct / len(matches))
    return rewards


REWARD_FUNCS = [format_reward, count_reward, iou_reward, cls_reward]
DEFAULT_WEIGHTS = [0.1, 0.1, 0.5, 0.3]
