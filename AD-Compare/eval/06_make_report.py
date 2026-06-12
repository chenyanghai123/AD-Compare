"""Task 7 — 生成 SiliconInstance Grounding 评估报告 REPORT.md。

依赖产物:
- pair_map.json / ng_index.json
- pred_raw.jsonl
- metrics.json / pr_curve.png / iou_hist.png / per_image_metrics.jsonl
- vis/*.jpg + vis_index.json
- task4_run.log
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))
from eval.utils import EVAL_OUT


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    metrics = json.loads((EVAL_OUT / "metrics.json").read_text())
    pair_map = json.loads((EVAL_OUT / "pair_map.json").read_text())
    vis_index = json.loads((EVAL_OUT / "vis_index.json").read_text())

    # 读 pred_raw 收集分布
    splits = Counter()
    sims = []
    gen_tokens = []
    durations = []
    label_count_pd = Counter()
    label_count_gt = Counter()
    for line in (EVAL_OUT / "pred_raw.jsonl").open():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "error" in rec:
            continue
        splits[rec.get("split", "?")] += 1
        gen_tokens.append(rec.get("gen_tokens", 0))
        durations.append(rec.get("dt", 0.0))
        for p in rec.get("pd", []):
            label_count_pd[p.get("label", "")] += 1
        for g in rec.get("gt", []):
            label_count_gt[g.get("label", "")] += 1

    for v in pair_map.values():
        sims.append(v["similarity"])

    avg_dt = sum(durations) / max(len(durations), 1)
    avg_tok = sum(gen_tokens) / max(len(gen_tokens), 1)

    # group vis_index by category
    by_cat: dict[str, list[dict]] = {}
    for item in vis_index:
        by_cat.setdefault(item["category"], []).append(item)

    # 构造 markdown
    lines: list[str] = []
    lines.append("# SiliconInstance Grounding 评估报告")
    lines.append("")
    lines.append("> 用 `stage3_multitask_sft_merged` 模型，在 663 张工业硅片缺陷 NG 图上做 grounding 评估。")
    lines.append("> 无成对参考图，自动从 15 380 张 OK 池检索 top-1 reference + 灰度模态对齐。")
    lines.append("")

    # 1. 数据集与实验配置
    lines.append("## 1. 数据集与实验配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 数据集 | `data/TODO/chenyanghai/EvalDatasets/InstanceSegmentation/SiliconInstance` |")
    lines.append(f"| NG 图像 | 663 张 RGB JPG，分辨率变长（典型 530×480 ~ 759×926），同名 `.tzjson` 多边形标注 |")
    lines.append(f"| OK 池 | 15 380 张 PNG，384×384 单通道灰度晶圆图 |")
    lines.append(f"| Train / Val | {splits.get('train', 0)} / {splits.get('val', 0)}（合并评估） |")
    lines.append(f"| 总 GT bbox | {metrics['total_gt_box']} 个；类别：{', '.join(label_count_gt.keys())}（共 {len(label_count_gt)} 类汉拼） |")
    lines.append(f"| 模型 | `stage3_multitask_sft_merged`（merge LoRA 后的最终权重） |")
    lines.append(f"| 推理 | bfloat16 / sdpa attention / greedy（do_sample=False, num_beams=1）/ max_new_tokens=512 |")
    lines.append(f"| 单张耗时 | 平均 {avg_dt:.2f} s/张，平均生成 {avg_tok:.1f} tokens |")
    lines.append("")
    lines.append("Prompt（与训练 grounding 任务一致）：")
    lines.append("```")
    lines.append("<image><image>Given the normal reference (first), identify and localize defects in the second image.")
    lines.append('Format: [{"bbox_2d": [x1,y1,x2,y2], "label": "type"}]')
    lines.append("```")
    lines.append("")

    # 2. 模态对齐 + Reference 检索
    lines.append("## 2. 模态对齐 + Reference 检索策略")
    lines.append("")
    lines.append("**模态对齐**（NG/OK 域差异极大，工业灰度 vs 训练 RGB）：")
    lines.append("```python")
    lines.append("ng = Image.open(ng_path).convert('L').convert('RGB')   # 保持原尺寸")
    lines.append("ok = Image.open(ok_path).convert('L').convert('RGB')")
    lines.append("ok = ok.resize(ng.size, Image.BILINEAR)                 # OK resize 到 NG 尺寸")
    lines.append("```")
    lines.append("")
    lines.append("**Reference 检索（外网不可用，未加载额外 ViT）**：将 NG / OK 灰度→64×64→flatten→L2-normalize，")
    lines.append("用 cosine 相似度从 15 380 张 OK 池为每张 NG 检索 top-1。")
    lines.append("")
    lines.append("- 相似度统计：min={:.3f} / mean={:.3f} / max={:.3f}".format(
        min(sims) if sims else 0.0, sum(sims) / max(len(sims), 1), max(sims) if sims else 0.0))
    lines.append(f"- OK feature 大小：(15380, 4096) float32 = 252 MB")
    lines.append("")

    # 3. 总体指标表
    lines.append("## 3. 总体指标")
    lines.append("")
    lines.append("**坐标系约定**：")
    lines.append("- GT bbox 在 NG 原图坐标系；")
    lines.append("- PD bbox 由模型在 [0, 1000] 归一化空间输出（Qwen-VL 训练惯例，已在 `stage3_real_clean.json` 验证最大坐标=1000）；")
    lines.append("- 评估前把 PD 用 `(orig_w/1000, orig_h/1000)` 缩到原图坐标系，统一在原图坐标系比 IoU。")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| **mAP@0.5** | **{metrics['mAP@0.5']:.4f}** |")
    lines.append(f"| **mAP@[.5:.95]** | **{metrics['mAP@[.5:.95]']:.4f}** |")
    lines.append(f"| Precision@0.5 | {metrics['Precision@0.5']:.4f} |")
    lines.append(f"| Recall@0.5 | {metrics['Recall@0.5']:.4f} |")
    lines.append(f"| F1@0.5 | {metrics['F1@0.5']:.4f} |")
    lines.append(f"| 评估样本数 | {metrics['n_records_eval']} / 663 |")
    lines.append(f"| 总 GT bbox | {metrics['total_gt_box']} |")
    lines.append(f"| 总 PD bbox | {metrics['total_pd_box']} |")
    lines.append(f"| 平均 GT/张 | {metrics['avg_gt_per_image']:.2f} |")
    lines.append(f"| 平均 PD/张 | {metrics['avg_pd_per_image']:.2f} |")
    lines.append(f"| empty pred 比例 | {fmt_pct(metrics['empty_pd_ratio'])} |")
    lines.append(f"| 推理失败数 | {metrics['n_with_error']} |")
    lines.append(f"| PD-level 平均 IoU | {metrics['iou_mean_per_pd']:.4f}（中位 {metrics['iou_median_per_pd']:.4f}） |")
    lines.append("")
    lines.append("**多 IoU 阈值 AP 曲线**：")
    lines.append("")
    lines.append("| IoU 阈值 | AP |")
    lines.append("|---|---|")
    for thr, ap in metrics["ap_per_thr"].items():
        lines.append(f"| {thr} | {ap:.4f} |")
    lines.append("")

    # 4. PR / IoU 图
    lines.append("## 4. PR 曲线与 IoU 分布")
    lines.append("")
    lines.append(f"![PR Curve](pr_curve.png)")
    lines.append("")
    lines.append(f"![IoU Histogram](iou_hist.png)")
    lines.append("")

    # 5. 抽样可视化
    lines.append("## 5. 典型样本可视化（每类 5 张）")
    lines.append("")
    cat_label = {
        "gold": "✅ 全命中（n_gt 与 PD IoU≥0.5 全部命中）",
        "partial": "🟡 部分命中（命中数 < n_gt）",
        "miss": "❌ 全失（最大 IoU < 0.5）",
        "fp": "⚠️ 过预测（PD 数远多于 GT）",
    }
    for cat in ["gold", "partial", "miss", "fp"]:
        items = by_cat.get(cat, [])
        if not items:
            continue
        lines.append(f"### 5.{['gold', 'partial', 'miss', 'fp'].index(cat) + 1} {cat_label[cat]}")
        lines.append("")
        for it in items:
            img_path = it["image"]
            # 写裸相对路径（不带 ./ 前缀也不带 / 前缀）。
            # 这是 Qoder 、VS Code 等 IDE 内置 Markdown 预览器
            # 对本地资源兼容性最好的写法。
            try:
                rel_path = Path(img_path).resolve().relative_to(EVAL_OUT.resolve())
                img_link = rel_path.as_posix()
            except ValueError:
                img_link = img_path
            ng_name = Path(it["ng_path"]).name
            lines.append(f"![{cat}: {ng_name}]({img_link})")
            lines.append("")
        lines.append("")

    # 6. 结论
    lines.append("## 6. 结论与失效模式分析")
    lines.append("")
    lines.append("**核心结论**（基于 663 张全量结果）：")
    lines.append(f"- mAP@0.5 = **{metrics['mAP@0.5']:.4f}**, F1@0.5 = **{metrics['F1@0.5']:.4f}**, empty rate = **{fmt_pct(metrics['empty_pd_ratio'])}**")
    if metrics["mAP@0.5"] >= 0.3:
        verdict = "模型在工业 OOD 灰度域仍能产出有意义的 grounding，但受限于训练数据 (mvtec/visa 等 RGB 域)，性能与训练域有显著差距。"
    elif metrics["mAP@0.5"] >= 0.1:
        verdict = "模型能在部分场景定位缺陷，但整体精度偏低，主要受工业灰度域+真实复杂背景影响。"
    else:
        verdict = "模型在工业灰度域几乎不能正确定位缺陷，强烈建议加入硅片域微调数据。"
    lines.append(f"- {verdict}")
    lines.append("")
    lines.append("**已识别的失效模式**：")
    lines.append("1. **PD 数偏少**：模型平均 PD={:.2f}，GT={:.2f}，召回主要瓶颈在 *漏检*（部分图只输出 1 个 bbox 即停）。".format(
        metrics["avg_pd_per_image"], metrics["avg_gt_per_image"]))
    lines.append("2. **PD label 漂移**：训练标签均为英文（scratch/cut/stain 等），与硅片 8 类汉拼标签完全不重合 — 因此采用 *class-agnostic* mAP，仅按 IoU 评估定位精度。")
    lines.append("3. **域漂移**：纯灰度晶圆图 + 长条形纹理（断/列纹）与训练域差异较大，部分细小缺陷被忽略。")
    lines.append("")
    lines.append("**改进方向**：")
    lines.append("- 在 SFT 阶段补入硅片真实数据（含汉拼标签）做 100~500 步 fine-tune；")
    lines.append("- prompt 中显式声明 \"输出 bbox 坐标在 [0,1000] 归一化空间\"，避免推理时坐标系误用；")
    lines.append("- 引入 SigLIP/CLIP 同源 ViT 抽取更语义化的 reference embedding，预期提升 reference 选取质量。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("**产物清单**（位于 `meta/eval_infos/silicon_instance_grounding/`）：")
    lines.append("")
    lines.append("- `ok_features.npy` (15380, 4096) float32 — OK 池 raw 灰度特征")
    lines.append("- `ok_paths.txt` — OK 池路径列表")
    lines.append("- `pair_map.json` — NG → top-1 OK reference 映射")
    lines.append("- `ng_index.json` — NG → (OK, GT, split, size) 索引")
    lines.append("- `pred_raw.jsonl` — 每张 NG 的 GT/PD/raw 输出")
    lines.append("- `metrics.json` — 总体指标")
    lines.append("- `per_image_metrics.jsonl` — 每张图 IoU 明细")
    lines.append("- `pr_curve.png` / `iou_hist.png` — 指标曲线图")
    lines.append("- `vis/*.jpg` — 抽样可视化")
    lines.append("- `task4_run.log` — 推理日志")
    lines.append("")

    out = EVAL_OUT / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"[task7] -> {out}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
