"""AD-Compare Gradio Web UI

用法:
    # 默认使用训练好的模型
    CUDA_VISIBLE_DEVICES=0 python app.py

    # 指定模型路径
    python app.py --model /path/to/model

    # 开启公网分享链接
    python app.py --share

    # 指定端口
    python app.py --port 8080
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont

from ad_compare import (
    AdCompareQwen3VLConfig,
    AdCompareQwen3VLForConditionalGeneration,
)
from ad_compare.dataset_ad_compare import load_processor

import gradio as gr

PRED_NORM_SCALE = 1000.0

GROUNDING_PROMPT = (
    '<image><image>Given the normal reference (first), identify and localize defects '
    'in the second image. Format: [{"bbox_2d": [x1,y1,x2,y2], "label": "type"}]'
)
CLASSIFICATION_PROMPT = (
    "<image><image>The first image is a normal reference sample. "
    "Is there any anomaly in the second image? A. Yes B. No. "
    "Please answer the letter only."
)
DESCRIPTION_PROMPT = (
    "<image><image>Compare the normal reference (first image) with the test image "
    "(second). Describe any defects you observe."
)

_RE_JSON_ARRAY = re.compile(r"\[\s*(?:\{[^\[\]]*?\}\s*,?\s*)+\]", re.DOTALL)
_RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
_RE_LABEL = re.compile(r'"label"\s*:\s*"([^"]*)"')

_model = None
_processor = None
_device = None


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
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


def pd_to_orig(pd_bbox, orig_w, orig_h):
    fx = orig_w / PRED_NORM_SCALE
    fy = orig_h / PRED_NORM_SCALE
    x1, y1, x2, y2 = pd_bbox
    return [x1 * fx, y1 * fy, x2 * fx, y2 * fy]


def find_font(size: int = 16) -> ImageFont.ImageFont:
    for cand in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(cand, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_bboxes(image: Image.Image, bboxes: List[Dict]) -> Image.Image:
    img = image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    orig_w, orig_h = img.size
    font = find_font(max(14, min(orig_w, orig_h) // 30))
    line_w = max(2, min(orig_w, orig_h) // 200)
    for p in bboxes:
        bb = pd_to_orig(p["bbox_2d"], orig_w, orig_h)
        x1, y1, x2, y2 = [int(round(v)) for v in bb]
        draw.rectangle([x1, y1, x2, y2], outline=(220, 30, 30), width=line_w)
        label = p.get("label", "")
        if label:
            draw.text((x1 + 2, max(0, y1 - 18)), label, fill=(220, 30, 30), font=font)
    return img


def build_messages(question: str, num_images: int):
    parts = question.split("<image>")
    contents = []
    img_idx = 0
    for i, seg in enumerate(parts):
        if i > 0 and img_idx < num_images:
            contents.append({"type": "image"})
            img_idx += 1
        if seg:
            contents.append({"type": "text", "text": seg})
    while img_idx < num_images:
        contents.insert(0, {"type": "image"})
        img_idx += 1
    return [{"role": "user", "content": contents}]


def resize_image(img: Image.Image, max_side: int) -> Image.Image:
    if max_side > 0 and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    return img


def run_inference(
    ref_image: Image.Image,
    test_image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    max_image_side: int,
) -> tuple[str, float, int]:
    images = [
        resize_image(ref_image, max_image_side),
        resize_image(test_image, max_image_side),
    ]
    msgs = build_messages(prompt, len(images))
    text = _processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {
        k: (v.to(_device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }
    t0 = time.time()
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=_processor.tokenizer.pad_token_id or _processor.tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    in_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, in_len:]
    pred = _processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return pred, dt, len(gen_ids)


def predict_grounding(ref_image, test_image, max_new_tokens, max_image_side):
    if ref_image is None or test_image is None:
        return None, "请上传参考图和测试图", ""
    pred, dt, n_tok = run_inference(
        ref_image, test_image, GROUNDING_PROMPT, max_new_tokens, max_image_side
    )
    bboxes = extract_json_from_text(pred)
    result_img = draw_bboxes(test_image, bboxes) if bboxes else test_image
    info = f"推理耗时: {dt:.2f}s | 生成 tokens: {n_tok} | 检测到 {len(bboxes)} 个缺陷"
    return result_img, pred, info


def predict_classification(ref_image, test_image, max_new_tokens, max_image_side):
    if ref_image is None or test_image is None:
        return "请上传参考图和测试图", ""
    pred, dt, n_tok = run_inference(
        ref_image, test_image, CLASSIFICATION_PROMPT, max_new_tokens, max_image_side
    )
    info = f"推理耗时: {dt:.2f}s | 生成 tokens: {n_tok}"
    return pred, info


def predict_description(ref_image, test_image, max_new_tokens, max_image_side):
    if ref_image is None or test_image is None:
        return "请上传参考图和测试图", ""
    pred, dt, n_tok = run_inference(
        ref_image, test_image, DESCRIPTION_PROMPT, max_new_tokens, max_image_side
    )
    info = f"推理耗时: {dt:.2f}s | 生成 tokens: {n_tok}"
    return pred, info


def load_model(model_path: str, attn_impl: str = "sdpa"):
    global _model, _processor, _device
    print(f"[loading] model from {model_path} ...")
    config = AdCompareQwen3VLConfig.from_pretrained(model_path)
    _model = AdCompareQwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    _processor = load_processor(model_path)
    if _processor.tokenizer.pad_token is None:
        _processor.tokenizer.pad_token = _processor.tokenizer.eos_token
    _device = _model.device
    print(f"[ok] model loaded, dtype={next(_model.parameters()).dtype}, device={_device}")


def build_ui():
    with gr.Blocks(title="AD-Compare 缺陷检测", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# AD-Compare: 工业异常检测\n"
            "上传正常参考图和测试图，自动检测缺陷。基于 Qwen3-VL-8B + Comparison Encoder。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                ref_input = gr.Image(label="正常参考图", type="pil", height=300)
                test_input = gr.Image(label="测试图", type="pil", height=300)
                with gr.Row():
                    max_tokens = gr.Slider(64, 1024, value=512, step=64, label="max_new_tokens")
                    max_side = gr.Slider(256, 1024, value=448, step=64, label="max_image_side")
                gr.Examples(
                    examples=[
                        [
                            "/data1/chenyanghai/vispectdl/data/AnomalyDatasets/mvtec/tile/train/good/178.png",
                            "/data1/chenyanghai/vispectdl/data/AnomalyDatasets/mvtec/tile/defect/glue_strip/002.png",
                        ],
                    ],
                    inputs=[ref_input, test_input],
                    label="示例（tile/glue_strip）",
                )

            with gr.Column(scale=1):
                with gr.Tab("缺陷定位"):
                    gr.Markdown("检测测试图中的缺陷位置，输出 bounding box。")
                    ground_btn = gr.Button("开始定位", variant="primary")
                    ground_out_img = gr.Image(label="检测结果", type="pil", height=300)
                    ground_out_text = gr.Textbox(label="原始输出", lines=6)
                    ground_info = gr.Textbox(label="状态", interactive=False)

                with gr.Tab("缺陷分类"):
                    gr.Markdown("判断测试图是否有缺陷（A. Yes / B. No）。")
                    cls_btn = gr.Button("开始分类", variant="primary")
                    cls_out = gr.Textbox(label="分类结果", lines=3)
                    cls_info = gr.Textbox(label="状态", interactive=False)

                with gr.Tab("缺陷描述"):
                    gr.Markdown("详细描述测试图中的缺陷。")
                    desc_btn = gr.Button("开始描述", variant="primary")
                    desc_out = gr.Textbox(label="描述结果", lines=8)
                    desc_info = gr.Textbox(label="状态", interactive=False)

        ground_btn.click(
            predict_grounding,
            inputs=[ref_input, test_input, max_tokens, max_side],
            outputs=[ground_out_img, ground_out_text, ground_info],
        )
        cls_btn.click(
            predict_classification,
            inputs=[ref_input, test_input, max_tokens, max_side],
            outputs=[cls_out, cls_info],
        )
        desc_btn.click(
            predict_description,
            inputs=[ref_input, test_input, max_tokens, max_side],
            outputs=[desc_out, desc_info],
        )

    return demo


def main():
    ap = argparse.ArgumentParser(description="AD-Compare Web UI")
    ap.add_argument("--model", default="outputs/stage4_grpo_merged", help="模型路径")
    ap.add_argument("--port", type=int, default=7860, help="服务端口")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址")
    ap.add_argument("--share", action="store_true", help="生成公网分享链接")
    ap.add_argument("--attn_implementation", default="sdpa", help="注意力实现")
    args = ap.parse_args()

    load_model(args.model, args.attn_implementation)
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
