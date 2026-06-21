#!/bin/bash
# AD-Compare: 串行评估 pipeline（6 步）
# 需先设置环境变量: EVAL_DATA_ROOT, MODEL_PATH
set -e

echo "===== AD-Compare Evaluation Pipeline ====="
echo "Start: $(date)"

# ============ 按需修改 ============
export CUDA_VISIBLE_DEVICES=0
export EVAL_DATA_ROOT=/path/to/eval_dataset       # 评估数据集根目录
export EVAL_OUT_DIR=./eval_outputs                # 评估输出目录
export MODEL_PATH=./checkpoints/stage3_multitask_sft_merged
# ==================================

echo "[1/6] Extracting OK pool features..."
python eval/01_extract_ok_features.py

echo "[2/6] Retrieving top-1 OK reference for each NG..."
python eval/02_retrieve_reference.py

echo "[3/6] Running grounding inference..."
python eval/03_infer_grounding.py

echo "[4/6] Computing mAP metrics..."
python eval/04_compute_map.py

echo "[5/6] Generating visualization samples..."
python eval/05_visualize_samples.py

echo "[6/6] Making evaluation report..."
python eval/06_make_report.py

echo "===== Evaluation COMPLETE ====="
echo "Results saved to: $EVAL_OUT_DIR"
echo "End: $(date)"
