#!/bin/bash
# AD-Compare: 推理测试
set -e

# ============ 按需修改 ============
export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=./checkpoints/stage3_multitask_sft_merged
# ==================================

python tools/infer.py --model_path $MODEL_PATH
