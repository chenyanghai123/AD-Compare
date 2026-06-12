#!/bin/bash
# AD-Compare: 构建初始 checkpoint（从 Qwen3-VL-8B-Instruct 注入 CE 模块）
# 仅需运行一次；已有则跳过
set -e

# ============ 请修改以下路径 ============
SRC=/path/to/Qwen3-VL-8B-Instruct      # Qwen3-VL 原始权重路径
DST=./checkpoints/ad_compare_init       # 输出初始 checkpoint 路径
# =======================================

if [ -d "$DST" ]; then
    echo "[skip] $DST already exists."
    exit 0
fi

python ad_compare/build_initial_checkpoint.py --src $SRC --dst $DST
echo "[done] Initial checkpoint saved to $DST"
