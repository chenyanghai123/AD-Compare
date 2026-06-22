#!/bin/bash
# AD-Compare: 串行五阶段训练 pipeline（含初始 checkpoint 构建 + LoRA merge + GRPO）
set -e
echo "===== AD-Compare Full Training Pipeline ====="
echo "Start: $(date)"

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

bash "$SCRIPT_DIR/run_build_init.sh"

echo "[pipeline] Stage 0 (CE pretrain) starting..."
bash "$SCRIPT_DIR/run_stage0.sh"

echo "[pipeline] Stage 0 done, Stage 1 (LLM LoRA) starting..."
bash "$SCRIPT_DIR/run_stage1.sh"

echo "[pipeline] Stage 1 done, Stage 2 (ViT+LLM LoRA) starting..."
bash "$SCRIPT_DIR/run_stage2.sh"

echo "[pipeline] Stage 2 done, Stage 3 (Multitask SFT) starting..."
bash "$SCRIPT_DIR/run_stage3.sh"

echo "[pipeline] Stage 3 done, Stage 4 (GRPO) starting..."
bash "$SCRIPT_DIR/run_stage4.sh"

echo "[pipeline] Stage 4 done."
echo "===== AD-Compare Pipeline COMPLETE ====="
echo "End: $(date)"
