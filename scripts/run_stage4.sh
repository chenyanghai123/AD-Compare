#!/bin/bash
# AD-Compare Stage 4: GRPO 强化学习 (多卡 ZeRO-2)
set -e

# ============ 按需修改 ============
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
# ==================================

python -m torch.distributed.run --nproc_per_node=$NPROC \
    tools/train_grpo.py \
    configs/stage4_grpo.yaml
