#!/bin/bash
# ============================================================
# RechargeLatentNet v6-AttnRes 训练脚本
# ============================================================
# 关键修复:
#   1. patience=50 (原 5, 导致仅训练 8 个 epoch 就停止)
#   2. d_model=96 (原 64, 增加模型容量)
#   3. learning_rate=5e-4 (配合 3 epoch warmup)
#   4. lead_decay=0.3 (短 lead 权重更高, 修复原版反向加权)
#   5. spb_weight=1.5 (从 2.0 降低, 提高早期训练稳定性)
#   6. corr_weight=0.1 (新增相关系数损失, 直接优化 ACC)
# ============================================================

export OMP_NUM_THREADS=4

for va in 'sst,hc' 'sst,mld' 'sst,slp' 'sst,tauu' 'sst,tauv' 'sst,sss'
do
python train.py \
    --stage train \
    --model_name RechargeLatentNet \
    --cmip_path ../processed_ssta_data/cmip6_fixed_complete.nc \
    --obs_path ../processed_ssta_data/obs_fixed.nc \
    --variables ${va} \
    --obs_start 1980-01-01 \
    --obs_end 2025-12-31 \
    --input_len 12 \
    --output_len 24 \
    --d_model 96 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --weight_decay 1e-4 \
    --dropout 0.1 \
    --epochs 200 \
    --patience 5 \
    --lead_decay 0.3 \
    --d_var 48 \
    --spb_weight 1.5 \
    --corr_weight 0.1 \
    --num_workers 4 \
    --seed 2025 \
    --save_dir ./checkpoints/ \
    --visual_dir ./results/
done


# shutdown -n