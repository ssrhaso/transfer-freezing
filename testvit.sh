#!/usr/bin/env bash
set -e

CONFIG="config.yaml"
SEED=1
VIT_MODES=(freeze_none freeze_patch freeze_patch_0_2 freeze_patch_0_5 freeze_patch_0_11)

mkdir -p outputs/logs
mkdir -p outputs/results

echo " STARTING ViT SMOKE TEST (SEED ${SEED})"

# 1. Run the 5 ViT configurations
for mode in "${VIT_MODES[@]}"; do
  echo "Running ViT | Mode: ${mode} | Seed: ${SEED}..."
  
  python train.py \
    --config "$CONFIG" \
    --model vit_b_16 \
    --freeze_mode "$mode" \
    --seed "$SEED" \
    2>&1 | tee "outputs/logs/vit_b_16_${mode}_seed${SEED}.log"
done

echo " TRAINING COMPLETE. GENERATING PLOT..."

# 2. Run the plotting script for this seed
python analysis/plot.py --seed "$SEED"

echo " DONE! CHECK outputs/results/plots/seed${SEED}_vit_barplot.png"