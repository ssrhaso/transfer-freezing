#!/usr/bin/env bash
set -e

CONFIG="config.yaml"
SEEDS=(1 10 15 20 25 30 35 40 45 50)

VIT_MODES=(freeze_none freeze_patch freeze_patch_0_2 freeze_patch_0_5 freeze_patch_0_11)
RESNET_MODES=(freeze_none freeze_0 freeze_0_1 freeze_0_1_2 freeze_0_1_2_3 freeze_0_1_2_3_4)

mkdir -p outputs/logs

for seed in "${SEEDS[@]}"; do
  echo " SEED ${seed} "

  for mode in "${VIT_MODES[@]}"; do
    python train.py --config "$CONFIG" --model vit_b_16 --freeze_mode "$mode" --seed "$seed" \
      2>&1 | tee "outputs/logs/vit_b_16_${mode}_seed${seed}.log"
  done

  for mode in "${RESNET_MODES[@]}"; do
    python train.py --config "$CONFIG" --model resnet50 --freeze_mode "$mode" --seed "$seed" \
      2>&1 | tee "outputs/logs/resnet50_${mode}_seed${seed}.log"
  done

  python analysis/plot_seed_sweeps.py --seed "$seed"
done

python analysis/aggregate_results.py
