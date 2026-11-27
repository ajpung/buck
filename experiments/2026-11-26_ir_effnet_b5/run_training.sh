#!/usr/bin/env bash
# Train the EfficientNet-B5 arm (600 px). Same call as the baseline except the
# model, --image-size 600, and the pinned --image-root. From the repo root:
#   bash experiments/2026-11-26_ir_effnet_b5/run_training.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=buck-env/Scripts/python.exe
OUT=benchmark_runs/ir_effnet_b5/b5
LOGS=experiments/2026-11-26_ir_effnet_b5/logs
mkdir -p "$OUT" "$LOGS"
for SEED in 42 43 44; do
  echo "=== seed $SEED start $(date -Iseconds)"
  PYTHONPATH=src "$PY" -u -m buck.benchmark.compare_architectures \
    --image-root benchmark_runs/ir_routing/pinned_images/squared \
    --manifest "trail cam/splits/holdout_test_v2.json" \
    --output "$OUT/efficientnet_b5_s$SEED" \
    --sources NDA --models efficientnet_b5 --image-size 600 --folds 10 --epochs 60 \
    --augmentation medium --train-multiplier 8 --seed "$SEED" --split-seed 1337 \
    --select-metric qwk --min-class-count 8 --checkpoint-policy final \
    --test-policy none --no-profile \
    > "$LOGS/train_s$SEED.log" 2>&1
  echo "=== seed $SEED done $(date -Iseconds)"
done
