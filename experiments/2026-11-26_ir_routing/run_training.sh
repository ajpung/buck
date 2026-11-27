#!/usr/bin/env bash
# Train the grayscale arm. Identical to the baseline's notebook call
# (trail cam/examples/261106 ... .ipynb, run_cv) except --grayscale and the
# pinned --image-root. Run from the repo root:
#   bash experiments/2026-11-26_ir_routing/run_training.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=buck-env/Scripts/python.exe
OUT=benchmark_runs/ir_routing/gray
LOGS=experiments/2026-11-26_ir_routing/logs
mkdir -p "$OUT" "$LOGS"
for SEED in 42 43 44; do
  echo "=== seed $SEED start $(date -Iseconds)"
  PYTHONPATH=src "$PY" -u -m buck.benchmark.compare_architectures \
    --image-root benchmark_runs/ir_routing/pinned_images/squared \
    --manifest "trail cam/splits/holdout_test_v2.json" \
    --output "$OUT/vit_base_dinov3_s$SEED" \
    --sources NDA --models vit_base_dinov3 --folds 10 --epochs 60 \
    --augmentation medium --train-multiplier 8 --seed "$SEED" --split-seed 1337 \
    --select-metric qwk --min-class-count 8 --checkpoint-policy final \
    --test-policy none --no-profile --grayscale \
    > "$LOGS/train_s$SEED.log" 2>&1
  echo "=== seed $SEED done $(date -Iseconds)"
done
