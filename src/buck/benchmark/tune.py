"""Nested-CV hyperparameter tuning: a tuned score you can actually report.

Tuning on the same folds you report is how this project has already produced
three separate phantom gains. Best-epoch checkpointing added +0.070 qwk,
greedy ensemble selection adds +0.065 accuracy, and the original notebook's
best-of-130 checkpoint rule turned 62% test into 80% "validation". All three
are the same mistake: taking a maximum over many noisy draws and reporting the
maximum. With ~46 images in a validation fold, one image is 2.2% accuracy, so
a 12-config grid will hand you a winner roughly 0.05 above its own true mean.

Nested cross-validation removes that. The configuration is chosen on *inner*
folds carved out of each outer training set, and scored on the outer
validation fold the selection never touched. The outer score is therefore an
estimate of "tune, then deploy", not of "tune and peek".

The grid is a module constant rather than a command-line argument, so it is
pre-registered: it cannot be widened after seeing which way the results went.

``--also-naive`` additionally runs a plain grid search on the outer folds and
prints the gap between the two. That gap is the number you would otherwise
have published. It is optional because it costs |grid| x outer_folds more
trainings, and it must train on the same data as the outer scores to mean
anything -- the inner scores cannot stand in, since those models see only
(inner-1)/inner of each training set and score lower for reasons unrelated to
selection bias.

Cost is the honest downside: outer x inner x |grid| trainings. The default
grid is 8 configurations at 3 inner folds over 5 outer folds, which is 125
fold-trainings, roughly 12-15 hours on one architecture.

Usage::

    python -m buck.benchmark.tune --model convnext_tiny --output benchmark_runs/tune_cnx
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from buck.benchmark import architectures as arch
from buck.benchmark.compare_architectures import (SCRATCH_BACKBONE_LR,
                                                  TRAIN_DEFAULTS, train_fold)
from buck.benchmark.data import (build_groups, decode_images,
                                 drop_rare_classes, load_records)
from buck.benchmark.ensemble import rebuild_splits
from buck.benchmark.metrics import ordinal_metrics

DEFAULT_IMAGE_ROOT = (
    Path(__file__).resolve().parents[3] / "trail cam" / "images" / "squared"
)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "trail cam" / "splits"
    / "holdout_test_v2.json"
)

# --------------------------------------------------------------------------
# PRE-REGISTERED GRID. Fixed before any result is seen, and deliberately small.
#
# Only knobs with a reason to matter on this corpus:
#   image_size    -- body proportions are a fine-detail judgement and 224px may
#                    simply be discarding the evidence. Never measured under
#                    the corrected pipeline (HANDOFF.md section 8).
#   augmentation  -- likewise unmeasured post-fix.
#   backbone_lr   -- how hard to disturb transferred features on 184 images.
#
# Widening this after seeing results reintroduces exactly the bias the nesting
# removes. Add configurations only with a stated reason, before a run.
# --------------------------------------------------------------------------
GRID = [
    dict(image_size=224, augmentation="medium", backbone_lr=1e-4),  # anchor
    dict(image_size=224, augmentation="light", backbone_lr=1e-4),
    dict(image_size=224, augmentation="heavy", backbone_lr=1e-4),
    dict(image_size=224, augmentation="medium", backbone_lr=3e-4),
    dict(image_size=320, augmentation="medium", backbone_lr=1e-4),
    dict(image_size=320, augmentation="heavy", backbone_lr=1e-4),
    dict(image_size=320, augmentation="medium", backbone_lr=3e-4),
    dict(image_size=384, augmentation="medium", backbone_lr=1e-4),
]


def label(cfg):
    return (f"{cfg['image_size']}px/{cfg['augmentation']}/"
            f"lr{cfg['backbone_lr']:.0e}")


def build_config(model_name, cfg, args):
    """Materialise one grid point into a train_fold config dict."""
    config = dict(TRAIN_DEFAULTS)
    config["max_epochs"] = args.epochs
    config["patience"] = args.patience
    config["augmentation"] = cfg["augmentation"]
    config["train_multiplier"] = args.train_multiplier
    config["pretrained"] = not args.no_pretrained
    config["loss"] = args.loss
    config["ordinal_sigma"] = 0.65
    config["mixup"] = 0.0
    config["backbone_lr"] = (SCRATCH_BACKBONE_LR if args.no_pretrained
                             else cfg["backbone_lr"])
    ceiling = int(arch.REGISTRY[model_name]["batch"] * args.batch_scale)
    # Larger inputs need a smaller batch to fit; scale by area against 224.
    scale = (224.0 / cfg["image_size"]) ** 2
    config["batch_size"] = max(2, min(int(ceiling * scale), args.max_batch))
    return config


def train_and_score(model_name, images, labels, train_idx, val_idx, class_ages,
                    config, device, seed, args):
    _, metrics, _, _ = train_fold(
        model_name,
        images[train_idx], labels[train_idx],
        images[val_idx], labels[val_idx],
        class_ages, config, device,
        seed=seed,
        select_metric=args.metric,
        tta=False,
        verbose=False,
        policy=args.checkpoint_policy,
        ema_decay=args.ema_decay,
    )
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="architecture to tune")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--configs", type=int, nargs="*", default=None,
                   help="indices into GRID; default is all of them")
    p.add_argument("--metric", default="accuracy",
                   choices=["accuracy", "qwk", "within_one", "macro_f1"],
                   help="what the inner folds select on")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--train-multiplier", type=int, default=8)
    p.add_argument("--batch-scale", type=float, default=1.0)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--checkpoint-policy", default="final", choices=["final", "best"])
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--loss", default="ce", choices=["ce", "ordinal"])
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--also-naive", action="store_true",
                   help="additionally run a plain grid search on the outer "
                        "folds, to measure the optimism nesting removes; "
                        "costs |grid| x outer_folds more fold-trainings")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--min-class-count", type=int, default=8)
    args = p.parse_args(argv)

    grid = [GRID[i] for i in args.configs] if args.configs else list(GRID)
    model_name = args.model

    if model_name not in arch.REGISTRY:
        raise SystemExit(f"unknown architecture {model_name!r}")

    # A fixed-input backbone silently ignores --image-size, which would make
    # several grid points identical and the "winner" among them pure noise.
    if arch.REGISTRY[model_name].get("fixed_input"):
        sizes = {c["image_size"] for c in grid}
        if len(sizes) > 1:
            native = arch.REGISTRY[model_name]["size"]
            seen, collapsed = set(), []
            for c in grid:
                key = (c["augmentation"], c["backbone_lr"])
                if key not in seen:
                    seen.add(key)
                    collapsed.append(dict(c, image_size=native))
            print(f"[grid] {model_name} has a fixed {native}px input; the "
                  f"resolution axis cannot be tuned. Collapsed "
                  f"{len(grid)} -> {len(collapsed)} configurations.")
            grid = collapsed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = drop_rare_classes(
        load_records(args.image_root), args.min_class_count)
    class_ages = sorted({r.age for r in records})
    groups = build_groups(records, verbose=False)
    labels, dev_idx, test_idx, outer = rebuild_splits(
        records, groups, class_ages, args.manifest, args.outer_folds,
        args.split_seed, args.test_fraction)

    sizes = sorted({arch.input_size(model_name, c["image_size"]) for c in grid})
    print(f"[data] {len(records)} records | dev {len(dev_idx)} | "
          f"outer {args.outer_folds} x inner {args.inner_folds} x "
          f"{len(grid)} configs")
    print(f"[data] decoding at {sizes}px")
    images_by_size = {s: decode_images(records, s) for s in sizes}

    print(f"[plan] {args.outer_folds * args.inner_folds * len(grid) + args.outer_folds}"
          f" fold-trainings total\n")

    started = time.time()
    outer_rows, inner_log = [], []

    for o, (o_tr, o_va) in enumerate(outer, start=1):
        tr_global, va_global = dev_idx[o_tr], dev_idx[o_va]
        print(f"{'=' * 66}\nOUTER FOLD {o}/{args.outer_folds}  "
              f"train {len(tr_global)} / val {len(va_global)}\n{'=' * 66}",
              flush=True)

        # --- inner selection, entirely within the outer training set
        inner = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                                     random_state=args.split_seed + o)
        splits = list(inner.split(np.zeros(len(tr_global)),
                                  labels[tr_global], groups[tr_global]))
        scores = []
        for ci, cfg in enumerate(grid):
            size = arch.input_size(model_name, cfg["image_size"])
            imgs = images_by_size[size]
            config = build_config(model_name, cfg, args)
            vals = []
            for i_tr, i_va in splits:
                m = train_and_score(
                    model_name, imgs, labels, tr_global[i_tr], tr_global[i_va],
                    class_ages, config, device, args.seed, args)
                vals.append(m[args.metric])
            mean = float(np.mean(vals))
            scores.append(mean)
            inner_log.append({"outer": o, "config": label(cfg),
                              "inner_mean": mean,
                              "inner_folds": [float(v) for v in vals]})
            print(f"   [inner] {label(cfg):24} {args.metric}={mean:.3f}",
                  flush=True)

        best = int(np.argmax(scores))
        chosen = grid[best]
        print(f"   -> selected {label(chosen)} "
              f"(inner {args.metric}={scores[best]:.3f})", flush=True)

        # --- honest outer score: retrain on the whole outer training set
        size = arch.input_size(model_name, chosen["image_size"])
        config = build_config(model_name, chosen, args)
        metrics = train_and_score(
            model_name, images_by_size[size], labels, tr_global, va_global,
            class_ages, config, device, args.seed, args)
        print(f"   OUTER {args.metric}={metrics[args.metric]:.3f} "
              f"acc={metrics['accuracy']:.3f} f1={metrics['macro_f1']:.3f}\n",
              flush=True)
        outer_rows.append({"outer": o, "chosen": label(chosen), **metrics})

    # --- results
    print("=" * 70)
    print("NESTED CV RESULT (this is the reportable number)")
    print("=" * 70)
    agg = {}
    for key in ("accuracy", "within_one", "qwk", "macro_f1", "mae_years"):
        vals = [r[key] for r in outer_rows]
        agg[key] = (float(np.mean(vals)), float(np.std(vals, ddof=1)))
        print(f"  {key:12} {agg[key][0]:.3f} +/- {agg[key][1]:.3f}")

    picked = [r["chosen"] for r in outer_rows]
    print(f"\n  config chosen per outer fold: {picked}")
    if len(set(picked)) > 1:
        print("  The selection is unstable across outer folds, which means the "
              "grid points are not distinguishable on this much data. Prefer "
              "the anchor configuration unless one wins consistently.")
    else:
        print(f"  Stable: every outer fold selected {picked[0]}.")

    # --- Optional: what plain grid-search-on-the-outer-folds would have said.
    #
    # This must train on the SAME data as the outer scores to be comparable.
    # Inner scores cannot stand in: those models see only (inner-1)/inner of
    # each outer training set, so they score lower for a reason that has
    # nothing to do with selection bias -- comparing against them reports a
    # negative "optimism", which is meaningless.
    naive_rows = None
    if args.also_naive:
        print()
        print("=" * 70)
        print("NAIVE GRID SEARCH (for contrast -- selects and reports on the "
              "same folds)")
        print("=" * 70)
        naive_rows = {}
        for cfg in grid:
            size = arch.input_size(model_name, cfg["image_size"])
            config = build_config(model_name, cfg, args)
            vals = []
            for o_tr, o_va in outer:
                m = train_and_score(
                    model_name, images_by_size[size], labels,
                    dev_idx[o_tr], dev_idx[o_va], class_ages, config, device,
                    args.seed, args)
                vals.append(m[args.metric])
            naive_rows[label(cfg)] = float(np.mean(vals))
            print(f"   {label(cfg):24} {args.metric}={naive_rows[label(cfg)]:.3f}",
                  flush=True)
        naive_best = max(naive_rows, key=naive_rows.get)
        naive_score = naive_rows[naive_best]
        nested_score = agg[args.metric][0]
        print()
        print(f"  naive winner:      {naive_best} -> {naive_score:.3f}")
        print(f"  nested estimate:   {nested_score:.3f}")
        print(f"  optimism:          {naive_score - nested_score:+.3f}")
        print("  The naive number is what tuning-and-reporting on the same "
              "folds would have published. Only the nested number is real.")
    else:
        print()
        print("  (pass --also-naive to measure how much a plain grid search "
              "on these folds would have overstated the result; it costs "
              f"{len(grid) * args.outer_folds} more fold-trainings)")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps({
        "protocol": "nested-cv-tuning",
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "grid": [label(c) for c in grid],
        "select_metric": args.metric,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "n_dev": int(len(dev_idx)),
        "nested": {k: {"mean": v[0], "sd": v[1]} for k, v in agg.items()},
        "chosen_per_fold": picked,
        "outer_rows": outer_rows,
        "inner_log": inner_log,
        "naive_grid": naive_rows,
        "minutes": (time.time() - started) / 60.0,
        "test": None,
    }, indent=2, default=float))
    print(f"\n[out] {args.output / 'results.json'}  "
          f"({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
