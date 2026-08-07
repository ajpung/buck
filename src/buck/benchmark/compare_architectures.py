"""Compare transfer-learning architectures for buck age estimation.

Replaces the sweep in ``trail cam/examples/251008 - all image.ipynb``, which
overstated its own accuracy in three ways: it flipped test images at random,
it oversampled the test set to an artificial class balance, and it selected
checkpoints using a score containing test accuracy. This module fixes all
three and adds a prospective backtest that matches how the model is actually
used week to week.

Protocols
---------
``holdout`` (default)
    A test set is carved once, written to a manifest, and frozen. Architectures
    are ranked by cross-validation on the remaining development pool. Only the
    winner touches the test set, and only once.

``temporal``
    Rolling-origin backtest. For each recent week, train on every datapoint
    collected strictly earlier and predict that week's deer. This is leak-free
    by construction and is the honest answer to "is my model slipping?", since
    it reproduces the weekly workflow exactly.

Examples
--------
    python -m buck.benchmark.compare_architectures --quick
    python -m buck.benchmark.compare_architectures --models all --folds 5
    python -m buck.benchmark.compare_architectures --protocol temporal \
        --models resnet50 --weeks 30
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from buck.benchmark import architectures as arch
from buck.benchmark.data import (
    EvalDataset,
    TrainDataset,
    assert_no_leakage,
    build_groups,
    decode_images,
    drop_rare_classes,
    load_or_create_holdout,
    load_records,
)
from buck.benchmark.deployment import (
    pareto_front,
    print_cost_table,
    profile_all,
)
from buck.benchmark.metrics import bootstrap_ci, confusion, ordinal_metrics

DEFAULT_IMAGE_ROOT = Path(__file__).resolve().parents[3] / "trail cam" / "images" / "squared"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "trail cam" / "splits" / "holdout_test_v1.json"

TRAIN_DEFAULTS = dict(
    backbone_lr=1e-4,
    classifier_lr=5e-4,
    weight_decay=0.05,
    label_smoothing=0.1,
    dropout=0.3,
    max_epochs=60,
    patience=15,
    augmentation="medium",
    train_multiplier=8,
    pretrained=True,
)

# Training from scratch at the fine-tuning backbone LR would rig the comparison:
# 1e-4 is chosen to *avoid* disturbing transferred features, which is exactly
# the wrong rate for weights that start as noise. Raise it with the backbone
# unfrozen so the no-pretraining arm gets a fair shot.
SCRATCH_BACKBONE_LR = 1e-3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Training and evaluation
# --------------------------------------------------------------------------


def _batch_size(model_name, args):
    """Registry batch is a VRAM ceiling; cap it so a tiny fold still yields
    a useful number of gradient steps per epoch."""
    ceiling = int(arch.REGISTRY[model_name]["batch"] * args.batch_scale)
    return max(2, min(ceiling, args.max_batch))


@torch.no_grad()
def predict(model, loader, device, use_amp, tta=False):
    """Return (labels, predictions) for a loader. Deterministic.

    ``tta`` averages the logits of the image and its mirror. It is off by
    default; when on it is applied identically to validation and test, and it
    is deterministic -- unlike the random flip it replaces.
    """
    model.eval()
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            if tta:
                logits = (logits + model(torch.flip(images, dims=[3]))) / 2
        all_preds.append(logits.float().argmax(1).cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def train_fold(
    model_name,
    train_images,
    train_labels,
    val_images,
    val_labels,
    class_ages,
    config,
    device,
    seed,
    select_metric="qwk",
    tta=False,
    verbose=True,
):
    """Train one fold. Returns (best_state_dict, best_val_metrics, epochs_run).

    The validation set is the only signal used for early stopping and
    checkpoint selection. No test data is visible from inside this function --
    it is not passed in, so it cannot leak.
    """
    set_seed(seed)
    num_classes = len(class_ages)

    train_ds = TrainDataset(
        train_images, train_labels, config["augmentation"], seed=seed
    )
    val_ds = EvalDataset(val_images, val_labels)

    # Balance classes by sampling rather than by inflating the dataset, and
    # draw several augmented views of each image per epoch. With ~180 training
    # images a one-view epoch would be two gradient steps, which never
    # converges. Redrawing is train-side only and each view is freshly
    # augmented, so it adds optimisation steps without adding information.
    weights = train_ds.class_weights()
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_ds) * config["train_multiplier"],
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )

    batch_size = config["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = arch.build_model(
        model_name, num_classes, config["dropout"],
        pretrained=config["pretrained"],
    ).to(device)

    backbone, head = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head if any(k in name for k in ("classifier", "fc", "head")) else backbone).append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone, "lr": config["backbone_lr"]},
            {"params": head, "lr": config["classifier_lr"]},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"], eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_state, best_metrics, stale = -np.inf, None, None, 0

    for epoch in range(config["max_epochs"]):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        y_true, y_pred = predict(model, val_loader, device, use_amp, tta)
        metrics = ordinal_metrics(y_true, y_pred, class_ages)
        scheduler.step()

        if metrics[select_metric] > best_score:
            best_score = metrics[select_metric]
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if verbose and epoch % 10 == 0:
            print(
                f"      epoch {epoch:3d}  val acc {metrics['accuracy']:.3f}  "
                f"qwk {metrics['qwk']:.3f}"
            )

        if stale >= config["patience"]:
            if verbose:
                print(f"      early stop at epoch {epoch}")
            break

    del model
    torch.cuda.empty_cache()
    return best_state, best_metrics, epoch + 1


# --------------------------------------------------------------------------
# Protocol: locked holdout + cross-validated ranking
# --------------------------------------------------------------------------


def run_holdout(args, records, groups, class_ages, device):
    labels = np.array([class_ages.index(r.age) for r in records])

    test_mask = load_or_create_holdout(
        records, groups, args.manifest, args.test_fraction, args.split_seed
    )
    dev_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    print(
        f"[split] development pool {len(dev_idx)} images | "
        f"locked test {len(test_idx)} images"
    )
    print(f"        test class mix: {dict(Counter(labels[test_idx].tolist()))}")

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.split_seed
    )
    folds = list(
        splitter.split(np.zeros(len(dev_idx)), labels[dev_idx], groups[dev_idx])
    )

    results = []
    for model_name in args.models:
        size = arch.input_size(model_name, args.image_size)
        print(f"\n{'=' * 70}\n{model_name}  ({size}px)\n{'=' * 70}")

        try:
            images = decode_images(records, size)
        except RuntimeError as exc:
            raise RuntimeError(f"decoding failed for {model_name}: {exc}") from exc

        config = dict(TRAIN_DEFAULTS)
        config["max_epochs"] = args.epochs
        config["patience"] = args.patience
        config["augmentation"] = args.augmentation
        config["batch_size"] = _batch_size(model_name, args)
        config["train_multiplier"] = args.train_multiplier
        config["pretrained"] = not args.no_pretrained
        if args.backbone_lr is not None:
            config["backbone_lr"] = args.backbone_lr
        elif args.no_pretrained:
            config["backbone_lr"] = SCRATCH_BACKBONE_LR
        if args.no_pretrained:
            print(f"   [scratch] random init, backbone lr {config['backbone_lr']:.0e}")

        fold_metrics, fold_states = [], []
        started = time.time()

        for fold, (tr, va) in enumerate(folds, start=1):
            train_idx, val_idx = dev_idx[tr], dev_idx[va]
            assert_no_leakage(train_idx, val_idx, test_idx, groups, records)

            print(f"   fold {fold}/{args.folds}  train {len(train_idx)} / val {len(val_idx)}")
            state, metrics, epochs = train_fold(
                model_name,
                images[train_idx], labels[train_idx],
                images[val_idx], labels[val_idx],
                class_ages, config, device,
                seed=args.seed + fold,
                select_metric=args.select_metric,
                tta=args.tta,
                verbose=args.verbose,
            )
            print(
                f"      best val: acc {metrics['accuracy']:.3f}  "
                f"+/-1yr {metrics['within_one']:.3f}  qwk {metrics['qwk']:.3f}  "
                f"MAE {metrics['mae_years']:.2f}yr  ({epochs} epochs)"
            )
            fold_metrics.append(metrics)
            fold_states.append(state)

        summary = {
            "model": model_name,
            "input_size": size,
            "minutes": (time.time() - started) / 60,
            "folds": len(fold_metrics),
        }
        for key in ("accuracy", "within_one", "qwk", "mae_years", "macro_f1"):
            values = [m[key] for m in fold_metrics]
            summary[f"cv_{key}"] = float(np.mean(values))
            summary[f"cv_{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        results.append(summary)
        print(
            f"   CV mean: acc {summary['cv_accuracy']:.3f} "
            f"+/- {summary['cv_accuracy_sd']:.3f} | qwk {summary['cv_qwk']:.3f} | "
            f"{summary['minutes']:.1f} min"
        )

        # Keep fold weights so the winner can be scored on test without
        # retraining. Written per-model to bound memory.
        ckpt_dir = args.output / "checkpoints" / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for i, state in enumerate(fold_states, start=1):
            torch.save({"model_state_dict": state, "model_name": model_name,
                        "input_size": size, "fold": i}, ckpt_dir / f"fold{i}.pth")

        del images, fold_states
        torch.cuda.empty_cache()

    if not args.no_profile:
        print("\n[cost] measuring deployment cost of each architecture")
        costs = {c["model"]: c for c in profile_all(
            [r["model"] for r in results], len(class_ages), args.image_size,
            with_onnx=not args.no_onnx,
        )}
        for r in results:
            r.update({k: v for k, v in costs[r["model"]].items()
                      if k not in ("model", "input_size")})

    results.sort(key=lambda r: r[f"cv_{args.select_metric}"], reverse=True)
    _print_leaderboard(results, args.select_metric)
    if not args.no_profile:
        _print_tradeoff(results, args.select_metric)

    test_report = _score_on_test(
        args, results, records, labels, test_idx, class_ages, device
    )

    payload = {
        "protocol": "holdout",
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": _serialisable_args(args),
        "class_ages": class_ages,
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "leaderboard": results,
        "test": test_report,
    }
    _write(args.output / "results.json", payload)
    return payload


def _score_on_test(args, results, records, labels, test_idx, class_ages, device):
    """Evaluate on the locked test set, once.

    Under the default ``winner-only`` policy exactly one architecture -- the
    one cross-validation chose -- is scored. Scoring all of them and then
    quoting the maximum is the same multiple-comparison error that produced the
    old inflated numbers, so ``--test-policy all`` prints a warning.
    """
    if args.test_policy == "none":
        print("\n[test] skipped (--test-policy none); locked test set untouched")
        return None

    to_score = results if args.test_policy == "all" else results[:1]
    if args.test_policy == "all":
        print(
            "\n[test] WARNING: scoring every architecture on the locked test "
            "set. Use the CV column to choose a model. Quoting the best test "
            "number across models re-introduces selection bias."
        )

    use_amp = device.type == "cuda"
    report = []

    for entry in to_score:
        model_name = entry["model"]
        size = entry["input_size"]
        images = decode_images(records, size)
        loader = DataLoader(
            EvalDataset(images[test_idx], labels[test_idx]),
            batch_size=_batch_size(model_name, args),
            shuffle=False,
            num_workers=0,
        )

        # Average the fold models' probabilities: this is the ensemble the
        # project already deploys, and it is what the CV score estimated.
        ckpt_dir = args.output / "checkpoints" / model_name
        prob_sum, y_true = None, None
        for ckpt in sorted(ckpt_dir.glob("fold*.pth")):
            # Weights are overwritten by the checkpoint, so skip the download.
            model = arch.build_model(
                model_name, len(class_ages), TRAIN_DEFAULTS["dropout"],
                pretrained=False,
            )
            model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
            model.to(device).eval()

            probs, labels_seen = [], []
            with torch.no_grad():
                for batch, batch_labels in loader:
                    batch = batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(batch)
                        if args.tta:
                            logits = (logits + model(torch.flip(batch, dims=[3]))) / 2
                    probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                    labels_seen.append(batch_labels.numpy())

            probs = np.concatenate(probs)
            prob_sum = probs if prob_sum is None else prob_sum + probs
            y_true = np.concatenate(labels_seen)
            del model
            torch.cuda.empty_cache()

        y_pred = prob_sum.argmax(1)
        metrics = ordinal_metrics(y_true, y_pred, class_ages)
        point, lo, hi = bootstrap_ci(y_true, y_pred, class_ages, "accuracy", seed=args.seed)

        entry_report = {
            "model": model_name,
            **metrics,
            "accuracy_ci95": [lo, hi],
            "confusion": confusion(y_true, y_pred, class_ages),
        }
        report.append(entry_report)

        print(
            f"\n[test] {model_name} on {metrics['n']} held-out images "
            f"(ensemble of {args.folds} folds, no augmentation):"
        )
        print(f"       accuracy   {metrics['accuracy']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"       within 1yr {metrics['within_one']:.3f}")
        print(f"       QWK        {metrics['qwk']:.3f}")
        print(f"       MAE        {metrics['mae_years']:.2f} years")

        del images
        torch.cuda.empty_cache()

    return report


# --------------------------------------------------------------------------
# Protocol: rolling-origin temporal backtest
# --------------------------------------------------------------------------


def run_temporal(args, records, groups, class_ages, device):
    """Prospective backtest that mirrors the weekly workflow.

    For each of the most recent ``--weeks`` collection dates, train on every
    image collected strictly earlier and predict that date's images. Nothing
    from the future is ever visible, so no held-out manifest is needed: the
    time ordering is the wall.
    """
    dated = [(r.collected_date, i) for i, r in enumerate(records) if r.collected_date]
    if len(dated) < len(records):
        print(
            f"[temporal] {len(records) - len(dated)} image(s) have an unparseable "
            f"collection date and are excluded from the backtest"
        )
    dated.sort()

    weeks = sorted({d for d, _ in dated})
    targets = weeks[-args.weeks:]
    labels = np.array([class_ages.index(r.age) for r in records])

    if len(args.models) != 1:
        print(
            f"[temporal] backtesting {len(args.models)} architectures over "
            f"{len(targets)} weeks retrains {len(args.models) * len(targets)} "
            f"models; consider --models <one>"
        )

    payload_models = []
    for model_name in args.models:
        size = arch.input_size(model_name, args.image_size)
        images = decode_images(records, size)

        config = dict(TRAIN_DEFAULTS)
        config.update(
            max_epochs=args.epochs,
            patience=args.patience,
            augmentation=args.augmentation,
            batch_size=_batch_size(model_name, args),
            train_multiplier=args.train_multiplier,
            pretrained=not args.no_pretrained,
        )
        if args.backbone_lr is not None:
            config["backbone_lr"] = args.backbone_lr
        elif args.no_pretrained:
            config["backbone_lr"] = SCRATCH_BACKBONE_LR

        print(f"\n{'=' * 70}\n{model_name} rolling backtest over {len(targets)} weeks\n{'=' * 70}")
        rows = []

        for week in targets:
            eval_idx = np.array([i for d, i in dated if d == week])
            hist_idx = np.array([i for d, i in dated if d < week])

            if len(hist_idx) < args.min_history:
                print(f"   {week}: only {len(hist_idx)} prior images, skipped")
                continue
            if len(np.unique(labels[hist_idx])) < len(class_ages):
                print(f"   {week}: history missing a class, skipped")
                continue

            # Carve a small validation slice from history for early stopping.
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(hist_idx))
            n_val = max(len(class_ages), int(0.15 * len(hist_idx)))
            val_idx, train_idx = hist_idx[order[:n_val]], hist_idx[order[n_val:]]

            assert_no_leakage(train_idx, val_idx, eval_idx, groups, records)

            state, _, _ = train_fold(
                model_name,
                images[train_idx], labels[train_idx],
                images[val_idx], labels[val_idx],
                class_ages, config, device,
                seed=args.seed,
                select_metric=args.select_metric,
                tta=args.tta,
                verbose=False,
            )

            model = arch.build_model(
                model_name, len(class_ages), config["dropout"], pretrained=False,
            )
            model.load_state_dict(state)
            model.to(device)
            loader = DataLoader(
                EvalDataset(images[eval_idx], labels[eval_idx]),
                batch_size=config["batch_size"], shuffle=False, num_workers=0,
            )
            y_true, y_pred = predict(model, loader, device, device.type == "cuda", args.tta)
            del model
            torch.cuda.empty_cache()

            for idx, truth, pred in zip(eval_idx, y_true, y_pred):
                rows.append({
                    "week": week.isoformat(),
                    "file": records[idx].filename,
                    "train_n": int(len(train_idx)),
                    "true_age": class_ages[truth],
                    "pred_age": class_ages[pred],
                    "correct": bool(truth == pred),
                    "abs_error_years": abs(class_ages[truth] - class_ages[pred]),
                })
            hit = "OK " if y_true[0] == y_pred[0] else "MISS"
            print(
                f"   {week}  n_train={len(train_idx):3d}  {hit}  "
                f"truth {class_ages[y_true[0]]} -> pred {class_ages[y_pred[0]]}"
            )

        if rows:
            truth = np.array([class_ages.index(r["true_age"]) for r in rows])
            pred = np.array([class_ages.index(r["pred_age"]) for r in rows])
            metrics = ordinal_metrics(truth, pred, class_ages)
            point, lo, hi = bootstrap_ci(truth, pred, class_ages, "accuracy", seed=args.seed)

            print(f"\n[temporal] {model_name} over {len(rows)} prospective predictions:")
            print(f"           accuracy   {metrics['accuracy']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
            print(f"           within 1yr {metrics['within_one']:.3f}")
            print(f"           MAE        {metrics['mae_years']:.2f} years")

            half = len(rows) // 2
            if half >= 4:
                early = ordinal_metrics(truth[:half], pred[:half], class_ages)
                late = ordinal_metrics(truth[half:], pred[half:], class_ages)
                print(
                    f"           drift check: first half {early['accuracy']:.3f} -> "
                    f"second half {late['accuracy']:.3f}"
                )

            payload_models.append({
                "model": model_name, **metrics,
                "accuracy_ci95": [lo, hi], "per_week": rows,
            })

        del images
        torch.cuda.empty_cache()

    payload = {
        "protocol": "temporal",
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": _serialisable_args(args),
        "class_ages": class_ages,
        "models": payload_models,
    }
    _write(args.output / "results_temporal.json", payload)
    return payload


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _print_tradeoff(results, select_metric):
    """Show accuracy against serving cost, and the Pareto front.

    The front is the shortlist worth deciding between: every model omitted from
    it is beaten outright by something both more accurate and smaller.
    """
    quality = f"cv_{select_metric}"
    size_key = "onnx_mb" if any(r.get("onnx_mb") for r in results) else "fp32_mb"

    print(f"\n{'=' * 84}")
    print(f"ACCURACY vs DEPLOYMENT COST   (quality = {quality}, size = {size_key})")
    print(f"{'=' * 84}")
    print(f"{'model':<22} {select_metric:>8} {'acc':>8} {'MB':>8} "
          f"{'CPU ms':>8} {select_metric + '/MB':>10}")
    print("-" * 84)
    for r in sorted(results, key=lambda r: r.get(size_key) or 0):
        size = r.get(size_key)
        per_mb = (r[quality] / size) if size else float("nan")
        print(
            f"{r['model']:<22} {r[quality]:>8.3f} {r['cv_accuracy']:>8.3f} "
            f"{(size if size else 0):>8.1f} {r['cpu_ms']:>8.1f} {per_mb:>10.4f}"
        )
    print("-" * 84)

    front = pareto_front(results, quality, size_key)
    print("\nPareto front (nothing is both better and smaller):")
    for r in front:
        print(
            f"   {r['model']:<22} {select_metric} {r[quality]:.3f}  "
            f"{r[size_key]:.1f} MB  {r['cpu_ms']:.0f} ms"
        )
    beaten = [r["model"] for r in results if r not in front]
    if beaten:
        print(f"\nDominated (do not ship these): {', '.join(beaten)}")


def _print_leaderboard(results, select_metric):
    print(f"\n{'=' * 96}")
    print("CROSS-VALIDATED LEADERBOARD  (development pool only; test set untouched)")
    print(f"ranked by {select_metric}")
    print(f"{'=' * 96}")
    header = (
        f"{'model':<22} {'px':>5} {'acc':>14} {'+/-1yr':>8} {'QWK':>8} "
        f"{'MAE yr':>8} {'min':>7}"
    )
    print(header)
    print("-" * 96)
    for r in results:
        print(
            f"{r['model']:<22} {r['input_size']:>5} "
            f"{r['cv_accuracy']:>7.3f}+/-{r['cv_accuracy_sd']:<5.3f} "
            f"{r['cv_within_one']:>8.3f} {r['cv_qwk']:>8.3f} "
            f"{r['cv_mae_years']:>8.2f} {r['minutes']:>7.1f}"
        )
    print("-" * 96)


def _serialisable_args(args):
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[out] {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Leak-free architecture comparison for BUCK age estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                   help="the 'squared' image tree; images/original is never read")
    p.add_argument("--sources", nargs="+", default=["NDA"],
                   help="labelling institutions to accept")
    p.add_argument("--channels", nargs="+", default=["color", "grayscale"])
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help="locked test-set manifest; created on first run, then frozen")
    p.add_argument("--output", type=Path,
                   default=Path("benchmark_runs") / datetime.now().strftime("%Y%m%d_%H%M%S"))

    p.add_argument("--protocol", choices=["holdout", "temporal"], default="holdout")
    p.add_argument("--models", nargs="+", default=None,
                   help="architecture names, 'all', or 'suite' (default suite)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--weeks", type=int, default=25,
                   help="temporal protocol: how many recent weeks to backtest")
    p.add_argument("--min-history", type=int, default=60,
                   help="temporal protocol: minimum prior images before a week is scored")

    p.add_argument("--epochs", type=int, default=TRAIN_DEFAULTS["max_epochs"])
    p.add_argument("--patience", type=int, default=TRAIN_DEFAULTS["patience"])
    p.add_argument("--augmentation", choices=["light", "medium", "heavy"], default="medium")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random init instead of ImageNet weights, and train the "
                        "whole backbone. Raises the backbone LR to "
                        f"{SCRATCH_BACKBONE_LR:.0e} unless --backbone-lr is given, "
                        "so the comparison is not rigged by a fine-tuning rate.")
    p.add_argument("--backbone-lr", type=float, default=None,
                   help=f"override the backbone learning rate (default "
                        f"{TRAIN_DEFAULTS['backbone_lr']:.0e} pretrained, "
                        f"{SCRATCH_BACKBONE_LR:.0e} scratch)")
    p.add_argument("--image-size", type=int, default=None,
                   help="override input size where the architecture allows it")
    p.add_argument("--batch-scale", type=float, default=1.0,
                   help="scale the per-model batch sizes for your VRAM")
    p.add_argument("--max-batch", type=int, default=32,
                   help="upper bound on batch size; small batches give a tiny "
                        "dataset more gradient steps per epoch")
    p.add_argument("--train-multiplier", type=int, default=8,
                   help="augmented views drawn per training image per epoch")
    p.add_argument("--select-metric", choices=["qwk", "accuracy", "within_one", "macro_f1"],
                   default="qwk",
                   help="validation metric for early stopping and ranking")
    p.add_argument("--tta", action="store_true",
                   help="deterministic horizontal-flip averaging at eval time")
    p.add_argument("--test-policy", choices=["winner-only", "all", "none"],
                   default="winner-only",
                   help="how many architectures may touch the locked test set")

    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--min-class-count", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--profile-only", action="store_true",
                   help="measure size and latency of the selected architectures "
                        "and exit; no training, no data needed beyond class count")
    p.add_argument("--no-profile", action="store_true",
                   help="skip deployment-cost measurement after training")
    p.add_argument("--no-onnx", action="store_true",
                   help="skip ONNX export sizing (fp32/int8 sizes still measured)")
    p.add_argument("--quick", action="store_true",
                   help="3 folds, 20 epochs, 4 architectures; for smoke-testing")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    if args.quick:
        args.folds, args.epochs, args.patience = 3, 20, 8
        args.models = args.models or ["resnet18", "efficientnet_b0", "convnext_tiny", "swin_t"]

    if args.models in (None, ["suite"]):
        args.models = list(arch.DEFAULT_SUITE)
    elif args.models == ["all"]:
        args.models = list(arch.REGISTRY)
    elif args.models == ["efficient"]:
        args.models = list(arch.EFFICIENT_SUITE)

    unknown = [m for m in args.models if m not in arch.REGISTRY]
    if unknown:
        p.error(f"unknown architecture(s): {unknown}\nknown: {sorted(arch.REGISTRY)}")

    return args


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[env] {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    else:
        print("[env] running on CPU; this will be slow")

    records = load_records(args.image_root, tuple(args.sources), tuple(args.channels))
    if not records:
        raise SystemExit(f"no images matched under {args.image_root}")
    records = drop_rare_classes(records, args.min_class_count)

    class_ages = sorted({r.age for r in records})
    counts = Counter(r.age for r in records)
    print(f"[data] {len(records)} images, {len(class_ages)} classes")
    print(f"[data] class mix: {dict(sorted(counts.items()))}")
    print(f"[data] channels: {dict(Counter(r.channel for r in records))}")

    if args.profile_only:
        records_cost = profile_all(
            args.models, len(class_ages), args.image_size,
            with_onnx=not args.no_onnx,
        )
        print_cost_table(records_cost)
        args.output.mkdir(parents=True, exist_ok=True)
        _write(args.output / "deployment_cost.json",
               {"created": datetime.now().isoformat(timespec="seconds"),
                "num_classes": len(class_ages), "models": records_cost})
        return

    groups = build_groups(records)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.protocol == "holdout":
        run_holdout(args, records, groups, class_ages, device)
    else:
        run_temporal(args, records, groups, class_ages, device)


if __name__ == "__main__":
    main()