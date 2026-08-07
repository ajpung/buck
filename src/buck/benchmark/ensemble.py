"""Build a cross-architecture ensemble from a finished benchmark sweep.

On a 284-image corpus the largest remaining accuracy gain is not a better
single backbone -- it is combining backbones whose errors are uncorrelated. A
ConvNeXt and a ShuffleNet misclassify different deer, and averaging their
probabilities recovers some of both.

Doing that honestly requires choosing the ensemble without looking at the test
set. This module therefore works from **out-of-fold** predictions: for fold *i*
the checkpoint was trained on the other folds, so its predictions on fold *i*
are genuinely unseen. Stacking those across folds yields one clean prediction
per development image, which is enough to choose members and weights.

No retraining is needed. ``StratifiedGroupKFold`` is deterministic given the
same records, groups and seeds, so the fold assignment used during the sweep is
reconstructed exactly and matched against the saved ``fold{i}.pth`` files.

Usage::

    python -m buck.benchmark.ensemble --run benchmark_runs/20260806_120000
    python -m buck.benchmark.ensemble --run <dir> --score-test   # spends ONE read
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from buck.benchmark import architectures as arch
from buck.benchmark.data import (
    EvalDataset,
    build_groups,
    decode_images,
    drop_rare_classes,
    load_or_create_holdout,
    load_records,
)
from buck.benchmark.metrics import bootstrap_ci, confusion, ordinal_metrics


def rebuild_splits(records, groups, class_ages, manifest, folds, split_seed,
                   test_fraction):
    """Reproduce the sweep's dev/test split and inner folds exactly."""
    labels = np.array([class_ages.index(r.age) for r in records])
    test_mask = load_or_create_holdout(
        records, groups, manifest, test_fraction, split_seed
    )
    dev_idx = np.flatnonzero(~test_mask)
    test_idx = np.flatnonzero(test_mask)

    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=split_seed
    )
    fold_list = list(
        splitter.split(np.zeros(len(dev_idx)), labels[dev_idx], groups[dev_idx])
    )
    return labels, dev_idx, test_idx, fold_list


@torch.no_grad()
def _probabilities(model, images, labels, batch_size, device, tta):
    loader = DataLoader(
        EvalDataset(images, labels), batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    use_amp = device.type == "cuda"
    out = []
    model.eval()
    for batch, _ in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch)
            if tta:
                logits = (logits + model(torch.flip(batch, dims=[3]))) / 2
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def collect_predictions(run_dir, model_name, records, labels, dev_idx, test_idx,
                        fold_list, class_ages, device, image_size=None,
                        tta=False, batch_size=32):
    """Return (oof_probs over dev, mean test probs) for one architecture.

    ``oof_probs[j]`` is produced by the only fold model that never saw
    development image ``dev_idx[j]`` during training.
    """
    ckpt_dir = Path(run_dir) / "checkpoints" / model_name
    checkpoints = sorted(ckpt_dir.glob("fold*.pth"))
    if len(checkpoints) != len(fold_list):
        raise RuntimeError(
            f"{model_name}: found {len(checkpoints)} checkpoints but the split "
            f"has {len(fold_list)} folds. The run's --folds must match --folds "
            f"here, or out-of-fold predictions would be mismatched."
        )

    size = arch.input_size(model_name, image_size)
    images = decode_images(records, size)

    oof = np.zeros((len(dev_idx), len(class_ages)), dtype=np.float64)
    test_accum = np.zeros((len(test_idx), len(class_ages)), dtype=np.float64)

    for (_, va), ckpt in zip(fold_list, checkpoints):
        model = arch.build_model(model_name, len(class_ages), pretrained=False)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["model_state_dict"])
        model.to(device)

        val_global = dev_idx[va]
        oof[va] = _probabilities(
            model, images[val_global], labels[val_global], batch_size, device, tta
        )
        test_accum += _probabilities(
            model, images[test_idx], labels[test_idx], batch_size, device, tta
        )

        del model
        torch.cuda.empty_cache()

    del images
    return oof, test_accum / len(checkpoints)


def greedy_selection(oof_by_model, y_dev, class_ages, metric="qwk",
                     rounds=25, verbose=True):
    """Caruana-style forward selection with replacement.

    Repeatedly adds whichever architecture most improves the out-of-fold score
    of the running average. Allowing replacement lets a strong model take more
    weight, and the whole search touches development data only.

    Returns:
        (weights dict, best score, history list)
    """
    names = list(oof_by_model)
    counts = {n: 0 for n in names}
    total = np.zeros_like(next(iter(oof_by_model.values())))
    chosen, history, best_overall = 0, [], -np.inf
    best_counts = dict(counts)

    for step in range(1, rounds + 1):
        scores = {}
        for name in names:
            candidate = (total + oof_by_model[name]) / (chosen + 1)
            scores[name] = ordinal_metrics(
                y_dev, candidate.argmax(1), class_ages
            )[metric]

        winner = max(scores, key=scores.get)
        total = total + oof_by_model[winner]
        chosen += 1
        counts[winner] += 1
        score = scores[winner]
        history.append({"step": step, "added": winner, metric: score})

        if score > best_overall:
            best_overall = score
            best_counts = dict(counts)

        if verbose:
            print(f"   step {step:2d}: +{winner:<22} oof {metric} {score:.4f}")

    weights = {n: c / sum(best_counts.values())
               for n, c in best_counts.items() if c > 0}
    return weights, best_overall, history


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build a cross-architecture ensemble from a sweep, using "
                    "out-of-fold predictions so the test set stays untouched.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", type=Path, required=True,
                   help="benchmark run directory containing checkpoints/")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset to consider (default: every checkpointed model)")
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--sources", nargs="+", default=["NDA"])
    p.add_argument("--channels", nargs="+", default=["color", "grayscale"])
    p.add_argument("--folds", type=int, default=None,
                   help="must match the sweep; read from results.json if present")
    p.add_argument("--metric", choices=["qwk", "accuracy", "within_one", "macro_f1"],
                   default="qwk")
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--tta", action="store_true",
                   help="deterministic horizontal-flip averaging")
    p.add_argument("--score-test", action="store_true",
                   help="evaluate the chosen ensemble on the locked test set. "
                        "This spends one read; do it once, at the end.")
    p.add_argument("--min-class-count", type=int, default=8)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)

    # Recover the sweep's configuration so the split is reproduced exactly.
    results_path = args.run / "results.json"
    config = {}
    if results_path.exists():
        config = json.loads(results_path.read_text()).get("config", {})
    folds = args.folds or int(config.get("folds", 5))
    image_root = args.image_root or Path(config["image_root"])
    manifest = args.manifest or Path(config["manifest"])
    image_size = config.get("image_size")
    split_seed = args.split_seed or int(config.get("split_seed", 1337))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = drop_rare_classes(
        load_records(image_root, tuple(args.sources), tuple(args.channels)),
        args.min_class_count,
    )
    class_ages = sorted({r.age for r in records})
    groups = build_groups(records, verbose=False)

    labels, dev_idx, test_idx, fold_list = rebuild_splits(
        records, groups, class_ages, manifest, folds, split_seed,
        args.test_fraction,
    )
    y_dev = labels[dev_idx]

    available = sorted(d.name for d in (args.run / "checkpoints").iterdir()
                       if d.is_dir())
    models = args.models or available
    missing = [m for m in models if m not in available]
    if missing:
        raise SystemExit(f"no checkpoints for {missing}; available: {available}")

    print(f"[ens] {len(models)} architectures, {len(dev_idx)} dev images, "
          f"{folds} folds")
    print("[ens] generating out-of-fold predictions (no test data touched)")

    oof_by_model, test_by_model, singles = {}, {}, []
    for name in models:
        oof, test_probs = collect_predictions(
            args.run, name, records, labels, dev_idx, test_idx, fold_list,
            class_ages, device, image_size, args.tta, args.batch_size,
        )
        oof_by_model[name] = oof
        test_by_model[name] = test_probs
        m = ordinal_metrics(y_dev, oof.argmax(1), class_ages)
        singles.append({"model": name, **m})
        print(f"   {name:<22} oof acc {m['accuracy']:.3f}  qwk {m['qwk']:.3f}")

    singles.sort(key=lambda r: r[args.metric], reverse=True)
    best_single = singles[0]

    print(f"\n[ens] greedy forward selection on out-of-fold {args.metric}")
    weights, best_oof, history = greedy_selection(
        oof_by_model, y_dev, class_ages, args.metric, args.rounds
    )

    blend = sum(w * oof_by_model[n] for n, w in weights.items())
    ens_oof = ordinal_metrics(y_dev, blend.argmax(1), class_ages)

    print(f"\n{'=' * 72}")
    print("SELECTED ENSEMBLE")
    print(f"{'=' * 72}")
    for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"   {weight:6.3f}  {name}")
    print(
        f"\n   out-of-fold: acc {ens_oof['accuracy']:.3f}  "
        f"+/-1yr {ens_oof['within_one']:.3f}  qwk {ens_oof['qwk']:.3f}  "
        f"MAE {ens_oof['mae_years']:.2f}yr"
    )
    print(
        f"   best single ({best_single['model']}): "
        f"acc {best_single['accuracy']:.3f}  qwk {best_single['qwk']:.3f}"
    )
    gain = ens_oof[args.metric] - best_single[args.metric]
    print(f"   ensemble gain on {args.metric}: {gain:+.4f}")

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "run": str(args.run),
        "metric": args.metric,
        "tta": args.tta,
        "weights": weights,
        "oof_ensemble": ens_oof,
        "oof_singles": singles,
        "selection_history": history,
    }

    if args.score_test:
        print(f"\n{'=' * 72}")
        print("LOCKED TEST SET  (one read)")
        print(f"{'=' * 72}")
        blend_test = sum(w * test_by_model[n] for n, w in weights.items())
        y_test = labels[test_idx]
        y_pred = blend_test.argmax(1)
        m = ordinal_metrics(y_test, y_pred, class_ages)
        point, lo, hi = bootstrap_ci(y_test, y_pred, class_ages, "accuracy")
        print(f"   accuracy   {m['accuracy']:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"   within 1yr {m['within_one']:.3f}")
        print(f"   QWK        {m['qwk']:.3f}")
        print(f"   MAE        {m['mae_years']:.2f} years")
        payload["test"] = {**m, "accuracy_ci95": [lo, hi],
                           "confusion": confusion(y_test, y_pred, class_ages)}
    else:
        print("\n[ens] test set untouched (pass --score-test when you have "
              "committed to this ensemble)")

    out = args.run / "ensemble.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
