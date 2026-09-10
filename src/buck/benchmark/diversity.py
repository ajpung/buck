"""Do these models make the same mistakes? Ensembling only pays if they do not.

A blend of several architectures can only recover images that some members get
right and others get wrong. On the 12-model suite that condition was barely
met -- 18% mean pairwise disagreement on errors, 26 development images that
every architecture missed -- and a uniform blend bought +0.009 accuracy.

That test had a limitation worth revisiting: all 12 members were
ImageNet-supervised torchvision models, one pretraining regime. The current
field spans masked autoencoding, image-text contrastive, self-distillation and
21k supervision, which is the first real chance for decorrelated errors. This
module measures whether that materialised, before anyone builds an ensemble on
the assumption that it did.

It also separates two things the old analysis conflated:

``uniform``
    Average every member. Chooses nothing from the data, so its
    out-of-fold score is honest.

``rule-based``
    Average one pre-specified member per pretraining family. The rule is fixed
    in advance, so this is honest too.

``greedy``
    Caruana forward selection. Reported here **only** to quantify its own
    optimism -- it picks members on the same out-of-fold predictions it then
    scores, the identical defect as best-epoch checkpointing. The gap between
    greedy and uniform is the selection bias, not a gain.

Usage::

    python -m buck.benchmark.diversity --run benchmark_runs/mega_a \\
        --log <path to the run's stdout log>
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from buck.benchmark import architectures as arch
from buck.benchmark.data import (EvalDataset, build_groups, decode_images,
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
# Corpus pinning
# --------------------------------------------------------------------------
def pin_records(image_root, run_log, extra_exclude=(), min_class_count=8):
    """Rebuild the record set the run actually saw.

    The corpus moves under a long sweep. The weekly add gives last week's
    ``xpx`` deer its age, which drops a new image into the development pool;
    ``StratifiedGroupKFold`` over 231 images bears no relation to the folds
    over 230, so every checkpoint gets scored on a fold containing its own
    training data and the leaderboard comes back at ~0.96.

    File mtime cannot detect this -- renaming preserves it -- so the pin is
    taken from the run's own log, which lists the filenames it skipped as
    unaged. Those images are excluded by their ``collected_photo_state``
    prefix, which survives the rename that gives them an age.
    """
    skipped = []
    if run_log is not None:
        text = Path(run_log).read_text(encoding="utf-8", errors="replace")
        skipped = re.findall(r"(\S+_xpx_\S+\.png): unusable age field", text)
        expected = re.search(r"^\[data\] (\d+) images,", text, re.M)
        expected = int(expected.group(1)) if expected else None
    else:
        expected = None

    # "260903_260901_WI_xpx_NDA.png" -> "260903_260901_WI_"
    prefixes = {"_".join(name.split("_")[:3]) + "_" for name in skipped}
    excluded = []

    records = []
    for rec in drop_rare_classes(load_records(image_root), min_class_count):
        name = Path(rec.path).name
        if name in extra_exclude or any(name.startswith(p) for p in prefixes):
            excluded.append(name)
            continue
        records.append(rec)

    print(f"[pin] {len(records)} records "
          f"({len(excluded)} excluded as added after the run began)")
    for name in excluded:
        print(f"        {name}")
    if expected is not None and len(records) != expected:
        raise SystemExit(
            f"[pin] FAILED: reconstructed {len(records)} records but the run "
            f"saw {expected}. The corpus has drifted in a way this cannot "
            f"infer -- pass the extra filenames with --exclude. Refusing to "
            f"continue, because mismatched folds silently report ~0.96."
        )
    if expected is not None:
        print(f"[pin] verified against the run log: {expected} records")
    return records


# --------------------------------------------------------------------------
# Pretraining regime, for the rule-based blend
# --------------------------------------------------------------------------
def pretraining_family(name):
    """Coarse label for how a backbone was pretrained."""
    spec = arch.REGISTRY.get(name, {})
    if "timm" not in spec:
        return "imagenet_sup_tv"
    tag = spec["timm"].lower()
    for needle, family in (
        ("dinov3", "self_distill_dino"),
        (".mae", "masked_autoencoder"),
        ("fcmae", "masked_autoencoder"),
        ("beitv2", "masked_image_model"),
        ("eva02", "masked_image_model"),
        ("siglip", "image_text_contrastive"),
        ("clip", "image_text_contrastive"),
        ("ssld", "distilled"),
        ("usi", "distilled"),
        ("dist", "distilled"),
        ("in22k", "imagenet21k_sup"),
    ):
        if needle in tag:
            return family
    return "imagenet_sup_timm"


# --------------------------------------------------------------------------
# Out-of-fold predictions
# --------------------------------------------------------------------------
@torch.no_grad()
def out_of_fold_probs(run_dir, name, records, labels, dev_idx, folds, class_ages,
                      device, batch_size=32):
    """Probabilities for every development image, from the fold that never saw it.

    Normalisation comes from the backbone's own config. ``ensemble.py`` does
    not do this yet and will mis-score the CLIP, SigLIP, Inception, Xception,
    EVA-02, CoAtNet, RegNetZ and MobileViT entries.
    """
    size = arch.input_size(name, None)
    mean, std = arch.normalisation(name)
    images = decode_images(records, size)
    probs = np.zeros((len(dev_idx), len(class_ages)), dtype=np.float64)
    use_amp = device.type == "cuda"

    for fold, (_, val_pos) in enumerate(folds, start=1):
        ckpt = torch.load(run_dir / "checkpoints" / name / f"fold{fold}.pth",
                          map_location="cpu")
        model = arch.build_model(name, len(class_ages), 0.3, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()
        vi = dev_idx[val_pos]
        loader = DataLoader(EvalDataset(images[vi], labels[vi], mean, std),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        out = []
        for batch, _ in loader:
            batch = batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        probs[val_pos] = np.concatenate(out)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del images
    return probs


# --------------------------------------------------------------------------
# Blends
# --------------------------------------------------------------------------
def score(probs, y, class_ages):
    return ordinal_metrics(y, probs.argmax(1), class_ages)


def greedy_selection(oof, y, class_ages, metric="accuracy", rounds=25):
    """Caruana forward selection -- reported only to measure its own optimism."""
    names = list(oof)
    counts = {n: 0 for n in names}
    total = np.zeros_like(next(iter(oof.values())))
    chosen, best_overall, best_counts = 0, -np.inf, dict(counts)
    for _ in range(rounds):
        best_name, best_score = None, -np.inf
        for n in names:
            cand = (total + oof[n]) / (chosen + 1)
            s = score(cand, y, class_ages)[metric]
            if s > best_score:
                best_name, best_score = n, s
        total = total + oof[best_name]
        chosen += 1
        counts[best_name] += 1
        if best_score > best_overall:
            best_overall, best_counts = best_score, dict(counts)
    return best_counts, best_overall


def main(argv=None):
    p = argparse.ArgumentParser(description="Error-diversity and honest blends.")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None,
                   help="the run's stdout log, used to pin the corpus")
    p.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="extra filenames added after the run began")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--min-class-count", type=int, default=8)
    p.add_argument("--cache", type=Path, default=None,
                   help="npz to store/reuse out-of-fold probabilities")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args(argv)

    records = pin_records(args.image_root, args.log, set(args.exclude),
                          args.min_class_count)
    groups = build_groups(records, verbose=False)
    class_ages = sorted({r.age for r in records})
    labels, dev_idx, test_idx, folds = rebuild_splits(
        records, groups, class_ages, args.manifest, args.folds,
        args.split_seed, args.test_fraction)
    y = labels[dev_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done = sorted(d.name for d in (args.run / "checkpoints").iterdir()
                  if d.is_dir() and len(list(d.glob("fold*.pth"))) == args.folds)
    print(f"[oof] {len(done)} completed model(s)\n")

    oof = {}
    if args.cache and args.cache.exists():
        cached = np.load(args.cache)
        oof = {k: cached[k] for k in cached.files if k in done}
        print(f"[oof] reused {len(oof)} from {args.cache}")
    for i, name in enumerate(done, 1):
        if name in oof:
            continue
        try:
            oof[name] = out_of_fold_probs(args.run, name, records, labels,
                                          dev_idx, folds, class_ages, device)
            print(f"  {i:3d}/{len(done)}  {name}", flush=True)
        except Exception as exc:
            print(f"  {i:3d}/{len(done)}  {name}: FAILED ({exc})")
    if args.cache:
        np.savez_compressed(args.cache, **oof)

    names = sorted(oof, key=lambda n: -score(oof[n], y, class_ages)["accuracy"])
    err = {n: (oof[n].argmax(1) != y) for n in names}

    # --- 1. single-model table
    print()
    print("=" * 76)
    print("SINGLE MODELS (out-of-fold, every dev image scored once)")
    print("=" * 76)
    print(f"{'model':26}{'family':26}{'acc':>7}{'macroF1':>9}")
    print("-" * 76)
    for n in names[:args.top]:
        m = score(oof[n], y, class_ages)
        print(f"{n:26}{pretraining_family(n):26}"
              f"{m['accuracy']:7.3f}{m['macro_f1']:9.3f}")

    # --- 2. diversity
    print()
    print("=" * 76)
    print("ERROR DIVERSITY")
    print("=" * 76)
    dis = [float((err[a] != err[b]).mean())
           for a, b in itertools.combinations(names, 2)]
    print(f"  mean pairwise disagreement on correctness: {np.mean(dis):.1%}")
    print(f"  range: {np.min(dis):.1%} to {np.max(dis):.1%}")
    all_wrong = np.ones(len(y), dtype=bool)
    for n in names:
        all_wrong &= err[n]
    any_right = ~all_wrong
    print(f"  images EVERY model misses: {int(all_wrong.sum())} of {len(y)}")
    print(f"  images at least one model gets right: {int(any_right.sum())} "
          f"({any_right.mean():.1%})  <- ceiling for any blend")

    print()
    print("  cross-family disagreement (higher = more complementary):")
    fams = {}
    for n in names:
        fams.setdefault(pretraining_family(n), []).append(n)
    fam_names = sorted(fams)
    print("    " + "".join(f"{f[:11]:>13}" for f in fam_names))
    for a in fam_names:
        row = []
        for b in fam_names:
            vals = [float((err[x] != err[z]).mean())
                    for x in fams[a] for z in fams[b] if x != z]
            row.append(f"{np.mean(vals):13.1%}" if vals else f"{'--':>13}")
        print(f"    {a[:11]:<11}" + "".join(row))

    # --- 3. honest blends
    print()
    print("=" * 76)
    print("BLENDS")
    print("=" * 76)
    best_single = score(oof[names[0]], y, class_ages)
    print(f"  {'best single (' + names[0] + ')':52} "
          f"acc={best_single['accuracy']:.3f} f1={best_single['macro_f1']:.3f}")

    uni = np.mean([oof[n] for n in names], axis=0)
    m = score(uni, y, class_ages)
    print(f"  {'uniform blend of all ' + str(len(names)):52} "
          f"acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f}  "
          f"({m['accuracy'] - best_single['accuracy']:+.3f})")

    rule = [max(v, key=lambda n: score(oof[n], y, class_ages)["accuracy"])
            for v in fams.values()]
    rb = np.mean([oof[n] for n in rule], axis=0)
    m = score(rb, y, class_ages)
    print(f"  {'rule: best of each of ' + str(len(rule)) + ' pretraining families':52} "
          f"acc={m['accuracy']:.3f} f1={m['macro_f1']:.3f}  "
          f"({m['accuracy'] - best_single['accuracy']:+.3f})")
    print(f"        members: {', '.join(sorted(rule))}")

    top3 = np.mean([oof[n] for n in names[:3]], axis=0)
    m3 = score(top3, y, class_ages)
    print(f"  {'top 3 by CV score (BIASED -- shown for contrast)':52} "
          f"acc={m3['accuracy']:.3f} f1={m3['macro_f1']:.3f}  "
          f"({m3['accuracy'] - best_single['accuracy']:+.3f})")

    counts, gscore = greedy_selection(oof, y, class_ages)
    picked = {k: v for k, v in counts.items() if v}
    print(f"  {'greedy selection (BIASED -- measures its own optimism)':52} "
          f"acc={gscore:.3f}")
    print(f"        picked: {picked}")
    print()
    print(f"  selection bias estimate (greedy - uniform): "
          f"{gscore - score(uni, y, class_ages)['accuracy']:+.3f} accuracy")
    print("  Treat that gap as inflation, not gain. Only the uniform and")
    print("  rule-based rows are quotable; both choose members without")
    print("  consulting the scores they report.")

    out = args.run / "diversity.json"
    out.write_text(json.dumps({
        "n_models": len(names),
        "mean_pairwise_disagreement": float(np.mean(dis)),
        "all_wrong": int(all_wrong.sum()),
        "n_dev": int(len(y)),
        "best_single": {"model": names[0], **best_single},
        "uniform": score(uni, y, class_ages),
        "rule_based": {"members": sorted(rule), **score(rb, y, class_ages)},
        "families": {k: sorted(v) for k, v in fams.items()},
    }, indent=2, default=float))
    print(f"\n[out] {out}")


if __name__ == "__main__":
    main()
