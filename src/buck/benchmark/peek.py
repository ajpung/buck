"""Reconstruct CV metrics for whichever models of a sweep have finished.

results.json is only written after every model, but each model's fold
checkpoints are saved as soon as that model completes, and StratifiedGroupKFold
is deterministic given the same records/groups/seed -- so the exact validation
fold each checkpoint was scored on can be rebuilt without retraining.

Run from the repo root::

    python benchmark_runs/peek_partial.py --run benchmark_runs/mega_a \\
        --log path/to/the/run/stdout.log

**Pass --log.** A long sweep outlives the corpus it started on. The weekly data
add gives last week's ``xpx`` deer its age, which drops a new image into the
development pool; StratifiedGroupKFold over 231 images bears no resemblance to
the folds over 230, so every checkpoint gets scored on a fold containing its own
training data and the leaderboard comes back at ~0.96 with no error raised.
File mtime cannot detect this because renaming preserves it. The run's own log
records which files it skipped as unaged, which is what ``pin_records`` uses,
and it verifies the reconstructed count against the run's before proceeding.

Without --log there is no pin and no verification, so a drifted corpus will
produce silently inflated numbers; a crude absurdity check is applied instead.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from buck.benchmark import architectures as arch
from buck.benchmark.compare_architectures import predict
from buck.benchmark.data import (EvalDataset, build_groups, decode_images,
                                 drop_rare_classes, load_records)
from buck.benchmark.diversity import pin_records
from buck.benchmark.ensemble import rebuild_splits
from buck.benchmark.metrics import ordinal_metrics

# Above this, a reconstruction is almost certainly scoring checkpoints on their
# own training data. The corpus has never supported anything close to it: the
# NDA panel itself scores 0.795, and no honest run here has passed 0.71.
ABSURD = 0.90


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True,
                   help="sweep directory holding checkpoints/")
    p.add_argument("--log", type=Path, default=None,
                   help="the run's stdout log; pins the corpus (strongly advised)")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="extra filenames added after the run began")
    p.add_argument("--manifest", type=Path,
                   default=Path('trail cam/splits/holdout_test_v2.json'))
    p.add_argument("--image-root", type=Path,
                   default=Path('trail cam/images/squared'))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--min-class-count", type=int, default=8)
    p.add_argument("--sort", default="macro_f1",
                   choices=["accuracy", "within_one", "qwk", "macro_f1"])
    args = p.parse_args()

    if args.log is not None:
        records = pin_records(args.image_root, args.log, set(args.exclude),
                              args.min_class_count)
    else:
        print("[pin] WARNING: no --log given, so the corpus is not pinned and "
              "not verified.\n"
              "      If a weekly data add landed since this run started, the "
              "folds below are\n"
              "      wrong and every score is inflated. Pass --log to be sure.")
        records = [r for r in drop_rare_classes(
            load_records(args.image_root), args.min_class_count)
            if Path(r.path).name not in set(args.exclude)]

    groups = build_groups(records, verbose=False)
    class_ages = sorted({r.age for r in records})
    labels, dev_idx, test_idx, folds = rebuild_splits(
        records, groups, class_ages, args.manifest, args.folds,
        args.split_seed, args.test_fraction)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    done = [d.name for d in sorted((args.run / 'checkpoints').iterdir())
            if d.is_dir() and len(list(d.glob('fold*.pth'))) == args.folds]
    print(f"reconstructing {len(done)} completed model(s)\n", flush=True)

    rows = []
    for name in done:
        try:
            size = arch.input_size(name, None)
            # Read from the backbone's own config: the CLIP, SigLIP, Inception,
            # Xception, EVA-02, CoAtNet, RegNetZ and MobileViT entries were not
            # trained under ImageNet constants, and feeding them the wrong ones
            # understates them without raising.
            mean, std = arch.normalisation(name)
            images = decode_images(records, size)
            per_fold = []
            for fold, (_, va) in enumerate(folds, start=1):
                val_idx = dev_idx[va]
                ckpt = torch.load(args.run / 'checkpoints' / name / f'fold{fold}.pth',
                                  map_location='cpu')
                model = arch.build_model(name, len(class_ages), 0.3,
                                         pretrained=False)
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(device).eval()
                loader = DataLoader(
                    EvalDataset(images[val_idx], labels[val_idx], mean, std),
                    batch_size=16, shuffle=False, num_workers=0)
                y_true, y_pred = predict(model, loader, device, use_amp, False)
                per_fold.append(ordinal_metrics(y_true, y_pred, class_ages))
                del model
                torch.cuda.empty_cache()
            agg = {k: float(np.mean([m[k] for m in per_fold]))
                   for k in ('accuracy', 'within_one', 'qwk', 'macro_f1',
                             'mae_years')}
            agg['acc_sd'] = float(np.std([m['accuracy'] for m in per_fold], ddof=1))
            agg['model'] = name
            agg['px'] = size
            rows.append(agg)
            del images
        except Exception as exc:
            print(f"  {name}: reconstruction failed "
                  f"({type(exc).__name__}: {exc})")

    if not rows:
        sys.exit("no models could be reconstructed")

    rows.sort(key=lambda r: -r[args.sort])
    print(f"{'model':<24}{'px':>5}{'accuracy':>16}{'+/-1yr':>9}{'QWK':>8}"
          f"{'macroF1':>9}{'MAE':>7}")
    print('-' * 78)
    for r in rows:
        print(f"{r['model']:<24}{r['px']:>5}"
              f"{r['accuracy']:>9.3f}+/-{r['acc_sd']:<5.3f}"
              f"{r['within_one']:>9.3f}{r['qwk']:>8.3f}"
              f"{r['macro_f1']:>9.3f}{r['mae_years']:>7.3f}")
    print('-' * 78)
    print(f"sorted by {args.sort}. {len(rows)} model(s) complete.")
    print("+/- is ACROSS FOLDS within this run -- much larger than the "
          "across-seed SD, and not a basis for ranking.")

    top = max(r['accuracy'] for r in rows)
    if top > ABSURD:
        print()
        print("=" * 78)
        print(f"WARNING: top accuracy {top:.3f} exceeds {ABSURD}, which this "
              "corpus does not support.")
        print("The fold reconstruction is almost certainly misaligned -- these "
              "checkpoints are")
        print("being scored on their own training data. The usual cause is a "
              "weekly data add")
        print("landing after the run started. Re-run with --log so the corpus "
              "is pinned and")
        print("verified, and treat every number above as invalid.")
        print("=" * 78)


if __name__ == '__main__':
    main()
