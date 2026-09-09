"""Short launcher for the 5-class accuracy experiments.

The benchmark's own command line is too long to paste without wrapping, so the
configurations under test live here by name instead:

    python sweep.py soft
    python sweep.py all
    python sweep.py list

Every preset uses --test-policy none, so the locked test set stays untouched no
matter which one is run. Compare results on the CV line.
"""
import sys

from buck.benchmark.compare_architectures import main

BASE = ["--models", "convnext_tiny", "--folds", "5",
        "--test-policy", "none", "--no-profile"]

PRESETS = {
    "base":   ([], "unchanged baseline -- run with 'rep' to measure the noise floor"),
    "gray":   (["--grayscale"], "luma input; lifts infrared, costs colour more -- net loss"),
    "soft":   (["--soft-labels"], "poll vote distribution as training target"),
    "ord":    (["--loss", "ordinal"], "distance-decayed ordinal target"),
    "mix":    (["--mixup", "0.2"], "mixup on target distributions"),
    "combo":  (["--soft-labels", "--loss", "ordinal", "--mixup", "0.2"],
               "all three together"),
    "res320": (["--image-size", "320"], "320px input"),
    "res384": (["--image-size", "384"], "384px input"),
    "heavy":  (["--image-size", "320", "--augmentation", "heavy"],
               "320px + heavy augmentation"),
    "heavy224": (["--augmentation", "heavy"], "heavy augmentation at 224px"),
    "light224": (["--augmentation", "light"], "light augmentation at 224px"),
    "tta":    (["--image-size", "320", "--tta"], "320px + flip TTA"),
    "big":    (["--models", "convnext_small"], "larger backbone"),
    "unfroze": (["--backbone-lr", "3e-4"], "hotter backbone, no stem freeze effect"),
}


def run(name):
    flags, why = PRESETS[name]
    argv = [a for a in BASE if a not in ()] + flags + ["--output", f"benchmark_runs/{name}"]
    # A preset may override --models; drop the base pair if so.
    if "--models" in flags:
        i = argv.index("--models")
        argv = argv[:i] + argv[i + 2:]
    print(f"\n{'#' * 70}\n# {name}: {why}\n{'#' * 70}", flush=True)
    main(argv)


# Measured 2026-09-03 over 3 seeds (42/43/44) under the corrected pipeline:
# same-animal grouping, holdout_test_v2, the SE parameter-group fix, and
# --checkpoint-policy final (EMA weights, fixed schedule, no best-epoch pick).
# The +/- values are ACROSS-RUN SD -- the same config on a different training
# seed -- which is the only thing a change should be judged against. The
# per-fold SD printed by the leaderboard is a different and larger quantity.
#
# For the record, the pre-fix figures were acc 0.675 +/- 0.023, qwk 0.823.
# Accuracy is unchanged; qwk fell 0.053. The mean selection gap measured on
# these same runs is +0.070 qwk, i.e. re-enabling best-epoch selection here
# would report ~0.840 -- so essentially all of the old qwk inflation came from
# checkpoint selection rather than from the frame leakage.
BASELINE = dict(model="convnext_tiny (224px)", cv_accuracy=0.677,
                cv_accuracy_sd=0.015, cv_qwk=0.770, cv_qwk_sd=0.018,
                cv_within_one=0.888, cv_macro_f1=0.669, cv_mae_years=0.462,
                selection_gap_qwk=0.070)
RUN_TO_RUN_QWK_SD = 0.018   # across seeds, same config


def compare():
    """Table every finished run in benchmark_runs/, ranked against baseline."""
    import json
    from pathlib import Path

    rows = []
    for path in sorted(Path("benchmark_runs").glob("*/results.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for entry in data.get("leaderboard", []):
            rows.append((path.parent.name, entry))

    print(f"{'run':14} {'model':17} {'acc':>15} {'QWK':>7} "
          f"{'+/-1yr':>7} {'MAE':>6} {'dQWK':>7}")
    print("-" * 80)
    b = BASELINE
    if b is None:
        print("(baseline)     -- not re-measured since the grouping, "
              "parameter-group and checkpoint-selection fixes.")
        print("               Run: python sweep.py rep base 3")
    else:
        print(f"{'(baseline)':14} {b['model']:17} "
              f"{b['cv_accuracy']:7.3f}+/-{b['cv_accuracy_sd']:.3f} "
              f"{b['cv_qwk']:7.3f} {b['cv_within_one']:7.3f} "
              f"{b['cv_mae_years']:6.3f} {'--':>7}")
    for name, e in sorted(rows, key=lambda r: -r[1]["cv_qwk"]):
        delta = f"{e['cv_qwk'] - b['cv_qwk']:+7.3f}" if b else f"{'n/a':>7}"
        print(f"{name[:14]:14} {e['model'][:17]:17} "
              f"{e['cv_accuracy']:7.3f}+/-{e['cv_accuracy_sd']:.3f} "
              f"{e['cv_qwk']:7.3f} {e['cv_within_one']:7.3f} "
              f"{e['cv_mae_years']:6.3f} {delta}")
    if b is None:
        print("\nNo dQWK column: the stored baseline predates the leakage and "
              "selection fixes and would make real changes look like "
              "regressions. Re-measure it before judging anything.")
    else:
        print("\ndQWK is vs baseline. Judge a change against the across-run SD "
              "from 'sweep.py rep', not against this mean.")


def repeat(name, n=3):
    """Run one preset n times under different seeds and report across-run SD.

    Training is nondeterministic even at a fixed seed -- cudnn.benchmark, TF32
    and AMP all admit run-to-run variation -- so a single number cannot tell a
    real effect from luck. The split is held fixed (--split-seed is untouched);
    only the training seed moves, which isolates optimisation variance.
    """
    import json
    import statistics
    from pathlib import Path

    flags, why = PRESETS[name]
    print(f"\n{'#' * 70}\n# {name} x{n}: {why}\n{'#' * 70}", flush=True)
    got = []
    for i in range(n):
        seed = 42 + i
        out = f"benchmark_runs/{name}_s{seed}"
        argv = list(BASE) + flags + ["--seed", str(seed), "--output", out]
        if "--models" in flags:
            j = argv.index("--models")
            argv = argv[:j] + argv[j + 2:]
        print(f"\n--- {name} seed {seed} ({i + 1}/{n}) ---", flush=True)
        main(argv)
        entry = json.loads(Path(out, "results.json").read_text())["leaderboard"][0]
        got.append((entry["cv_accuracy"], entry["cv_qwk"]))

    accs = [g[0] for g in got]
    qwks = [g[1] for g in got]
    sd = statistics.stdev if len(got) > 1 else (lambda v: 0.0)
    print(f"\n{'=' * 60}\n{name}: {n} runs\n{'=' * 60}")
    print(f"  accuracy {statistics.mean(accs):.3f} +/- {sd(accs):.3f}   "
          f"(runs: {', '.join(f'{a:.3f}' for a in accs)})")
    print(f"  QWK      {statistics.mean(qwks):.3f} +/- {sd(qwks):.3f}   "
          f"(runs: {', '.join(f'{q:.3f}' for q in qwks)})")
    if BASELINE is not None:
        print(f"\nCompare against baseline QWK {BASELINE['cv_qwk']:.3f}. A "
              f"difference smaller than ~2x this SD is not a result.")
    else:
        print(f"\nPaste these into BASELINE at the top of this file if this was "
              f"the 'base' preset. A difference smaller than ~2x the SD above "
              f"is not a result.")


if __name__ == "__main__":
    args = sys.argv[1:] or ["list"]
    if args[0] == "rep":
        if len(args) < 2 or args[1] not in PRESETS:
            sys.exit("usage: python sweep.py rep <preset> [n]")
        repeat(args[1], int(args[2]) if len(args) > 2 else 3)
        sys.exit(0)
    if args[0] == "compare":
        compare()
        sys.exit(0)
    if args[0] == "list":
        b = BASELINE
        if b is None:
            print("baseline: NOT MEASURED under the current pipeline -- "
                  "run 'python sweep.py rep base 3' first\n")
        else:
            print(f"baseline: acc {b['cv_accuracy']:.3f}+/-{b['cv_accuracy_sd']:.3f}  "
                  f"qwk {b['cv_qwk']:.3f}+/-{b['cv_qwk_sd']:.3f}  "
                  f"({b['model']}, 3 seeds, corrected pipeline)\n")
        for k, (_, why) in PRESETS.items():
            print(f"  {k:9} {why}")
        sys.exit(0)
    names = list(PRESETS) if args[0] == "all" else args
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        sys.exit(f"unknown preset(s): {', '.join(unknown)}")
    for n in names:
        run(n)
