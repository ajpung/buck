# BUCK — work in progress, 2026-09-03

Picking-up notes for the benchmark rebuild. Everything below is measured unless
it says otherwise.

---

## 1. Is something still running?

A 12-model sweep was launched at 16:16 and was still going when this was
written (10 of 12 done; `vit_b_16` and `maxvit_t` outstanding, ETA ~23:05):

```
python -m buck.benchmark.compare_architectures --folds 5 --test-policy none \
    --output benchmark_runs/suite_v2
```

**Check first:** if `benchmark_runs/suite_v2/results.json` exists, it finished —
that file has the full leaderboard, the accuracy-vs-cost table and the Pareto
front. If it does not exist, the run was interrupted.

**If it was interrupted, nothing is lost.** Fold checkpoints are written per
model as each one completes, so the finished models are still scoreable without
retraining. `StratifiedGroupKFold` is deterministic given the same records,
groups and seed, so the exact validation fold each checkpoint was scored on can
be rebuilt. A script that does exactly this is saved at
**`buck.benchmark.peek`** (it used to live at `benchmark_runs/peek_partial.py`,
which is gitignored, so it was never under version control):

```
python -m buck.benchmark.peek --run <sweep dir> --log <that run's stdout log>
```

**Pass `--log`.** A sweep that runs for days outlives the corpus it started on:
the weekly add gives last week's `xpx` deer its age, the development pool goes
230 → 231, and `StratifiedGroupKFold` over 231 images bears no resemblance to
the folds over 230. Every checkpoint then gets scored on a fold holding its own
training data and the leaderboard comes back around 0.96 with nothing raised.
File mtime cannot catch this — renaming preserves it — so the pin is read from
the run's own log of skipped `xpx` files and verified against its record count.
Without `--log` there is no pin; only a crude "top accuracy above 0.90" check
fires. If that file is gone, the recipe is:

```python
from buck.benchmark.ensemble import rebuild_splits   # gives labels, dev_idx, test_idx, folds
# for each model dir with 5 fold*.pth: load fold i, evaluate on dev_idx[folds[i][1]],
# score with buck.benchmark.metrics.ordinal_metrics
```

Only directories holding **exactly 5** `fold*.pth` files are complete;
checkpoints are written in one go after a model's folds finish.

**Gotcha:** stdout was buffered because the command was piped, so
`tasks/*.output` and `scratchpad/suite.log` are **empty**. That is cosmetic —
there is no lost error information — but do not read the absence of log output
as a crash. Add `python -u` next time.

---

## 2. Uncommitted work

Last commit is `02ecfcf` ("chore: correcting architecture"), which contains the
first batch of fixes. Still uncommitted:

| file | what changed |
|---|---|
| `src/buck/benchmark/data.py` | embedding-based `build_groups` + `_embed`; rewritten `load_vote_targets` |
| `src/buck/benchmark/compare_architectures.py` | `WeightEMA`, `--checkpoint-policy`, `--ema-decay`, `arch.split_parameters` call, group-aware temporal splits, holdout collision guard, macro-F1 leaderboard column |
| `src/buck/benchmark/README.md` | corrected noise-floor table; rewritten `--loss ordinal` section |
| `sweep.py` | `BASELINE` re-measured; `compare`/`list` updated |

`trail cam/splits/holdout_test_v2.json` **is** committed (in `02ecfcf`). Keep it.

---

## 3. What was fixed, and why

Four defects, in rough order of how much they mattered:

**a. Checkpoint selection (biggest effect).** `train_fold` kept whichever of ~60
epochs scored best on a ~45-image validation fold and reported *that* score — a
maximum over 60 noisy draws, biased upward by construction. Replaced with
`--checkpoint-policy final` (default): fixed cosine schedule, no early stopping,
`WeightEMA` weights, scored once at the end. Validation now selects nothing.
`--checkpoint-policy best` reproduces the old behaviour so the bias stays
measurable; every run reports a `selection_gap`.

**b. Same-animal grouping.** `build_groups` used a perceptual hash at Hamming
≤2, which merged **1 pair out of 288**. Part of the corpus is video, so one buck
appears as several frames of a clip that a pHash rates at Hamming 10–12.
Replaced with pHash ∪ (cosine > 0.90 on ResNet-50 features, constrained to the
same age class): **288 images → 253 groups**, 25 multi-image groups covering 60
images. Verified visually on the closest pairs — all are the same animal.
Embeddings run on CPU float32 and cache to
`trail cam/splits/duplicate_embeddings.npz` (gitignored); CPU is deliberate
because `ensemble.py` recomputes groups to rebuild fold assignment, and a
borderline pair flipping between GPU and CPU would silently re-partition the
data. Verified identical across from-scratch recomputes.

**c. Parameter groups.** The optimizer split backbone from head with
`any(k in name for k in ("classifier","fc","head"))`. torchvision names
squeeze-excitation blocks `fc1`/`fc2`, so **64 of EfficientNet-B0's 70**
head-group tensors, **108 of RegNet-Y-1.6GF's 114** and **32 of
MobileNetV3-Large's 38** trained at 5× the intended rate. Replaced with a
`_buck_head` module marker and `arch.split_parameters()` matching on parameter
identity. Verified across all 42 registry entries: 6 head tensors each (8 for
ConvNeXt, whose replacement LayerNorm is also freshly initialised).

**d. Vote-target ambiguity (latent, was NOT corrupting anything).**
`load_vote_targets` keyed on the collection date, which is ambiguous for the
four bulk archive batches (23–65 images each). Measured: **0 broadcasts** — none
of those batches appear in the vote CSV, and 67 images matched from 67 distinct
rows. Fixed anyway by requiring an unambiguous match. Do **not** key strictly on
`(collected, photodate, state)`: that drops 8 legitimate matches which are CSV
transcription slips, not different deer (see §7).

**Also fixed while verifying:** `run_temporal` carved its validation slice with
a plain permutation (not group-aware) and assumed time ordering was a
sufficient wall — four same-animal groups span two collection dates. Both fixed.
A guard was added to `run_holdout` for the future case where a newly added
weekly image turns out to be another frame of a locked test deer.

**Test manifest re-locked.** `holdout_test_v1.json` had **11 clusters straddling
the wall** under corrected grouping — roughly a fifth of its 57 test images had
a sibling in training. `holdout_test_v2.json` (58 images, 0 straddling) is now
the default. v1 is kept as the record of what older numbers were measured
against; **do not use it for new runs**.

---

## 4. Measured results

**Baseline** — `convnext_tiny` 224px, 3 seeds (42/43/44), corrected pipeline.
`±` is across-run SD, which is the only thing a change should be judged against:

| metric | mean | SD |
|---|---|---|
| accuracy | 0.677 | ±0.015 |
| within 1yr | 0.888 | ±0.007 |
| QWK | 0.770 | ±0.018 |
| macro F1 | 0.669 | ±0.011 |
| MAE years | 0.462 | ±0.025 |
| **selection gap** | **+0.070 QWK** | |

Pre-fix figures were acc 0.675 ±0.023, qwk 0.823. **Accuracy is unchanged;** QWK
fell 0.053, and the selection gap accounts for essentially all of it. Both
defects were real, but only checkpoint selection was moving this number.

**`--loss ordinal`** — 3 seeds each. The old README claim ("0.925 → 0.946,
perfectly separated, does not move exact accuracy") is **wrong on both counts**:

| metric | base | ordinal | delta | vs SD | read |
|---|---|---|---|---|---|
| accuracy | 0.677 ±0.015 | 0.643 ±0.009 | −0.033 | 2.8× | real, and a **cost** |
| within 1yr | 0.888 ±0.007 | 0.920 ±0.029 | +0.032 | 1.8× | **unresolved** |
| QWK | 0.770 ±0.018 | 0.790 ±0.032 | +0.019 | 0.8× | noise |
| macro F1 | 0.669 ±0.011 | 0.643 ±0.008 | −0.026 | 2.6× | real, and a cost |

It trades exact accuracy for near-misses, which is what a distance-decayed
target does mechanically. The runs are **not** separated (base max 0.896,
ordinal min 0.896 — they touch).

**Suite, first 8 models** (reconstructed from checkpoints; `±` here is across
folds within one run, much larger than across-seed SD — do not rank on it):

| model | accuracy | ±1yr | QWK | macroF1 | | old rank → new |
|---|---|---|---|---|---|---|
| regnet_y_1_6gf | 0.665 | 0.874 | **0.771** | 0.657 | SE | 6 → 1 |
| efficientnet_b3 | 0.665 | 0.870 | 0.719 | 0.660 | SE | 8 → 2 |
| densenet121 | 0.652 | 0.839 | 0.718 | 0.634 | | 5 → 3 |
| resnet50 | 0.635 | 0.848 | 0.708 | 0.626 | | 12 → 4 |
| efficientnet_v2_s | 0.643 | 0.852 | 0.699 | 0.634 | SE | 4 → 5 |
| resnet18 | 0.648 | 0.835 | 0.692 | 0.644 | | 10 → 6 |
| efficientnet_b0 | 0.643 | 0.830 | 0.674 | 0.633 | SE | 9 → 7 |
| mobilenet_v3_large | 0.639 | 0.813 | 0.659 | 0.628 | SE | 11 → 8 |

`regnet_y_1_6gf` is the only model that did not drop, going 6th → 1st. It is
the most SE-dense model in the suite, so it had the most to gain from fix (c).
But the SE fix is not uniformly kind — `efficientnet_b0` and
`mobilenet_v3_large` dropped the most. Everything from 0.719 down to 0.659 is
one cluster; two models need ~0.036 QWK between them to be distinguishable.

---

## 5. Open decisions

1. **Is `--loss ordinal` worth the trade?** Needs 3 more seeds
   (`python sweep.py rep ord 3` with seeds 45–47, ~75 min) to settle within-one
   at 1.8×SD. Then a judgement call: the paper quotes exact accuracy (76.7%),
   where ordinal costs 3.3 points; the README argues within-one is the
   field-practical number, where it gains 3.2. **Quote both or neither.**
2. **When to spend the test read.** `holdout_test_v2.json` is untouched. Spend
   it only once a training configuration *and* an architecture are committed —
   fold checkpoints are saved, so `ensemble.py --score-test` can do it without
   retraining. Reading it during tuning turns it back into a validation set.
3. **`convnext_tiny` sanity check.** It ran inside the suite under the same
   config as the baseline; its suite number should reproduce 0.770 ±0.018. If it
   does not, something is configuration-dependent and worth chasing.

---

## 6. Suggested order from here

1. Read `benchmark_runs/suite_v2/results.json` (or reconstruct — §1).
2. `pip install onnxscript`, then
   `python -m buck.benchmark.compare_architectures --profile-only --models suite`
   — ~1 min, fills the empty `onnx MB` column (what a browser downloads) without
   redoing training.
3. Resolve ordinal with 3 more seeds.
4. Commit the four modified files.
5. Then the larger open items in §8.

---

## 7. Data-quality items surfaced, not fixed

`load_vote_targets` now prints these every run. They are transcription slips in
`trail cam/image_metadata.csv`, not different deer — worth fixing at source:

```
250410_240916_VA_2p5   file state VA        csv VT
250612_241116_WI_3p5   file photo 241116    csv 250612   <- collection date in the photo columns
250619_251005_OH_5p5   file photo 251005    csv 250619   <- same
250626_241229_NC_4p5   file photo 241229    csv 250626   <- same
250911_241014_MI_3p5   file photo 241014    csv 241114
250925_250912_WI_5p5   file state WI        csv GA
251204_241210_MI_5p5   file state MI        csv MS
260319_251031_MT_2p5   file photo 251031    csv 251119
```

---

## 8. Known and unfixed

**Training is dataloader-bound, not GPU-bound.** During the suite the GPU sat at
1–3% utilisation while ~11 CPU cores were saturated. `train_fold` uses
`num_workers=0`, so every augmentation (rotation, flip, desaturate, brightness,
gamma, noise — all cv2) is produced serially in the main process, ~1,470 per
epoch. Roughly a 2× speedup is available, but **it is not a one-line change**:
`TrainDataset` holds a seeded `random.Random`, so `num_workers>0` forks a copy
per worker and changes the augmentation stream — results shift, not just speed.
Needs `worker_init_fn` seeding, and the baseline must be re-measured afterwards.

**`sweep.py` presets other than `base` and `ord` have not been re-run** under
the corrected pipeline. All pre-fix runs were moved to `benchmark_runs/pre_fix/`
(25 of them) so `sweep.py compare` no longer mixes eras. Those numbers are not
comparable to anything current.

**Ideas assessed and rejected** (so they don't get re-proposed):

- *Photo date / state as model inputs.* Measured: metadata alone scores 0.292
  accuracy vs a 0.243 majority-class floor with QWK ≈ 0.01, and on the 124
  images that have real date+state it scores **below** the floor. Median photo
  month is **10 for every age class** — the NDA posts rut-season photos
  regardless of age, so there is no seasonal contrast to exploit. Worse, 56% of
  images are `UUUUUU`/`UU`, and those have a different age mix, so a missingness
  flag alone scores 0.302 — a pure batch-prior shortcut that would inflate CV
  and be worthless on the website. Only main effects were tested; an
  interaction with image features is untested but implausible at n=288.
- *Widening `--sources` beyond NDA.* 184 non-NDA images exist. Adding them
  lowers exact accuracy 0.618 → 0.569 while raising QWK 0.702 → 0.740 — the
  signature of a labelling-convention difference between institutions. Would
  need a source-held-out study first.
- *The `temporal` protocol as a "weekly" backtest.* It buckets on collection
  date, but 176 of 289 images sit at four timestamps, and one of those batches
  contains photos from 2005–2019. It is leak-free but does not reproduce the
  weekly workflow the docstring claims.

**~~Still untried, and probably the largest remaining lever:~~ MEASURED AND
REJECTED, 2026-09-09.** The idea was to normalise the crop with an animal
detector (e.g. MegaDetector), on the reasoning that AOTH is a claim about body
*proportions* while the deer occupies a different fraction of every frame. The
premise is false: MegaDetector boxes on all 289 NDA images show the deer already
centred to ±0.03 and spanning >=94% of the width in three quarters of the
corpus, leaving a ±10-15% residual zoom and no correlation with age. Squaring a
tightly-zoomed rectangular original produces that framing by construction. See
*Measured and rejected* in `src/buck/benchmark/README.md` for the numbers.

---

## 9. Corrections to earlier analysis

Recorded so they are not repeated as fact:

- **"Expect accuracy to drop 5–8 points from the grouping fix."** It did not
  move at all (0.675 → 0.677). That figure came from a frozen linear probe and
  did not transfer to the fine-tuned pipeline.
- **"The soft-labels bug is confirmed, not just suspected."** It was not firing.
  67 images matched from 67 distinct CSV rows, zero broadcasts. The defect was
  latent only.
- **"Season should help, since neck swelling varies with the rut."** Sound
  biology, but the corpus has no seasonal variation to exploit (see §8).
