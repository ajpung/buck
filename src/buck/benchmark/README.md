# BUCK architecture benchmark

Leak-free comparison of transfer-learning backbones for whitetail buck age
estimation. Replaces the sweep in `trail cam/examples/251008 - all image.ipynb`.

## Why this exists

The previous sweep reported ~80% validation against ~62% test and the gap kept
widening against fresh weekly datapoints. Three defects in the harness, not the
data, explain it:

| Defect | Where | Effect |
|---|---|---|
| Test images randomly h-flipped | `OptimizedDataset.__getitem__` | Test score depended on a coin flip; not reproducible |
| Test set oversampled to equal class counts | `length = num_classes * target_per_class` | Score measured a class mix that does not exist in the field |
| Checkpoints ranked by `val x test` | `multiplicative_score` | Best-of-130 selection against the test set; the "held-out" number was a second validation number |

Two further issues inflated apparent coverage: the 10 "folds" were 10
overlapping random splits rather than a partition, and 20 of the 33 declared
architectures never ran because a fixed-224 input error was swallowed by a bare
`except Exception: continue`.

Two more were found later, in this package rather than the notebook, and are
fixed as of the `holdout_test_v2` manifest:

| Defect | Where | Effect |
|---|---|---|
| Grouping by perceptual hash only | `build_groups` | Merged 1 pair in 288. Video frames of one buck scored Hamming 10-12 and were split across the wall; 11 clusters straddled `holdout_test_v1` |
| Parameter groups split by name substring | `train_fold` | `"fc" in name` matched every squeeze-excitation `fc1`/`fc2`, training 64 of EfficientNet-B0's 70 head-group tensors -- and 108 of RegNet-Y's 114 -- at 5x the intended backbone rate |

The second one matters most for `--models efficient`: `EFFICIENT_SUITE` is
almost entirely SE-based, so the table used to choose a model to ship was the
one most affected. Leaderboards produced before this fix cannot be compared
across families.

## The four rules

1. **Augmentation is train-only.** `EvalDataset` contains no stochastic code
   path at all — not a disabled flag, an absent one.
2. **Evaluation sets are never oversampled or rebalanced.** One tensor per real
   image, fixed order, true class mix.
3. **Images of the same animal are grouped**, and the group is the unit of
   splitting, so a deer cannot appear on both sides of the wall. Grouping is
   done in feature space, not by perceptual hash -- see below.
4. **The test set is locked to a manifest.** Created once, reused verbatim
   forever. New weekly images join the development pool automatically.

`assert_no_leakage()` re-checks 1–3 before every single fold, and raises rather
than warns.

### Why grouping is not a perceptual hash

Part of the corpus comes from video, so one buck can appear as several frames
of a single clip -- same pose, same background, same second. A DCT perceptual
hash does not see those as duplicates: on the 288-image NDA corpus a Hamming
threshold of 2 merged exactly **one** pair, while pairs that are visibly the
same animal score Hamming 10-12.

`build_groups` therefore unions two passes: the hash (for byte-level copies and
re-encodes) and cosine similarity above 0.90 between ResNet-50 features,
additionally required to agree on age class. That finds 25 multi-image clusters
covering 60 of 288 images. Every merge is printed, with its reason, so the
decisions are auditable rather than implicit.

Embeddings are computed on CPU in float32 and cached by content digest in
`trail cam/splits/duplicate_embeddings.npz`. CPU is deliberate: group ids feed
`StratifiedGroupKFold`, and `ensemble.py` reconstructs a finished sweep's fold
assignment by recomputing them, so a borderline pair flipping between a GPU and
a CPU run would silently re-partition the data. The cache is derived data and
is gitignored; deleting it costs a minute, not a result.

Measured cost of getting this wrong, via a frozen-feature linear probe scored
under both groupings:

| backbone (frozen + logistic head) | per-image groups | same-animal groups |
|---|---|---|
| convnext_tiny.fb_in1k | 0.597 | 0.521 |
| convnext_tiny.fb_in22k | 0.653 | 0.583 |
| vit_base_patch14_dinov2 | 0.660 | 0.635 |

That gap is the leak. The ImageNet backbones lose the most, which is what you
would expect if part of what they were matching on was the background.

## Usage

```bash
# Smoke test: 3 folds, 20 epochs, 4 architectures
python -m buck.benchmark.compare_architectures --quick

# Real comparison across a spread of families
python -m buck.benchmark.compare_architectures --folds 5

# Everything in the registry (36 architectures; slow)
python -m buck.benchmark.compare_architectures --models all --folds 5

# Prospective backtest: does the model actually work week to week?
python -m buck.benchmark.compare_architectures \
    --protocol temporal --models resnet50 --weeks 30
```

From a notebook:

```python
from buck.benchmark.compare_architectures import main
main(["--quick"])
```

## One run is not a result

Training is **nondeterministic even at a fixed seed**: `cudnn.benchmark`, TF32
and AMP all admit run-to-run variation. Measured over three seeds on the same
configuration (`convnext_tiny`, 224px, defaults):

| metric | mean | across-run SD | observed range |
|---|---|---|---|
| accuracy | 0.677 | ±0.015 | 0.661 – 0.691 |
| within 1yr | 0.888 | ±0.007 | 0.883 – 0.896 |
| qwk | 0.770 | ±0.018 | 0.758 – 0.791 |
| macro F1 | 0.669 | ±0.011 | 0.658 – 0.681 |

So an unmodified baseline can hand you anything from 0.758 to 0.791 qwk while
nothing has changed. **A single-run difference smaller than about 0.036 qwk is
not evidence of anything.** This has already produced two false positives: a
"+0.034 qwk" for `--loss ordinal` and a "+0.010" for `--soft-labels`, both of
which vanished under repetition.

The figures above were re-measured on 2026-09-03 under the corrected pipeline
(same-animal grouping, `holdout_test_v2`, the SE parameter-group fix, and
`--checkpoint-policy final`). The previous ones — accuracy 0.675 ±0.023, qwk
0.823 ±0.021 — were measured under best-epoch checkpoint selection. Accuracy is
unchanged; qwk fell 0.053, and the mean selection gap on the new runs is +0.070
qwk, so essentially all of the old qwk inflation came from selecting the
best-scoring epoch rather than from the frame leakage. Both defects were real;
only one of them was moving this number.

Note this is a *different* quantity from the `+/-` printed in the leaderboard,
which is the spread across CV folds within one run. That column says nothing
about whether the run would reproduce.

Use `sweep.py rep <preset> <n>` in the repo root to run a configuration under
several seeds and get the across-run SD. Judge changes against that.

### `--loss ordinal`, re-measured

This used to read: *"the one effect that has survived repetition is `--loss
ordinal` on within-one-year accuracy: 0.925 -> 0.946, perfectly separated
across three seeds. It does not move exact accuracy (t = -0.01)."* Both halves
of that were measured under best-epoch selection and both are wrong under the
corrected pipeline. Three seeds each, same split seed:

| metric | base | ordinal | delta | vs SD | read |
|---|---|---|---|---|---|
| accuracy | 0.677 ±0.015 | 0.643 ±0.009 | **-0.033** | 2.8x | real, and a **cost** |
| within 1yr | 0.888 ±0.007 | 0.920 ±0.029 | +0.032 | 1.8x | suggestive, not established |
| qwk | 0.770 ±0.018 | 0.790 ±0.032 | +0.019 | 0.8x | noise |
| macro F1 | 0.669 ±0.011 | 0.643 ±0.008 | **-0.026** | 2.6x | real, and a cost |
| MAE years | 0.462 ±0.025 | 0.455 ±0.033 | -0.007 | 0.3x | noise |

So ordinal loss **trades exact accuracy for near-misses**, which is what a
distance-decayed target does mechanically: probability mass moves onto
neighbouring classes, so fewer predictions land exactly and more land next
door. The old claim that it costs nothing on exact accuracy was an artifact of
picking the best epoch per fold.

The within-one gain is also weaker than it looks. The runs are **not**
separated -- base tops out at 0.896 and ordinal bottoms out at 0.896, touching
exactly -- and ordinal's spread is four times the baseline's (±0.029 vs
±0.007), driven by one seed at 0.952. At 1.8x the pooled SD this needs more
seeds before it goes in a paper.

Whether the trade is worth taking depends on which number the deployed tool is
judged on. If within-one-year is the field-practical metric, ordinal is
arguably the right default; if exact accuracy is quoted, it is not. Quote both
or neither -- reporting ordinal's within-one gain without its accuracy cost
would repeat exactly the error this section documents.

## What actually moves the number: corpus size

Measured 2026-09-08. `convnext_tiny`, 5 folds, 3 seeds per point. Only the
**training** portion of each fold was subsampled (stratified by class); the
validation fold was always full and unchanged, so every point is scored on
identical images.

| images/fold | accuracy | infrared (n=50) | colour (n=180) |
|---|---|---|---|
| 23 | 0.358 ±0.023 | 0.360 ±0.020 | 0.357 ±0.023 |
| 46 | 0.471 ±0.013 | 0.413 ±0.061 | 0.487 ±0.017 |
| 92 | 0.567 ±0.028 | 0.460 ±0.111 | 0.596 ±0.009 |
| 138 | 0.600 ±0.012 | 0.520 ±0.000 | 0.622 ±0.015 |
| **184 (current)** | **0.677 ±0.015** | 0.533 ±0.081 | 0.717 ±0.006 |

Fit: **`accuracy = -0.092 + 0.100 * log2(n)`**, residual SD 0.014.

**Roughly +0.10 accuracy per doubling of training data, with no sign of
saturation anywhere in the measured range.** The corpus is the binding
constraint; the model is not near the flat part of this curve.

Extrapolating (with the caveats below): ~440 total images for ~0.72, ~536 for
~0.75, ~731 for ~0.795 -- the last being human-crowd parity.

Three things that make those projections soft:

- Epochs and `train_multiplier` were held constant across fractions, so small
  fractions overfit more than a tuned model would. That depresses the low end
  and **steepens the fit**, so +0.10/doubling is likely an overestimate.
- The projections run 2-4x past the measured range. Learning curves saturate
  eventually; this one simply has not started to.
- The label itself has a ceiling. The NDA panel's mean vote share on the
  *correct* class is 0.544 and its plurality scores 0.795, so a hundred experts
  are split on the average deer. Treat ~0.80 as the practical ceiling.

Data must come from the NDA panel. The 184 non-NDA images in the corpus are not
a substitute: they carry a different institution's judgement, which is a
different target function rather than a noisy reading of this one. Training on
them measurably pulls accuracy toward that other definition (0.618 -> 0.569).

## The infrared deficit

Part of the corpus is infrared -- the `grayscale` channel is IR, not a colour
transform. It is 50 of 230 development images and it is much harder:

| subset | n | accuracy |
|---|---|---|
| infrared | 50 | 0.533 ±0.081 |
| colour | 180 | 0.717 ±0.006 |

An 18-point gap, reproducible three independent ways (saturation quartiles, the
channel flag, and two separate grayscale-input arms). Note the variances: the
colour subset is remarkably stable across seeds while infrared swings twelve
times as much, so **any infrared number needs more seeds than usual**.

This is a data problem, not a perception problem. Two pieces of evidence:

- Grad-CAM on out-of-fold checkpoints puts attention on the shoulder, neck and
  chest for both the images the model gets right and the 26 that *every*
  architecture misses. There is no background shortcut and no confusion about
  where to look.
- On the low-saturation development images the NDA panel scores 0.783 -- its
  normal rate -- with the same plurality share it shows on bright images. The
  information is in the photo; people extract it; the model does not.

The gap also widens with corpus size (absent at 23 images/fold, 18 points at
184), consistent with infrared simply being further back on the same curve.

**One confound to control for.** The two subsets are not framed alike. Detector
boxes (`trail cam/detector_boxes.json`) put the infrared images at 0.928 ±0.060
of the frame against colour's 0.754 ±0.156 -- 65 vs 218 of the confidently
detected NDA images, a 0.174 difference at roughly t = 13. **This is
unintentional**, an artifact of how those originals were cropped rather than a
property of the capture mode. So the 18-point gap is not a clean colour-vs-IR
contrast; it also contrasts tight framing against loose. The evidence above
still points at data rather than perception, but an IR experiment should
control for framing instead of assuming the channel is the only difference.

## Measured and rejected

Recorded so they are not re-proposed. All measured under the corrected
pipeline; see `HANDOFF.md` for the earlier set.

| idea | result |
|---|---|
| **Cross-architecture ensembling** | Uniform blend of all 12: +0.009 accuracy over the best single. The greedy-selected blend's +0.069 qwk is **selection bias** -- it picks members on the same out-of-fold data it reports, the same defect class as best-epoch checkpointing. Models are too correlated: 18% mean pairwise disagreement on errors, and 26 dev images that all of them miss. |
| **Architecture search** | 12 backbones across a 4x parameter range span 0.654-0.771 qwk, one cluster. `maxvit_t` placed 3rd at 2.3x the cost of `convnext_tiny`. |
| **Detector-normalised crops** | Measured 2026-09-09 as a pre-check, before building anything. MegaDetector v6 boxes on all 289 NDA images (99.7% found, median conf 0.945) show the framing is *already* normalised: the deer is centred at 0.497 ±0.027 / 0.522 ±0.057 and spans >=94% of the image width in three quarters of the corpus. Equalising every deer's area to the median needs a 0.95-1.12x rescale across the IQR (0.92-1.29x at p5-p95) -- a +/-10-15% zoom, against a +12deg rotation the augmentation already applies. Framing is also not a shortcut: corr(age, area) = -0.045, corr(age, aspect) = +0.055. This is structural, not luck -- squaring a tightly-zoomed rectangular original yields a square necessarily smaller than the rectangle, so the animal fills the frame by construction. See `trail cam/detector_boxes.json`. |
| **Flip TTA** | +1 correct image out of 230, scored on identical weights (exactly paired, so training noise cancels). Changes 4.6% of predictions; accuracy and macro-F1 up, within-one and qwk down. Doubles inference cost for nothing. |
| **Grayscale input** (`--grayscale`) | Net loss of ~3.3 images out of 230. Lifts infrared (+0.067, up on all 3 seeds) but costs colour more (-0.037, down on all 3), and colour is 78% of the corpus. See `decode_images()`. |
| **Hand-built body proportions** | Given the NDA panel's own stated justifications *as ground truth*, a bag-of-concepts encoding predicts age at 0.462 -- well below the 0.700 the pixels achieve. The published AOTH-style criteria are less informative than the image. |

## Loss and target options

| Flag | Effect |
|---|---|
| `--loss ordinal` | Trains against a Gaussian kernel over neighbouring age classes instead of a one-hot. Errors land next door rather than two classes away. `--loss ce` (default) is numerically identical to the previous `CrossEntropyLoss(label_smoothing=)` path. |
| `--ordinal-sigma` | Width of that kernel in class units; as it approaches 0 it converges back on hard CE. |
| `--mixup ALPHA` | Mixes target *distributions*, so it composes with `--loss ordinal` rather than fighting it. |
| `--soft-labels` | Uses the NDA weekly poll's vote distribution as the training target where it exists (63 of 284 images). Training signal only -- validation and test still score against the recorded label. |
| `--no-pretrained` | Random init, whole backbone trainable, backbone LR raised to 1e-3 so the comparison is not rigged by a fine-tuning rate. Costs 12-17 accuracy points; see below. |

Transfer learning is not optional on this corpus. Measured over 5 folds with a
*doubled* epoch budget for the scratch arm: resnet18 0.643 -> 0.524, and
efficientnet_b0 0.661 -> 0.489. That is 15x the run-to-run noise floor and the
only unambiguous effect the harness has ever measured.

## Reading the output

Rank architectures by the **CV columns**. That is the whole point of the
cross-validated leaderboard: it is computed on the development pool, so you may
compare as many models as you like against it without biasing anything.

The **test block** is a one-shot confirmation of the winner. By default
(`--test-policy winner-only`) exactly one architecture is scored on it. Running
`--test-policy all` and then quoting the best test number reintroduces precisely
the selection bias this package removes; the script prints a warning if you do.

With ~57 held-out images the 95% CI on test accuracy spans roughly ±12 points.
Two architectures within ~10 points of each other are **not** distinguishable on
this test set — use the CV mean and its standard deviation to separate them, and
treat the test number as a sanity check rather than a ranking.

Metrics are ordinal-aware, because predicting 2.5 for a 5.5-year-old is not the
same kind of error as predicting 3.5:

- `accuracy` — exact class match
- `within_one` — within one age class (the field-practical number)
- `qwk` — quadratic weighted kappa; default ranking metric, more stable than
  accuracy on small validation folds
- `mae_years` — mean absolute error in years

## Choosing a model for the website

Accuracy alone cannot pick a backbone to serve. Every run also measures what
each architecture costs to ship, and the leaderboard is followed by an
accuracy-vs-cost table and a Pareto front.

```bash
# Cost only -- no training, runs in about a minute
python -m buck.benchmark.compare_architectures --profile-only --models efficient

# Accuracy AND cost across the small-model suite
python -m buck.benchmark.compare_architectures --models efficient --folds 5
```

Measured per architecture, never looked up:

| Column | Meaning |
|---|---|
| `params M` | Parameter count, millions |
| `fp32 MB` | Serialised weights on disk |
| `int8 MB` | After dynamic int8 quantisation — the usual shipping format |
| `onnx MB` | The exported graph; what a browser downloads |
| `CPU ms` | Median latency, one image, one CPU thread |

Latency is batch 1 and single-thread on purpose. A hunter uploads one photo and
waits for one answer, so throughput at batch 32 would flatter the big models
misleadingly.

`onnx MB` needs `pip install onnx onnxscript`. Without it the column is empty
and the script says so explicitly rather than leaving a silent blank.

The **Pareto front** is the shortlist worth deciding between: a model is
dropped only when something else is both more accurate *and* smaller. Anything
listed as dominated should not be shipped, whatever its headline accuracy.

`--models efficient` selects `EFFICIENT_SUITE` — modern, ImageNet-competitive
backbones that stay small (MobileNetV3, ShuffleNetV2, MNASNet, EfficientNet-B0
/ V2-S, RegNet-Y, ResNet18, ConvNeXt-Tiny). The point of running them together
is to find where BUCK's accuracy actually starts to fall off as capacity drops,
rather than assuming the biggest backbone is required.

Note that int8 savings vary sharply by family: ConvNeXt-Tiny drops 108 MB to 33
MB because it is Linear-heavy, while convolution-dominated backbones like
RegNet barely move. Quantise before concluding a model is too large.

### `DEFAULT_SUITE`, complete

All 12, `benchmark_runs/suite_v2`, 5 folds, seed 42, reconstructed from fold
checkpoints by `python -m buck.benchmark.peek`. The `±` is across folds within
one run -- much larger than the across-seed SD -- so **do not rank on it**.

| model | px | accuracy | ±1yr | QWK | macroF1 | MAE | min |
|---|---|---|---|---|---|---|---|
| regnet_y_1_6gf | 224 | 0.665 | 0.874 | **0.771** | 0.657 | 0.483 | 15 |
| convnext_tiny | 224 | **0.700** | 0.878 | 0.763 | **0.687** | **0.457** | 22 |
| maxvit_t | 224 | 0.678 | 0.870 | 0.756 | 0.674 | 0.483 | 51 |
| vit_b_16 | 224 | 0.661 | 0.874 | 0.734 | 0.654 | 0.509 | |
| swin_t | 224 | 0.643 | 0.835 | 0.721 | 0.636 | 0.552 | |
| efficientnet_b3 | 300 | 0.665 | 0.870 | 0.719 | 0.660 | 0.513 | |
| densenet121 | 224 | 0.652 | 0.839 | 0.718 | 0.634 | 0.535 | |
| resnet50 | 224 | 0.635 | 0.848 | 0.708 | 0.626 | 0.552 | |
| efficientnet_v2_s | 384 | 0.643 | 0.852 | 0.699 | 0.634 | 0.552 | |
| resnet18 | 224 | 0.648 | 0.835 | 0.692 | 0.644 | 0.565 | |
| efficientnet_b0 | 224 | 0.643 | 0.830 | 0.674 | 0.633 | 0.583 | |
| mobilenet_v3_large | 224 | 0.639 | 0.813 | 0.659 | 0.628 | 0.604 | |

The whole field spans 0.117 qwk against a ~0.036 threshold for
distinguishability, and **the top three are tied**. On the tiebreakers that
need no significance test, `convnext_tiny` leads on accuracy, macro-F1 and MAE
at less than half the cost of `maxvit_t`. It is the right default, and there is
no longer an untested architecture that might displace it.

Architecture is exhausted as a lever -- a 4x parameter range buys ~0.06
accuracy. See *What actually moves the number* above.

## Protocols

**`holdout`** ranks by `StratifiedGroupKFold` CV on the development pool, then
scores the winner once on the locked test set, using the ensemble of the fold
models.

**`temporal`** is a rolling-origin backtest. For each recent collection date it
trains on every image collected strictly earlier and predicts that date's deer.
It needs no manifest — time ordering is the wall — and it is the closest
available proxy for the weekly workflow, so it is the right protocol for
answering "is the model getting worse?". It also prints a first-half/second-half
drift check.

Note that the two protocols are independent evaluations. A temporal run trains
on images that belong to the holdout manifest's test set; that is legitimate
within the temporal protocol, but do not mix numbers between the two.

## The manifest

`trail cam/splits/holdout_test_v2.json` records the held-out filenames plus a
content digest of each. It is deliberately exempted from the repo's `*.json`
ignore rule and **must stay in version control**. The script refuses to run if a
listed image is missing or its bytes changed.

`holdout_test_v1.json` is kept alongside it as the record of what earlier
reported numbers were measured against. **Do not use it for new runs.** Under
the corrected grouping, 11 of its clusters straddle the wall: roughly a fifth of
its 57 test images have a sibling frame in the development pool, so every
held-out number ever reported against v1 is optimistic by an unknown amount.
v2 holds 58 images and straddles zero groups.

Deleting it silently re-randomises the test set and makes every previously
reported held-out number incomparable. If you ever need to refresh it — say the
corpus has doubled — bump the version to `holdout_test_v2.json` and state which
manifest each reported result used.

## Data scope

Reads `trail cam/images/squared/{color,grayscale}` and, by default, only
`*_NDA.png`, whose labels are the project's ground truth. `--sources` widens
this.

`trail cam/images/original/` is **never** read; it holds the separate uncropped
imagery experiment.

Images whose age field is `xpx` (the current week's not-yet-aged deer) are
skipped automatically and reported.