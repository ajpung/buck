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
| **Architecture search** | Settled at 85 models, not 12. 83 valid entries span 0.494-0.697 accuracy, and against a ~0.015 across-seed SD the top twenty are indistinguishable. Capacity actively hurts: ResNet and RegNet both peak at their smallest variant, and `resnet18` (17 min) beats `efficientnet_b7` (244 min). See *The 85-model sweep*. |
| **Alternative pretraining** | Seven regimes, and plain ImageNet supervision wins all of them. On one ViT-B/16 body: supervised 0.610 > BEiT-v2 0.591 > CLIP 0.590 > MAE 0.569 > SigLIP 0.558. Elsewhere FCMAE 0.646-0.648, in22k 0.638-0.649, EVA-02 0.611, DINOv3 0.596 against an identical `convnext_tiny` at 0.654. The best model in the sweep is a distilled ImageNet model. |
| **Ensembling, re-tested at 83 models** | Still a loss, now with seven pretraining families represented. Uniform blend of all 83: **-0.048** accuracy against the best single. A pre-specified best-per-family blend: **-0.013**. Greedy forward selection reports +0.052 -- which is **+0.100 over the uniform blend, and pure selection bias**, the same defect class as best-epoch checkpointing. Mean pairwise disagreement is 20.8%, and cross-family disagreement (19-23%) is indistinguishable from within-family (18-24%): different pretraining objectives do not make different mistakes. Only 9 of 230 images are missed by every model, so the ceiling is 96% -- unreachable because the members are too correlated. `buck.benchmark.diversity` measures this. |
| **Detector-normalised crops** | Measured 2026-09-09 as a pre-check, before building anything. MegaDetector v6 boxes on all 289 NDA images (99.7% found, median conf 0.945) show the framing is *already* normalised: the deer is centred at 0.497 ±0.027 / 0.522 ±0.057 and spans >=94% of the image width in three quarters of the corpus. Equalising every deer's area to the median needs a 0.95-1.12x rescale across the IQR (0.92-1.29x at p5-p95) -- a +/-10-15% zoom, against a +12deg rotation the augmentation already applies. Framing is also not a shortcut: corr(age, area) = -0.045, corr(age, aspect) = +0.055. This is structural, not luck -- squaring a tightly-zoomed rectangular original yields a square necessarily smaller than the rectangle, so the animal fills the frame by construction. See `trail cam/detector_boxes.json`. |
| **Detector-normalised crops, second look** | The pre-check below stands, but note the related *colour shortcut* finding: detector boxes revealed that background pixels alone predict age at 0.593. Crop normalisation does not fix that and may not even touch it. |
| **The published training recipe** | Three seeds on `convnextv2_tiny`, matched folds: **-0.038 accuracy and -0.072 qwk** against the harness default, at 5.6x the runtime. Both halves hurt and they stack -- augmentation volume -0.020/-0.034, the published learning rates -0.019/-0.037. Neutral on `resnet18` (0.606 vs 0.610), harmful on a better backbone: tuned around the smaller model. See *The published recipe, isolated*. |
| **Augmentation volume** (`--train-multiplier 40`) | **-0.020 accuracy, -0.034 qwk on matched folds**, at 6x the runtime. Also raises the selection gap (+0.044 -> +0.084) and seed-to-seed prediction disagreement (0.205 -> 0.273, above the 18-24% between different architectures). The default 8x stands. An earlier entry called this open; it compared across a change in the record set. |
| **Flip TTA** | +1 correct image out of 230, scored on identical weights (exactly paired, so training noise cancels). Changes 4.6% of predictions; accuracy and macro-F1 up, within-one and qwk down. Doubles inference cost for nothing. |
| **Seed ensembling** | The one ensembling result that pays, and it is small: **+0.011** accuracy at 5 folds, **+0.023** at 10, saturating at k=3. Averaged over k-subsets, not scored on one subset -- doing the latter is what produced an earlier **+0.026** claim. Its real value is variance reduction: subset-to-subset SD 0.019 -> 0.007. |
| **More than 3 ensemble seeds** | Nothing. k=4 through k=8 are flat inside noise. Seeds 45-49 bought no accuracy. |
| **20-fold cross-validation** | **-0.017 accuracy** against 10-fold at 2x the cost. 12 of 20 validation folds end up missing at least one age class (8-15 images across 5 classes), so stratification fails and the extra 11 training images per fold do not pay for it. 10 folds is a real optimum, not a plateau. |
| **Single-seed model ranking** | Rejected as a method. Two sweep leaders have now failed replication: `regnet_y_1_6gf` (1st of 12, then 46th of 85) and `edgenext_small` (0.697 at 1 seed, 0.644 at 3, losing to `convnextv2_tiny` on every seed). The across-seed SD is of the same order as the spread across an entire top twenty. |
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
at less than half the cost of `maxvit_t`.

**Superseded by the 85-model sweep below.** This table is kept because earlier
numbers were measured against it, but it covers 12 torchvision backbones at one
seed and its ordering did not survive: `regnet_y_1_6gf`, first here, placed
46th of 85.

## The 85-model sweep

Measured 2026-09-10 to 09-12, `benchmark_runs/mega_a`, 5 folds, seed 42, 55.7
GPU-hours, zero failures. 85 entries: torchvision's 42 plus 43 from timm chosen
to widen the *pretraining* axis rather than only capacity -- masked
autoencoding (MAE, FCMAE, BEiT-v2), image-text contrastive (CLIP, SigLIP),
self-distillation (DINOv3), 21k supervision, and USI/SSLD distillation.

| model | px | accuracy | +/-1yr | QWK | macroF1 | MAE | pretraining |
|---|---|---|---|---|---|---|---|
| edgenext_small | 256 | **0.697** | 0.873 | **0.787** | **0.673** | 0.443 | distilled |
| densenet201 | 224 | 0.683 | 0.847 | 0.719 | 0.654 | 0.507 | in1k |
| resnet18 | 224 | 0.672 | 0.832 | 0.687 | 0.650 | 0.550 | in1k |
| resnet34 | 224 | 0.656 | 0.855 | 0.702 | 0.642 | 0.531 | in1k |
| regnet_y_400mf | 224 | 0.679 | 0.858 | 0.724 | 0.640 | 0.505 | in1k |
| convnext_base | 224 | 0.670 | **0.876** | 0.771 | 0.640 | **0.476** | in1k |
| convnext_tiny | 224 | 0.654 | 0.858 | 0.737 | 0.624 | 0.523 | in1k |

Excluding two runs that collapsed (see below), 83 valid models span **0.494 to
0.697 accuracy**. Against a ~0.015 across-seed SD the top twenty are mutually
indistinguishable, so read the extremes, not the ordering.

Three things the full field settles:

- **Capacity is not the lever, and often hurts.** ResNet peaks at 18 (0.672,
  0.656, 0.605, 0.614 for 18/34/50/101); RegNet peaks at 400mf. The four most
  expensive entries -- `efficientnet_b7` (600px, 244 min), `b6` (166 min),
  `v2_m` (161 min), `b5` (116 min) -- score 0.613, 0.617, 0.589, 0.639.
  **`resnet18` beats `efficientnet_b7` at a fourteenth of the compute.**
- **Pure transformers underperform.** Swin, SwinV2, CrossViT, Visformer, PiT,
  PVT-v2, Twins, MViTv2 and TinyViT all land mid-table or below; only
  conv-hybrids (`davit_tiny`, `maxvit`, `caformer`) reach the upper half. At
  184 training images the convolutional prior still pays for itself.
- **Recency does not predict performance.** A 2015 ResNet-18 places third;
  several 2022-23 designs sit in the bottom twelve.

### Pretraining regime is not the lever either

`vit_b_16` appears five times with one architecture, 86M parameters and a
224px input, differing only in how it was pretrained:

| pretraining | accuracy | QWK |
|---|---|---|
| ImageNet supervised | **0.610** | 0.675 |
| BEiT-v2 (masked image) | 0.591 | 0.686 |
| CLIP (LAION-2B) | 0.590 | 0.648 |
| MAE (masked autoencoder) | 0.569 | 0.613 |
| SigLIP (WebLI) | 0.558 | 0.673 |

Supervised wins. The same holds across families: FCMAE 0.646-0.648, in22k
0.638-0.649, EVA-02 0.611, and DINOv3 0.596 against `convnext_tiny`'s 0.654 on
an identical body. The best model in the whole sweep is a *distilled ImageNet*
model. **No self-supervised or multimodal regime beats plain supervision here.**

### The DINOv3 learning-rate trap

`convnext_tiny_dinov3` first scored 0.206 accuracy at qwk **0.014** -- the
signature of collapse onto one constant class, a different one per fold. That
was not a result about pretraining. Measured on fold 1, one variable at a time:

| trial | accuracy | QWK |
|---|---|---|
| `backbone_lr` 1e-4 (default) | 0.135 | 0.000 |
| `backbone_lr` 3e-5 | **0.692** | 0.730 |
| `backbone_lr` 1e-5 | 0.673 | 0.674 |
| backbone frozen | 0.673 | 0.747 |
| `weight_decay` 0.0 | 0.346 | 0.000 |
| ImageNet reference | 0.731 | 0.780 |

The learning rate, then -- not weight decay, and not the harness. These
checkpoints carry LayerNorm gains averaging 2.80 against ImageNet's 0.88 and
emit pooled features roughly 9x larger, so a step size chosen for supervised
weights destroys them in the first epochs. `REGISTRY` entries may now carry a
`backbone_lr` override; both DINOv3 entries use 3e-5 and rerun to 0.596 and
0.619.

**The general lesson: a backbone scoring near the majority-class floor with QWK
near zero has not lost, it has failed to train.** Check for that signature
before reporting any model as weak.

### Without transfer learning

`benchmark_runs/mega_b`, 10 architectures, `--no-pretrained`, same folds.
`convnext_tiny` falls from 0.654 to 0.490; the best from-scratch model is
`regnet_y_1_6gf` at 0.510 and the worst is `swin_t` at **0.191 with QWK
-0.012** -- the majority-class floor with no ordinal signal at all.
**Pretraining is worth about 0.15 accuracy, more than every architecture
choice in the registry combined.**

### Classical classifiers, re-measured

`benchmark_runs/mega_c`, 8 feature sets times 24 classifiers = 192 rows under
the benchmark's own splits. The project's original "20 canned classifiers"
predate every leak fix and are not comparable to anything current.

| feature set | best accuracy |
|---|---|
| colour histograms | 0.588 |
| frozen ResNet-50 features | 0.569 |
| frozen ConvNeXt-Tiny features | 0.522 |
| frozen ConvNeXt DINOv3 features | 0.515 |
| all hand-crafted concatenated | 0.458 |
| raw 32x32 pixels | 0.364 |
| HOG | 0.353 |
| LBP | 0.338 |
| majority-class floor | 0.185 |

Fine-tuning buys 0.08 to 0.14 over any frozen representation. But see **The
colour shortcut** below -- the histogram result does not mean what it looks
like.

## The published recipe, isolated

The paper's own training recipe against the harness default on
`convnextv2_tiny`, `--checkpoint-policy final` throughout. The question was
whether the published learning rates and exponential schedule help a better
backbone, or were tuned around `resnet18`.

They were tuned around `resnet18`. On that backbone they are neutral -- 0.606
against the harness default's 0.610 -- and on a better one they cost accuracy
and qwk together.

All three rows below are on one record set (232 dev images) with identical
folds, so the comparison is matched. An earlier version of this section
compared across a change in the corpus and drew a different conclusion; see
*Record sets* below, because that failure is the more useful lesson.

| config | acc | qwk | macro F1 | MAE | sel. gap | min |
|---|---|---|---|---|---|---|
| **default** -- tm 8, cosine, 1e-4/5e-4 | **0.660** | **0.759** | **0.648** | **0.485** | **+0.044** | **27** |
| **aug40** -- tm 40, cosine, 1e-4/5e-4 | 0.640 | 0.725 | 0.633 | 0.533 | +0.084 | 160 |
| **paper** -- tm 40, exponential, 3e-4/1e-3 | 0.622 | 0.688 | 0.617 | 0.560 | +0.102 | 150 |

Per seed (42/43/44): default 0.650 / 0.656 / 0.675; aug40 0.640 / 0.620 /
0.660; paper 0.620 / 0.623 (two seeds -- the third is on the older record set).

Isolating one change at a time:

    tm 8 -> tm 40 (augmentation volume)   acc -0.020   qwk -0.034   6.0x cost
    default LRs -> paper LRs              acc -0.019   qwk -0.037
    tm 8 default -> full paper recipe     acc -0.038   qwk -0.072   5.6x cost

**Both halves of the recipe are harmful, in the same way, and they stack.**
Each costs about 0.02 accuracy and about 0.035 qwk, and the full recipe loses
roughly the sum of the two. The default wins every column at a sixth of the
runtime.

An earlier reading of these runs claimed the two components failed in
*different* metrics -- augmentation costing accuracy only, learning rates
costing qwk only. That dissociation was an artifact of comparing across record
sets and does not survive matched folds. The real story is duller: both changes
are simply bad.

The selection gap rises with distance from the default, 0.044 -> 0.084 ->
0.102. The published configuration is both worse on the corrected metric and
more inflated by the early stopping it pairs with: under its own best-epoch
rule it would report roughly 0.62 + 0.10 qwk and look competitive against a
corrected 0.660.

### What the seeds vary, and what they do not

Not the split. `--seed` feeds `set_seed()` only -- `random`, `numpy`, `torch`.
The fold partition comes from `--split-seed` (`compare_architectures.py:463`),
which has been 1337 in every run in this repository. **Within one record set,
every seed shares identical folds.** Three seeds are three training runs on one
partition, not three partitions.

The original rationale for three seeds -- "three independent splits, so each
image is held out three times" -- was therefore never true. To vary the
partition, vary `--split-seed`.

### Record sets

`StratifiedGroupKFold` repartitions completely when the record count changes,
so a single added image invalidates every baseline measured before it. This has
now caused two wrong conclusions in this project and is worth stating plainly:
**a result is only comparable to another result on the same record count.**

| set | dev images | runs |
|---|---|---|
| 230 | 230 | `mega_a` (the 85-model sweep) |
| 231 | 231 | `ens10_*`, `paper_geom_s42` |
| 232 | 232 | `paper_geom_s43/s44`, `aug40_*`, `default_poolB_*`, `edgenext_poolB_*` |
| 229 | 229 | the masking arms (`--require-boxes` drops 4 of 291) |
| 233 | 233 | `fold5_*`, `fold10_*`, `fold20_*` |

Two measured examples of what a change costs. Between the 231 and 232 sets, the
same configuration at the same seed moved **-0.068** accuracy (`paper_geom`
seed 42 at 0.689 against its siblings at 0.620 and 0.623) -- and for a
different configuration the same change was **-0.008 accuracy but +0.033 qwk**,
moving the two metrics in opposite directions. The shift is neither small nor a
constant offset, so it cannot be corrected for after the fact.

`mega_a` and `ens10_s42` differ only by one image joining the pool and score
0.648 against 0.669. That 0.021 is the same size as effects this project has
spent days chasing.

### Model choice, re-tested

The 85-model sweep's leader was never seed-replicated, and when it was, it lost.

| model | sweep (1 seed, 230) | 3 seeds, 232 set | 3-seed ensemble |
|---|---|---|---|
| `edgenext_small` | **0.697** / qwk 0.787 | 0.644 / qwk 0.705 | 0.647 (+0.000) |
| `convnextv2_tiny` | 0.648 / qwk 0.750 | **0.660** / qwk 0.759 | **0.690** |

Per seed, head to head: 0.629 / 0.632 / 0.672 against 0.650 / 0.656 / 0.675.
`convnextv2_tiny` wins all three on accuracy and all three on qwk, by -0.055
qwk on average -- which clears the 0.036 qwk threshold this README sets for a
real difference. The accuracy margin, -0.016, is about one across-seed SD and
is the weaker half of the claim.

Note that `edgenext_small` gains **nothing** from seed-ensembling (+0.000)
while `convnextv2_tiny` gains, despite near-identical seed disagreement (0.200
against 0.205). Diversity is not what separates them; its members are simply
individually weaker.

**This is the second sweep ranking to fail replication.** `regnet_y_1_6gf` led
the 12-model suite and placed 46th of 85. A single-seed leaderboard position on
this corpus is not evidence; the across-seed SD is of the same order as the
spread across the entire top twenty.

### Reproducing it

    # the published recipe
    --scheduler exponential --lr-gamma 0.95 --backbone-lr 3e-4       --classifier-lr 1e-3 --epochs 70 --patience 20 --train-multiplier 40

    # augmentation volume alone, everything else at harness defaults
    --epochs 60 --patience 15 --train-multiplier 40

`--scheduler exponential`, `--lr-gamma` and `--classifier-lr` exist only to
express this recipe; cosine and `TRAIN_DEFAULTS` remain the defaults and no
existing run changes. Exponential decay is paired in the paper with early
stopping, i.e. best-epoch selection, so the two are kept separable rather than
bundled into a preset -- `--checkpoint-policy` controls the second
independently.

Cost note: of the paper recipe's 149 minutes, 130 are the augmentation
multiplier, so the extra 10 epochs are nearly free and the entire 5.6x is
`--train-multiplier`. One `aug40` seed took 206 minutes against 130 and 144 for
identical work -- almost certainly thermal; that cell is not usable for timing
comparisons.

## The colour shortcut

A 384-bin colour histogram with an SVM scores 0.588 accuracy, within 0.065 of a
fully fine-tuned CNN, using no spatial information whatsoever. The obvious
confounds do not explain it:

| shortcut-only baseline | accuracy |
|---|---|
| majority class | 0.185 |
| channel flag alone (colour vs IR) | 0.250 |
| collection-batch one-hot (98 batches) | 0.148 |
| colorhist, colour images only | **0.643** (floor 0.175) |

So the signal is real and large. But it is **not on the animal**. Splitting the
histogram by MegaDetector box, colour images only:

| region | accuracy | QWK |
|---|---|---|
| inside the animal box | 0.612 | 0.633 |
| **outside -- background only** | **0.593** | 0.587 |

A histogram of background pixels alone -- median 16.6% of the frame, containing
no deer -- predicts the deer's age at 0.593 against a 0.188 floor. Nor is it
trivial exposure: mean brightness lifts 0.032 over the floor, mean RGB 0.107,
saturation 0.210, against the full histogram's 0.42.

### Resolved by masking, 2026-09-28

The decisive experiment -- retrain with the animal masked out, and again with
the background masked out -- has now run. Three seeds per arm, 229 dev images,
`convnextv2_tiny` at harness defaults, `--mask` using the MegaDetector boxes.
Majority floor on this record set is **0.240**.

| arm | per seed | acc | qwk | macro F1 | MAE |
|---|---|---|---|---|---|
| **control** (untouched) | 0.625 / 0.669 / 0.695 | **0.663** | 0.746 | 0.641 | 0.491 |
| **deer only** (background blanked) | 0.647 / 0.634 / 0.678 | **0.653** | 0.707 | 0.637 | 0.525 |
| **background only** (deer blanked) | 0.498 / 0.481 / 0.490 | **0.489** | 0.348 | 0.477 | 0.951 |
| majority floor | | 0.240 | 0.000 | | |

    removing the background   acc -0.010   qwk -0.039
    removing the deer         acc -0.174   qwk -0.398

**The model reads the deer.** Blanking the background costs 0.010 accuracy,
inside the control's own 0.070 seed spread. Blanking the deer costs 0.174 and
collapses qwk from 0.746 to 0.348. The claim that the model reads body
proportions is no longer unverified; the deer alone reproduces the full model.

**And the shortcut is real.** Background alone reaches 0.489 against a 0.240
floor -- **59% of the control's above-floor margin** -- from 4-17% of the
pixels, since the deer fills most of the frame by construction. Its seed spread
is 0.017, the tightest of any cell in this project. That is a working model,
not degenerate guessing.

Both being true is coherent: the signal is **duplicated**. The deer and its
surround each predict age independently, so a model with both available leans
on the deer, and masking either leaves the other to carry it. The arms do not
decompose -- 0.653 + 0.489 has no reason to equal 0.663 -- because masking
destroys information rather than redistributing it.

**What still matters for field use.** The background signal is not age; it is
whatever co-varies with age in the collection -- site, season, camera, batch.
The threat is not that the current model uses it, but that
`StratifiedGroupKFold` groups by animal and cannot detect it: a context cue
shared across the split is leakage this protocol does not catch, and dev
accuracy looks identical whether the model generalises or not. Folds grouped by
site or collection batch would test this, and have not been run.

Deer-only costs 0.010 accuracy -- nothing, against a 0.070 seed spread -- and
is structurally immune. With boxes on 99.7% of images, masking at inference is
already feasible, which makes **0.653 a better field estimate than 0.663**.

## Fold count and ensemble size

Two levers that cost only compute, measured together on 233 dev images because
they share a baseline. `convnextv2_tiny` at harness defaults, unmasked.

All numbers below are **pooled out-of-fold**: one metric computed over all 233
development images, not a mean of per-fold metrics. That matters here, because
comparing fold counts on `cv_accuracy` would compare a mean over five 46-image
folds against a mean over twenty 12-image ones, which are different estimators.

| folds | train/fold | 1 seed | 3-seed ensemble | qwk (k=3) | min/seed |
|---|---|---|---|---|---|
| 5 | 186 | 0.643 | 0.654 | 0.746 | 27 |
| **10** | 210 | **0.668** | **0.691** | 0.764 | 59 |
| 20 | 221 | 0.651 | 0.670 | 0.806 | 123 |

**10 folds is the optimum, and it is a real optimum** -- performance rises to
it and falls after, rather than plateauing. Against 5 folds it is worth +0.025
as a single model and +0.037 ensembled, for 2x the compute. That is the largest
non-data improvement this project has measured.

**20 folds is worse**, by -0.017 single and -0.021 ensembled, at 2x the cost
again. The learning-curve extrapolation predicted +0.007 and got the sign
wrong. The reason is visible in the splits: at 20 folds the validation folds
hold 8-15 images across 5 classes, and **12 of the 20 are missing at least one
age class entirely**. Stratification cannot hold, and whatever the extra 11
training images buy is lost to worse-conditioned folds. Each doubling of the
fold count also adds less data than the last -- 5->10 gains 24 images per fold,
10->20 gains 11, 20->40 would gain about 6 -- so the lever is nearly exhausted
at this corpus size regardless.

One oddity, recorded but not trusted: the 20-fold arm produced the highest qwk
in the project (0.806) while its accuracy fell. It rests on a single 3-subset
and on pooled predictions from many tiny folds. The accuracy drop is the
reliable read.

### How much seed-ensembling is worth

Eight seeds at 5 folds, scoring every k-subset (up to 20 sampled per k) and
averaging, so each row is what a k-seed ensemble is *expected* to be worth
rather than what one lucky subset was worth.

| k | acc | sd across subsets | qwk |
|---|---|---|---|
| 1 | 0.643 | 0.019 | 0.730 |
| 2 | 0.652 | 0.017 | 0.742 |
| **3** | **0.654** | 0.015 | 0.746 |
| 4 | 0.655 | 0.009 | 0.747 |
| 5 | 0.656 | 0.007 | 0.744 |
| 6 | 0.650 | 0.008 | 0.746 |
| 7 | 0.659 | 0.006 | 0.755 |
| 8 | 0.648 | -- | 0.749 |

**It saturates at k=3 and the gain is small**: +0.011 at 5 folds, +0.023 at 10.
Everything past k=3 is flat inside noise; the k=8 dip is an artifact, since
only one 8-subset exists and it gets no averaging. Seeds 45-49 bought nothing.

An earlier note in this file put seed-ensembling at **+0.026**. That came from
scoring the one 3-seed subset that happened to be on hand, whose members were
the strong ones. Averaged across subsets the expectation is **+0.011**. Score
subsets, not the subset you have.

What ensembling does buy reliably is **stability**: subset-to-subset SD falls
from 0.019 at k=1 to 0.007 at k=5. For a project that has been misled by
single-seed results three times -- `regnet_y_1_6gf`, `edgenext_small`, and
`paper_geom` seed 42 -- that is worth more than the accuracy.

### Current best

**`convnextv2_tiny`, 10 folds, 3-seed ensemble: 0.691** out-of-fold on 233 dev
images. That is +0.048 over the 5-fold single model, for about three hours of
compute.

    --models convnextv2_tiny --folds 10 --epochs 60 --patience 15       --train-multiplier 8 --checkpoint-policy final --seed 42|43|44

Ranked by what paid:

    10 folds instead of 5          +0.025 single, +0.037 ensembled
    3-seed ensembling (at 10)      +0.023
    20 folds instead of 10         -0.017            <- rejected

The locked test set has still never been read. Every number here, and
everywhere else in this file, is development performance.

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