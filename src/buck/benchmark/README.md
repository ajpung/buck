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

## The four rules

1. **Augmentation is train-only.** `EvalDataset` contains no stochastic code
   path at all — not a disabled flag, an absent one.
2. **Evaluation sets are never oversampled or rebalanced.** One tensor per real
   image, fixed order, true class mix.
3. **Near-duplicate images are grouped**, and the group is the unit of
   splitting, so an image cannot appear on both sides of the wall.
4. **The test set is locked to a manifest.** Created once, reused verbatim
   forever. New weekly images join the development pool automatically.

`assert_no_leakage()` re-checks 1–3 before every single fold, and raises rather
than warns.

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
| accuracy | 0.675 | ±0.023 | 0.656 – 0.701 |
| qwk | 0.823 | ±0.021 | 0.799 – 0.837 |

So an unmodified baseline can hand you anything from 0.799 to 0.837 qwk while
nothing has changed. **A single-run difference smaller than about 0.04 qwk is
not evidence of anything.** This has already produced two false positives: a
"+0.034 qwk" for `--loss ordinal` and a "+0.010" for `--soft-labels`, both of
which vanished under repetition.

Note this is a *different* quantity from the `+/-` printed in the leaderboard,
which is the spread across CV folds within one run. That column says nothing
about whether the run would reproduce.

Use `sweep.py rep <preset> <n>` in the repo root to run a configuration under
several seeds and get the across-run SD. Judge changes against that.

The one effect that has survived repetition so far is `--loss ordinal` on
within-one-year accuracy: 0.925 -> 0.946, perfectly separated across three
seeds. It does **not** move exact accuracy (t = -0.01).

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

`trail cam/splits/holdout_test_v1.json` records the held-out filenames plus a
content digest of each. It is deliberately exempted from the repo's `*.json`
ignore rule and **must stay in version control**. The script refuses to run if a
listed image is missing or its bytes changed.

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