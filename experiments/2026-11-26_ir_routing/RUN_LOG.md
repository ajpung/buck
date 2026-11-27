# Run log: IR routing experiment

Every step actually run, in order. Times are local (2026-11-26). All Python is
`buck-env/Scripts/python.exe` from the repo root unless stated.

## 1. Protocol written

`PROTOCOL.md` written before any grayscale model existed. Two edits made
before any experimental result existed (still pre-registration):

- dropped "saved locked-test predictions 58/58" from gate 2, so no test image
  is decoded anywhere in the experiment;
- made the leakage-gate thresholds numeric (off-diagonal minus diagonal > 0.15;
  diagonal within 0.10 of mean per-seed OOF).

## 2. Corpus pinned

`pin_corpus.py`: 297 images copied to
`benchmark_runs/ir_routing/pinned_images/squared/`, SHA-256 verified against
the live files, record list and ages identical. Manifest: `corpus_manifest.json`.

## 3. Gate 2 (pin verification): `oof.py --arm baseline --verify`

| | recorded (executed.ipynb) | reproduced |
|---|---|---|
| seed 42 | 0.682 | 0.682 |
| seed 43 | 0.653 | 0.653 |
| seed 44 | 0.678 | **0.682** |
| ensemble | 0.669 | 0.669 |

Leakage matrix (baseline): own-fold 0.667, other folds 1.000. **Pass.**

**Deviation 1: seed 44 differs by one image.** Investigated with
`check_seed44.py`: predicting per fold (as the notebook did) reproduces 0.678
exactly. Predicting all 239 dev images in one pass changes batch composition,
which changes floating-point results by at most 0.0018 in probability. One
image, `241205_240927_TX_3p5_NDA.png`, is an exact tie (0.407 / 0.407 between
3.5 and 5.5+) and flips. The pin is therefore correct. Both arms are scored the
same all-at-once way by `oof.py`, so the comparison between arms is
like-for-like. Gate 2 treated as passed on this evidence.

## 4. Grayscale arm training started

`bash experiments/2026-11-26_ir_routing/run_training.sh` (background). Logs in
`logs/train_s{42,43,44}.log`. Startup confirmed: 297 images, class mix
{1.5: 38, 2.5: 51, 3.5: 71, 4.5: 65, 5.5: 72}, channels {color: 229,
grayscale: 68}, identical to the baseline.

## 5. Router feasibility: `router.py`

Mean-saturation rule (pre-registered): fit agreement 0.953 (297 NDA),
held-out agreement 0.949 (234 non-NDA images the fit never saw).

**Deviation 2: the IR folder holds blue-tinted IR.** The 14 fit misroutes
are all `grayscale`-folder images from batches 250512/250513 with mean
saturation 59-161. Viewed two (`250512_121010_NC`, `250513_131110_MI`): genuine
night IR, uniformly blue-tinted (likely screen captures of trail-cam video). The
folder labels are right; saturation is the wrong cue for these. Two extra
features were defined *before* looking at their scores (`rank1_residual`,
`hue_spread`) and all three reported. Selection rule fixed in advance: highest
fit agreement. Results:

| feature | fit agreement | held-out agreement |
|---|---|---|
| saturation | **0.953** | 0.949 |
| rank1_residual | 0.946 | 0.983 |
| hue_spread | 0.933 | 0.979 |

Saturation is kept, per the rule. Not switching to `rank1_residual` on the
strength of its held-out score: that would be choosing on the evaluation set.
Saturation's fit misroutes are all IR -> colour model, which only reproduces
today's behaviour, so the cost is bounded. Added a secondary endpoint, *routed
with the pixel router*, so the deployable system is measured, not just the
folder-label oracle.

## 6. Analysis dry run

`analyze.py` run with the baseline standing in for both arms (`IR_OUT` pointed
at a scratch copy). All differences 0, all McNemar p = 1. Baseline reproduces
the notebook: IR 0.481, colour 0.722, overall 0.669. Weekly deer usable without
reading the test set: 54 (website 24 exact, baseline 26).

## 7. Training finished

Seeds 42 / 43 / 44: 14:10-16:37, 16:37-19:04, 19:04-21:28. Exit 0, 10
checkpoints each. (Estimate was ~67 min per seed; actual ~2 h 25 min.)

## 8. Gate 5 on the grayscale arm: `oof.py --arm gray`

OOF 0.661 / 0.628 / 0.661, ensemble 0.657. Own-fold 0.644 vs other folds
1.000. **Pass.** Partial results were not examined before all three seeds
finished.

## 9. Analysis: `analyze.py`

Run once, 21:30. Outcome in RESULTS.md: routing not recommended.
