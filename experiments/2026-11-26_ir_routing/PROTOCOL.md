# Experiment: route infrared images to a grayscale-input model

**Pre-registered 2026-11-26, before any grayscale model was trained or scored.**
Nothing below may be changed after results exist. Deviations go in
`RESULTS.md` under *Deviations*, with the reason, never silently into this file.

## Question

The current best model (`vit_base_dinov3`, 10 folds x 3 seeds, run
`benchmark_runs/nda_dino_yolo`, 2026-11-08) is much weaker on infrared (IR)
images (0.481 out-of-fold) than on colour (0.722). Does routing IR images to a
model trained on grayscale input, while colour images keep the current colour
model, improve accuracy?

## Prior evidence, treated as a hypothesis only

`src/buck/benchmark/data.py` (`decode_images` docstring) reports, for
`convnext_tiny` on an older 230-image record set: grayscale input raised IR
0.533 -> 0.600 (up on 3/3 seeds, paired p=0.098) and lowered colour 0.717 ->
0.680. Different architecture, different record set, not significant. Earlier
project findings have needed correction more than once (see CHANGELOG), so this
experiment re-measures everything it relies on and imports no conclusion.

## Design

Two arms, identical in everything except `--grayscale`:

| | baseline arm | grayscale arm |
|---|---|---|
| checkpoints | existing `benchmark_runs/nda_dino_yolo/final/vit_base_dinov3_s{42,43,44}` | new, `benchmark_runs/ir_routing/gray/vit_base_dinov3_s{42,43,44}` |
| input | colour | BT.601 luma replicated to 3 channels, train and eval |
| everything else | 10 folds, seeds 42/43/44, split seed 1337, 60 epochs, medium augmentation, train multiplier 8, `checkpoint_policy final`, `select_metric qwk`, min class count 8, `test_policy none` | identical |

No hyperparameter is tuned for the grayscale arm. There is no selection step.

**Routed system** (the thing being evaluated): images in the `grayscale` (IR)
channel are predicted by the grayscale arm; `color` images by the baseline arm.
By construction the routed system equals the baseline on every colour image, so
any difference is entirely on IR images.

## Data and leakage controls

1. **Pinned corpus.** The baseline was trained on 297 NDA images (239 dev / 58
   locked test). The live image folder has since gained aged images, which would
   change `StratifiedGroupKFold` completely. The exact 297 files are copied to
   `benchmark_runs/ir_routing/pinned_images/squared/{color,grayscale}` and both
   arms read only that folder. `corpus_manifest.json` (filename, channel,
   SHA-256) is committed here.
2. **Pin verification (gate, before training).** From the pinned folder, the
   baseline checkpoints must reproduce the notebook's recorded out-of-fold
   accuracies exactly (per seed 0.682 / 0.653 / 0.678; ensemble 0.669). If
   not, stop. (No test image is decoded at any point in this experiment, not
   even to compare against the saved test predictions.)
3. **Interpreter.** All training and scoring run under
   `buck-env/Scripts/python.exe` (scikit-learn 1.7.1). System Python
   (scikit-learn 1.8.0) produces different folds; that silently leaked
   (OOF 0.962) during the 2026-11-26 same-deer comparison.
4. **Group-aware folds.** Animal groups from `build_groups` (perceptual hash +
   embedding similarity, cached by content digest); `assert_no_leakage` runs
   inside training for every fold.
5. **Empirical leakage check (gate, after training).** Score every grayscale
   checkpoint on every fold. Each checkpoint must be clearly worst on its own
   held-out fold. Pass requires mean off-diagonal accuracy minus mean diagonal
   accuracy > 0.15, and the diagonal mean within 0.10 of the arm's mean
   per-seed OOF accuracy. Same check run on the baseline.
6. **Locked test set is not read.** `holdout_test_v2.json` was read once on
   2026-11-08 for the baseline. Reading it again to evaluate routing would make
   it a selection set. The decision is made on out-of-fold data only. Any test
   claim about the routed system requires a new locked set (`v3`) built from
   images not yet collected.
7. **Weekly website deer** (secondary): each image's prediction comes only from
   models that never trained on it (its held-out fold for dev images; all 30
   models for images added after the pin).

## Endpoints

All on the 239 development images, out-of-fold, 3-seed probability-averaged
ensemble unless stated.

- **Primary:** accuracy difference, routed minus baseline, on the IR subset
  (n = 52). Exact two-sided McNemar test on the paired per-image outcomes, and a
  95% CI from a cluster bootstrap over animal groups (10,000 resamples, seed 0).
- **Secondary:**
  - overall accuracy difference on all 239 (same tests; equals the IR
    difference x 52/239);
  - per-seed IR difference (seed s grayscale vs seed s baseline);
  - within-one, QWK, macro-F1, MAE on IR and overall;
  - grayscale arm on colour images (replication of the prior colour cost;
    descriptive, not used for the decision);
  - the 64 matched weekly website deer: website vs baseline vs routed.
- **Router feasibility:** the production router cannot see folder names. A
  pixel rule (mean HSV saturation below a threshold) is fit to the channel label
  only, never to age, on the pinned corpus; agreement with the folder label is
  reported. It uses no age information, so it cannot leak.

## Power, stated in advance

With 52 IR images and an expected gain of ~+0.067 (~3.5 images net), an exact
McNemar test at alpha 0.05 has low power (roughly 20-30%). A non-significant
result is the expected outcome even if the effect is real, and will be reported
as *inconclusive*, not as *no effect*.

## Decision rule

Recommend routing for the website if **all** of:

1. ensemble IR accuracy difference > 0;
2. IR difference > 0 on at least 2 of 3 seeds;
3. overall within-one and QWK not worse than baseline;
4. the leakage gates (2, 5) pass.

Report the McNemar p and bootstrap CI regardless. If 1-4 hold but p >= 0.05 the
recommendation is labelled *provisional* and is re-checked on the weekly
website log once it has accumulated new IR deer.

## Files

- `pin_corpus.py`: build and verify the pinned folder and manifest
- `oof.py`: every checkpoint of an arm on every dev image (-> out-of-fold
  predictions, checkpoint x fold leakage matrix) plus post-pin weekly images;
  `--verify` runs gate 2
- `router.py`: saturation router fit and agreement
- `analyze.py`: endpoints and statistics -> `results.json`
- `RUN_LOG.md`: every command actually run, in order, with timestamps
- `RESULTS.md`: results, deviations, conclusion
