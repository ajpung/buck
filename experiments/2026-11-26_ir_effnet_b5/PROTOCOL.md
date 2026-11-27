# Experiment: route infrared images to EfficientNet-B5

**Pre-registered 2026-11-26 ~22:00, before any EfficientNet-B5 was trained
under this protocol.** Do not edit after results exist; deviations go in
`RESULTS.md`.

Builds on `../2026-11-26_ir_routing/` and reuses its pinned corpus, baseline
predictions and leakage machinery. Read that PROTOCOL.md first.

## Where the hypothesis came from (and why that matters)

In the routing experiment's weekly-deer table (an exploratory, non-registered
look), the live website model (EfficientNet-B5, `deer-age-api/251809_enb5.pth`)
got 13 of 27 weekly IR deer exactly right against 9 for the new
`vit_base_dinov3`, while losing on colour 11 vs 17. Hypothesis: **an
EfficientNet-B5 is better than `vit_base_dinov3` on IR images, so routing IR to
B5 and colour to the ViT would beat the ViT alone.**

Those 27 IR deer generated the hypothesis, so they cannot also confirm it. They
are listed in `hyp_split.json` (`hypothesis_generating`: 26 dev images plus 1
post-pin image).

## Why the deployed checkpoint is not tested directly

`251809_enb5.pth` came from `trail cam/examples/RTX5090 - setup.ipynb`:
random image-level `train_test_split` (no animal grouping, so video frames of
one deer can straddle train and test), on the NDA images of ~2025-09; fold 5
was chosen by validation x test accuracy (test used for selection; its recorded
78.4% test is not a clean estimate). It trained on much of our development set,
so it cannot be scored on it without leakage. Instead **the recipe is retrained
under this project's protocol**, so every prediction is out-of-fold.

## Design

| | baseline (reused) | B5 arm (new) |
|---|---|---|
| model | `vit_base_dinov3` | `efficientnet_b5` (torchvision, ImageNet weights) |
| input | 256 px colour | **600 px** colour (the website's resolution) |
| runs | `benchmark_runs/nda_dino_yolo/final` | `benchmark_runs/ir_effnet_b5/b5/efficientnet_b5_s{42,43,44}` |
| shared | pinned 297-image corpus, split seed 1337, 10 group-aware folds, seeds 42/43/44, 60 epochs, medium augmentation, multiplier 8, checkpoint policy final, select metric qwk, min class 8, `test_policy none`, `buck-env` | identical |

The harness training recipe is used, not the website notebook's (optimiser
settings, label smoothing, 1000-per-class augmentation). This tests the
architecture at the website's resolution under one controlled recipe. If it
wins, the model that would ship is this retrained B5, not the old checkpoint.
No tuning, no selection step. Batch size is the registry's 16.

**Routed system:** IR (`grayscale` folder) -> B5 3-seed ensemble; colour ->
baseline ensemble. Equal to baseline on all colour images by construction.

## Leakage controls

As in the routing experiment: pinned corpus (`../2026-11-26_ir_routing/corpus_manifest.json`),
`buck-env` (sklearn 1.7.1, asserted), `assert_no_leakage` per fold, the locked
test set never decoded, and gate 5 (each checkpoint x each fold; off-diagonal
minus diagonal accuracy > 0.15, and diagonal within 0.10 of mean per-seed OOF).

## Endpoints

- **Primary (confirmatory):** IR accuracy difference, routed minus baseline,
  on the **26 IR dev images not in `hypothesis_generating`** (`hyp_split.json`,
  `confirmatory_ir_dev`). Exact McNemar; cluster bootstrap over animal groups
  (10,000, seed 0).
- **Secondary:**
  - the same on all 52 IR dev images, and on the 26 hypothesis-generating ones
    (the latter is expected to favour B5 and is reported only for completeness);
  - per-seed IR difference on the confirmatory 26;
  - overall (239) accuracy, within-one, QWK, MAE for routed vs baseline;
  - B5 on colour images and overall (descriptive);
  - the pixel router from the routing experiment (saturation < 20.59) in place
    of folder labels.
- **Prospective (after this experiment):** log B5 and ViT predictions on every
  new weekly NDA deer before its age is published.

## Power, stated in advance

26 images. Even a large true gain (+15 points, ~4 net images) has well under
50% power at alpha 0.05. A non-significant primary result is expected and will
be reported as inconclusive. The decision therefore rests mainly on
direction and consistency (rule below), and the prospective log is the real
confirmation.

## Decision rule

Recommend IR -> B5 routing if **all** of:

1. ensemble IR accuracy difference on the confirmatory 26 > 0;
2. that difference > 0 on at least 2 of 3 seeds;
3. overall (239) within-one and QWK of routed >= baseline;
4. gate 5 passes for the B5 arm.

If 1-4 hold with p >= 0.05, the recommendation is *provisional*, pending the
prospective log. Also reported regardless: serving cost (routing means hosting
two models).
