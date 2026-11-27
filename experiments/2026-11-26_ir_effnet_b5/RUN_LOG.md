# Run log: IR -> EfficientNet-B5 experiment

Local time, 2026-11-26. Python is `buck-env/Scripts/python.exe` from repo root.

## 1. Provenance of the deployed model (before designing anything)

`deer-age-api/251809_enb5.pth` checkpoint metadata: torchvision
`efficientnet_b5`, image size 600x600, fold_number 5, train 100.0 / val 80.4 /
test 78.4, multiplicative_score 6303 (= val x test). Trained by
`trail cam/examples/RTX5090 - setup.ipynb`: images from
`buck/images/squared/{color,grayscale}/*_NDA.png` (older path), split with
`train_test_split(test_size=0.2, random_state=fold*42, stratify=age)` then an
inner 0.2 val split; no animal grouping. Conclusion: cannot be scored on our dev
set without leakage, so the recipe is retrained (PROTOCOL.md). Nothing in
`deer-age-api` was modified; only the checkpoint was read.

## 2. Hypothesis split

`hyp_split.json` from `../2026-11-26_ir_routing/outputs/results.json`
(weekly-deer rows with channel `grayscale`): 27 hypothesis-generating images
(26 dev, 1 post-pin), leaving 26 confirmatory IR dev images.

## 3. Protocol written

PROTOCOL.md, before any B5 training under this protocol. Resolution 600 px
chosen by the user (over 456 px, ~11 h) to match the website.

## 4. Smoke test

2 folds x 2 epochs at 600 px to the scratchpad (discarded): trains, no OOM at
batch 16, 415 s wall time. Not used for anything else.

## 5. Training launched

`bash experiments/2026-11-26_ir_effnet_b5/run_training.sh` (background).
Startup log matches the pinned corpus (297 images, colour 229 / grayscale 68).

## 6. Analysis dry run (before any B5 result)

`analyze.py` with `B5_NPZ` pointed at the baseline's npz: all differences 0,
all p = 1. The dry run shows the baseline's IR accuracy is 0.615 on the
confirmatory 26 and 0.346 on the hypothesis-generating 26. The hypothesis was
generated on the hardest IR images, so regression to the mean would favour
B5 there. This is why the confirmatory subset is primary.
