# Results: IR routing experiment

Analysed 2026-11-26 21:30 by `analyze.py`; raw numbers in `outputs/results.json`.

## Conclusion

**Routing is not recommended.** The pre-registered decision rule fails on two
of its four conditions. The prior finding that grayscale input lifts IR accuracy
(+0.067 on `convnext_tiny`) **did not replicate** on `vit_base_dinov3`: the
grayscale model was no better than the colour model on IR at any single seed.

| decision condition | result | pass |
|---|---|---|
| 1. ensemble IR accuracy difference > 0 | +0.038 (+2 images of 52) | yes |
| 2. IR improved on >= 2 of 3 seeds | 0 of 3 (two worse, one tied) | **no** |
| 3a. overall within-one not worse | 0.895 vs 0.895 | yes |
| 3b. overall QWK not worse | 0.784 vs 0.787 | **no** |
| 4. leakage gates | baseline 0.667 own-fold / 1.000 other; gray 0.644 / 1.000 | yes |

## Gates

- Pin (gate 2): passed, with deviation 1 (see RUN_LOG.md).
- Leakage (gate 5): both arms pass. Grayscale arm OOF per seed 0.661 / 0.628 /
  0.661, ensemble 0.657; own-fold 0.644 vs other folds 1.000.

## Primary endpoint: IR subset (n = 52, 52 animal groups)

| | accuracy | within-one | QWK | MAE (yr) |
|---|---|---|---|---|
| baseline (colour model) | 0.481 | 0.788 | 0.607 | 0.750 |
| routed (gray model on IR) | 0.519 | 0.788 | 0.580 | 0.731 |

Paired: 3 images only routed right, 1 only baseline right; exact McNemar
p = 0.625. Cluster-bootstrap 95% CI for the difference: -0.038 to +0.115.

Per seed, IR accuracy, baseline vs gray: 42: 0.519 vs 0.500; 43: 0.442 vs
0.442; 44: 0.558 vs 0.538. The ensemble's +2 images come from averaging, not
from a model that is better at any seed. That is the pattern of noise.

## Secondary endpoints

- Overall (n = 239): 0.669 -> 0.678 (folder routing), CI -0.008 to +0.026;
  0.682 with the pixel router (3 vs 0 discordant, p = 0.25, CI 0.000 to 0.029).
  The pixel router does slightly *better* than folder routing because it sends 13
  tinted-IR images to the colour model. That is not a meaningful difference.
- Grayscale model on colour images (descriptive): 0.722 -> 0.695 (8 vs 3
  discordant, p = 0.23). Same direction as the prior colour cost.
- Weekly website deer usable without reading the test set: 54 of 64.

| exact / 54 | all | colour (27) | IR (27) |
|---|---|---|---|
| website (EfficientNet-B5) | 24 | 11 | **13** |
| baseline (vit_base_dinov3) | 26 | **17** | 9 |
| gray model | 26 | 15 | 11 |
| routed | 28 | 17 | 11 |

## Exploratory, not pre-registered: treat as a hypothesis only

On the 27 weekly IR deer, the **website's EfficientNet-B5 beat the new model,
13 vs 9**, while the new model won on colour 17 vs 11. This was not an endpoint.
It is 27 images, was found by looking, and the website saw user-cropped uploads
rather than the identical files. It would justify a pre-registered test of
"route IR to EfficientNet-B5", scored out-of-fold on the dev IR images. It
does not justify shipping that.

## Interpretation

The IR gap (0.48 vs 0.72) is not closed by withholding colour from the model.
That fits the README's diagnosis that IR is a data problem: 52 IR dev images is
too few. The power warning in PROTOCOL.md applies: a real effect of a few
points could still exist, but this experiment gives no evidence for one, and
the per-seed results argue against it.

## What this means for the website

Ship the single `vit_base_dinov3` model without routing, if it is shipped. On
the 54 comparable weekly deer it scores 26 vs 24 exact, 49 vs 44 within one
class, MAE 0.61 vs 0.76. Its IR weakness relative to the current site is the
open risk.
