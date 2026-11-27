"""Router feasibility: can pixels alone tell IR from colour?

Production cannot see folder names, so routing needs a pixel rule. Feature:
mean HSV saturation (0-255) of the image. The threshold is fit to the CHANNEL
label only (never age), so it cannot leak age information. Fit on the pinned
corpus, then checked on the post-pin weekly images and the non-NDA images in
the live folder, which the fit never saw.

Usage: buck-env/Scripts/python.exe router.py
"""
import json, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from buck.benchmark.data import load_records

PINNED = ROOT / "benchmark_runs" / "ir_routing" / "pinned_images" / "squared"
LIVE = ROOT / "trail cam" / "images" / "squared"


def saturation(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())


# Deviation (see RESULTS.md): mean saturation misroutes blue-tinted IR video
# captures. Tinted IR is one hue scaled by brightness, so its pixels lie near a
# single line in RGB. Candidate features, fixed before looking at their scores:
def rank1_residual(path):
    """1 - share of RGB pixel variance on the first principal axis (x1000)."""
    img = cv2.resize(cv2.imread(str(path), cv2.IMREAD_COLOR), (128, 128)).reshape(-1, 3).astype(np.float64)
    ev = np.linalg.eigvalsh(np.cov(img.T))
    return float(1000 * (1 - ev[-1] / ev.sum()))


def hue_spread(path):
    """Circular SD of hue over pixels with saturation > 40 (degrees); 0 if none."""
    hsv = cv2.cvtColor(cv2.resize(cv2.imread(str(path), cv2.IMREAD_COLOR), (128, 128)), cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][hsv[..., 1] > 40].astype(np.float64) * 2 * np.pi / 180
    if h.size < 50:
        return 0.0
    r = np.hypot(np.cos(h).mean(), np.sin(h).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(r, 1e-9)))))


FEATURES = {"saturation": saturation, "rank1_residual": rank1_residual, "hue_spread": hue_spread}


def is_ir(sat, threshold):
    return sat < threshold


fit = load_records(PINNED, ("NDA",), ("color", "grayscale"))
fit_names = {r.filename for r in fit}
ir = np.array([r.channel == "grayscale" for r in fit])
all_sources = tuple({p.stem.split("_")[4] for p in LIVE.glob("*/*.png") if len(p.stem.split("_")) >= 5})
held = [r for r in load_records(LIVE, all_sources, ("color", "grayscale"))
        if r.filename not in fit_names]
hir = np.array([r.channel == "grayscale" for r in held])


def fit_threshold(x, y):
    """IR = feature below threshold. Accuracy-maximising cut on channel labels only."""
    c = np.unique(x); c = (c[:-1] + c[1:]) / 2
    acc = np.array([((x < t) == y).mean() for t in c])
    return float(np.median(c[acc == acc.max()]))


res = {"selection_rule": "highest fit agreement; held-out agreement reported, not used to choose"}
for name, f in FEATURES.items():
    x = np.array([f(r.path) for r in fit]); hx = np.array([f(r.path) for r in held])
    t = fit_threshold(x, ir)
    res[name] = {
        "threshold": t,
        "fit_agreement": float(((x < t) == ir).mean()),
        "heldout_n": int(len(held)),
        "heldout_agreement": float(((hx < t) == hir).mean()),
        "fit_misrouted": [(r.filename, r.channel, round(v, 2)) for r, v in zip(fit, x) if (v < t) != (r.channel == "grayscale")],
        "heldout_misrouted": [(r.filename, r.channel, round(v, 2)) for r, v in zip(held, hx) if (v < t) != (r.channel == "grayscale")],
    }
res["chosen"] = max(FEATURES, key=lambda n: res[n]["fit_agreement"])
(HERE / "outputs").mkdir(exist_ok=True)
(HERE / "outputs" / "router.json").write_text(json.dumps(res, indent=1))
for n in FEATURES:
    print(n, {k: v for k, v in res[n].items() if "misrouted" not in k}, "fit-miss", len(res[n]["fit_misrouted"]), "held-miss", len(res[n]["heldout_misrouted"]))
print("chosen:", res["chosen"])
