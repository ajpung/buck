"""Endpoints from PROTOCOL.md. Reads outputs/{baseline,gray}.npz and router.json.

Usage: buck-env/Scripts/python.exe analyze.py
"""
import csv, json, sys
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
import os
OUT = Path(os.environ.get("IR_OUT", HERE / "outputs"))  # override only for dry runs
sys.path.insert(0, str(ROOT / "src"))
from buck.benchmark.metrics import ordinal_metrics

B, G = np.load(OUT / "baseline.npz"), np.load(OUT / "gray.npz")
for k in ("labels", "files", "channels", "groups", "fold", "late_files"):
    assert (B[k] == G[k]).all(), f"arms disagree on {k}"
for arm in ("baseline", "gray"):
    gates = json.loads((OUT / f"{arm}_gates.json").read_text())
    assert gates["leak_gate_pass"], f"{arm} failed the leakage gate"

ages = B["class_ages"].tolist()
y, groups, files = B["labels"], B["groups"], B["files"]
ir = B["channels"] == "grayscale"
router = json.loads((OUT / "router.json").read_text())
thr = router["saturation"]["threshold"]
PINNED = ROOT / "benchmark_runs" / "ir_routing" / "pinned_images" / "squared"
LIVE = ROOT / "trail cam" / "images" / "squared"


def sat(path):
    return float(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2HSV)[..., 1].mean())


pix_ir = np.array([sat(PINNED / ch / f) < thr for f, ch in zip(files, B["channels"])])

b_seed, g_seed = B["oof"].argmax(-1), G["oof"].argmax(-1)       # (seed, n)
b, g = B["oof"].mean(0).argmax(1), G["oof"].mean(0).argmax(1)  # 3-seed ensembles
routed = np.where(ir, g, b)
routed_pix = np.where(pix_ir, g, b)


def metrics(pred, mask):
    return {k: float(v) for k, v in ordinal_metrics(y[mask], pred[mask], ages).items()}


def mcnemar(a, c, mask):
    a_ok, c_ok = a[mask] == y[mask], c[mask] == y[mask]
    only_a, only_c = int((a_ok & ~c_ok).sum()), int((~a_ok & c_ok).sum())
    p = binomtest(only_c, only_a + only_c, 0.5).pvalue if only_a + only_c else 1.0
    return {"only_baseline_right": only_a, "only_new_right": only_c, "exact_p": float(p)}


def cluster_boot(a, c, mask, n=10_000, seed=0):
    """Accuracy difference (c - a) on `mask`, resampling animal groups."""
    rng = np.random.default_rng(seed)
    d = ((c == y).astype(float) - (a == y).astype(float))
    ug = np.unique(groups[mask])
    by = {u: np.flatnonzero(mask & (groups == u)) for u in ug}
    sums = np.array([d[by[u]].sum() for u in ug]); cnts = np.array([len(by[u]) for u in ug])
    draws = rng.integers(0, len(ug), (n, len(ug)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    return {"diff": float(d[mask].mean()), "ci95": [float(np.percentile(boot, 2.5)),
                                                   float(np.percentile(boot, 97.5))], "n_groups": int(len(ug))}


all_ = np.ones_like(ir)
res = {"created": datetime.now().isoformat(timespec="seconds"), "n_dev": int(len(y)),
       "n_ir": int(ir.sum()), "n_color": int((~ir).sum())}

res["primary_ir"] = {"baseline": metrics(b, ir), "routed": metrics(routed, ir),
                     "mcnemar": mcnemar(b, routed, ir), "cluster_bootstrap": cluster_boot(b, routed, ir)}
res["overall"] = {"baseline": metrics(b, all_), "routed": metrics(routed, all_),
                  "routed_pixel_router": metrics(routed_pix, all_),
                  "mcnemar": mcnemar(b, routed, all_), "cluster_bootstrap": cluster_boot(b, routed, all_),
                  "pixel_router_mcnemar": mcnemar(b, routed_pix, all_),
                  "pixel_router_bootstrap": cluster_boot(b, routed_pix, all_)}
res["per_seed_ir_accuracy"] = {
    str(s): {"baseline": float((b_seed[i][ir] == y[ir]).mean()), "gray": float((g_seed[i][ir] == y[ir]).mean())}
    for i, s in enumerate((42, 43, 44))}
res["seeds_ir_improved"] = int(sum(v["gray"] > v["baseline"] for v in res["per_seed_ir_accuracy"].values()))
res["gray_arm_on_color_descriptive"] = {"baseline": metrics(b, ~ir), "gray": metrics(g, ~ir),
                                        "mcnemar": mcnemar(b, g, ~ir)}
res["pixel_router_on_dev"] = {"ir_sent_to_color_model": int((ir & ~pix_ir).sum()),
                              "color_sent_to_gray_model": int((~ir & pix_ir).sum())}

o, rp = res["overall"], res["primary_ir"]
res["decision"] = {
    "1_ir_diff_positive": rp["cluster_bootstrap"]["diff"] > 0,
    "2_ir_improved_on_2of3_seeds": res["seeds_ir_improved"] >= 2,
    "3_within_one_not_worse": o["routed"]["within_one"] >= o["baseline"]["within_one"],
    "3_qwk_not_worse": o["routed"]["qwk"] >= o["baseline"]["qwk"],
    "4_leakage_gates": True,  # asserted above; pin gate documented in RUN_LOG
}
res["decision"]["recommend"] = all(res["decision"].values())
res["decision"]["provisional"] = res["decision"]["recommend"] and rp["mcnemar"]["exact_p"] >= 0.05

# ---- secondary: weekly website deer (dev + post-pin only; test images not read)
pred_of = {f: (ages[b[i]], ages[g[i]], ages[routed[i]], ages[routed_pix[i]], B["channels"][i], "dev")
           for i, f in enumerate(files)}
for j, f in enumerate(B["late_files"]):
    ch = B["late_channels"][j]
    pb, pg = ages[B["late"][j].argmax()], ages[G["late"][j].argmax()]
    pir = sat(LIVE / ch / f) < thr
    pred_of[f] = (pb, pg, pg if ch == "grayscale" else pb, pg if pir else pb, ch, "post-pin")


def d(s):
    return datetime.strptime(s, "%y%m%d")


rows, skipped = [], []
with open(ROOT / "trail cam" / "image_metadata.csv", encoding="utf-8-sig") as fh:
    for x in csv.DictReader(fh):
        c = x["Collected"].strip()
        if not c or c < "250724" or not x["Correct"].strip() or not x["AgeMyDeer"].strip():
            continue
        truth = min(float(x["Correct"]), 5.5)
        cands = [(abs((d(f[:6]) - d(c)).days), f) for f, p in pred_of.items()
                 if abs((d(f[:6]) - d(c)).days) <= 4 and float(f.split("_")[3].replace("p", ".")) == truth]
        if not cands:
            skipped.append(f"{c} {x['Location']}")
            continue
        f = min(cands)[1]
        rows.append((c, f, truth, float(x["AgeMyDeer"]), *pred_of[f]))
n = len(rows)


def score(k, subset=None):
    rs = [r for r in rows if subset is None or r[8] == subset]
    if not rs:
        return None
    return {"n": len(rs), "exact": sum(r[2] == r[k] for r in rs),
            "within_one": sum(abs(r[2] - r[k]) <= 1 for r in rs),
            "mae": round(sum(abs(r[2] - r[k]) for r in rs) / len(rs), 3)}


res["weekly_deer"] = {
    "note": "excludes CSV rows whose image is in the locked test set (not read) or has no image",
    "unmatched_or_test": skipped,
    **{name: {"all": score(k), "color": score(k, "color"), "ir": score(k, "grayscale")}
       for name, k in (("website", 3), ("baseline", 4), ("gray_arm", 5), ("routed", 6), ("routed_pixel", 7))},
    "rows": rows,
}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps({k: v for k, v in res.items() if k != "weekly_deer"}, indent=1, default=str))
print(json.dumps({k: v for k, v in res["weekly_deer"].items() if k != "rows"}, indent=1))
