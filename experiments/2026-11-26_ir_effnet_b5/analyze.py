"""Endpoints from PROTOCOL.md. Written and dry-run before any B5 result existed.

Usage: buck-env/Scripts/python.exe analyze.py
       (B5_NPZ=<path> overrides the B5 arm's file, for dry runs only)
"""
import json, os, sys
from datetime import datetime
from pathlib import Path
import cv2, numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from buck.benchmark.metrics import ordinal_metrics

ROUTING = HERE.parent / "2026-11-26_ir_routing" / "outputs"
B = np.load(ROUTING / "baseline.npz")
b5_path = Path(os.environ.get("B5_NPZ", HERE / "outputs" / "b5.npz"))
E = np.load(b5_path)
for k in ("labels", "files", "channels", "groups", "fold"):
    assert (B[k] == E[k]).all(), f"arms disagree on {k}"
if "B5_NPZ" not in os.environ:
    assert json.loads((HERE / "outputs" / "b5_gates.json").read_text())["leak_gate_pass"]

split = json.loads((HERE / "hyp_split.json").read_text())
ages = B["class_ages"].tolist()
y, groups, files = B["labels"], B["groups"], B["files"]
ir = B["channels"] == "grayscale"
conf = np.isin(files, split["confirmatory_ir_dev"])
hyp = ir & ~conf
assert conf.sum() == 26 and hyp.sum() == 26 and not (conf & ~ir).any()

thr = json.loads((ROUTING / "router.json").read_text())["saturation"]["threshold"]
PINNED = ROOT / "benchmark_runs" / "ir_routing" / "pinned_images" / "squared"
pix_ir = np.array([float(cv2.cvtColor(cv2.imread(str(PINNED / c / f)), cv2.COLOR_BGR2HSV)[..., 1].mean()) < thr
                   for f, c in zip(files, B["channels"])])

b = B["oof"].mean(0).argmax(1)
e = E["oof"].mean(0).argmax(1)
b_seed, e_seed = B["oof"].argmax(-1), E["oof"].argmax(-1)
routed = np.where(ir, e, b)
routed_pix = np.where(pix_ir, e, b)


def metrics(pred, mask):
    return {k: float(v) for k, v in ordinal_metrics(y[mask], pred[mask], ages).items()}


def paired(a, c, mask, n=10_000, seed=0):
    a_ok, c_ok = a == y, c == y
    oa, oc = int((a_ok & ~c_ok & mask).sum()), int((~a_ok & c_ok & mask).sum())
    p = binomtest(oc, oa + oc, 0.5).pvalue if oa + oc else 1.0
    d = c_ok.astype(float) - a_ok.astype(float)
    ug = np.unique(groups[mask])
    sums = np.array([d[mask & (groups == u)].sum() for u in ug])
    cnts = np.array([(mask & (groups == u)).sum() for u in ug])
    draws = np.random.default_rng(seed).integers(0, len(ug), (n, len(ug)))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    return {"n": int(mask.sum()), "baseline_acc": float(a_ok[mask].mean()), "new_acc": float(c_ok[mask].mean()),
            "diff": float(d[mask].mean()), "only_baseline_right": oa, "only_new_right": oc,
            "exact_p": float(p), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


allm = np.ones_like(ir)
res = {"created": datetime.now().isoformat(timespec="seconds"), "b5_file": str(b5_path)}
res["primary_confirmatory_ir"] = paired(b, routed, conf)
res["per_seed_confirmatory"] = {str(s): {"baseline": float((b_seed[i][conf] == y[conf]).mean()),
                                         "b5": float((e_seed[i][conf] == y[conf]).mean())}
                                for i, s in enumerate((42, 43, 44))}
res["seeds_improved"] = int(sum(v["b5"] > v["baseline"] for v in res["per_seed_confirmatory"].values()))
res["secondary"] = {
    "all_ir_52": paired(b, routed, ir),
    "hypothesis_generating_ir_26": paired(b, routed, hyp),
    "overall_239": {"baseline": metrics(b, allm), "routed": metrics(routed, allm),
                    "routed_pixel_router": metrics(routed_pix, allm), "paired": paired(b, routed, allm),
                    "paired_pixel_router": paired(b, routed_pix, allm)},
    "b5_on_color_descriptive": paired(b, e, ~ir),
    "b5_alone_overall_descriptive": {"metrics": metrics(e, allm), "paired": paired(b, e, allm)},
    "ir_metrics": {"baseline": metrics(b, ir), "routed": metrics(routed, ir)},
}
o = res["secondary"]["overall_239"]
res["decision"] = {
    "1_confirmatory_diff_positive": res["primary_confirmatory_ir"]["diff"] > 0,
    "2_improved_on_2of3_seeds": res["seeds_improved"] >= 2,
    "3_within_one_not_worse": o["routed"]["within_one"] >= o["baseline"]["within_one"],
    "3_qwk_not_worse": o["routed"]["qwk"] >= o["baseline"]["qwk"],
    "4_leakage_gate": True,  # asserted above
}
res["decision"]["recommend"] = all(res["decision"].values())
res["decision"]["provisional"] = res["decision"]["recommend"] and res["primary_confirmatory_ir"]["exact_p"] >= 0.05

out = HERE / "outputs" / ("results.json" if "B5_NPZ" not in os.environ else "dryrun_results.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
