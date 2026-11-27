"""Score every checkpoint of one arm on every development image.

From the full (seed x checkpoint x image) probability tensor we derive both the
out-of-fold predictions (checkpoint k on fold k) and the leakage matrix
(checkpoint k on every fold). Also predicts the post-pin weekly images (never
trained on by either arm) with all 30 checkpoints.

Never decodes a locked-test image.

Usage:
  buck-env/Scripts/python.exe oof.py --arm baseline --verify
  buck-env/Scripts/python.exe oof.py --arm gray
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, sklearn, torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from buck.benchmark import architectures as arch
from buck.benchmark.data import (assert_no_leakage, build_groups, decode_images,
                                 drop_rare_classes, load_or_create_holdout, load_records)
from buck.benchmark.ensemble import _probabilities

assert sklearn.__version__ == "1.7.1", f"run under buck-env (sklearn {sklearn.__version__})"

PINNED = ROOT / "benchmark_runs" / "ir_routing" / "pinned_images" / "squared"
LIVE = ROOT / "trail cam" / "images" / "squared"
MANIFEST = ROOT / "trail cam" / "splits" / "holdout_test_v2.json"
ARMS = {"baseline": (ROOT / "benchmark_runs" / "nda_dino_yolo" / "final", False),
        "gray": (ROOT / "benchmark_runs" / "ir_routing" / "gray", True)}
MODEL, SEEDS, FOLDS, SPLIT_SEED = "vit_base_dinov3", (42, 43, 44), 10, 1337
RECORDED = {42: 0.682, 43: 0.653, 44: 0.678, "ensemble": 0.669}  # executed.ipynb


def split():
    records = drop_rare_classes(load_records(PINNED, ("NDA",), ("color", "grayscale")), 8)
    class_ages = sorted({r.age for r in records})
    labels = np.array([class_ages.index(r.age) for r in records])
    groups = build_groups(records, verbose=False)
    test_mask = load_or_create_holdout(records, groups, MANIFEST, seed=SPLIT_SEED)
    test_idx, dev_idx = np.flatnonzero(test_mask), np.flatnonzero(~test_mask)
    tg = set(groups[test_idx].tolist())
    dev_idx = dev_idx[~np.array([g in tg for g in groups[dev_idx]])]
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SPLIT_SEED)
                 .split(np.zeros(len(dev_idx)), labels[dev_idx], groups[dev_idx]))
    for tr, va in folds:
        assert_no_leakage(dev_idx[tr], dev_idx[va], test_idx, groups, records)
    assert (len(records), len(dev_idx), len(test_idx)) == (297, 239, 58)
    return records, class_ages, labels, groups, dev_idx, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    run_dir, gray = ARMS[a.arm]

    records, class_ages, labels, groups, dev_idx, folds = split()
    dev = [records[i] for i in dev_idx]
    y = labels[dev_idx]
    fold_of = np.empty(len(dev_idx), int)
    for k, (_, va) in enumerate(folds):
        fold_of[va] = k

    pinned_names = {r.filename for r in records}
    late = [r for r in drop_rare_classes(load_records(LIVE, ("NDA",), ("color", "grayscale")), 8)
            if r.filename not in pinned_names]

    size, (mean, std) = arch.input_size(MODEL), arch.normalisation(MODEL)
    imgs = decode_images(dev, size, grayscale=gray)
    late_imgs = decode_images(late, size, grayscale=gray) if late else None
    device = torch.device("cuda")

    full = np.zeros((len(SEEDS), FOLDS, len(dev), len(class_ages)), np.float32)
    late_p = np.zeros((len(SEEDS), FOLDS, len(late), len(class_ages)), np.float32)
    for s, seed in enumerate(SEEDS):
        for k in range(FOLDS):
            path = run_dir / f"{MODEL}_s{seed}" / "checkpoints" / MODEL / f"fold{k + 1}.pth"
            ck = torch.load(path, map_location="cpu")
            assert ck.get("fold") in (None, k + 1), (path, ck.get("fold"))
            m = arch.build_model(MODEL, len(class_ages), pretrained=False)
            m.load_state_dict(ck["model_state_dict"]); m.to(device)
            full[s, k] = _probabilities(m, imgs, y, 32, device, False, mean, std)
            if late:
                late_p[s, k] = _probabilities(m, late_imgs, np.zeros(len(late), int), 32,
                                              device, False, mean, std)
            del m, ck; torch.cuda.empty_cache()
            print(f"{a.arm} seed {seed} fold {k + 1}", flush=True)

    oof = full[:, fold_of, np.arange(len(dev))]          # (seed, image, class)
    acc = {seed: float((oof[s].argmax(1) == y).mean()) for s, seed in enumerate(SEEDS)}
    acc["ensemble"] = float((oof.mean(0).argmax(1) == y).mean())
    # leakage matrix: accuracy of checkpoint k on fold j, averaged over seeds
    pred = full.argmax(-1)
    mat = np.array([[(pred[:, k, fold_of == j] == y[fold_of == j]).mean() for j in range(FOLDS)]
                    for k in range(FOLDS)])
    diag, off = np.diag(mat).mean(), mat[~np.eye(FOLDS, dtype=bool)].mean()
    print("OOF accuracy:", {k: round(v, 3) for k, v in acc.items()})
    print(f"leakage matrix: own-fold {diag:.3f} vs other folds {off:.3f}")

    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"{a.arm}.npz", oof=oof, labels=y, fold=fold_of,
                        groups=groups[dev_idx], late=late_p.mean((0, 1)),
                        files=np.array([r.filename for r in dev]),
                        channels=np.array([r.channel for r in dev]),
                        late_files=np.array([r.filename for r in late]),
                        late_channels=np.array([r.channel for r in late]),
                        late_ages=np.array([r.age for r in late]),
                        class_ages=np.array(class_ages), leak_matrix=mat)
    gates = {"oof_accuracy": acc, "leak_own_fold": float(diag), "leak_other_folds": float(off),
             "leak_gate_pass": bool(off - diag > 0.15 and abs(diag - np.mean([acc[k] for k in SEEDS])) < 0.10)}
    if a.verify:
        gates["pin_gate_pass"] = all(abs(acc[k] - v) < 0.0015 for k, v in RECORDED.items())
    (out / f"{a.arm}_gates.json").write_text(json.dumps(gates, indent=1))
    print(json.dumps(gates, indent=1))
    if not all(v for k, v in gates.items() if k.endswith("pass")):
        raise SystemExit("GATE FAILED")


if __name__ == "__main__":
    main()
