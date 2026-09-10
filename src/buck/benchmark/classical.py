"""Non-CNN baselines: classical classifiers on deep and hand-crafted features.

The original project reported "20 canned classifiers" alongside the CNNs, but
those numbers predate every leak fix in this package -- they were measured
without same-animal grouping, without a locked test set, and on evaluation
sets that were oversampled to equal class counts. They are not comparable to
anything current, so this module re-measures them under the benchmark's own
splits.

Comparability is the whole point, so nothing here re-derives its own split.
``rebuild_splits`` reconstructs the identical ``StratifiedGroupKFold``
assignment the CNN sweep uses, from the same records, the same groups and the
same seed, and every score is the same :func:`ordinal_metrics` averaged the
same way. A row here can be read directly against a row in a
``compare_architectures`` leaderboard.

Two families of input:

``deep_*``
    Pooled features from a frozen pretrained backbone. The CNN arms fine-tune
    the backbone; these do not, so the gap between them measures what
    fine-tuning buys on ~184 training images.

everything else
    Hand-crafted descriptors with no learned component at all -- raw pixels,
    colour histograms, HOG, LBP. These establish the floor: whatever a
    gradient-boosted tree on colour histograms scores is what the pixels give
    up without any representation learning.

Usage::

    python -m buck.benchmark.classical --output benchmark_runs/mega_c
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from buck.benchmark.data import (
    build_groups,
    decode_images,
    drop_rare_classes,
    load_records,
    normalisation_arrays,
)
from buck.benchmark.ensemble import rebuild_splits
from buck.benchmark.metrics import ordinal_metrics

DEFAULT_IMAGE_ROOT = (
    Path(__file__).resolve().parents[3] / "trail cam" / "images" / "squared"
)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "trail cam" / "splits"
    / "holdout_test_v2.json"
)

# Backbones used as frozen feature extractors. Named by timm id so the
# supervised and self-supervised ConvNeXt-Tiny pair stays exactly comparable:
# same architecture, same width, different pretraining.
DEEP_BACKBONES = {
    "deep_resnet50": "resnet50.tv_in1k",
    "deep_convnext_tiny": "convnext_tiny.fb_in1k",
    "deep_convnext_dinov3": "convnext_tiny.dinov3_lvd1689m",
}


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _gray(images, size):
    out = np.empty((len(images), size, size), dtype=np.uint8)
    for i, img in enumerate(images):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        out[i] = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return out


def feat_pixels32(images):
    """Raw downsampled luma. The crudest possible representation."""
    return _gray(images, 32).reshape(len(images), -1).astype(np.float32) / 255.0


def feat_colorhist(images):
    """Per-channel RGB and HSV histograms, L1-normalised."""
    out = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        parts = []
        for src in (img, hsv):
            for channel in range(3):
                hist = cv2.calcHist([src], [channel], None, [64], [0, 256]).ravel()
                parts.append(hist / max(hist.sum(), 1.0))
        out.append(np.concatenate(parts))
    return np.asarray(out, dtype=np.float32)


def feat_hog(images):
    from skimage.feature import hog

    grays = _gray(images, 128)
    return np.asarray(
        [
            hog(g, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True)
            for g in grays
        ],
        dtype=np.float32,
    )


def feat_lbp(images):
    from skimage.feature import local_binary_pattern

    grays = _gray(images, 128)
    out = []
    for g in grays:
        lbp = local_binary_pattern(g, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        out.append(hist)
    return np.asarray(out, dtype=np.float32)


def feat_handcrafted(images):
    """Everything hand-built, concatenated."""
    return np.concatenate(
        [feat_colorhist(images), feat_hog(images), feat_lbp(images)], axis=1
    )


HANDCRAFTED = {
    "pixels32": feat_pixels32,
    "colorhist": feat_colorhist,
    "hog": feat_hog,
    "lbp": feat_lbp,
    "handcrafted": feat_handcrafted,
}


@torch.no_grad()
def deep_features(timm_id, records, size, device, batch=32):
    """Pooled features from a frozen pretrained backbone.

    Normalisation is read from the checkpoint's own config rather than
    assumed -- the same trap ``architectures.normalisation`` exists to close.
    """
    import timm

    model = timm.create_model(timm_id, pretrained=True, num_classes=0)
    cfg = model.pretrained_cfg or {}
    mean, std = normalisation_arrays(cfg.get("mean"), cfg.get("std"))
    model.eval().to(device)

    images = decode_images(records, size)
    out = []
    for start in range(0, len(images), batch):
        chunk = images[start:start + batch].astype(np.float32) / 255.0
        chunk = chunk.transpose(0, 3, 1, 2)
        chunk = (chunk - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        out.append(model(tensor.to(device)).float().cpu().numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------
def build_classifiers(seed):
    """The canned-classifier field, plus a majority-class floor.

    ``DummyClassifier`` is included deliberately: on a 5-class problem whose
    largest class is 30% of the corpus, several of these will not beat it, and
    a leaderboard without that line makes 0.30 look like a result.
    """
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        AdaBoostClassifier,
        BaggingClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import (
        LogisticRegression,
        RidgeClassifier,
        SGDClassifier,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    classifiers = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=2000, C=1.0),
        "logreg_l1": LogisticRegression(
            max_iter=2000, penalty="l1", solver="liblinear", C=1.0
        ),
        "ridge": RidgeClassifier(),
        "sgd_hinge": SGDClassifier(loss="hinge", max_iter=3000, random_state=seed),
        "linear_svc": LinearSVC(max_iter=5000),
        "svc_rbf": SVC(kernel="rbf", C=10.0, gamma="scale"),
        "svc_poly": SVC(kernel="poly", degree=3, C=10.0, gamma="scale"),
        "knn5": KNeighborsClassifier(n_neighbors=5),
        "knn15": KNeighborsClassifier(n_neighbors=15),
        "gaussian_nb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "grad_boost": GradientBoostingClassifier(random_state=seed),
        "hist_gb": HistGradientBoostingClassifier(random_state=seed),
        "adaboost": AdaBoostClassifier(random_state=seed),
        "bagging": BaggingClassifier(n_estimators=50, random_state=seed, n_jobs=-1),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128), max_iter=1500, random_state=seed
        ),
    }

    try:
        from xgboost import XGBClassifier

        classifiers["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, tree_method="hist",
            verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=400, learning_rate=0.05, random_state=seed, verbose=-1
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        classifiers["catboost"] = CatBoostClassifier(
            iterations=400, depth=4, learning_rate=0.05, random_seed=seed,
            verbose=0, allow_writing_files=False,
        )
    except ImportError:
        pass

    return classifiers


def make_pipeline(clf, n_features, n_train, seed):
    """Scale always; reduce only when the feature count dwarfs the sample count.

    With ~184 training rows a 2048-d deep feature is a badly underdetermined
    problem for most of these classifiers. PCA to at most 128 components keeps
    them tractable, and it is applied identically to every classifier so the
    comparison between them stays fair.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("scale", StandardScaler())]
    if n_features > 256:
        steps.append(
            ("pca", PCA(n_components=min(128, n_features, n_train - 1),
                        random_state=seed))
        )
    steps.append(("clf", clf))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(features, labels, dev_idx, fold_list, class_ages, seed, verbose):
    """Score every classifier on one feature set over the shared folds."""
    rows = []
    for clf_name, clf in build_classifiers(seed).items():
        started = time.time()
        per_fold = []
        try:
            for train_pos, val_pos in fold_list:
                train_idx = dev_idx[train_pos]
                val_idx = dev_idx[val_pos]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe = make_pipeline(
                        clf, features.shape[1], len(train_idx), seed
                    )
                    pipe.fit(features[train_idx], labels[train_idx])
                    # CatBoost returns (n, 1); everything else returns (n,).
                    pred = np.asarray(pipe.predict(features[val_idx])).ravel()
                per_fold.append(ordinal_metrics(labels[val_idx], pred, class_ages))
        except Exception as exc:
            # One classifier failing must not take the sweep down with it.
            print(f"      {clf_name}: FAILED ({type(exc).__name__}: {exc})")
            continue

        row = {
            "classifier": clf_name,
            "minutes": (time.time() - started) / 60.0,
            "folds": len(per_fold),
        }
        for key in ("accuracy", "within_one", "qwk", "mae_years", "macro_f1"):
            values = [m[key] for m in per_fold]
            row[f"cv_{key}"] = float(np.mean(values))
            row[f"cv_{key}_sd"] = float(np.std(values))
        rows.append(row)
        if verbose:
            print(f"      {clf_name:16s} acc={row['cv_accuracy']:.3f} "
                  f"qwk={row['cv_qwk']:.3f}", flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classical-classifier arm.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sources", nargs="+", default=["NDA"])
    parser.add_argument("--channels", nargs="+", default=["color", "grayscale"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-class-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--features", nargs="+", default=None,
                        help="subset of feature sets; default is all of them")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device {device}")

    records = drop_rare_classes(
        load_records(args.image_root, tuple(args.sources), tuple(args.channels)),
        args.min_class_count,
    )
    class_ages = sorted({r.age for r in records})
    groups = build_groups(records, verbose=False)
    labels, dev_idx, test_idx, fold_list = rebuild_splits(
        records, groups, class_ages, args.manifest, args.folds,
        args.split_seed, args.test_fraction,
    )
    print(f"[data] {len(records)} images, {len(class_ages)} classes, "
          f"{len(dev_idx)} dev / {len(test_idx)} test, {args.folds} folds")
    print("[data] fold assignment rebuilt from the same seed as the CNN sweep")

    wanted = args.features or (list(HANDCRAFTED) + list(DEEP_BACKBONES))
    raw = decode_images(records, args.image_size)

    leaderboard = []
    for feat_name in wanted:
        print(f"\n{'=' * 70}\n{feat_name}\n{'=' * 70}", flush=True)
        started = time.time()
        if feat_name in HANDCRAFTED:
            features = HANDCRAFTED[feat_name](raw)
        elif feat_name in DEEP_BACKBONES:
            features = deep_features(
                DEEP_BACKBONES[feat_name], records, args.image_size, device
            )
        else:
            print(f"   unknown feature set {feat_name!r}, skipping")
            continue
        print(f"   features {features.shape} in "
              f"{(time.time() - started) / 60:.1f} min", flush=True)

        for row in evaluate(features, labels, dev_idx, fold_list, class_ages,
                            args.seed, not args.quiet):
            row["model"] = f"{feat_name}+{row['classifier']}"
            row["features"] = feat_name
            row["n_features"] = int(features.shape[1])
            row["input_size"] = args.image_size
            leaderboard.append(row)

    leaderboard.sort(key=lambda r: -r["cv_qwk"])
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "holdout-classical",
        "created": datetime.now().isoformat(timespec="seconds"),
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "class_ages": class_ages,
        "n_dev": int(len(dev_idx)),
        "n_test": int(len(test_idx)),
        "leaderboard": leaderboard,
        "test": None,
    }
    (args.output / "results.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'=' * 78}")
    print("CLASSICAL LEADERBOARD (development pool only; test set untouched)")
    print("=" * 78)
    print(f"{'feature+classifier':40} {'acc':>7} {'+/-1yr':>7} {'QWK':>7} "
          f"{'macroF1':>8}")
    print("-" * 78)
    for row in leaderboard[:30]:
        print(f"{row['model'][:40]:40} {row['cv_accuracy']:7.3f} "
              f"{row['cv_within_one']:7.3f} {row['cv_qwk']:7.3f} "
              f"{row['cv_macro_f1']:8.3f}")
    print(f"\n[out] {args.output / 'results.json'}  ({len(leaderboard)} rows)")
    return payload


if __name__ == "__main__":
    main()
