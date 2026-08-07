"""Leak-free data handling for the BUCK architecture benchmark.

Four rules are enforced structurally here, not by convention:

1. Augmentation exists only inside :class:`TrainDataset`. :class:`EvalDataset`
   has no stochastic code path whatsoever -- no random flip, no jitter.
2. Evaluation sets are never oversampled or class-rebalanced. One tensor per
   real image, in a fixed order, so the reported score is an estimate of
   accuracy on the true class mix.
3. Near-duplicate images are clustered and the cluster, not the image, is the
   unit of splitting. A duplicate pair can never straddle the train/test wall.
4. The held-out test set is written to a manifest on first creation and reused
   verbatim forever after. New weekly images join the development pool; they
   never silently enter the test set, and the test set never drifts to flatter
   a model.

Filenames carry all metadata and follow::

    <collected>_<photodate>_<state>_<age>_<source>.png
    260226_251020_MO_3p5_NDA.png

``collected`` is the YYMMDD the datapoint was received (the weekly cadence),
``photodate`` is when the trail cam fired (``UUUUUU`` if unknown), ``state`` is
a two-letter code (``UU`` if unknown), ``age`` is years with ``p`` for the
decimal point, and ``source`` is the labelling institution.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold

# Ages at or above this are pooled into a single open-ended top class. Field
# guides stop distinguishing beyond 5.5 and the sample counts get too thin.
MAX_AGE = 5.5

# Images whose perceptual hashes differ by at most this many bits are treated
# as the same animal. 0-2 catches re-encodes and exact copies without merging
# genuinely different deer that happen to share a pose.
PHASH_MERGE_DISTANCE = 2

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class ImageRecord:
    """One labelled trail-cam image."""

    path: str
    collected: str
    photo: str
    state: str
    age: float
    source: str
    channel: str  # 'color' or 'grayscale'

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    @property
    def collected_date(self):
        """``collected`` as a real date, or None if unparseable."""
        return _parse_yymmdd(self.collected)


def _parse_yymmdd(token: str):
    if not token or len(token) != 6 or not token.isdigit():
        return None
    yy, mm, dd = int(token[:2]), int(token[2:4]), int(token[4:])
    try:
        return date(2000 + yy, mm, dd)
    except ValueError:
        return None


def load_records(image_root, sources=("NDA",), channels=("color", "grayscale")):
    """Discover and parse labelled images under ``image_root``.

    Args:
        image_root: Path to ``trail cam/images/squared``. Only the ``squared``
            tree is read -- ``images/original`` holds an unrelated
            uncropped-image experiment and is deliberately ignored.
        sources: Labelling institutions to accept. Defaults to NDA only, whose
            labels are the project's ground truth.
        channels: Subdirectories to read.

    Returns:
        List of :class:`ImageRecord`, sorted by filename for determinism.
    """
    image_root = Path(image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")

    records, skipped = [], []
    for channel in channels:
        channel_dir = image_root / channel
        if not channel_dir.is_dir():
            continue
        for path in sorted(glob.glob(str(channel_dir / "*.png"))):
            parts = Path(path).stem.split("_")
            if len(parts) < 5:
                skipped.append((path, "filename has fewer than 5 fields"))
                continue

            collected, photo, state, age_token, source = parts[:5]
            if source not in sources:
                continue
            if "p" not in age_token.lower() or "xpx" in age_token.lower():
                skipped.append((path, f"unusable age field {age_token!r}"))
                continue
            try:
                age = float(age_token.lower().replace("p", "."))
            except ValueError:
                skipped.append((path, f"unparseable age {age_token!r}"))
                continue

            records.append(
                ImageRecord(
                    path=path,
                    collected=collected,
                    photo=photo,
                    state=state,
                    age=min(age, MAX_AGE),
                    source=source,
                    channel=channel,
                )
            )

    if skipped:
        print(f"[data] skipped {len(skipped)} file(s) with unusable names:")
        for path, why in skipped[:10]:
            print(f"        {os.path.basename(path)}: {why}")

    records.sort(key=lambda r: r.filename)
    return records


def drop_rare_classes(records, min_count=8):
    """Remove age classes too rare to appear in every CV fold.

    A class with fewer members than the fold count cannot be stratified, and a
    class with a handful of members produces per-fold scores dominated by
    sampling noise. Dropping is reported loudly because it changes the task.
    """
    counts = {}
    for r in records:
        counts[r.age] = counts.get(r.age, 0) + 1

    keep = {age for age, n in counts.items() if n >= min_count}
    dropped = {age: n for age, n in counts.items() if age not in keep}
    if dropped:
        print(
            f"[data] dropping age classes with < {min_count} images: "
            + ", ".join(f"{a}yr (n={n})" for a, n in sorted(dropped.items()))
        )
    return [r for r in records if r.age in keep]


def phash(path, hash_size=8):
    """64-bit DCT perceptual hash of an image's luminance."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (hash_size * 4, hash_size * 4)).astype(np.float32)
    dct = cv2.dct(resized)[:hash_size, :hash_size].flatten()
    # Exclude the DC term from the median so overall brightness doesn't
    # dominate the threshold.
    return dct > np.median(dct[1:])


def build_groups(records, merge_distance=PHASH_MERGE_DISTANCE, verbose=True):
    """Assign a group id to each record, merging near-duplicate images.

    Each weekly datapoint is a distinct animal, so the default group is the
    image itself. This step exists to catch the case where the same photo was
    entered twice under different filenames -- splitting such a pair across the
    train/test wall would leak an exact answer.

    Returns:
        ``np.ndarray`` of integer group ids, parallel to ``records``.
    """
    hashes = [phash(r.path) for r in records]

    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    merged = []
    for i in range(len(records)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(records)):
            if hashes[j] is None:
                continue
            distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            if distance <= merge_distance:
                union(i, j)
                merged.append((distance, records[i].filename, records[j].filename))

    roots = {}
    groups = np.empty(len(records), dtype=int)
    for i in range(len(records)):
        root = find(i)
        groups[i] = roots.setdefault(root, len(roots))

    if verbose:
        if merged:
            print(f"[data] merged {len(merged)} near-duplicate pair(s) into groups:")
            for distance, a, b in merged:
                print(f"        hamming={distance}  {a}  <->  {b}")
        else:
            print("[data] no near-duplicate images found")
        print(f"[data] {len(records)} images in {len(roots)} groups")

    return groups


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _content_digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def load_or_create_holdout(
    records, groups, manifest_path, test_fraction=0.2, seed=1337
):
    """Return boolean mask marking the locked held-out test images.

    On first call this carves a group-aware, age-stratified test set and writes
    it to ``manifest_path``. On every later call it reads that manifest back,
    so the test set is byte-identical across architectures, across sweeps, and
    across weeks. That permanence is what makes the number trustworthy: a model
    cannot be selected against a target that was fixed before it existed.

    Images added after the manifest was written are placed in the development
    pool. To fold new data into the test set you must delete the manifest
    deliberately, which invalidates comparisons against earlier runs.
    """
    manifest_path = Path(manifest_path)
    filenames = [r.filename for r in records]
    ages = np.array([r.age for r in records])
    # Stratification needs discrete class codes, not the float ages.
    class_ages = sorted(set(ages.tolist()))
    codes = np.array([class_ages.index(a) for a in ages.tolist()])

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        held = set(manifest["test_files"])

        known = set(filenames)
        missing = held - known
        if missing:
            raise RuntimeError(
                f"{len(missing)} image(s) in the test manifest are no longer on "
                f"disk, e.g. {sorted(missing)[:3]}. The locked test set is "
                f"broken; restore the files or delete {manifest_path} and "
                f"accept that results are no longer comparable to earlier runs."
            )

        # Detect edited images: same name, different bytes.
        digests = manifest.get("test_digests", {})
        changed = [
            r.filename
            for r in records
            if r.filename in held
            and r.filename in digests
            and _content_digest(r.path) != digests[r.filename]
        ]
        if changed:
            raise RuntimeError(
                f"test image(s) changed on disk since the manifest was written: "
                f"{changed[:3]}. Restore them or delete {manifest_path}."
            )

        mask = np.array([f in held for f in filenames])
        print(
            f"[split] reusing locked test set from {manifest_path.name}: "
            f"{mask.sum()} test / {(~mask).sum()} dev"
        )
        added = len(records) - int(manifest.get("n_total", len(records)))
        if added > 0:
            print(
                f"        {added} image(s) added since the manifest was written; "
                f"all joined the development pool, as intended"
            )
        return mask

    # First run: carve the test set and freeze it.
    n_splits = max(2, int(round(1.0 / test_fraction)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_idx = next(splitter.split(np.zeros(len(records)), codes, groups))

    mask = np.zeros(len(records), dtype=bool)
    mask[test_idx] = True

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": None,  # filled by caller if desired; kept out for reproducibility
        "seed": seed,
        "test_fraction": test_fraction,
        "n_total": len(records),
        "test_files": sorted(r.filename for r, m in zip(records, mask) if m),
        "test_digests": {
            r.filename: _content_digest(r.path) for r, m in zip(records, mask) if m
        },
        "class_distribution": {
            str(age): int(((ages == age) & mask).sum()) for age in sorted(set(ages))
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"[split] created and LOCKED a new test set of {mask.sum()} images -> "
        f"{manifest_path}"
    )
    print("        This file must be kept under version control. Deleting it "
          "invalidates comparison against every earlier run.")
    return mask


def assert_no_leakage(train_idx, val_idx, test_idx, groups, records):
    """Fail loudly if any index or group appears on both sides of a wall.

    Cheap insurance. A silent overlap here is the difference between a real
    62% and a fictional 80%.
    """
    splits = {"train": np.asarray(train_idx), "val": np.asarray(val_idx)}
    if test_idx is not None:
        splits["test"] = np.asarray(test_idx)

    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]

            shared_idx = np.intersect1d(splits[a], splits[b])
            if shared_idx.size:
                raise AssertionError(
                    f"{a}/{b} share {shared_idx.size} image(s), e.g. "
                    f"{[records[k].filename for k in shared_idx[:3]]}"
                )

            shared_groups = np.intersect1d(groups[splits[a]], groups[splits[b]])
            if shared_groups.size:
                examples = [
                    records[k].filename
                    for k in splits[a]
                    if groups[k] in shared_groups[:1]
                ][:2]
                raise AssertionError(
                    f"{a}/{b} share {shared_groups.size} duplicate-group(s); "
                    f"near-identical images would straddle the split, e.g. {examples}"
                )


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def augment(image, strength="medium", rng=random):
    """Photometric and small-geometric augmentation. Training use only.

    Deliberately conservative on geometry: a deer's body proportions are the
    signal, so aggressive scaling or shear would destroy the label.
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    presets = {
        "light": dict(rot=0.5, flip=0.3, bright=0.6, gamma=0.2, noise=0.1,
                      gray=0.15, rot_range=8, bright_range=(0.85, 1.15)),
        "medium": dict(rot=0.7, flip=0.5, bright=0.8, gamma=0.4, noise=0.3,
                       gray=0.30, rot_range=12, bright_range=(0.75, 1.25)),
        "heavy": dict(rot=0.8, flip=0.6, bright=0.9, gamma=0.5, noise=0.4,
                      gray=0.40, rot_range=18, bright_range=(0.65, 1.35)),
    }
    p = presets[strength]

    if rng.random() < p["rot"]:
        angle = rng.uniform(-p["rot_range"], p["rot_range"])
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
        )

    if rng.random() < p["flip"]:
        image = cv2.flip(image, 1)

    # The corpus mixes colour and IR-grayscale captures, so randomly
    # desaturating teaches invariance to the capture mode rather than noise.
    if image.ndim == 3 and image.shape[2] == 3 and rng.random() < p["gray"]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if rng.random() < p["bright"]:
        alpha = rng.uniform(*p["bright_range"])
        beta = rng.randint(-20, 20)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if rng.random() < p["gamma"]:
        gamma = rng.uniform(0.85, 1.15)
        table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(256)]
        ).astype(np.uint8)
        image = cv2.LUT(image, table)

    if rng.random() < p["noise"]:
        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def _to_tensor(image):
    """uint8 HWC -> normalised float32 CHW tensor."""
    array = image.astype(np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array.transpose(2, 0, 1)
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


class TrainDataset(Dataset):
    """Training images with on-the-fly augmentation.

    Length equals the number of real training images. Class imbalance is
    handled by a ``WeightedRandomSampler`` at the DataLoader level rather than
    by inflating the dataset, which keeps "one epoch" a meaningful unit and
    stops minority images being memorised through sheer repetition.
    """

    def __init__(self, images, labels, strength="medium", seed=0):
        if len(images) != len(labels):
            raise ValueError("images and labels differ in length")
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.strength = strength
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = augment(self.images[idx].copy(), self.strength, self._rng)
        return _to_tensor(image), int(self.labels[idx])

    def class_weights(self):
        """Per-sample weights that equalise class frequency during sampling."""
        counts = np.bincount(self.labels, minlength=self.labels.max() + 1)
        per_class = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
        return per_class[self.labels]


class EvalDataset(Dataset):
    """Validation and test images. Deterministic and untouched.

    There is intentionally no augmentation parameter, no flip, and no
    resampling: exactly one tensor per real image, always in the same order.
    Anything that would make an evaluation score depend on a random draw is
    absent by construction rather than by a disabled flag.
    """

    def __init__(self, images, labels):
        if len(images) != len(labels):
            raise ValueError("images and labels differ in length")
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return _to_tensor(self.images[idx]), int(self.labels[idx])


def decode_images(records, size):
    """Decode and resize records into one uint8 array of shape (N, H, W, 3)."""
    out = np.empty((len(records), size, size, 3), dtype=np.uint8)
    for i, record in enumerate(records):
        image = cv2.imread(record.path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {record.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return out