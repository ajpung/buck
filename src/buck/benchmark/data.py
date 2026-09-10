"""Leak-free data handling for the BUCK architecture benchmark.

Four rules are enforced structurally here, not by convention:

1. Augmentation exists only inside :class:`TrainDataset`. :class:`EvalDataset`
   has no stochastic code path whatsoever -- no random flip, no jitter.
2. Evaluation sets are never oversampled or class-rebalanced. One tensor per
   real image, in a fixed order, so the reported score is an estimate of
   accuracy on the true class mix.
3. Images of the same animal are clustered and the cluster, not the image, is
   the unit of splitting. The corpus is drawn partly from video, so one buck
   can appear as several frames of one clip; a perceptual hash does not catch
   those, so clustering is done in feature space. See :func:`build_groups`.
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

import csv
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

# Cosine similarity above which two same-age images are treated as the same
# animal. A perceptual hash only catches near-identical *pixels*; frames from
# one video clip differ enough in pixels to score Hamming 10-12 while being
# obviously the same buck, so the real duplicate test is done in feature space.
# 0.90 was chosen by inspecting the ranked pair list: every pair above it on
# this corpus is visibly one animal, and the merged-image count is not
# knife-edge around it (19 images at 0.92, 58 at 0.90, 85 at 0.88).
EMBEDDING_MERGE_SIMILARITY = 0.90

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


DEFAULT_METADATA = Path(__file__).resolve().parents[3] / "trail cam" / "image_metadata.csv"
DEFAULT_EMBEDDING_CACHE = (
    Path(__file__).resolve().parents[3] / "trail cam" / "splits" / "duplicate_embeddings.npz"
)


def load_vote_targets(records, class_ages, csv_path=None, verbose=True):
    """Per-image target distribution from the NDA weekly poll.

    The poll records how many voters chose each age class for that week's deer.
    That distribution is a measured statement about how ambiguous the animal
    is, which a one-hot label throws away: an image where 82% of voters said
    1.5 and one where 32% said 3.5 carry very different amounts of evidence.

    One CSV row describes exactly one deer, so it may be attached to exactly
    one image. Rows are keyed on the collection date, which is the weekly
    cadence and is unique per poll -- but four of the collection dates in the
    corpus are bulk archival back-fills holding 23 to 65 images, and nothing
    about the key prevents a row from being broadcast across all of them. No
    such row is present today, so this has never corrupted a run; the guard
    below closes the hole rather than fixing live damage.

    A row is used only when it identifies its image unambiguously: exactly one
    image in that collection batch carries the poll's answer as its label.
    Where a batch offers several candidates the row is dropped and reported,
    because guessing which deer the votes describe would put fabricated
    training signal on the other candidates.

    ``Year_taken``/``Month_taken``/``Day_taken`` and ``Location`` are checked
    against the filename's photo date and state where both exist. Disagreements
    are reported but do not drop the row: all eight in the corpus today are
    transcription slips on a single field (``VA``/``VT``, the collection date
    typed into the photo-date columns) rather than a different animal, so
    keying on them strictly would discard real training signal.

    Returns:
        (targets, has_vote) where ``targets`` is (N, K) float64 -- vote shares
        for images the poll covered, zeros elsewhere -- and ``has_vote`` is an
        (N,) bool mask.
    """
    csv_path = Path(csv_path or DEFAULT_METADATA)
    ages = list(class_ages)
    targets = np.zeros((len(records), len(ages)), dtype=np.float64)
    has_vote = np.zeros(len(records), dtype=bool)
    if not csv_path.exists():
        print(f"[votes] {csv_path} not found; falling back to hard labels")
        return targets, has_vote

    by_week = {}
    duplicate_rows = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            try:
                votes = np.array([float(row[str(a)]) for a in ages])
                answer = float(row["Correct"])
            except (KeyError, TypeError, ValueError):
                continue
            week = row["Collected"].strip()
            if not week or votes.sum() <= 0 or answer not in ages:
                continue
            # Photo date as YYMMDD and state, for the consistency check only.
            try:
                photo = "{:02d}{:02d}{:02d}".format(
                    int(row["Year_taken"]) % 100,
                    int(row["Month_taken"]),
                    int(row["Day_taken"]),
                )
            except (KeyError, TypeError, ValueError):
                photo = None
            state = (row.get("Location") or "").strip().upper() or None

            if week in by_week:
                duplicate_rows += 1
                continue  # keep the first; a second row for one week is unusable
            by_week[week] = (votes / votes.sum(), answer, photo, state)

    # Group candidate images by collection batch before assigning anything, so
    # an ambiguous batch can be recognised as ambiguous.
    candidates = {}
    for i, rec in enumerate(records):
        entry = by_week.get(rec.collected)
        if entry is None:
            continue
        if abs(min(entry[1], MAX_AGE) - rec.age) > 1e-6:
            continue
        candidates.setdefault(rec.collected, []).append(i)

    assigned, ambiguous, mismatched = 0, [], []
    for week, idx in candidates.items():
        votes, answer, photo, state = by_week[week]
        if len(idx) > 1:
            ambiguous.append((week, [records[i].filename for i in idx]))
            continue
        i = idx[0]
        rec = records[i]
        if photo is not None and rec.photo != photo and "U" not in rec.photo:
            mismatched.append((rec.filename, f"photo date {rec.photo} vs CSV {photo}"))
        elif state is not None and rec.state.upper() not in (state, "UU"):
            mismatched.append((rec.filename, f"state {rec.state} vs CSV {state}"))
        targets[i] = votes
        has_vote[i] = True
        assigned += 1

    if verbose:
        skipped = sum(
            1
            for rec in records
            if rec.collected in by_week
            and abs(min(by_week[rec.collected][1], MAX_AGE) - rec.age) > 1e-6
        )
        print(f"[votes] {assigned} of {len(records)} images carry a poll distribution")
        if skipped:
            print(f"        {skipped} image(s) skipped: label disagrees with the "
                  f"poll's recorded answer")
        if duplicate_rows:
            print(f"        {duplicate_rows} CSV row(s) ignored: a second row for a "
                  f"collection date already seen")
        if ambiguous:
            print(f"        {len(ambiguous)} poll row(s) dropped as ambiguous -- the "
                  f"collection batch holds several deer at the answer age:")
            for week, names in ambiguous[:5]:
                print(f"          {week}: {len(names)} candidates, e.g. {names[:2]}")
        if mismatched:
            print(f"        {len(mismatched)} row(s) attached despite disagreeing with "
                  f"the filename (likely CSV transcription errors, worth fixing):")
            for name, why in mismatched:
                print(f"          {name}: {why}")

    return targets, has_vote


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


def _embed(records, cache_path=None, verbose=True):
    """L2-normalised ImageNet features for each record, one row per image.

    Used only to decide which images show the same animal. ResNet-50 is enough
    for that -- the pairs being separated here are near-identical frames, not
    subtle distinctions -- and torchvision is already a dependency, so this
    adds one weight file and no new package.

    Runs on CPU in float32 on purpose. Group ids feed ``StratifiedGroupKFold``,
    and ``ensemble.py`` reconstructs a finished sweep's fold assignment by
    recomputing them, so a borderline pair flipping between a GPU run and a CPU
    run would silently re-partition the data. Results are cached by content
    digest, so the cost is paid once and new weekly images embed incrementally.
    """
    import torchvision.models as tvm

    digests = [_content_digest(r.path) for r in records]
    cached = {}
    if cache_path is not None and Path(cache_path).exists():
        with np.load(cache_path, allow_pickle=False) as blob:
            cached = dict(zip(blob["digests"].tolist(), blob["vectors"]))

    todo = [i for i, d in enumerate(digests) if d not in cached]
    if todo:
        if verbose:
            print(f"[data] embedding {len(todo)} image(s) for duplicate detection "
                  f"({len(digests) - len(todo)} cached)")
        try:
            model = tvm.resnet50(weights="IMAGENET1K_V2")
        except Exception as exc:  # weights absent from disk and no network
            raise RuntimeError(
                "could not load ResNet-50 weights, which are required to group "
                "images of the same animal. The perceptual-hash pass alone "
                "merged 1 pair out of 288 and let same-deer video frames "
                "straddle the train/test wall, so running without this is not "
                f"offered. Original error: {exc}"
            ) from exc
        model.fc = torch.nn.Identity()
        model.eval()

        computed = []
        with torch.no_grad():
            for chunk in range(0, len(todo), 32):
                tensors = []
                for i in todo[chunk:chunk + 32]:
                    image = cv2.imread(records[i].path, cv2.IMREAD_COLOR)
                    if image is None:
                        raise RuntimeError(f"failed to decode {records[i].path}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                    tensors.append(_to_tensor(image))
                computed.append(model(torch.stack(tensors)).numpy())
        for i, vector in zip(todo, np.concatenate(computed)):
            cached[digests[i]] = vector.astype(np.float32)

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            keys = sorted(cached)
            np.savez_compressed(
                cache_path,
                digests=np.array(keys),
                vectors=np.stack([cached[k] for k in keys]),
            )

    features = np.stack([cached[d] for d in digests]).astype(np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def build_groups(
    records,
    merge_distance=PHASH_MERGE_DISTANCE,
    similarity=EMBEDDING_MERGE_SIMILARITY,
    cache_path=DEFAULT_EMBEDDING_CACHE,
    verbose=True,
):
    """Assign a group id to each record, merging images of the same animal.

    The unit of splitting must be the *animal*, not the file. The corpus is
    built partly from video, so one buck can appear as several frames of a
    single clip: same pose, same background, same second. Splitting those
    across the train/test wall hands the model an answer it has already seen.

    Two passes, unioned:

    ``perceptual hash``
        Hamming distance <= ``merge_distance`` over a 64-bit DCT hash. Catches
        byte-level copies and re-encodes.
    ``embedding similarity``
        Cosine similarity > ``similarity`` between ImageNet features, further
        required to agree on age class -- two images of one deer always carry
        the same label, so the constraint costs nothing and stops genuinely
        different bucks that share a pose from being merged.

    The hash pass alone was previously the only one, and it is not sufficient.
    On the 288-image NDA corpus it merges a single pair, while the embedding
    pass finds 24 multi-image clusters covering 58 images (20%); inspecting the
    closest pairs confirms they are consecutive frames of one animal, and the
    hash rates several of them at Hamming 10-12. Eleven of those clusters
    straddled the ``holdout_test_v1`` wall, so held-out numbers reported
    against that manifest were measured with roughly a fifth of the test set
    having a sibling in training.

    Returns:
        ``np.ndarray`` of integer group ids, parallel to ``records``.
    """
    hashes = [phash(r.path) for r in records]
    features = _embed(records, cache_path, verbose)
    ages = np.array([r.age for r in records])

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

    cosine = features @ features.T

    merged = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            reason = None
            if hashes[i] is not None and hashes[j] is not None:
                distance = int(np.count_nonzero(hashes[i] != hashes[j]))
                if distance <= merge_distance:
                    reason = f"hamming={distance}"
            if reason is None and ages[i] == ages[j] and cosine[i, j] > similarity:
                reason = f"cos={cosine[i, j]:.3f}"
            if reason is not None:
                union(i, j)
                merged.append((reason, records[i].filename, records[j].filename))

    roots = {}
    groups = np.empty(len(records), dtype=int)
    for i in range(len(records)):
        groups[i] = roots.setdefault(find(i), len(roots))

    if verbose:
        if merged:
            print(f"[data] merged {len(merged)} same-animal pair(s):")
            for reason, a, b in merged:
                print(f"        {reason:12s} {a}  <->  {b}")
        else:
            print("[data] no near-duplicate images found")
        sizes = np.bincount(groups)
        print(f"[data] {len(records)} images in {len(roots)} groups "
              f"({int((sizes > 1).sum())} group(s) hold more than one image, "
              f"covering {int(sizes[sizes > 1].sum())} images)")

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


def normalisation_arrays(mean=None, std=None):
    """Coerce (mean, std) 3-tuples into (3,1,1) arrays, defaulting to ImageNet.

    Precomputed once per dataset rather than per sample: ``_to_tensor`` runs
    on every image of every epoch.
    """
    mean = IMAGENET_MEAN if mean is None else np.asarray(
        mean, dtype=np.float32).reshape(3, 1, 1)
    std = IMAGENET_STD if std is None else np.asarray(
        std, dtype=np.float32).reshape(3, 1, 1)
    return mean, std


def _to_tensor(image, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """uint8 HWC -> normalised float32 CHW tensor.

    ``mean``/``std`` default to ImageNet, which every torchvision backbone
    wants. They are parameters because timm's CLIP, SigLIP and TF-ported
    Inception/Xception weights were trained under different constants; feeding
    those ImageNet values does not fail, it just quietly costs accuracy and
    looks like the architecture underperforming.
    """
    array = image.astype(np.float32) / 255.0
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    array = array.transpose(2, 0, 1)
    array = (array - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


class TrainDataset(Dataset):
    """Training images with on-the-fly augmentation.

    Length equals the number of real training images. Class imbalance is
    handled by a ``WeightedRandomSampler`` at the DataLoader level rather than
    by inflating the dataset, which keeps "one epoch" a meaningful unit and
    stops minority images being memorised through sheer repetition.
    """

    def __init__(self, images, labels, strength="medium", seed=0,
                 return_index=False, mean=None, std=None):
        if len(images) != len(labels):
            raise ValueError("images and labels differ in length")
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.strength = strength
        self.return_index = return_index
        self._rng = random.Random(seed)
        self.mean, self.std = normalisation_arrays(mean, std)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = augment(self.images[idx].copy(), self.strength, self._rng)
        tensor = _to_tensor(image, self.mean, self.std)
        # The index lets the caller attach a per-sample training target without
        # this class needing to know anything about the loss.
        if self.return_index:
            return tensor, int(self.labels[idx]), int(idx)
        return tensor, int(self.labels[idx])

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

    def __init__(self, images, labels, mean=None, std=None):
        if len(images) != len(labels):
            raise ValueError("images and labels differ in length")
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mean, self.std = normalisation_arrays(mean, std)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return (_to_tensor(self.images[idx], self.mean, self.std),
                int(self.labels[idx]))


def decode_images(records, size, grayscale=False):
    """Decode and resize records into one uint8 array of shape (N, H, W, 3).

    ``grayscale`` collapses every image to BT.601 luma and replicates it across
    all three channels. The array shape and the ImageNet normalisation that
    follows are unchanged, so this is purely a decision to withhold chroma from
    the model -- not a different input pipeline.

    **This did not pay off. Left in place so the result stays measurable, but
    do not turn it on expecting a win.** Measured over 3 seeds on
    ``convnext_tiny`` (``sweep.py rep gray 3`` vs ``rep base 3``):

    ==================  ===============  ===============
    subset              colour input     grayscale input
    ==================  ===============  ===============
    infrared (n=50)     0.533 +/- 0.081  0.600 +/- 0.040
    colour (n=180)      0.717 +/- 0.006  0.680 +/- 0.022
    overall (n=230)     0.677 +/- 0.015  0.662 +/- 0.024
    ==================  ===============  ===============

    Infrared improves (+0.067, up on all three seeds) and colour images lose
    more than infrared gains (-0.037, down on all three seeds). Because colour
    is 78% of the corpus that is a **net loss of ~3.3 images out of 230**.
    Neither subset effect clears p=0.05 on a paired per-image test (IR p=0.098,
    colour p=0.051) and the overall change is null (p=0.331).

    An earlier 3-seed arm measured the colour cost at only -0.002. It did not
    replicate; two independent 3-seed grayscale arms disagree by 3.5 points on
    the colour subset. Note the SDs above -- grayscale training is markedly
    less stable than colour (colour subset +/-0.022 against +/-0.006), so it
    needs more seeds than usual before any grayscale number means anything.

    The infrared deficit itself is real and reproducible (0.533 against 0.717
    on the same runs). Withholding chroma is simply not the fix for it; the
    corpus has only 50 infrared development images and that is the constraint.
    """
    out = np.empty((len(records), size, size, 3), dtype=np.uint8)
    for i, record in enumerate(records):
        image = cv2.imread(record.path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {record.path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if grayscale:
            luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.cvtColor(luma, cv2.COLOR_GRAY2RGB)
        out[i] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return out