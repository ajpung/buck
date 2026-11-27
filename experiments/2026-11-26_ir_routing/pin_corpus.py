"""Copy the exact 297-image record set the 2026-11-08 baseline was trained on.

The live folder keeps growing (weekly `xpx` files are renamed once NDA ages
them, preserving mtime), so the record set is pinned by collection date: the
baseline run's own log shows 261105 was still `xpx` when it ran. Correctness of
the pin is not assumed from this rule -- `oof.py --verify` checks it against the
run's recorded numbers.

Usage: buck-env/Scripts/python.exe pin_corpus.py
"""
import hashlib, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from buck.benchmark.data import drop_rare_classes, load_records

LIVE = ROOT / "trail cam" / "images" / "squared"
PINNED = ROOT / "benchmark_runs" / "ir_routing" / "pinned_images" / "squared"
CUTOFF = "261105"  # first batch NOT in the baseline run
HERE = Path(__file__).parent

records = [r for r in drop_rare_classes(load_records(LIVE, ("NDA",), ("color", "grayscale")), 8)
           if r.collected < CUTOFF]
assert len(records) == 297, len(records)

if PINNED.exists():
    raise SystemExit(f"{PINNED} exists; refusing to overwrite a pin. Delete it deliberately.")
manifest = []
for r in records:
    dst = PINNED / r.channel / Path(r.path).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(r.path, dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    assert sha == hashlib.sha256(Path(r.path).read_bytes()).hexdigest()
    manifest.append({"file": dst.name, "channel": r.channel, "age": r.age, "sha256": sha})

pinned = drop_rare_classes(load_records(PINNED, ("NDA",), ("color", "grayscale")), 8)
assert [p.filename for p in pinned] == [r.filename for r in records]
assert [p.age for p in pinned] == [r.age for r in records]
(HERE / "corpus_manifest.json").write_text(json.dumps(
    {"cutoff_exclusive": CUTOFF, "n": len(manifest), "images": manifest}, indent=1))
print(f"pinned {len(manifest)} images -> {PINNED}")
