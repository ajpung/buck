"""Deployment cost measurement: what each architecture costs to actually serve.

Accuracy alone cannot pick a model for a website. A backbone that scores two
points higher but ships 90 MB more and takes 400 ms longer per request is the
worse choice for BUCK, where a hunter uploads one photo and waits for one
answer.

Everything here is measured, not looked up:

``params``
    Total parameter count.
``fp32_mb``
    Real serialised size of the weights on disk.
``int8_mb``
    Size after post-training dynamic quantisation, the usual shipping format
    for ONNX Runtime Web / TF.js. Roughly a quarter of fp32, but measured
    because per-layer overhead varies by architecture.
``onnx_mb``
    Size of the actual exported artefact, when the exporter is available. This
    is the number your users download.
``cpu_ms``
    Median single-image, single-thread CPU latency. Single thread is the
    conservative, comparable figure and approximates a browser or a shared
    server handling concurrent requests.
"""

from __future__ import annotations

import os
import statistics
import tempfile
import time
import warnings

import torch

from buck.benchmark import architectures as arch


def _tempfile_size_mb(save_fn):
    """Serialise via ``save_fn(path)`` and return the resulting size in MB."""
    handle, path = tempfile.mkstemp(suffix=".bin")
    os.close(handle)
    try:
        save_fn(path)
        return os.path.getsize(path) / (1024 * 1024)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def measure_cpu_latency(model, size, runs=20, warmup=5, threads=1):
    """Median milliseconds to classify one image on CPU.

    Batch size 1, because a website scores one upload at a time; throughput
    benchmarks at batch 32 would flatter the larger models misleadingly.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    dummy = torch.randn(1, 3, size, size)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

            timings = []
            for _ in range(runs):
                started = time.perf_counter()
                model(dummy)
                timings.append((time.perf_counter() - started) * 1000)
    finally:
        torch.set_num_threads(previous)

    return statistics.median(timings)


def measure_int8_mb(model):
    """Size after dynamic int8 quantisation of Linear layers.

    Dynamic quantisation is the cheapest path to a smaller artefact and needs
    no calibration data. Convolution-heavy backbones benefit less than
    transformer or MLP-headed ones, which is exactly the trade-off worth
    seeing in the table.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantised = torch.ao.quantization.quantize_dynamic(
                model.cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8
            )
        return _tempfile_size_mb(lambda p: torch.save(quantised.state_dict(), p))
    except Exception as exc:  # pragma: no cover - backend dependent
        print(f"        int8 quantisation unavailable: {exc}")
        return None


_onnx_warning_shown = False


def measure_onnx_mb(model, size):
    """Size of the exported ONNX graph, the artefact a browser downloads.

    Returns None if export is unavailable, but says so once rather than
    leaving a blank column. A silently empty measurement reads as "measured
    and fine", which is how the previous sweep lost 20 architectures.
    """
    global _onnx_warning_shown

    model = model.cpu().eval()
    dummy = torch.randn(1, 3, size, size)

    def export(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                model,
                dummy,
                path,
                input_names=["image"],
                output_names=["logits"],
                opset_version=17,
                dynamo=False,
            )

    try:
        return _tempfile_size_mb(export)
    except Exception as exc:
        if not _onnx_warning_shown:
            _onnx_warning_shown = True
            print(
                f"\n   [!] ONNX export unavailable, so the 'onnx MB' column "
                f"will be empty.\n"
                f"       Reason: {type(exc).__name__}: {str(exc)[:120]}\n"
                f"       Install it with:  pip install onnx onnxscript\n"
                f"       Until then, use fp32/int8 MB as the size proxy; the "
                f"ONNX graph is normally within a few percent of fp32.\n"
            )
        return None


def profile(model_name, num_classes=5, image_size=None, runs=20,
            with_onnx=True, with_int8=True):
    """Measure the serving cost of one architecture."""
    size = arch.input_size(model_name, image_size)
    # Pretrained weights are irrelevant to size and speed; skip the download.
    model = arch.build_model(model_name, num_classes, pretrained=False)

    trainable, frozen = arch.parameter_counts(model)
    record = {
        "model": model_name,
        "input_size": size,
        "params_m": (trainable + frozen) / 1e6,
        "fp32_mb": _tempfile_size_mb(lambda p: torch.save(model.state_dict(), p)),
        "cpu_ms": measure_cpu_latency(model, size, runs=runs),
    }
    record["int8_mb"] = measure_int8_mb(model) if with_int8 else None
    record["onnx_mb"] = measure_onnx_mb(model, size) if with_onnx else None
    return record


def profile_all(model_names, num_classes=5, image_size=None, runs=20,
                with_onnx=True, with_int8=True):
    records = []
    for name in model_names:
        print(f"   profiling {name} ...", flush=True)
        records.append(
            profile(name, num_classes, image_size, runs, with_onnx, with_int8)
        )
    return records


def pareto_front(records, quality_key, cost_key, higher_is_better=True):
    """Return the records not dominated on (quality, cost).

    A model is dominated when another is at least as good on quality *and* at
    least as cheap on cost, and strictly better on one of them. What survives
    is the set worth choosing between; everything else is beaten outright.
    """
    usable = [r for r in records if r.get(quality_key) is not None
              and r.get(cost_key) is not None]

    def better_quality(a, b):
        return a[quality_key] >= b[quality_key] if higher_is_better \
            else a[quality_key] <= b[quality_key]

    front = []
    for candidate in usable:
        dominated = any(
            other is not candidate
            and better_quality(other, candidate)
            and other[cost_key] <= candidate[cost_key]
            and (other[quality_key] != candidate[quality_key]
                 or other[cost_key] != candidate[cost_key])
            for other in usable
        )
        if not dominated:
            front.append(candidate)

    return sorted(front, key=lambda r: r[cost_key])


def print_cost_table(records, sort_key="fp32_mb"):
    def fmt(value, spec):
        return format(value, spec) if value is not None else "  --"

    print(f"\n{'=' * 84}")
    print("DEPLOYMENT COST  (batch 1, single CPU thread)")
    print(f"{'=' * 84}")
    print(
        f"{'model':<22} {'px':>5} {'params M':>9} {'fp32 MB':>9} "
        f"{'int8 MB':>9} {'onnx MB':>9} {'CPU ms':>9}"
    )
    print("-" * 84)
    for r in sorted(records, key=lambda r: r[sort_key]):
        print(
            f"{r['model']:<22} {r['input_size']:>5} {r['params_m']:>9.1f} "
            f"{fmt(r['fp32_mb'], '>9.1f')} {fmt(r['int8_mb'], '>9.1f')} "
            f"{fmt(r['onnx_mb'], '>9.1f')} {r['cpu_ms']:>9.1f}"
        )
    print("-" * 84)