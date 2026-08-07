"""Leak-free transfer-learning architecture benchmark for BUCK.

The modules here replace the ad-hoc sweep in
``trail cam/examples/251008 - all image.ipynb``. See ``data.py`` for the
specific leakage rules this package enforces.
"""

from buck.benchmark.data import (
    ImageRecord,
    build_groups,
    load_records,
    load_or_create_holdout,
)
from buck.benchmark.metrics import ordinal_metrics, bootstrap_ci

__all__ = [
    "ImageRecord",
    "build_groups",
    "load_records",
    "load_or_create_holdout",
    "ordinal_metrics",
    "bootstrap_ci",
]