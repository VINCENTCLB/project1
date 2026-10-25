"""Transparent .npy cache around the course's load_csv_data.

Parsing the raw CSVs with np.genfromtxt costs about four minutes. The parsed
arrays are cached as .npy files next to the data so later runs load in seconds.
The cache is keyed on the size and modification time of the three source CSVs,
so replacing a file invalidates it automatically.
"""

import csv
import json
import numpy as np
from pathlib import Path

from helpers import load_csv_data


CACHE_DIRNAME = "cache"
CACHE_VERSION = 1
SOURCE_FILES = ("x_train.csv", "y_train.csv", "x_test.csv")
ARRAY_NAMES = ("x_train", "x_test", "y_train", "train_ids", "test_ids")


def load_feature_names(data_path):
    """Return the x_train.csv column names, excluding the leading Id column."""
    with open(Path(data_path) / "x_train.csv", newline="") as handle:
        return np.array(next(csv.reader(handle))[1:])


def _signature(data_path):
    """Size and modification time of every source CSV, plus a format version."""
    stats = {}
    for name in SOURCE_FILES:
        info = (Path(data_path) / name).stat()
        stats[name] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    return {"version": CACHE_VERSION, "sources": stats}


def _subsample(x_train, x_test, y_train, train_ids, test_ids):
    """Apply the same ::50 training subsample that load_csv_data performs."""
    return x_train[::50], x_test, y_train[::50], train_ids[::50], test_ids


def _read_cache(cache_dir, signature):
    """Return the cached arrays, or None if the cache is absent or stale."""
    manifest = cache_dir / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        if json.loads(manifest.read_text()) != signature:
            return None
        return tuple(
            np.load(cache_dir / f"{name}.npy", allow_pickle=False)
            for name in ARRAY_NAMES
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None  # corrupt or partially written cache -> reparse


def _write_cache(cache_dir, signature, arrays):
    """Write the arrays, then the manifest last so a crash leaves it stale."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, array in zip(ARRAY_NAMES, arrays):
        np.save(cache_dir / f"{name}.npy", array, allow_pickle=False)
    (cache_dir / "manifest.json").write_text(json.dumps(signature, indent=2))


def load_csv_data_cached(data_path, sub_sample=False, use_cache=True, verbose=True):
    """Drop-in replacement for load_csv_data that caches the parsed arrays.

    Args:
        data_path (str or Path): Folder holding x_train.csv, y_train.csv, x_test.csv.
        sub_sample (bool): Keep every 50th training row, as load_csv_data does.
        use_cache (bool): Set False to always reparse the CSVs.
        verbose (bool): Print whether the cache was used.

    Returns:
        (x_train, x_test, y_train, train_ids, test_ids), identical to
        load_csv_data. The full arrays are cached; sub_sample is applied after
        loading, so one cache serves both modes.
    """
    data_path = Path(data_path)
    cache_dir = data_path / CACHE_DIRNAME

    if use_cache:
        signature = _signature(data_path)
        arrays = _read_cache(cache_dir, signature)
        if arrays is not None:
            if verbose:
                print(f"Loaded parsed arrays from cache: {cache_dir}")
            return _subsample(*arrays) if sub_sample else arrays

    if verbose:
        print("Parsing CSVs with the course helper (no usable cache)...")
    arrays = load_csv_data(data_path, sub_sample=False)

    if use_cache:
        _write_cache(cache_dir, signature, arrays)
        if verbose:
            print(f"Wrote cache: {cache_dir}")

    return _subsample(*arrays) if sub_sample else arrays
