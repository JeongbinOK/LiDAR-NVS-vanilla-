"""Compact validation window subsets.

Full-val validation at every epoch is dominated by near-duplicate windows: the
flat index steps one keyframe at a time, so consecutive entries share most of
their sweeps, and the split is overwhelmingly static.  A *compact* subset is a
manifest of explicitly chosen windows (typically 100 static + 100 dynamic) that
``NuScenesNVSDataset`` filters its index down to.

A window is addressed by the ``(start, end)`` LIDAR_TOP ``sample_data`` token
pair, which is stable across runs and independent of how the flat index happens
to be ordered.  Because the token pair *is* a function of the sampling contract
(``pair_mode``/``pair_kf_stride``/``window_us``/``sample_gap_us``), the manifest
records that contract and loading re-checks it: a manifest built for one
sampling contract must never be silently applied to another.

Built by ``tools/build_compact_val_windows.py``.
"""

import json
import os

MANIFEST_VERSION = 1

# Sampling-contract keys that determine which (start, end) token pairs exist.
_CONTRACT_KEYS = (
    "version",
    "split",
    "window_us",
    "sample_gap_us",
    "pair_mode",
    "pair_kf_stride",
)


def window_key(start_lidar_token: str, end_lidar_token: str) -> str:
    """Stable identifier for one window: its two endpoint LIDAR_TOP tokens."""
    return f"{start_lidar_token}|{end_lidar_token}"


def dataset_contract(dataset) -> dict:
    """Sampling contract of an already-constructed dataset."""
    return {
        "version": dataset.version,
        "split": dataset.split,
        "window_us": int(dataset.window_us),
        "sample_gap_us": int(dataset.sample_gap_us),
        "pair_mode": str(dataset.pair_mode),
        "pair_kf_stride": int(dataset.pair_kf_stride),
    }


def load_manifest(path: str) -> dict:
    with open(path) as f:
        manifest = json.load(f)
    if int(manifest.get("manifest_version", 0)) != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest_version={manifest.get('manifest_version')} "
            f"but this build expects {MANIFEST_VERSION}"
        )
    if not manifest.get("windows"):
        raise ValueError(f"{path}: manifest contains no windows")
    return manifest


def assert_contract_matches(manifest: dict, dataset, path: str) -> None:
    """Fail loudly when the manifest was built under different sampling rules."""
    stored = manifest.get("contract", {})
    actual = dataset_contract(dataset)
    mismatched = [
        f"{key}: manifest={stored.get(key)!r} != dataset={actual[key]!r}"
        for key in _CONTRACT_KEYS
        if stored.get(key) != actual[key]
    ]
    if mismatched:
        raise ValueError(
            f"Compact-val manifest {path} was built for a different sampling "
            "contract, so its window ids do not address the same windows:\n  "
            + "\n  ".join(mismatched)
            + "\nRebuild it with tools/build_compact_val_windows.py."
        )


def apply_manifest(dataset, path: str, verbose: bool = False) -> dict:
    """Filter ``dataset.index`` down to the manifest's windows, in place.

    Returns the loaded manifest.  Dataset order is preserved (scene-major), so
    the compact loader stays deterministic regardless of manifest ordering.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Compact-val manifest not found: {path}")
    manifest = load_manifest(path)
    assert_contract_matches(manifest, dataset, path)

    wanted = {}
    for entry in manifest["windows"]:
        wanted[entry["key"]] = entry
    if len(wanted) != len(manifest["windows"]):
        raise ValueError(f"{path}: manifest contains duplicate window keys")

    kept, kept_labels = [], []
    for i in range(len(dataset.index)):
        key = dataset.window_key(i)
        entry = wanted.pop(key, None)
        if entry is not None:
            kept.append(dataset.index[i])
            kept_labels.append(str(entry.get("label", "unknown")))

    if not kept:
        raise ValueError(
            f"{path}: none of the manifest's {len(manifest['windows'])} windows "
            f"exist in the {dataset.split} index"
        )
    if wanted:
        raise ValueError(
            f"{path}: {len(wanted)} manifest window(s) are absent from the "
            f"{dataset.split} index (e.g. {sorted(wanted)[0]}); the dataset "
            "build no longer matches the manifest"
        )

    dataset.index = kept
    dataset.window_labels = kept_labels
    if verbose:
        n_static = sum(1 for lab in kept_labels if lab == "static")
        n_dynamic = sum(1 for lab in kept_labels if lab == "dynamic")
        print(
            f"[{dataset.split}] compact-val: {len(kept)} windows "
            f"({n_static} static / {n_dynamic} dynamic) from {path}"
        )
    return manifest
