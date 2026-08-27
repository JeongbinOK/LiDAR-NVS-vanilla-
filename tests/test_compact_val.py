from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.dataloader import compact_val


class _StubDataset:
    """Minimal stand-in for NuScenesNVSDataset's window-addressing surface.

    Keeps the test free of a nuScenes install: `apply_manifest` only touches
    `index`, `window_key`, and the contract attributes.
    """

    def __init__(self, n=6, pair_mode="keyframe"):
        self.version = "v1.0-trainval"
        self.split = "val"
        self.window_us = 1_000_000
        self.sample_gap_us = 100_000
        self.pair_mode = pair_mode
        self.pair_kf_stride = 2
        self.index = [(0, i, i + 20) for i in range(n)]

    def window_key(self, idx):
        _, start, end = self.index[idx]
        return compact_val.window_key(f"tok{start}", f"tok{end}")


def _manifest(dataset, entries):
    return {
        "manifest_version": compact_val.MANIFEST_VERSION,
        "contract": compact_val.dataset_contract(dataset),
        "windows": entries,
    }


def _write(tmpdir, manifest):
    path = Path(tmpdir) / "manifest.json"
    path.write_text(json.dumps(manifest))
    return str(path)


class CompactValTest(unittest.TestCase):
    def test_filters_index_and_records_labels(self):
        ds = _StubDataset(n=6)
        entries = [
            {"key": ds.window_key(4), "label": "dynamic"},
            {"key": ds.window_key(1), "label": "static"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            compact_val.apply_manifest(ds, _write(tmp, _manifest(ds, entries)))
        # Dataset order wins over manifest order, so the loader is deterministic.
        self.assertEqual(ds.index, [(0, 1, 21), (0, 4, 24)])
        self.assertEqual(ds.window_labels, ["static", "dynamic"])

    def test_rejects_manifest_from_a_different_sampling_contract(self):
        built = _StubDataset(pair_mode="keyframe")
        manifest = _manifest(built, [{"key": built.window_key(0), "label": "static"}])
        applied = _StubDataset(pair_mode="sweep")
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, manifest)
            with self.assertRaises(ValueError) as ctx:
                compact_val.apply_manifest(applied, path)
        self.assertIn("pair_mode", str(ctx.exception))

    def test_rejects_window_absent_from_the_index(self):
        ds = _StubDataset(n=2)
        entries = [
            {"key": ds.window_key(0), "label": "static"},
            {"key": compact_val.window_key("nope", "nope"), "label": "dynamic"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, _manifest(ds, entries))
            with self.assertRaises(ValueError) as ctx:
                compact_val.apply_manifest(ds, path)
        self.assertIn("absent", str(ctx.exception))

    def test_rejects_duplicate_keys(self):
        ds = _StubDataset(n=2)
        entries = [{"key": ds.window_key(0), "label": "static"}] * 2
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, _manifest(ds, entries))
            with self.assertRaises(ValueError) as ctx:
                compact_val.apply_manifest(ds, path)
        self.assertIn("duplicate", str(ctx.exception))

    def test_rejects_unknown_manifest_version(self):
        ds = _StubDataset(n=2)
        manifest = _manifest(ds, [{"key": ds.window_key(0), "label": "static"}])
        manifest["manifest_version"] = compact_val.MANIFEST_VERSION + 1
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, manifest)
            with self.assertRaises(ValueError) as ctx:
                compact_val.apply_manifest(ds, path)
        self.assertIn("manifest_version", str(ctx.exception))


class ShippedManifestTest(unittest.TestCase):
    """The checked-in manifest must stay usable and balanced."""

    def test_shipped_manifest_is_20_static_180_dynamic(self):
        path = Path(__file__).resolve().parents[1] / "config" / "compact_val_windows.json"
        manifest = compact_val.load_manifest(str(path))
        labels = [w["label"] for w in manifest["windows"]]
        self.assertEqual(labels.count("static"), 20)
        self.assertEqual(labels.count("dynamic"), 180)
        self.assertEqual(len({w["key"] for w in manifest["windows"]}), 200)
        self.assertEqual(manifest["contract"]["split"], "val")

    def test_shipped_dynamic_pool_spans_every_difficulty_tier(self):
        """The point of the subset is coverage, not just count."""
        path = Path(__file__).resolve().parents[1] / "config" / "compact_val_windows.json"
        manifest = compact_val.load_manifest(str(path))
        dynamic = [w for w in manifest["windows"] if w["label"] == "dynamic"]
        self.assertEqual({w["tier"] for w in dynamic}, {0, 1, 2, 3})
        difficulty = [w["difficulty"] for w in dynamic]
        self.assertLess(min(difficulty), 0.15)
        self.assertGreater(max(difficulty), 0.85)

    def test_shipped_manifest_matches_the_default_sampling_contract(self):
        from omegaconf import OmegaConf

        from src.config_loader import compose_fresh_config

        path = Path(__file__).resolve().parents[1] / "config" / "compact_val_windows.json"
        manifest = compact_val.load_manifest(str(path))
        cfg, _ = compose_fresh_config(OmegaConf.create({}))
        for key in ("window_us", "sample_gap_us", "pair_mode", "pair_kf_stride"):
            self.assertEqual(manifest["contract"][key], cfg.data[key], msg=key)


if __name__ == "__main__":
    unittest.main()
