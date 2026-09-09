"""LiDAR4D / GS-LiDAR nuScenes test set (Table S6).

These 5 sequences live in **v1.0-trainval** (they need GT range images for
evaluation). Each spans 5.0 s at 0.1 s spacing (51 LiDAR4D "frames"). nuScenes
LIDAR_TOP sweeps are 0.05 s, so one LiDAR4D frame = every 2nd sweep -- verified:
the sequence end timestamp lands exactly on sweep index 100 (= 5.0 s).

Evaluation protocol (1-second temporal NVS, per user spec):
  for each target second T in {1, 2, 3, 4}:
      inputs  = frames at (T - 0.5)s and (T + 0.5)s   (the two endpoints)
      target  = the frame at T s                       (novel temporal view)
This reuses NuScenesNVSDataset's 1.0 s / 0.1 s window machinery: a window
anchored at sweep (T-0.5)s = sweep (T*20 - 10) gives 11 frames at 0.1 s, input =
the 2 endpoints, and the window midpoint = T s.
"""
from __future__ import annotations

import json
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from .nuscene import (
    NuScenesNVSDataset,
    _cfg_get,
    _resolve_bbox_json_path,
    _sensor_to_world,
    NUSCENES_SWEEP_US,
    MAX_WIN_US,
    MIN_GAP_US,
)

# (name, start LIDAR_TOP sweep timestamp in microseconds). The start frame is
# LiDAR4D "frame index 0" of each Table S6 sequence.
LIDAR4D_SEQUENCES = [
    ("seq_0450_0500", 1533151606948845),  # n008-2018-08-01-15-16-36-0400
    ("seq_1250_1300", 1535489300196771),  # n008-2018-08-28-16-43-51-0400 (ego-stationary)
    ("seq_1600_1650", 1535657110100754),  # n008-2018-08-30-15-16-55-0400
    ("seq_2200_2250", 1538448756347324),  # n015-2018-10-02-10-50-40+0800
    ("seq_3180_3230", 1542800849297337),  # n015-2018-11-21-19-38-26+0800
]

# Integer-second targets to evaluate (skip the 0 s / 5 s endpoints).
TARGET_SECONDS = (1, 2, 3, 4)


class LiDAR4DNuScenesTestDataset(NuScenesNVSDataset):
    def __init__(self, cfg, split: str = "test",
                 sequences=LIDAR4D_SEQUENCES, target_seconds=TARGET_SECONDS,
                 use_gt_boxes: bool = False):
        # NOTE: we intentionally do NOT call super().__init__ -- the parent builds
        # an index over a whole nuScenes split. We replicate the small amount of
        # setup it needs and inject our own scene_frames / index.
        self.cfg = cfg
        self.dataroot = cfg.dataroot
        self.split = split
        self.mode = getattr(cfg, "mode", "bbox")
        self.verbose = bool(getattr(cfg, "verbose", False))
        # Table S6 sequences are trainval scenes.
        self.version = _cfg_get(cfg, "version", "v1.0-trainval")
        self.window_us = int(_cfg_get(cfg, "window_us", MAX_WIN_US))
        self.sample_gap_us = int(_cfg_get(cfg, "sample_gap_us", MIN_GAP_US))
        self.gt_middle_count = 1  # only the midpoint (= target T) is evaluated
        self.pair_mode = "sweep"  # this test set uses fixed-anchor sweep windows

        self.sample_hop = max(1, self.sample_gap_us // NUSCENES_SWEEP_US)
        self.window_hop = self.window_us // NUSCENES_SWEEP_US
        self.window_frame_count = self.window_us // self.sample_gap_us + 1
        self.window_sample_hops = [
            i * self.sample_hop for i in range(self.window_frame_count)
        ]

        self.nusc = NuScenes(version=self.version, dataroot=cfg.dataroot, verbose=False)

        # BBox source. DEFAULT = predicted tracking. These 5 sequences span the
        # nuScenes *train* (2) and *val* (3) splits, and the two tracking files
        # use DIFFERENT cadences/keys:
        #   * tracking_train.json : keyframe-only, keyed by keyframe sample_token
        #   * tracking_val.json   : 0.1s (every 2nd sweep), keyed by sweep lidar_token
        # so we load + merge BOTH (sample tokens are globally unique). The earlier
        # 0-box symptom came from loading tracking_{split=test}.json, whose tokens
        # belong to the nuScenes test split and never match these trainval frames.
        self.use_gt_boxes = bool(use_gt_boxes)
        self.bbox_data = None
        if not self.use_gt_boxes:
            self.bbox_data = {}
            for sp in ("train", "val"):
                p = _resolve_bbox_json_path(cfg, sp)
                if p and os.path.exists(p):
                    with open(p) as f:
                        self.bbox_data.update(json.load(f)["results"])
                    if self.verbose:
                        print(f"[lidar4d-test] loaded predicted bbox: {p}")
            if not self.bbox_data:
                self.bbox_data = None
        elif self.verbose:
            print(f"[lidar4d-test] using GT boxes (mode={self.mode})")

        # sweeps per second (1.0 s / 0.05 s = 20) and the half-second offset.
        sweeps_per_sec = 1_000_000 // NUSCENES_SWEEP_US
        half = sweeps_per_sec // 2  # 0.5 s in sweeps
        max_anchor = max(target_seconds) * sweeps_per_sec - half
        need_len = max_anchor + self.window_hop + 1

        start_tokens = self._resolve_start_tokens([ts for _, ts in sequences])

        # nuScenes official train/val split membership of each sequence's scene.
        # IMPORTANT: 2 of the 5 LiDAR4D Table S6 seqs are in the nuScenes *train*
        # split -> for a feed-forward model trained on `train`, those are SEEN
        # data (leakage). Only the `val` seqs are a clean held-out comparison.
        splits = create_splits_scenes()
        train_set, val_set = set(splits["train"]), set(splits["val"])

        self.scene_frames = []          # List[List[(lidar_tok, sample_tok, ts_us, is_key)]]
        self.index = []                 # List[(seq_idx, anchor_sweep_idx)]
        self.index_meta = []            # List[{"seq_name", "target_s", "scene", "nuscenes_split"}]
        for s_idx, (name, ts_start) in enumerate(sequences):
            start_tok = start_tokens[ts_start]
            frames = self._walk_chain(start_tok, need_len + 4)
            if len(frames) < need_len:
                raise RuntimeError(
                    f"{name}: sweep chain too short ({len(frames)} < {need_len})")
            sd0 = self.nusc.get("sample_data", start_tok)
            scene = self.nusc.get(
                "scene", self.nusc.get("sample", sd0["sample_token"])["scene_token"])
            sname = scene["name"]
            nsplit = ("train" if sname in train_set
                      else "val" if sname in val_set else "other")
            self.scene_frames.append(frames)
            for T in target_seconds:
                anchor = T * sweeps_per_sec - half  # window start = (T-0.5)s sweep
                self.index.append((s_idx, anchor))
                self.index_meta.append({"seq_name": name, "target_s": int(T),
                                        "scene": sname, "nuscenes_split": nsplit})

        # Phase-align off-grid frames to the predicted tracking (see method doc).
        if self.bbox_data is not None:
            self._alias_offgrid_frames()

        if self.verbose:
            print(f"[lidar4d-test] {len(self.scene_frames)} sequences, "
                  f"{len(self.index)} windows (version={self.version})")

    def _alias_offgrid_frames(self):
        """val/test predictions are on a 0.1s grid (every 2nd sweep) aligned to
        keyframes; some LiDAR4D sequences start 1 sweep (0.05s) off that grid
        (e.g. seq_1250), so their window frames never coincide with a tracked
        sweep. For each frame whose lidar_token/sample_token is not in the
        tracking data, alias its lidar_token to the boxes of the nearest tracked
        sweep in the same chain (<= 2 sweeps / 0.1s away). Train frames already
        resolve via sample_token (parent keyframe) and are skipped."""
        bd = self.bbox_data
        n_alias = 0
        for frames in self.scene_frames:
            n = len(frames)
            for i, (lid, samp, ts, isk) in enumerate(frames):
                if lid in bd or samp in bd:
                    continue
                hit = None
                for d in (1, 2):
                    for j in (i - d, i + d):
                        if 0 <= j < n and frames[j][0] in bd:
                            hit = frames[j][0]
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    bd[lid] = bd[hit]
                    n_alias += 1
        if self.verbose:
            print(f"[lidar4d-test] phase-aliased {n_alias} off-grid frames")

    # ── target frame inside the selected GT cameras ─────────────────────────
    @property
    def target_cam_index(self) -> int:
        """gt cameras are [endpoint0, midpoint, endpoint1]; midpoint = target T."""
        return 1

    def _select_gt_indices(self, window):
        # endpoints (needed so input frames are a subset of the loaded GT frames)
        # + the exact window midpoint, which is the novel target view at T.
        V = len(window)
        return [0, V // 2, V - 1]

    def target_boxes_ref(self, window_index: int):
        """GT boxes at the target-second frame, in the window reference frame.

        The collated ``gt["bbox"]`` comes from ``_boxes_in_sensor_frame``, which
        reads the raw *keyframe* ``sample['anns']`` — for a non-keyframe target
        sweep (T rarely lands on a 2 Hz keyframe) a moving object's box is stale
        by up to ~0.5 s (metres of drift). ``nusc.get_boxes`` instead linearly
        interpolates each annotation to the sweep timestamp, so the boxes line
        up with the target-time LiDAR. Returned as ``(M, 7)``
        ``[x,y,z,w,l,h,yaw]`` in the ref frame = the (T-0.5)s endpoint sensor
        frame (the same ref ``__getitem__`` builds its poses/points in).
        """
        scene_idx, anchor = self.index[window_index]
        window = self._sample_frames(scene_idx, anchor)
        used = [window[i] for i in self._select_gt_indices(window)]
        ref_token = used[0][0]
        target_token = used[self.target_cam_index][0]

        world_to_ref = np.linalg.inv(
            _sensor_to_world(self.nusc, ref_token).astype(np.float64)
        )
        rot, trans = world_to_ref[:3, :3], world_to_ref[:3, 3]
        yaw_offset = float(np.arctan2(rot[1, 0], rot[0, 0]))

        boxes = []
        for box in self.nusc.get_boxes(target_token):   # global frame, interpolated
            cx, cy, cz = (rot @ np.asarray(box.center, np.float64) + trans).tolist()
            w, l, h = (float(v) for v in box.wlh)
            yaw = float(box.orientation.yaw_pitch_roll[0]) + yaw_offset
            boxes.append([cx, cy, cz, w, l, h, yaw])
        return np.asarray(boxes, np.float32).reshape(-1, 7)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _walk_chain(self, start_token: str, n: int):
        frames, cur = [], start_token
        while cur and len(frames) < n:
            sd = self.nusc.get("sample_data", cur)
            frames.append((sd["token"], sd["sample_token"], sd["timestamp"],
                           bool(sd["is_key_frame"])))
            cur = sd["next"]
        return frames

    def _resolve_start_tokens(self, timestamps):
        targets = {str(int(t)): int(t) for t in timestamps}
        found = {}
        for sd in self.nusc.sample_data:
            fn = sd["filename"]
            if "LIDAR_TOP" not in fn:
                continue
            for s, ti in targets.items():
                if s in fn:
                    found[ti] = sd["token"]
        missing = [int(t) for t in timestamps if int(t) not in found]
        if missing:
            raise RuntimeError(
                f"LIDAR_TOP start timestamps not found in {self.version}: {missing}")
        return found
