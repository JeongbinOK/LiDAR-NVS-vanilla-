import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from ..models_new import utonia 
from ..models_new.utils.camera import Camera
# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
 
US_PER_SEC = 1_000_000          # nuScenes timestamps are in microseconds
NUSCENES_SWEEP_US = 50_000      # LIDAR_TOP sample_data chain is 20Hz.
MIN_GAP_US = 100_000            # 0.1 s minimum gap between sampled frames
MAX_WIN_US = 1_000_000          # 1.0 s maximum window
DEFAULT_GT_MIDDLE_COUNT = 3


def relative_time_coordinates(timestamps_us, window_start_us, window_end_us):
    """Return normalized time, relative seconds, and physical window duration."""
    span_us = max(int(window_end_us) - int(window_start_us), 1)
    # Preserve the legacy normalized-time arithmetic exactly: integer
    # microsecond delta divided directly by the integer window span, followed by
    # the float32 tensor conversion. Physical seconds are an additional view of
    # the same timestamps and must not perturb old bbox checkpoints.
    normalized = torch.tensor(
        [
            (int(timestamp) - int(window_start_us)) / span_us
            for timestamp in timestamps_us
        ],
        dtype=torch.float32,
    )
    duration_sec = torch.tensor(span_us / US_PER_SEC, dtype=torch.float32)
    relative_sec = torch.tensor(
        [
            (int(timestamp) - int(window_start_us)) / US_PER_SEC
            for timestamp in timestamps_us
        ],
        dtype=torch.float32,
    )
    return normalized, relative_sec, duration_sec
 
 
def _sensor_to_world(nusc, sample_data_token: str) -> np.ndarray:
    """Returns (4,4) float32 T_sensor_to_world."""
    sd  = nusc.get('sample_data', sample_data_token)
    cs  = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep  = nusc.get('ego_pose', sd['ego_pose_token'])
 
    s2e = np.eye(4, dtype=np.float32)
    s2e[:3, :3] = Quaternion(cs['rotation']).rotation_matrix
    s2e[:3,  3] = cs['translation']
 
    e2w = np.eye(4, dtype=np.float32)
    e2w[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
    e2w[:3,  3] = ep['translation']
 
    return (e2w @ s2e).astype(np.float32)
 
 
def _load_points(
    nusc,
    dataroot: str,
    sample_data_token: str,
    ego_radius: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns points Tensor(N, 4): x, y, z, intensity and ring Tensor(N,)."""
    sd  = nusc.get('sample_data', sample_data_token)
    raw = np.fromfile(os.path.join(dataroot, sd['filename']), dtype=np.float32)
    if raw.size % 5 == 0:
        scan5 = raw.reshape(-1, 5)
        scan = scan5[:, :4]
        ring = scan5[:, 4].astype(np.int64)
    elif raw.size % 4 == 0:
        scan = raw.reshape(-1, 4)
        ring = np.full((scan.shape[0],), -1, dtype=np.int64)
    else:
        raise ValueError(f"Unexpected point cloud size {raw.size}")
    pts = torch.from_numpy(np.ascontiguousarray(scan))
    ring_t = torch.from_numpy(np.ascontiguousarray(ring))
    # nuScenes LiDAR intensity는 0~255 정수 범위 → [0,1]로 정규화.
    # 여기 한 곳에서 나누면 입력 point feature와 GT range image가 모두 같은 col 3에서
    # 파생되므로 양쪽에 동시에 반영된다.
    pts[:, 3] = pts[:, 3] / 255.0
    if ego_radius > 0:
        keep = torch.linalg.norm(pts[:, :3], dim=1) > float(ego_radius)
        pts = pts[keep].contiguous()
        ring_t = ring_t[keep].contiguous()
    return pts, ring_t
 
 
def _boxes_in_sensor_frame(nusc, sample_token, lidar_token):
    """
    Returns GT boxes expressed in the LiDAR sensor frame.
    boxes          : Tensor(B, 7)  [x,y,z,w,l,h,yaw]
    instance_tokens: List[str]
    """
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
 
    q_s2e = Quaternion(cs['rotation'])
    t_s2e = np.array(cs['translation'])
    q_e2w = Quaternion(ep['rotation'])
    t_e2w = np.array(ep['translation'])
 
    q_s2w = q_e2w * q_s2e
    t_s2w = q_e2w.rotate(t_s2e) + t_e2w
    q_w2s = q_s2w.inverse
    t_w2s = q_w2s.rotate(-t_s2w)
 
    w2s = np.eye(4, dtype=np.float32)
    w2s[:3, :3] = q_w2s.rotation_matrix
    w2s[:3,  3] = t_w2s
 
    sample = nusc.get('sample', sample_token)
    boxes, instance_tokens = [], []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        c   = np.array(ann['translation'] + [1.0], dtype=np.float32)
        cx, cy, cz = (w2s @ c)[:3].tolist()
        w, l, h    = ann['size']
        yaw        = (q_w2s * Quaternion(ann['rotation'])).yaw_pitch_roll[0]
        boxes.append([cx, cy, cz, w, l, h, yaw])
        instance_tokens.append(ann['instance_token'])
 
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), instance_tokens
    return torch.zeros((0, 7), dtype=torch.float32), []
 
 
def _pred_boxes_in_sensor_frame(nusc, bbox_data, sample_token, lidar_token):
    """
    Returns predicted boxes in the LiDAR sensor frame.
    boxes    : Tensor(B, 7)
    track_ids: List[str]

    Lookup key depends on the tracking-json cadence:
      * val/test predictions are at 0.1 s (every 2nd sweep) and are keyed by the
        LiDAR sample_data token (`lidar_token`);
      * train predictions are keyframe-only and keyed by the keyframe
        `sample_token`.
    Try lidar_token first, then sample_token, so both formats resolve.
    """
    if lidar_token in bbox_data:
        entries = bbox_data[lidar_token]
    elif sample_token in bbox_data:
        entries = bbox_data[sample_token]
    else:
        entries = []
    if not entries:
        return torch.zeros((0, 7), dtype=torch.float32), []
 
    sd = nusc.get('sample_data', lidar_token)
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    ep = nusc.get('ego_pose', sd['ego_pose_token'])
 
    q_s2e = Quaternion(cs['rotation'])
    t_s2e = np.array(cs['translation'])
    q_e2w = Quaternion(ep['rotation'])
    t_e2w = np.array(ep['translation'])
 
    q_s2w = q_e2w * q_s2e
    t_s2w = q_e2w.rotate(t_s2e) + t_e2w
    q_w2s = q_s2w.inverse
    t_w2s = q_w2s.rotate(-t_s2w)
 
    w2s = np.eye(4, dtype=np.float32)
    w2s[:3, :3] = q_w2s.rotation_matrix
    w2s[:3,  3] = t_w2s
 
    boxes, track_ids = [], []
    for e in entries:
        c           = np.array(e['translation'] + [1.0], dtype=np.float32)
        cx, cy, cz  = (w2s @ c)[:3].tolist()
        w, l, h     = e['size']
        yaw         = (q_w2s * Quaternion(e['rotation'])).yaw_pitch_roll[0]
        boxes.append([cx, cy, cz, w, l, h, yaw])
        track_ids.append(str(e['tracking_id']))
 
    return torch.tensor(boxes, dtype=torch.float32), track_ids
 
 
def _assign_instance_ids(tokens_list):
    """
    Assign consistent integer IDs across multiple frames.
    tokens_list: List[List[str]]
    Returns List[List[int]]
    """
    all_tokens = sorted(set(t for tokens in tokens_list for t in tokens))
    tok2id     = {tok: i for i, tok in enumerate(all_tokens)}
    return [[tok2id[t] for t in tokens] for tokens in tokens_list]
 
 
def _point_iid_for_frame(points: torch.Tensor, boxes: torch.Tensor,
                          instance_ids: list) -> torch.Tensor:
    """
    Assign each point an instance id based on axis-aligned bounding box check.
    Points outside all boxes get id = -1.
 
    points      : Tensor(N, 4)  x,y,z,i  in sensor frame
    boxes       : Tensor(B, 7)  x,y,z,w,l,h,yaw  in sensor frame
    instance_ids: List[int] length B
 
    Returns Tensor(N,) int64
    """
    N   = points.shape[0]
    iid = torch.full((N,), -1, dtype=torch.int64)
    if boxes.shape[0] == 0:
        return iid
 
    xyz = points[:, :3]   # (N, 3)
    for b_idx, (box, inst_id) in enumerate(zip(boxes, instance_ids)):
        cx, cy, cz, w, l, h, yaw = box.tolist()
 
        # Rotate points into box-local frame (around z-axis)
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        dx = xyz[:, 0] - cx
        dy = xyz[:, 1] - cy
        dz = xyz[:, 2] - cz
 
        lx = cos_y * dx - sin_y * dy
        ly = sin_y * dx + cos_y * dy
 
        # nuScenes size = (w, l, h) and yaw rotates the LENGTH axis onto local
        # +x, so x is bounded by l/2 and y by w/2 (see boxes.point_in_box).
        inside = (lx.abs() <= l / 2) & (ly.abs() <= w / 2) & (dz.abs() <= h / 2)
        iid[inside] = inst_id
 
    return iid
 
 
def _cfg_get(cfg, key: str, default=None):
    return getattr(cfg, key, default)


def _resolve_nuscenes_version(cfg, split: str) -> str:
    test_split = _cfg_get(cfg, "test_split", "test")
    if split == test_split or split == "test":
        return _cfg_get(cfg, "test_version", _cfg_get(cfg, "version", "v1.0-test"))
    return _cfg_get(cfg, "version", "v1.0-trainval")


def _resolve_bbox_json_path(cfg, split: str):
    bbox_json_paths = _cfg_get(cfg, "bbox_json_paths", None)
    if bbox_json_paths is not None and split in bbox_json_paths:
        return bbox_json_paths[split]

    bbox_json_path = _cfg_get(cfg, "bbox_json_path", None)
    if not bbox_json_path:
        return None
    return bbox_json_path.format(split=split)


# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────
 
class NuScenesNVSDataset(Dataset):
    """
    Builds a 1-second LiDAR_TOP window at 0.1-second spacing from sample_data
    sweeps. Inputs are the two endpoints; loss GT uses both endpoints plus a
    small subset of the middle sweep frames.
 
    Output keys
    -----------
    lidar_points : Tensor(N, 4)      all frames concat, in ref-frame coords
    offset       : Tensor(V_i * B,)  cumulative point count per (batch, frame)
    batch_idx    : Tensor(N,)        which batch sample each point belongs to
    bbox         : List[Tensor(B_f, 7)]  per-frame boxes in sensor frame
    point_iid    : Tensor(N,)        instance id per point (-1 = background)
    pose         : Tensor(V, 4, 4)   frame_i → ref_frame (frame_0) rel pose
    timestamps   : Tensor(V,)        normalised to [0, 1]
    timestamps_sec: Tensor(V,)       seconds relative to window start
    window_duration_sec: Tensor(())  actual endpoint interval in seconds
    """
    #n_input = 2 + max_input_extra
    def __init__(self, cfg, split: str):
        self.cfg           = cfg
        self.dataroot      = cfg.dataroot
        self.split         = split
        self.mode          = getattr(cfg, 'mode', 'bbox')   # 'nvs' | 'bbox'
        self.verbose       = bool(getattr(cfg, 'verbose', False))
        self.version       = _resolve_nuscenes_version(cfg, split)
        self.window_us     = int(_cfg_get(cfg, "window_us", MAX_WIN_US))
        self.sample_gap_us = int(_cfg_get(cfg, "sample_gap_us", MIN_GAP_US))
        self.gt_middle_count = int(_cfg_get(cfg, "gt_middle_count", DEFAULT_GT_MIDDLE_COUNT))
        self.sample_hop = max(1, self.sample_gap_us // NUSCENES_SWEEP_US)
        self.window_hop = self.window_us // NUSCENES_SWEEP_US
        self.window_frame_count = self.window_us // self.sample_gap_us + 1
        self.window_sample_hops = [
            i * self.sample_hop for i in range(self.window_frame_count)
        ]
        # Window pairing mode:
        #   "sweep"    (default) -- anchor at a keyframe, take fixed 0.1s hops.
        #              The 2nd input is usually a non-keyframe sweep, so in train
        #              it inherits *stale* parent-keyframe boxes (up to ~0.45s).
        #   "keyframe" -- BOTH inputs are exact-time keyframes `pair_kf_stride`
        #              apart (~0.5s/step); GT/middle frames are the 0.1s-grid
        #              sweeps between them (~9 candidates, like sweep mode). No
        #              box staleness; pair count is ~unchanged (+0.6% on val).
        self.pair_mode = str(_cfg_get(cfg, "pair_mode", "sweep"))
        self.pair_kf_stride = int(_cfg_get(cfg, "pair_kf_stride", 2))

        self.nusc          = NuScenes(version=self.version,
                                      dataroot=cfg.dataroot, verbose=False)
 
        # Optional predicted BBox JSON
        self.bbox_data = None
        bbox_json_path = _resolve_bbox_json_path(cfg, split)
        if bbox_json_path is not None and os.path.exists(bbox_json_path):
            with open(bbox_json_path) as f:
                self.bbox_data = json.load(f)['results']
            if self.verbose:
                print(f"Loaded predicted BBox from {bbox_json_path} "
                      f"({len(self.bbox_data)} samples)")
        elif bbox_json_path is not None:
            if split == _cfg_get(cfg, "test_split", "test") or split == "test":
                if self.verbose:
                    print(f"[{split}] BBox JSON not found, using empty boxes: {bbox_json_path}")
            else:
                raise FileNotFoundError(f"BBox JSON not found: {bbox_json_path}")
 
        # ── Build scene index ───────────────────────────────────────────────
        splits        = create_splits_scenes()
        split_scenes  = splits[split]
        filtered      = [s for s in self.nusc.scene if s['name'] in split_scenes]
        if self.verbose:
            print(f"[{split}] {len(filtered)} scenes")
 
        # For each scene, follow the LIDAR_TOP sample_data chain.  This includes
        # non-keyframe sweeps, unlike sample['next'] which only visits keyframes.
        self.scene_frames = []    # List[List[(lidar_token, sample_token, timestamp_us, is_key_frame)]]
        for scene in filtered:
            frames = []
            first_sample = self.nusc.get('sample', scene['first_sample_token'])
            curr = first_sample['data']['LIDAR_TOP']
            while curr:
                sd = self.nusc.get('sample_data', curr)
                frames.append((curr, sd['sample_token'], sd['timestamp'], bool(sd['is_key_frame'])))
                curr = sd['next']
            if len(frames) >= self.window_frame_count:
                self.scene_frames.append(frames)
 
        # Phase-align off-grid frames to predicted tracking. val/test tracking is
        # keyed by lidar_token on a 0.1s even-sweep grid aligned to the scene
        # start; dropped sweeps make some frames (incl. keyframes) land on an odd
        # chain index that is absent from the file -> borrow the nearest tracked
        # sweep's boxes (<= 2 sweeps / 0.1s away). Train frames resolve via the
        # keyframe sample_token and are skipped, so this is a no-op for train.
        if self.bbox_data is not None:
            self._alias_offgrid_frames()

        # Build a flat index.
        #   sweep mode    : entry = (scene_idx, anchor_keyframe_idx)
        #   keyframe mode : entry = (scene_idx, kf_start_idx, kf_end_idx)
        self.index = []
        if self.pair_mode == "keyframe":
            for s_idx, frames in enumerate(self.scene_frames):
                kf = [i for i in range(len(frames)) if frames[i][3]]
                for a in range(len(kf) - self.pair_kf_stride):
                    self.index.append((s_idx, kf[a], kf[a + self.pair_kf_stride]))
        else:
            for s_idx, frames in enumerate(self.scene_frames):
                for f_idx in range(len(frames)):
                    if not frames[f_idx][3]:
                        continue
                    if f_idx + self.window_hop < len(frames):
                        self.index.append((s_idx, f_idx))

        if self.verbose:
            print(f"[{split}] {len(self.index)} anchors")

        
 
    # ── Core sampling ────────────────────────────────────────────────────────
    

    def _alias_offgrid_frames(self):
        """Make every chain frame resolvable against the predicted tracking.

        For each frame whose lidar_token (val/test cadence) and sample_token
        (train cadence) are both missing from `self.bbox_data`, alias its
        lidar_token to the boxes of the nearest tracked sweep in the same chain
        (<= 2 sweeps away). Fixes val windows anchored at an odd-parity keyframe,
        which otherwise get zero predicted boxes for the whole window."""
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
            print(f"[{self.split}] phase-aliased {n_alias} off-grid frames "
                  f"to nearest tracked sweep (<=0.1s)")

    def _sample_frames(self, scene_idx: int, start_idx: int, end_idx: int = None):
        """0.1초(2 sweep) 간격 window 선택.

        sweep mode (end_idx=None): 키프레임 앵커에서 고정 hop.
        keyframe mode (end_idx 주어짐): [start, end] 사이의 모든 키프레임을
        반드시 포함(=항상 GT 후보)하고, 인접 키프레임 half-interval마다 0.1s
        (2 sweep) 그리드로 subsample.
        """
        frames = self.scene_frames[scene_idx]
        if end_idx is None:
            selected = [start_idx + hop for hop in self.window_sample_hops]
            if selected[-1] >= len(frames):
                raise RuntimeError("Indexed nuScenes window became invalid.")
            return [frames[i] for i in selected]
        # start/end are keyframes (by index construction); add any interior ones.
        kfs = ([start_idx]
               + [j for j in range(start_idx + 1, end_idx) if frames[j][3]]
               + [end_idx])
        selected = []
        for a in range(len(kfs) - 1):
            selected += list(range(kfs[a], kfs[a + 1], self.sample_hop))
        selected.append(kfs[-1])
        return [frames[i] for i in selected]

    def _select_input_indices(self, V: int) -> list:
        """
        전체 window 중 input으로 쓸 index 선택: 항상 양끝 2프레임.
        """
        return [0, V - 1]

    def _select_gt_indices(self, window) -> list:
        """Loss GT window-index 선택 (항상 input 양끝 2프레임 포함).

        keyframe mode: 중간 keyframe(들)을 GT로 반드시 포함하고, 그 좌/우
          half-interval에서 나머지를 뽑음 (train=random, val/test=구간 중앙 고정).
          예) input=(k_i, k_{i+2}) → GT중 1개는 항상 k_{i+1}, 나머지는 (k_i,k_{i+1})
          과 (k_{i+1},k_{i+2}) 사이 sweep에서 각각.
        sweep mode: 중간에서 train=random / val=균등 고정.
        """
        V = len(window)
        is_train = (self.split == _cfg_get(self.cfg, "train_split", "train"))

        if self.pair_mode == "keyframe":
            interior_kf = [p for p in range(1, V - 1) if window[p][3]]
            if interior_kf:
                p_mid = interior_kf[len(interior_kf) // 2]   # central keyframe
                left = list(range(1, p_mid))
                right = list(range(p_mid + 1, V - 1))
                n_rest = max(0, self.gt_middle_count - 1)
                n_left = n_rest // 2 + (n_rest % 2)
                sel = ([p_mid]
                       + self._pick_from(left, n_left, is_train)
                       + self._pick_from(right, n_rest - n_left, is_train))
                return sorted(set([0] + sel + [V - 1]))

        middle = list(range(1, V - 1))
        n_middle = min(self.gt_middle_count, len(middle))
        if n_middle <= 0:
            selected_middle = []
        elif is_train:
            selected_middle = sorted(
                np.random.choice(middle, n_middle, replace=False).tolist())
        else:
            positions = np.linspace(1, len(middle), n_middle + 2)[1:-1]
            selected_middle = [middle[int(round(p)) - 1] for p in positions]
        return sorted(set([0] + selected_middle + [V - 1]))

    def _pick_from(self, seg, k, is_train):
        """Pick k indices from segment `seg`: random for train, evenly-spaced
        and reproducible for val/test."""
        if k <= 0 or not seg:
            return []
        if k >= len(seg):
            return list(seg)
        if is_train:
            return sorted(np.random.choice(seg, k, replace=False).tolist())
        pos = np.linspace(0, len(seg) - 1, k + 2)[1:-1]
        return [seg[int(round(p))] for p in pos]

 
    # ── __getitem__ ──────────────────────────────────────────────────────────
 
    def __len__(self):
        return len(self.index)
 
    def __getitem__(self, idx):
        scene_idx, *span = self.index[idx]      # (anchor,) sweep | (start,end) keyframe
        window = self._sample_frames(scene_idx, *span)
        V_window = len(window)
        input_window_indices = self._select_input_indices(V_window)
        gt_window_indices = self._select_gt_indices(window)
        used = [window[i] for i in gt_window_indices]
        window_to_used = {window_idx: used_idx for used_idx, window_idx in enumerate(gt_window_indices)}
        input_indices = [window_to_used[i] for i in input_window_indices]

        lidar_tokens  = [f[0] for f in used]
        sample_tokens = [f[1] for f in used]
        timestamps_us = [f[2] for f in used]
        source_window_indices = [int(i) for i in gt_window_indices]
 
        # ── Poses ────────────────────────────────────────────────────────────
        s2w_list  = [_sensor_to_world(self.nusc, t) for t in lidar_tokens]
        world_to_ref = np.linalg.inv(s2w_list[0])              # ref = frame 0
 
        poses = []
        for s2w in s2w_list:
            rel = (world_to_ref @ s2w).astype(np.float32)      # frame_i → frame_0
            poses.append(torch.from_numpy(rel))
        pose = torch.stack(poses)                               # (V, 4, 4)
 
        # Keep both coordinate systems. Legacy bbox/V1 experiments consume the
        # per-window normalized coordinate; physical-velocity V3 conditions on
        # relative seconds (with one fixed reference-second scale) and transports
        # Gaussians in seconds, preserving irregular nuScenes keyframe gaps.
        t0 = window[0][2]
        t_last = window[-1][2]
        timestamps, timestamps_sec, window_duration_sec = (
            relative_time_coordinates(timestamps_us, t0, t_last)
        )
 
        # ── Points (transform each frame into ref frame) ──────────────────
        ego_radius = float(_cfg_get(self.cfg, "ego_radius", 0.0) or 0.0)
        loaded_points = [
            _load_points(self.nusc, self.dataroot, t, ego_radius=ego_radius)
            for t in lidar_tokens
        ]
        all_pts_sensor = [item[0] for item in loaded_points]     # List[Tensor(N_i,4)]
        all_lidar_ring = [item[1] for item in loaded_points]     # List[Tensor(N_i,)]
 
        all_pts_ref = []
        for pts, rel_pose in zip(all_pts_sensor, poses):
            xyz1 = torch.cat(
                [pts[:, :3], torch.ones(pts.shape[0], 1)], dim=1)  # (N,4)
            xyz_ref = (rel_pose @ xyz1.T).T[:, :3]                  # (N,3)
            all_pts_ref.append(
                torch.cat([xyz_ref, pts[:, 3:4]], dim=1))           # (N,4)
 
        # ── BBoxes & point_iid ───────────────────────────────────────────
        bbox_list       = []
        instance_tokens_list = []
 
        for lidar_tok, sample_tok in zip(lidar_tokens, sample_tokens):
            if self.mode == 'bbox':
                if self.bbox_data is not None:
                    boxes, ids = _pred_boxes_in_sensor_frame(
                        self.nusc, self.bbox_data, sample_tok, lidar_tok)
                else:
                    boxes, ids = _boxes_in_sensor_frame(
                        self.nusc, sample_tok, lidar_tok)
            else:
                boxes = torch.zeros((0, 7), dtype=torch.float32)
                ids   = []
            bbox_list.append(boxes)
            instance_tokens_list.append(ids)
 
        # Consistent integer instance ids across frames
        assigned = _assign_instance_ids(instance_tokens_list)  # List[List[int]]
 
        # point_iid: assign per point using per-sensor-frame geometry
        iid_list = []
        for pts_sensor, boxes, inst_ids in zip(all_pts_sensor, bbox_list, assigned):
            iid_list.append(_point_iid_for_frame(pts_sensor, boxes, inst_ids))
 
        # ── input frames ─────────────────────────────────────────────────────
        input_pts_ref    = [all_pts_ref[i]    for i in input_indices]
        input_pts_sensor = [all_pts_sensor[i] for i in input_indices]
        input_bbox       = [bbox_list[i]      for i in input_indices]
        input_bbox_iids  = [torch.tensor(assigned[i], dtype=torch.long) for i in input_indices]
        input_iid        = [iid_list[i]       for i in input_indices]
        input_ring       = [all_lidar_ring[i] for i in input_indices]
        input_pose       = pose[input_indices]                  # (n_input, 4, 4)
        input_timestamps = timestamps[input_indices]            # (n_input,)
        input_timestamps_sec = timestamps_sec[input_indices]    # (n_input,)
        input_frame_counts = torch.tensor([p.shape[0] for p in input_pts_ref])
        ref_to_sensor = torch.linalg.inv(pose)

        # ── gt = input endpoints + selected middle sweeps ───────────────────
        gt_frame_counts = torch.tensor([p.shape[0] for p in all_pts_ref])

        # ── 렌더링 supervision에 사용하는 GT Camera 객체 ───────────────────
        gt_cameras = [
            Camera.from_nuscenes(
                nusc=self.nusc,
                lidar_token=lidar_tok,
                timestamp_normalized=ts_norm,
                timestamp_seconds=ts_sec,
                pts_sensor=pts_sensor,
                lidar_ring=ring,
                cfg=self.cfg,
                uid=uid,
                ref_to_sensor=ref_pose.numpy(),
            )
            for uid, (lidar_tok, pts_sensor, ring, ts_norm, ts_sec, ref_pose) in enumerate(
                zip(
                    lidar_tokens,
                    all_pts_sensor,
                    all_lidar_ring,
                    timestamps.tolist(),
                    timestamps_sec.tolist(),
                    ref_to_sensor,
                )
            )
        ]

        return {
            "input": {
            # ── input ────────────────────────────────────────────────────────
            'lidar_points':           torch.cat(input_pts_ref,    dim=0),  # (N_in, 4)
            'lidar_points_sensor':    torch.cat(input_pts_sensor, dim=0),  # (N_in, 4)
            'lidar_ring':             torch.cat(input_ring, dim=0),
            'frame_counts':           input_frame_counts,                  # (n_input,)
            'bbox':                   input_bbox,
            'bbox_instance_ids':       input_bbox_iids,
            'point_iid':              torch.cat(input_iid, dim=0),
            'pose':                   input_pose,                          # (n_input, 4, 4)
            'timestamps':             input_timestamps,                    # (n_input,)
            'timestamps_sec':         input_timestamps_sec,                # (n_input,)
            'window_duration_sec':    window_duration_sec,                 # scalar
            'input_indices':          torch.tensor(input_indices),         # selected GT 안에서 input 위치
            'input_window_indices':   torch.tensor(input_window_indices),  # 11-frame window 안에서 input 위치
            },
            "gt":{
            # ── gt (input endpoints + selected middle sweeps) ───────────────
            'gt_lidar_points':        torch.cat(all_pts_ref,    dim=0),   # (N_all, 4)
            'gt_lidar_points_sensor': torch.cat(all_pts_sensor, dim=0),
            'gt_lidar_ring':          torch.cat(all_lidar_ring, dim=0),
            'gt_frame_counts':        gt_frame_counts,                    # (V,)
            'gt_bbox':                bbox_list,
            'gt_bbox_instance_ids':    [torch.tensor(ids, dtype=torch.long) for ids in assigned],
            'gt_point_iid':           torch.cat(iid_list, dim=0),
            'gt_pose':                pose,                               # (V, 4, 4)
            'gt_timestamps':          timestamps,                         # (V,)
            'gt_timestamps_sec':      timestamps_sec,                     # (V,)
            'window_duration_sec':    window_duration_sec,                # scalar
            'gt_cameras':             gt_cameras,
            'window_indices':         torch.tensor(source_window_indices),
            }
        }

 
# ────────────────────────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────────────────────────

transform = utonia.transform.default(0.2, apply_z_positive=False, keep_strength=True)

def multiframe_collate_fn(batch):
    all_input_pts        = []
    all_input_sensor_pts = []
    all_input_ring       = []
    all_input_iid        = []
    all_input_bidx       = []
    all_input_counts     = []
    all_input_bbox       = []
    all_input_bbox_iids  = []
    all_input_pose       = []
    all_input_timestamps = []
    all_input_timestamps_sec = []
    all_input_window_duration_sec = []
    all_input_indices    = []
    all_input_window_indices = []

    all_gt_pts        = []
    all_gt_sensor_pts = []
    all_gt_ring       = []
    all_gt_iid        = []
    all_gt_counts     = []
    all_gt_bbox       = []
    all_gt_bbox_iids  = []
    all_gt_pose       = []
    all_gt_timestamps = []
    all_gt_timestamps_sec = []
    all_gt_window_duration_sec = []
    all_gt_cameras    = []
    all_gt_window_indices = []

    for b_idx, item in enumerate(batch):
        inp = item["input"]
        gt  = item["gt"]

        N_in = inp["lidar_points"].shape[0]
        all_input_pts.append(inp["lidar_points"])
        all_input_sensor_pts.append(inp["lidar_points_sensor"])
        all_input_ring.append(inp["lidar_ring"])
        all_input_iid.append(inp["point_iid"])
        all_input_bidx.append(torch.full((N_in,), b_idx))
        all_input_counts.append(inp["frame_counts"])
        all_input_bbox.append(inp["bbox"])
        all_input_bbox_iids.append(inp["bbox_instance_ids"])
        all_input_pose.append(inp["pose"])
        all_input_timestamps.append(inp["timestamps"])
        all_input_timestamps_sec.append(inp["timestamps_sec"])
        all_input_window_duration_sec.append(inp["window_duration_sec"])
        all_input_indices.append(inp["input_indices"])
        all_input_window_indices.append(inp["input_window_indices"])

        all_gt_pts.append(gt["gt_lidar_points"])
        all_gt_sensor_pts.append(gt["gt_lidar_points_sensor"])
        all_gt_ring.append(gt["gt_lidar_ring"])
        all_gt_iid.append(gt["gt_point_iid"])
        all_gt_counts.append(gt["gt_frame_counts"])
        all_gt_bbox.append(gt["gt_bbox"])
        all_gt_bbox_iids.append(gt["gt_bbox_instance_ids"])
        all_gt_pose.append(gt["gt_pose"])
        all_gt_timestamps.append(gt["gt_timestamps"])
        all_gt_timestamps_sec.append(gt["gt_timestamps_sec"])
        all_gt_window_duration_sec.append(gt["window_duration_sec"])
        all_gt_cameras.append(gt["gt_cameras"])
        all_gt_window_indices.append(gt["window_indices"])

    # ── input concat ─────────────────────────────────────────────────────────
    lidar_points        = torch.cat(all_input_pts,        dim=0)
    lidar_points_sensor = torch.cat(all_input_sensor_pts, dim=0)
    lidar_ring          = torch.cat(all_input_ring,       dim=0)
    point_iid           = torch.cat(all_input_iid,        dim=0)
    batch_idx           = torch.cat(all_input_bidx,       dim=0)
    flat_counts         = torch.cat(all_input_counts,     dim=0)
    offset              = torch.cumsum(flat_counts, dim=0)

    # ── gt concat ────────────────────────────────────────────────────────────
    gt_lidar_points        = torch.cat(all_gt_pts,        dim=0)
    gt_lidar_points_sensor = torch.cat(all_gt_sensor_pts, dim=0)
    gt_lidar_ring          = torch.cat(all_gt_ring,       dim=0)
    gt_point_iid           = torch.cat(all_gt_iid,        dim=0)
    gt_flat_counts         = torch.cat(all_gt_counts,     dim=0)
    gt_offset              = torch.cumsum(gt_flat_counts, dim=0)

    # ── ptv3 input (input frames만) ───────────────────────────────────────────
    ptv3_coords, ptv3_grid_coords, ptv3_colors, ptv3_inverses, ptv3_feats = [], [], [], [], []
    ptv3_strengths = []
    starts = torch.cat([torch.tensor([0]), offset[:-1]])

    for start, end in zip(starts.tolist(), offset.tolist()):
        pts_slice_sensor = lidar_points_sensor[start:end]
        coords_sensor    = pts_slice_sensor[:, :3]
        coords_np = coords_sensor.detach().cpu().numpy().copy()
        # per-point intensity ("strength"): rides through GridSample so the KEPT
        # point's intensity is aligned to the sampled coord (Utonia's own point).
        strength_np = pts_slice_sensor[:, 3:4].detach().cpu().numpy().copy()
        d = {
            "coord": coords_np,
            "color": np.zeros_like(coords_np),
            "normal": np.zeros_like(coords_np),
            "strength": strength_np,
            "batch": offset.detach().cpu().numpy().copy(),
        }
        d = transform(d)
        ptv3_coords.append(d["coord"])
        ptv3_grid_coords.append(d["grid_coord"])
        ptv3_colors.append(d["color"])
        ptv3_inverses.append(d["inverse"])
        ptv3_feats.append(d["feat"])
        ptv3_strengths.append(d["strength"])

    new_counts  = torch.tensor([c.shape[0] for c in ptv3_coords])
    ptv3_offset = torch.cumsum(new_counts, dim=0)
    ptv3_input  = {
        "coord":      torch.cat(ptv3_coords,      dim=0),
        "grid_coord": torch.cat(ptv3_grid_coords, dim=0),
        "color":      torch.cat(ptv3_colors,      dim=0),
        "inverse":    torch.cat(ptv3_inverses,    dim=0),
        "feat":       torch.cat(ptv3_feats,       dim=0),
        "strength":   torch.cat(ptv3_strengths,   dim=0),
        "offset":     ptv3_offset,
    }

    return {
        "input": {
            "lidar_points":        lidar_points,         # (N_in, 4)
            "lidar_points_sensor": lidar_points_sensor,  # (N_in, 4)
            "lidar_ring":          lidar_ring,           # (N_in,)
            "point_iid":           point_iid,            # (N_in,)
            "batch_idx":           batch_idx,            # (N_in,)
            "offset":              offset,               # (n_input * B,)
            "bbox":                all_input_bbox,       # List[List[Tensor(B_f, 7)]]
            "bbox_instance_ids":    all_input_bbox_iids,  # List[List[Tensor(B_f,)]]
            "pose":                all_input_pose,       # List[Tensor(n_input, 4, 4)]
            "timestamps":          all_input_timestamps, # List[Tensor(n_input,)]
            "timestamps_sec":      all_input_timestamps_sec,
            "window_duration_sec": torch.stack(all_input_window_duration_sec),
            "input_indices":       all_input_indices,    # List[Tensor]
            "input_window_indices": all_input_window_indices,
            "ptv3_input":          ptv3_input,
        },
        "gt": {
            "lidar_points":        gt_lidar_points,         # (N_all, 4)
            "lidar_points_sensor": gt_lidar_points_sensor,  # (N_all, 4)
            "lidar_ring":          gt_lidar_ring,           # (N_all,)
            "point_iid":           gt_point_iid,            # (N_all,)
            "offset":              gt_offset,               # (V * B,)
            "bbox":                all_gt_bbox,             # List[List[Tensor(B_f, 7)]]
            "bbox_instance_ids":    all_gt_bbox_iids,        # List[List[Tensor(B_f,)]]
            "pose":                all_gt_pose,             # List[Tensor(V, 4, 4)]
            "timestamps":          all_gt_timestamps,       # List[Tensor(V,)]
            "timestamps_sec":      all_gt_timestamps_sec,
            "window_duration_sec": torch.stack(all_gt_window_duration_sec),
            "cameras":             all_gt_cameras,          # List[List[Camera]]
            "window_indices":      all_gt_window_indices,
        },
    }
