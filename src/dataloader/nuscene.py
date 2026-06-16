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
MIN_GAP_US = 100_000            # 0.1 s minimum gap between sampled frames
MAX_WIN_US = 1_000_000          # 1.0 s maximum window
 
 
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
 
 
def _load_points(nusc, dataroot: str, sample_data_token: str) -> torch.Tensor:
    """Returns Tensor(N, 4): x, y, z, intensity."""
    sd  = nusc.get('sample_data', sample_data_token)
    raw = np.fromfile(os.path.join(dataroot, sd['filename']), dtype=np.float32)
    if raw.size % 5 == 0:
        scan = raw.reshape(-1, 5)[:, :4]
    elif raw.size % 4 == 0:
        scan = raw.reshape(-1, 4)
    else:
        raise ValueError(f"Unexpected point cloud size {raw.size}")
    return torch.from_numpy(scan)
 
 
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
    """
    entries = bbox_data.get(sample_token, [])
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
 
        inside = (lx.abs() <= w / 2) & (ly.abs() <= l / 2) & (dz.abs() <= h / 2)
        iid[inside] = inst_id
 
    return iid
 
 
# ────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────
 
class NuScenesNVSDataset(Dataset):
    """
    For each sample, randomly selects V frames (V in [2, max_frame_gap])
    within a 1-second window, with at least 0.1 s between consecutive frames.
 
    Output keys
    -----------
    lidar_points : Tensor(N, 4)      all frames concat, in ref-frame coords
    offset       : Tensor(V_i * B,)  cumulative point count per (batch, frame)
    batch_idx    : Tensor(N,)        which batch sample each point belongs to
    bbox         : List[Tensor(B_f, 7)]  per-frame boxes in sensor frame
    point_iid    : Tensor(N,)        instance id per point (-1 = background)
    pose         : Tensor(V, 4, 4)   frame_i → ref_frame (frame_0) rel pose
    timestamps   : Tensor(V,)        normalised to [0, 1]
    """
    #n_input = 2 + max_input_extra
    def __init__(self, cfg, split: str):
        self.cfg           = cfg
        self.dataroot      = cfg.dataroot
        self.split         = split
        self.mode          = getattr(cfg, 'mode', 'bbox')   # 'nvs' | 'bbox'
        self.nusc          = NuScenes(version=cfg.version,
                                      dataroot=cfg.dataroot, verbose=False)
 
        # Optional predicted BBox JSON
        self.bbox_data = None
        bbox_json_path = getattr(cfg, 'bbox_json_path', None)
        if bbox_json_path is not None:
            with open(bbox_json_path) as f:
                self.bbox_data = json.load(f)['results']
            print(f"Loaded predicted BBox from {bbox_json_path} "
                  f"({len(self.bbox_data)} samples)")
 
        # ── Build scene index ───────────────────────────────────────────────
        splits        = create_splits_scenes()
        split_scenes  = splits[split]
        filtered      = [s for s in self.nusc.scene if s['name'] in split_scenes]
        print(f"[{split}] {len(filtered)} scenes")
 
        # For each scene, collect all sample_data tokens (key frames only)
        # together with their timestamps.  We index by scene so __getitem__
        # can do random sampling at runtime.
        self.scene_frames = []    # List[List[(lidar_token, sample_token, timestamp_us)]]
        for scene in filtered:
            frames = []
            curr   = scene['first_sample_token']
            while curr:
                sample      = self.nusc.get('sample', curr)
                lidar_token = sample['data']['LIDAR_TOP']
                ts          = self.nusc.get('sample_data', lidar_token)['timestamp']
                frames.append((lidar_token, curr, ts))
                curr = sample['next']
            if len(frames) >= 2:
                self.scene_frames.append(frames)
 
        # Build a flat index: each entry is (scene_idx, anchor_frame_idx)
        # The anchor is the first frame in the 1-second window.

        self.index = []
        for s_idx, frames in enumerate(self.scene_frames):
            for f_idx in range(len(frames)):
                t0 = frames[f_idx][2]
                available = [
                    i for i in range(f_idx, len(frames))
                    if frames[i][2] - t0 <= MAX_WIN_US
                ]
                if len(available) >= 2:
                    self.index.append((s_idx, f_idx))

        print(f"[{split}] {len(self.index)} anchors ")

        
 
    # ── Core sampling ────────────────────────────────────────────────────────
    

    def _sample_frames(self, scene_idx: int, anchor_idx: int):
        """
        0.1초 간격으로 전체 프레임 선택.
        """
        frames     = self.scene_frames[scene_idx]
        t0         = frames[anchor_idx][2]
        candidates = [
            i for i in range(anchor_idx, len(frames))
            if frames[i][2] - t0 <= MAX_WIN_US
        ]

        selected = [anchor_idx]
        for i in candidates[1:]:
            t_last = frames[selected[-1]][2]
            if frames[i][2] - t_last >= MIN_GAP_US:
                selected.append(i)

        return [frames[i] for i in selected]

    def _select_input_indices(self, V: int) -> list:
        """
        전체 V개 프레임 중 input으로 쓸 index 선택.
        - 항상 포함: 0 (맨 앞), V-1 (맨 뒤)
        - 중간에서 random으로 n_input_extra개 추가
        """
        n_extra = np.random.randint(0, self.cfg.max_input_extra + 1)
        middle  = list(range(1, V - 1))
        if middle and n_extra > 0:
            n_extra  = min(n_extra, len(middle))
            extra    = sorted(np.random.choice(middle, n_extra, replace=False).tolist())
        else:
            extra = []

        input_indices = sorted(set([0] + extra + [V - 1]))
        return input_indices

 
    # ── __getitem__ ──────────────────────────────────────────────────────────
 
    def __len__(self):
        return len(self.index)
 
    def __getitem__(self, idx):
        scene_idx, anchor_idx = self.index[idx]
        sampled = self._sample_frames(scene_idx, anchor_idx)
        V       = len(sampled)

        lidar_tokens  = [f[0] for f in sampled]
        sample_tokens = [f[1] for f in sampled]
        timestamps_us = [f[2] for f in sampled]
 
        # ── Poses ────────────────────────────────────────────────────────────
        s2w_list  = [_sensor_to_world(self.nusc, t) for t in lidar_tokens]
        world_to_ref = np.linalg.inv(s2w_list[0])              # ref = frame 0
 
        poses = []
        for s2w in s2w_list:
            rel = (world_to_ref @ s2w).astype(np.float32)      # frame_i → frame_0
            poses.append(torch.from_numpy(rel))
        pose = torch.stack(poses)                               # (V, 4, 4)
 
        # Normalised timestamps: 0 = frame 0, 1 = last frame
        t0, t_last = timestamps_us[0], timestamps_us[-1]
        span       = max(t_last - t0, 1)
        timestamps = torch.tensor(
            [(t - t0) / span for t in timestamps_us], dtype=torch.float32)  # (V,)
 
        # ── Points (transform each frame into ref frame) ──────────────────
        all_pts_sensor = [_load_points(self.nusc, self.dataroot, t)
                          for t in lidar_tokens]                # List[Tensor(N_i,4)]
 
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
                    print("bbox is not none")
                    boxes, ids = _pred_boxes_in_sensor_frame(
                        self.nusc, self.bbox_data, sample_tok, lidar_tok)
                    print(boxes)
                else:
                    print("bbox is none")
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
 
        # ── input index 선택 ─────────────────────────────────────────────────
        input_indices = self._select_input_indices(V)   # ex) [0, 2, V-1]

        # ── input frames ─────────────────────────────────────────────────────
        input_pts_ref    = [all_pts_ref[i]    for i in input_indices]
        input_pts_sensor = [all_pts_sensor[i] for i in input_indices]
        input_bbox       = [bbox_list[i]      for i in input_indices]
        input_iid        = [iid_list[i]       for i in input_indices]
        input_pose       = pose[input_indices]                  # (n_input, 4, 4)
        input_timestamps = timestamps[input_indices]            # (n_input,)
        input_tokens     = [lidar_tokens[i]   for i in input_indices]

        input_frame_counts = torch.tensor([p.shape[0] for p in input_pts_ref])

        # ── gt = 전체 ────────────────────────────────────────────────────────
        gt_frame_counts = torch.tensor([p.shape[0] for p in all_pts_ref])

        # ── Camera 객체 ──────────────────────────────────────────────────────
        input_cameras = [
            Camera.from_nuscenes(
                nusc=self.nusc,
                lidar_token=lidar_tok,
                timestamp_normalized=ts_norm,
                pts_sensor=pts_sensor,
                cfg=self.cfg,
                uid=uid,
            )
            for uid, (lidar_tok, pts_sensor, ts_norm) in enumerate(
                zip(input_tokens, input_pts_sensor, input_timestamps.tolist())
            )
        ]

        gt_cameras = [
            Camera.from_nuscenes(
                nusc=self.nusc,
                lidar_token=lidar_tok,
                timestamp_normalized=ts_norm,
                pts_sensor=pts_sensor,
                cfg=self.cfg,
                uid=uid,
            )
            for uid, (lidar_tok, pts_sensor, ts_norm) in enumerate(
                zip(lidar_tokens, all_pts_sensor, timestamps.tolist())
            )
        ]

        return {
            "input": {
            # ── input ────────────────────────────────────────────────────────
            'lidar_points':           torch.cat(input_pts_ref,    dim=0),  # (N_in, 4)
            'lidar_points_sensor':    torch.cat(input_pts_sensor, dim=0),  # (N_in, 4)
            'frame_counts':           input_frame_counts,                  # (n_input,)
            'bbox':                   input_bbox,
            'point_iid':              torch.cat(input_iid, dim=0),
            'pose':                   input_pose,                          # (n_input, 4, 4)
            'timestamps':             input_timestamps,                    # (n_input,)
            'cameras':                input_cameras,
            'input_indices':          torch.tensor(input_indices),         # 어떤 프레임이 input인지
            },
            "gt":{
            # ── gt (전체 프레임) ──────────────────────────────────────────────
            'gt_lidar_points':        torch.cat(all_pts_ref,    dim=0),   # (N_all, 4)
            'gt_lidar_points_sensor': torch.cat(all_pts_sensor, dim=0),
            'gt_frame_counts':        gt_frame_counts,                    # (V,)
            'gt_bbox':                bbox_list,
            'gt_point_iid':           torch.cat(iid_list, dim=0),
            'gt_pose':                pose,                               # (V, 4, 4)
            'gt_timestamps':          timestamps,                         # (V,)
            'gt_cameras':             gt_cameras,
            }
        }

 
# ────────────────────────────────────────────────────────────────────────────
# Collate
# ────────────────────────────────────────────────────────────────────────────

transform = utonia.transform.default(0.2, apply_z_positive=False)    

def ptv3_mod(lidar_points, offset):
    coords = lidar_points[:, :3]
    return {
        "coord" : coords.numpy(),
        "color" : torch.zeros_like(coords).numpy(),
        "normal" : torch.zeros_like(coords).numpy(),
        "batch": offset.numpy(),
    }

def multiframe_collate_fn(batch):
    all_input_pts        = []
    all_input_sensor_pts = []
    all_input_iid        = []
    all_input_bidx       = []
    all_input_counts     = []
    all_input_bbox       = []
    all_input_pose       = []
    all_input_timestamps = []
    all_input_cameras    = []
    all_input_indices    = []

    all_gt_pts        = []
    all_gt_sensor_pts = []
    all_gt_iid        = []
    all_gt_counts     = []
    all_gt_bbox       = []
    all_gt_pose       = []
    all_gt_timestamps = []
    all_gt_cameras    = []

    for b_idx, item in enumerate(batch):
        inp = item["input"]
        gt  = item["gt"]

        N_in = inp["lidar_points"].shape[0]
        all_input_pts.append(inp["lidar_points"])
        all_input_sensor_pts.append(inp["lidar_points_sensor"])
        all_input_iid.append(inp["point_iid"])
        all_input_bidx.append(torch.full((N_in,), b_idx))
        all_input_counts.append(inp["frame_counts"])
        all_input_bbox.append(inp["bbox"])
        all_input_pose.append(inp["pose"])
        all_input_timestamps.append(inp["timestamps"])
        all_input_cameras.append(inp["cameras"])
        all_input_indices.append(inp["input_indices"])

        all_gt_pts.append(gt["gt_lidar_points"])
        all_gt_sensor_pts.append(gt["gt_lidar_points_sensor"])
        all_gt_iid.append(gt["gt_point_iid"])
        all_gt_counts.append(gt["gt_frame_counts"])
        all_gt_bbox.append(gt["gt_bbox"])
        all_gt_pose.append(gt["gt_pose"])
        all_gt_timestamps.append(gt["gt_timestamps"])
        all_gt_cameras.append(gt["gt_cameras"])

    # ── input concat ─────────────────────────────────────────────────────────
    lidar_points        = torch.cat(all_input_pts,        dim=0)
    lidar_points_sensor = torch.cat(all_input_sensor_pts, dim=0)
    point_iid           = torch.cat(all_input_iid,        dim=0)
    batch_idx           = torch.cat(all_input_bidx,       dim=0)
    flat_counts         = torch.cat(all_input_counts,     dim=0)
    offset              = torch.cumsum(flat_counts, dim=0)

    # ── gt concat ────────────────────────────────────────────────────────────
    gt_lidar_points        = torch.cat(all_gt_pts,        dim=0)
    gt_lidar_points_sensor = torch.cat(all_gt_sensor_pts, dim=0)
    gt_point_iid           = torch.cat(all_gt_iid,        dim=0)
    gt_flat_counts         = torch.cat(all_gt_counts,     dim=0)
    gt_offset              = torch.cumsum(gt_flat_counts, dim=0)

    # ── ptv3 input (input frames만) ───────────────────────────────────────────
    ptv3_coords, ptv3_grid_coords, ptv3_colors, ptv3_inverses, ptv3_feats = [], [], [], [], []
    starts = torch.cat([torch.tensor([0]), offset[:-1]])

    for start, end in zip(starts.tolist(), offset.tolist()):
        pts_slice_sensor = lidar_points_sensor[start:end]
        coords_sensor    = pts_slice_sensor[:, :3]
        d = {
            "coord":  coords_sensor.numpy(),
            "color":  np.zeros_like(coords_sensor.numpy()),
            "normal": np.zeros_like(coords_sensor.numpy()),
            "batch":  offset.numpy()
        }
        d = transform(d)
        ptv3_coords.append(d["coord"])
        ptv3_grid_coords.append(d["grid_coord"])
        ptv3_colors.append(d["color"])
        ptv3_inverses.append(d["inverse"])
        ptv3_feats.append(d["feat"])

    new_counts  = torch.tensor([c.shape[0] for c in ptv3_coords])
    ptv3_offset = torch.cumsum(new_counts, dim=0)
    ptv3_input  = {
        "coord":      torch.cat(ptv3_coords,      dim=0),
        "grid_coord": torch.cat(ptv3_grid_coords, dim=0),
        "color":      torch.cat(ptv3_colors,      dim=0),
        "inverse":    torch.cat(ptv3_inverses,    dim=0),
        "feat":       torch.cat(ptv3_feats,       dim=0),
        "offset":     ptv3_offset,
    }

    return {
        "input": {
            "lidar_points":        lidar_points,         # (N_in, 4)
            "lidar_points_sensor": lidar_points_sensor,  # (N_in, 4)
            "point_iid":           point_iid,            # (N_in,)
            "batch_idx":           batch_idx,            # (N_in,)
            "offset":              offset,               # (n_input * B,)
            "bbox":                all_input_bbox,       # List[List[Tensor(B_f, 7)]]
            "pose":                all_input_pose,       # List[Tensor(n_input, 4, 4)]
            "timestamps":          all_input_timestamps, # List[Tensor(n_input,)]
            "cameras":             all_input_cameras,    # List[List[Camera]]
            "input_indices":       all_input_indices,    # List[Tensor]
            "ptv3_input":          ptv3_input,
        },
        "gt": {
            "lidar_points":        gt_lidar_points,         # (N_all, 4)
            "lidar_points_sensor": gt_lidar_points_sensor,  # (N_all, 4)
            "point_iid":           gt_point_iid,            # (N_all,)
            "offset":              gt_offset,               # (V * B,)
            "bbox":                all_gt_bbox,             # List[List[Tensor(B_f, 7)]]
            "pose":                all_gt_pose,             # List[Tensor(V, 4, 4)]
            "timestamps":          all_gt_timestamps,       # List[Tensor(V,)]
            "cameras":             all_gt_cameras,          # List[List[Camera]]
        },
    }