import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

class NuScenesNVSDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.dataroot = cfg.dataroot
        self.split = split
        self.frame_gap = cfg.frame_gap
        self.mode = cfg.mode        # 'nvs' | 'bbox'
        self.nusc = NuScenes(version=cfg.version, dataroot=dataroot, verbose=False)
        self.bbox_json_path = cfg.bbox_json_path
        # Predicted BBox source (LK3D + MCTrack results).
        # If None and mode='bbox', GT annotations are used instead.
        self.bbox_data = None
        if bbox_json_path is not None:
            with open(bbox_json_path) as f:
                self.bbox_data = json.load(f)['results']   # dict[sample_token → List[box_dict]]
            print(f"Loaded predicted BBox from {bbox_json_path} "
                  f"({len(self.bbox_data)} samples)")

        # 1. Filter scenes by split
        splits = create_splits_scenes()
        if split not in splits:
            raise ValueError(f"Split '{split}' not found in nuScenes splits")

        split_scenes = splits[split]
        filtered_scenes = [s for s in self.nusc.scene if s['name'] in split_scenes]
        print(f"[{split}] Found {len(filtered_scenes)} scenes.")

        # 2. Build (input_0, input_1, gts) triplets
        self.data_infos = []
        for scene in filtered_scenes:
            scene_samples = []
            curr_sample_token = scene['first_sample_token']
            while curr_sample_token:
                scene_samples.append(curr_sample_token)
                curr_sample_token = self.nusc.get('sample', curr_sample_token)['next']

            for i in range(len(scene_samples) - frame_gap):
                start_token = scene_samples[i]
                end_token   = scene_samples[i + frame_gap]

                start_lidar_token = self.nusc.get('sample', start_token)['data']['LIDAR_TOP']
                end_lidar_token   = self.nusc.get('sample', end_token)['data']['LIDAR_TOP']

                # Collect all sample_data entries strictly between start and end.
                # The chain includes both key samples (is_key_frame=True) and sweeps.
                # With frame_gap >= 2, intermediate key samples are included here
                # but their BBoxes are not loaded (only input_0 and input_1 have BBoxes).
                intermediate_lidar_tokens = []
                curr = self.nusc.get('sample_data', start_lidar_token)['next']
                while curr and curr != end_lidar_token:
                    intermediate_lidar_tokens.append(curr)
                    curr = self.nusc.get('sample_data', curr)['next']

                self.data_infos.append({
                    'input_0':       start_lidar_token,
                    'input_1':       end_lidar_token,
                    'gts':           intermediate_lidar_tokens,
                    'sample_token_0': start_token,   # needed for GT BBox lookup
                    'sample_token_1': end_token,
                })

        print(f"[{split}] Created {len(self.data_infos)} pairs with frame_gap={frame_gap}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_sensor_to_world_pose(self, sample_data_token):
        """
        Returns a (4, 4) float32 torch tensor:
            T_sensor_to_world = T_ego_to_world  @  T_sensor_to_ego
        Transforms points from the LIDAR_TOP sensor frame to the global world frame.
        """
        sd_record = self.nusc.get('sample_data', sample_data_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        sensor_to_ego = np.eye(4)
        sensor_to_ego[:3, :3] = Quaternion(cs_record['rotation']).rotation_matrix
        sensor_to_ego[:3,  3] = np.array(cs_record['translation'])

        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = Quaternion(ep_record['rotation']).rotation_matrix
        ego_to_world[:3,  3] = np.array(ep_record['translation'])

        return torch.from_numpy((ego_to_world @ sensor_to_ego).astype(np.float32))

    def _get_timestamp(self, sample_data_token):
        """Returns timestamp in microseconds for a sample_data record."""
        return self.nusc.get('sample_data', sample_data_token)['timestamp']

    def _load_points(self, sample_data_token):
        """
        Loads a LiDAR point cloud from the .bin file.

        Returns Tensor(N, 5) for nuScenes (x, y, z, intensity, ring) or
        Tensor(N, 4) (x, y, z, intensity) for sensors that don't expose ring.
        Other worktrees that only consume the first four columns are unaffected.
        """
        sd_record = self.nusc.get('sample_data', sample_data_token)
        pcl_path  = os.path.join(self.dataroot, sd_record['filename'])
        raw       = np.fromfile(pcl_path, dtype=np.float32)
        if raw.size and raw.size % 5 == 0:
            scan = raw.reshape(-1, 5)            # x, y, z, intensity, ring
        elif raw.size and raw.size % 4 == 0:
            scan = raw.reshape(-1, 4)            # x, y, z, intensity
        else:
            raise ValueError(
                f"unexpected point cloud size {raw.size} in {pcl_path}; "
                "expected a multiple of 4 or 5 floats"
            )
        return torch.from_numpy(scan)

    def _load_boxes_in_sensor_frame(self, sample_token, lidar_token):
        """
        Load GT bounding boxes for a key sample, expressed in that frame's
        LiDAR sensor coordinate system.

        The transform is computed purely via quaternion arithmetic (no matrix
        inversion) to avoid floating-point orthogonality errors.

        Args:
            sample_token : nuScenes sample token (key sample only)
            lidar_token  : LIDAR_TOP sample_data token for this frame

        Returns:
            boxes          : Tensor(B, 7)  [x, y, z, w, l, h, yaw]
            instance_tokens: List[str] of length B
        """
        sd_record = self.nusc.get('sample_data', lidar_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Build sensor → world transform via quaternions
        q_s2e = Quaternion(cs_record['rotation'])    # sensor → ego
        t_s2e = np.array(cs_record['translation'])
        q_e2w = Quaternion(ep_record['rotation'])    # ego → world
        t_e2w = np.array(ep_record['translation'])

        q_s2w = q_e2w * q_s2e                        # sensor → world
        t_s2w = q_e2w.rotate(t_s2e) + t_e2w

        # Exact inverse: world → sensor
        q_w2s = q_s2w.inverse
        t_w2s = q_w2s.rotate(-t_s2w)

        R_w2s = q_w2s.rotation_matrix                # always orthogonal
        world_to_sensor        = np.eye(4, dtype=np.float32)
        world_to_sensor[:3, :3] = R_w2s
        world_to_sensor[:3,  3] = t_w2s

        # Load annotations and transform to sensor frame
        sample = self.nusc.get('sample', sample_token)
        boxes, instance_tokens = [], []

        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Center: global → sensor frame via homogeneous transform
            center_h      = np.array(ann['translation'] + [1.0], dtype=np.float32)
            center_sensor = world_to_sensor @ center_h
            x, y, z       = center_sensor[:3].tolist()

            # Size: nuScenes wlh = [width, length, height]
            w, l, h = ann['size']

            # Yaw: compose world quaternion into sensor frame, extract Z rotation
            q_global = Quaternion(ann['rotation'])
            yaw      = (q_w2s * q_global).yaw_pitch_roll[0]

            boxes.append([x, y, z, w, l, h, yaw])
            instance_tokens.append(ann['instance_token'])

        boxes_tensor = (torch.tensor(boxes, dtype=torch.float32)
                        if boxes else torch.zeros((0, 7), dtype=torch.float32))
        return boxes_tensor, instance_tokens

    def _load_pred_boxes_in_sensor_frame(self, sample_token, lidar_token):
        """
        Load predicted bounding boxes from self.bbox_data (LK3D + MCTrack results),
        expressed in the given frame's LiDAR sensor coordinate system.

        Uses the same world→sensor quaternion arithmetic as _load_boxes_in_sensor_frame,
        but reads from the pre-computed JSON instead of nuScenes GT annotations.

        Args:
            sample_token : nuScenes sample token (used to look up bbox_data)
            lidar_token  : LIDAR_TOP sample_data token for this frame

        Returns:
            boxes     : Tensor(B, 7)  [x, y, z, w, l, h, yaw] in LiDAR sensor frame
            track_ids : List[str]     tracking_id strings (cross-frame ID mapping)
        """
        entries = self.bbox_data.get(sample_token, [])
        if not entries:
            return torch.zeros((0, 7), dtype=torch.float32), []

        sd_record = self.nusc.get('sample_data', lidar_token)
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        ep_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

        # Build sensor → world via quaternions (same as _load_boxes_in_sensor_frame)
        q_s2e = Quaternion(cs_record['rotation'])
        t_s2e = np.array(cs_record['translation'])
        q_e2w = Quaternion(ep_record['rotation'])
        t_e2w = np.array(ep_record['translation'])

        q_s2w = q_e2w * q_s2e
        t_s2w = q_e2w.rotate(t_s2e) + t_e2w

        q_w2s = q_s2w.inverse
        t_w2s = q_w2s.rotate(-t_s2w)

        R_w2s              = q_w2s.rotation_matrix
        world_to_sensor        = np.eye(4, dtype=np.float32)
        world_to_sensor[:3, :3] = R_w2s
        world_to_sensor[:3,  3] = t_w2s

        boxes, track_ids = [], []
        for e in entries:
            # Center: global → sensor frame
            center_h      = np.array(e['translation'] + [1.0], dtype=np.float32)
            center_sensor = world_to_sensor @ center_h
            x, y, z       = center_sensor[:3].tolist()

            # Size: nuScenes wlh = [width, length, height]
            w, l, h = e['size']

            # Yaw: compose object quaternion into sensor frame
            q_obj_w = Quaternion(e['rotation'])
            yaw     = (q_w2s * q_obj_w).yaw_pitch_roll[0]

            boxes.append([x, y, z, w, l, h, yaw])
            track_ids.append(str(e['tracking_id']))

        return torch.tensor(boxes, dtype=torch.float32), track_ids

    @staticmethod
    def _assign_instance_ids(tokens_0, tokens_1):
        """
        Assign consistent integer IDs across two frames.
        All instance_tokens from both frames are sorted into a shared mapping so
        the same object gets the same integer ID in both frames.

        Returns:
            ids_0 : List[int]
            ids_1 : List[int]
        """
        all_tokens = sorted(set(tokens_0) | set(tokens_1))
        tok2id     = {tok: i for i, tok in enumerate(all_tokens)}
        return [tok2id[t] for t in tokens_0], [tok2id[t] for t in tokens_1]

    # ──────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        info = self.data_infos[idx]

        # ── Frame 0 (reference frame) ──────────────────────────────────────
        input_0_pts  = self._load_points(info['input_0'])
        input_0_pose = self._get_sensor_to_world_pose(info['input_0'])

        # All poses will be expressed relative to LiDAR_0.
        # world → LiDAR_0  (used to bring other frames into LiDAR_0 frame)
        world_to_input_0  = torch.linalg.inv(input_0_pose)
        rel_input_0_pose  = torch.eye(4)               # LiDAR_0 → LiDAR_0 = Identity

        # ── Frame 1 ────────────────────────────────────────────────────────
        input_1_pts  = self._load_points(info['input_1'])
        input_1_pose = self._get_sensor_to_world_pose(info['input_1'])
        # LiDAR_1 → World → LiDAR_0
        rel_input_1_pose = world_to_input_0 @ input_1_pose

        # ── Timestamps ────────────────────────────────────────────────────
        t0_us     = self._get_timestamp(info['input_0'])
        t1_us     = self._get_timestamp(info['input_1'])
        time_span = max(t1_us - t0_us, 1)             # guard against div-by-zero

        # ── Intermediate sweeps (GTs) ──────────────────────────────────────
        # All entries in the sample_data chain strictly between input_0 and input_1.
        # For frame_gap >= 2 this also includes any intermediate key samples,
        # though those intermediate key samples do NOT carry BBox annotations here.
        gts_pts, rel_gts_poses, gts_timestamps = [], [], []
        for gt_token in info['gts']:
            gts_pts.append(self._load_points(gt_token))
            gt_pose = self._get_sensor_to_world_pose(gt_token)
            rel_gts_poses.append(world_to_input_0 @ gt_pose)
            gt_us = self._get_timestamp(gt_token)
            # Normalised to [0, 1]: 0 = t0, 1 = t1 (dimensionless ratio)
            gts_timestamps.append((gt_us - t0_us) / time_span)

        gts_timestamps    = torch.tensor(gts_timestamps, dtype=torch.float32)
        input_1_timestamp = torch.tensor(1.0, dtype=torch.float32)

        result = {
            'input_0':           input_0_pts,         # Tensor(N0, 4)  LiDAR_0 frame
            'input_0_pose':      rel_input_0_pose,    # Tensor(4, 4)   Identity
            'input_1':           input_1_pts,         # Tensor(N1, 4)  LiDAR_1 frame
            'input_1_pose':      rel_input_1_pose,    # Tensor(4, 4)   LiDAR_1 → LiDAR_0
            'gts':               gts_pts,             # List[Tensor(K, 4)]
            'gts_poses':         rel_gts_poses,       # List[Tensor(4, 4)]  sweep → LiDAR_0
            'gts_timestamps':    gts_timestamps,      # Tensor(G,)  in [0, 1)
            'input_1_timestamp': input_1_timestamp,   # Tensor scalar = 1.0
        }

        # ── BBox mode ──────────────────────────────────────────────────────
        if self.mode == 'bbox':
            if self.bbox_data is not None:
                # Predicted BBox (LK3D + MCTrack offline results)
                boxes_0, ids_str_0 = self._load_pred_boxes_in_sensor_frame(
                    info['sample_token_0'], info['input_0'])
                boxes_1, ids_str_1 = self._load_pred_boxes_in_sensor_frame(
                    info['sample_token_1'], info['input_1'])
                ids_0, ids_1 = self._assign_instance_ids(ids_str_0, ids_str_1)
            else:
                # GT annotations (validation / debug)
                boxes_0, tokens_0 = self._load_boxes_in_sensor_frame(
                    info['sample_token_0'], info['input_0'])
                boxes_1, tokens_1 = self._load_boxes_in_sensor_frame(
                    info['sample_token_1'], info['input_1'])
                ids_0, ids_1 = self._assign_instance_ids(tokens_0, tokens_1)

            result['boxes_0']        = boxes_0                                  # Tensor(B0, 7) LiDAR sensor frame
            result['instance_ids_0'] = torch.tensor(ids_0, dtype=torch.int64)  # Tensor(B0,)
            result['boxes_1']        = boxes_1                                  # Tensor(B1, 7) LiDAR sensor frame
            result['instance_ids_1'] = torch.tensor(ids_1, dtype=torch.int64)  # Tensor(B1,)

        return result


def nvs_collate_fn(batch):
    """
    Custom collate function for NuScenesNVSDataset.

    Point clouds have variable N per frame, so they stay as lists.
    Pose matrices (fixed 4×4) are stacked into (B, 4, 4) tensors.
    BBox fields (variable number of boxes) also stay as lists when present.
    """
    result = {
        'input_0':      [item['input_0'] for item in batch],          # List[Tensor(N0, 4)]
        'input_0_pose':  torch.stack([item['input_0_pose'] for item in batch]),  # (B, 4, 4)
        'input_1':      [item['input_1'] for item in batch],          # List[Tensor(N1, 4)]
        'input_1_pose':  torch.stack([item['input_1_pose'] for item in batch]),  # (B, 4, 4)
        'gts':          [item['gts'] for item in batch],              # List[List[Tensor]]
        'gts_poses':    [torch.stack(item['gts_poses']) if item['gts_poses']
                         else torch.empty(0, 4, 4) for item in batch],  # List[Tensor(G,4,4)]
        'gts_timestamps': [item['gts_timestamps'] for item in batch], # List[Tensor(G,)]
        'input_1_timestamp': torch.stack(                             # Tensor(B,)
            [item['input_1_timestamp'] for item in batch]),
    }

    if 'boxes_0' in batch[0]:
        result['boxes_0']        = [item['boxes_0']        for item in batch]  # List[Tensor(B0,7)]
        result['instance_ids_0'] = [item['instance_ids_0'] for item in batch]  # List[Tensor(B0,)]
        result['boxes_1']        = [item['boxes_1']        for item in batch]  # List[Tensor(B1,7)]
        result['instance_ids_1'] = [item['instance_ids_1'] for item in batch]  # List[Tensor(B1,)]

    return result


if __name__ == '__main__':
    print("Initializing DataLoader test...")
    dataset = NuScenesNVSDataset(
        dataroot='/data1/nuScenes', version='v1.0-trainval',
        split='train', frame_gap=1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=nvs_collate_fn)

    print("\nFetching 1 batch to verify output shapes and types...")
    for batch in dataloader:
        print("\n" + "=" * 50)
        print("Batch Validation Results")
        print("=" * 50)

        print("[Input 0 - Base Frame]")
        print(f"- Batch size:          {len(batch['input_0'])}")
        print(f"- Point cloud (item 0): {batch['input_0'][0].shape}  (N, 4)")
        print(f"- Pose matrix:          {batch['input_0_pose'].shape}  (B, 4, 4)")

        print("\n[Input 1]")
        print(f"- Point cloud (item 0): {batch['input_1'][0].shape}  (M, 4)")
        print(f"- Pose matrix:          {batch['input_1_pose'].shape}  (B, 4, 4)")

        print("\n[GT Sweeps]")
        print(f"- Intermediate frames (item 0): {len(batch['gts'][0])}")
        if len(batch['gts'][0]) > 0:
            print(f"- First sweep shape:  {batch['gts'][0][0].shape}  (K, 4)")
        print(f"- GT poses (item 0):  {batch['gts_poses'][0].shape}  (G, 4, 4)")
        print(f"- Timestamps (item 0): {batch['gts_timestamps'][0].shape}  range "
              f"[{batch['gts_timestamps'][0].min():.3f}, {batch['gts_timestamps'][0].max():.3f}]  (expected ~[0, 1])")
        print("=" * 50)
        break
