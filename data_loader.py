"""nuScenes .pcd.bin data loading."""

import glob
import os

import numpy as np
import torch


def load_pcd_bin(path: str, device: str = "cuda") -> dict:
    """Load a nuScenes .pcd.bin file.

    Format: N x 5 float32 (x, y, z, intensity, ring).

    Returns dict with 'xyz' [N,3], 'intensity' [N], 'ring' [N] tensors.
    """
    data = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    xyz = torch.from_numpy(data[:, :3]).to(device)
    intensity = torch.from_numpy(data[:, 3]).to(device)
    ring = torch.from_numpy(data[:, 4]).to(device)
    return {"xyz": xyz, "intensity": intensity, "ring": ring}


def list_lidar_files(nuscenes_root: str) -> list:
    """List all LIDAR_TOP .pcd.bin files under nuScenes root."""
    pattern = os.path.join(nuscenes_root, "samples", "LIDAR_TOP", "*.pcd.bin")
    files = sorted(glob.glob(pattern))
    return files
