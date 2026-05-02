"""Interactive HTML visualization: LiDAR 2-frame static points + spherical voxel boundaries.

Reuses data loading / processing logic from phase_a_voxel_stats.py.
Output: /data/jeongbin/qgs/scripts/viz_voxel_b_sample.html
"""

import os
import sys
import random
import numpy as np
import torch

# Reuse helpers from phase_a_voxel_stats
sys.path.insert(0, '/data/jeongbin/qgs/scripts')
from phase_a_voxel_stats import (
    transform_xyz,
    points_in_box,
    static_mask,
    spherical_voxelize,
)

# Dataset loader
sys.path.insert(0, '/data1/nuScenes/loader')
from dataset import NuScenesNVSDataset  # noqa: E402

import plotly.graph_objects as go

# ─── Config ──────────────────────────────────────────────────────────────────
DPHI_DEG    = 3.0
DTHETA_DEG  = 4.0
DR_M        = 3.0
K_MIN       = 8
SEED        = 42
PREFERRED_IDX = 12345
OUTPUT_PATH = '/data/jeongbin/qgs/scripts/viz_voxel_b_sample.html'
EDGE_SAMPLES = 2      # points per edge (arcs are ≤4° so n=2 is visually indistinguishable)

# ─── Spherical helpers ────────────────────────────────────────────────────────

def sph_to_cart(r, theta, phi):
    """theta = elevation, phi = azimuth (radians). Supports broadcasting scalars/arrays."""
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    # Broadcast all to the same shape
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return np.stack([x, y, z], axis=-1)


def voxel_edges(r_lo, r_hi, theta_lo, theta_hi, phi_lo, phi_hi, n=EDGE_SAMPLES):
    """Return list of (N,3) arrays, one per edge of the voxel in Cartesian space.

    12 edges:
      - 4 radial  (r varies, theta/phi fixed at corners)  — straight
      - 4 phi-arc (phi varies, theta/r fixed at corners)  — arc
      - 4 theta-arc (theta varies, phi/r fixed at corners) — arc
    """
    t = np.linspace(0, 1, n)
    edges = []

    # radial edges — 4 combos of (theta, phi) corners
    for th in (theta_lo, theta_hi):
        for ph in (phi_lo, phi_hi):
            r_vals = r_lo + t * (r_hi - r_lo)
            edges.append(sph_to_cart(r_vals, th, ph))

    # phi-arc edges — 4 combos of (r, theta) corners
    for r in (r_lo, r_hi):
        for th in (theta_lo, theta_hi):
            ph_vals = phi_lo + t * (phi_hi - phi_lo)
            edges.append(sph_to_cart(r, th, ph_vals))

    # theta-arc edges — 4 combos of (r, phi) corners
    for r in (r_lo, r_hi):
        for ph in (phi_lo, phi_hi):
            th_vals = theta_lo + t * (theta_hi - theta_lo)
            edges.append(sph_to_cart(r, th_vals, ph))

    return edges  # list of 12 arrays, each (n, 3)


def build_voxel_trace(occupied_voxel_indices, dphi, dtheta, dr, n_phi, n_theta):
    """Build a single Scatter3d trace for ALL occupied voxel edges using None separators."""
    n_draw = len(occupied_voxel_indices)

    xs, ys, zs = [], [], []
    for vid in occupied_voxel_indices:
        iphi_v   = int(vid % n_phi)
        itheta_v = int((vid // n_phi) % n_theta)
        ir_v     = int(vid // (n_phi * n_theta))

        phi_lo   = iphi_v * dphi - np.pi
        phi_hi   = phi_lo + dphi
        theta_lo = itheta_v * dtheta - np.pi / 2
        theta_hi = theta_lo + dtheta
        r_lo     = ir_v * dr
        r_hi     = r_lo + dr

        for edge_pts in voxel_edges(r_lo, r_hi, theta_lo, theta_hi, phi_lo, phi_hi):
            ep = np.round(edge_pts, 2)
            xs.extend(ep[:, 0].tolist() + [None])
            ys.extend(ep[:, 1].tolist() + [None])
            zs.extend(ep[:, 2].tolist() + [None])

    trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(dash='dash', width=2, color='#9b59b6'),
        name=f'Voxel boundaries (all occupied, {n_draw} cells)',
        hoverinfo='skip',
    )
    return trace, n_draw


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rng_np = np.random.default_rng(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading dataset (train, frame_gap=2, mode=bbox)...")
    ds = NuScenesNVSDataset(
        dataroot='/data1/nuScenes',
        version='v1.0-trainval',
        split='train',
        frame_gap=2,
        mode='bbox',
        bbox_json_path=None,
    )
    print(f"Dataset size: {len(ds)}")

    idx = PREFERRED_IDX if PREFERRED_IDX < len(ds) else int(rng_np.integers(0, len(ds)))
    print(f"Using sample index: {idx}")

    item = ds[idx]
    pts0 = item['input_0']          # (N0, 4) LiDAR_0 frame
    pts1 = item['input_1']          # (N1, 4) LiDAR_1 frame
    T_1to0 = item['input_1_pose']   # (4, 4)

    boxes_0 = item.get('boxes_0', torch.zeros((0, 7)))
    boxes_1 = item.get('boxes_1', torch.zeros((0, 7)))
    ids_0 = item.get('instance_ids_0', torch.zeros((0,), dtype=torch.int64)).tolist()
    ids_1 = item.get('instance_ids_1', torch.zeros((0,), dtype=torch.int64)).tolist()
    common_ids = set(ids_0) & set(ids_1)

    mask0 = torch.tensor([_id in common_ids for _id in ids_0], dtype=torch.bool)
    mask1 = torch.tensor([_id in common_ids for _id in ids_1], dtype=torch.bool)
    dyn_boxes_0 = boxes_0[mask0] if boxes_0.shape[0] > 0 else boxes_0
    dyn_boxes_1 = boxes_1[mask1] if boxes_1.shape[0] > 0 else boxes_1

    keep0 = static_mask(pts0[:, :3], dyn_boxes_0)
    keep1 = static_mask(pts1[:, :3], dyn_boxes_1)
    pts0_static = pts0[keep0]
    pts1_static = pts1[keep1]
    xyz1_in0 = transform_xyz(pts1_static[:, :3], T_1to0)

    n0 = pts0_static.shape[0]
    n1 = pts1_static.shape[0]
    print(f"Static points — Frame0: {n0}, Frame1: {n1}, dynamic instances: {len(common_ids)}")

    # ── Voxelize combined cloud to find valid voxels ─────────────────────────
    all_xyz = torch.cat([pts0_static[:, :3], xyz1_in0], dim=0)
    dphi   = np.deg2rad(DPHI_DEG)
    dtheta = np.deg2rad(DTHETA_DEG)
    dr     = DR_M
    n_phi   = int(np.ceil(2 * np.pi / dphi)) + 1
    n_theta = int(np.ceil(np.pi / dtheta)) + 1

    voxel_id, _ = spherical_voxelize(all_xyz, DPHI_DEG, DTHETA_DEG, DR_M)
    unique_vids, counts = torch.unique(voxel_id, return_counts=True)

    valid_mask = counts >= K_MIN
    valid_vids = unique_vids[valid_mask].numpy()
    n_valid = len(valid_vids)
    print(f"Valid voxels (count≥{K_MIN}): {n_valid}")

    # All occupied voxels (count >= 1)
    occupied_vids = unique_vids.numpy()
    n_occupied = len(occupied_vids)
    print(f"Occupied voxels (count≥1): {n_occupied}")

    # ── Build Plotly traces ──────────────────────────────────────────────────
    xyz0_np = pts0_static[:, :3].numpy()
    xyz1_np = xyz1_in0.numpy()

    trace_f0 = go.Scatter3d(
        x=xyz0_np[:, 0], y=xyz0_np[:, 1], z=xyz0_np[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='rgba(50,100,255,0.6)', symbol='circle'),
        name=f'Frame 0 ({n0:,} pts)',
    )
    trace_f1 = go.Scatter3d(
        x=xyz1_np[:, 0], y=xyz1_np[:, 1], z=xyz1_np[:, 2],
        mode='markers',
        marker=dict(size=1.5, color='rgba(255,60,60,0.6)', symbol='circle'),
        name=f'Frame 1 ({n1:,} pts)',
    )

    voxel_trace, n_drawn = build_voxel_trace(occupied_vids, dphi, dtheta, dr, n_phi, n_theta)
    print(f"Drawing {n_drawn} voxel boxes (all occupied)")

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = go.Figure(data=[trace_f0, trace_f1, voxel_trace])
    fig.update_layout(
        title=dict(
            text=(
                f'LiDAR Static Points + Spherical Voxel Boundaries<br>'
                f'<sup>Sample idx={idx} | Δφ={DPHI_DEG}° Δθ={DTHETA_DEG}° Δr={DR_M}m | '
                f'Valid voxels={n_valid} | Occupied cells={n_occupied} (all drawn)</sup>'
            ),
            font=dict(size=14),
        ),
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (m)', showgrid=True, gridcolor='#444'),
            yaxis=dict(title='Y (m)', showgrid=True, gridcolor='#444'),
            zaxis=dict(title='Z (m)', showgrid=True, gridcolor='#444'),
            bgcolor='rgb(20,20,30)',
        ),
        paper_bgcolor='rgb(30,30,40)',
        plot_bgcolor='rgb(30,30,40)',
        font=dict(color='white'),
        legend=dict(
            x=0.01, y=0.98,
            bgcolor='rgba(40,40,50,0.8)',
            bordercolor='gray',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # ── Write HTML ────────────────────────────────────────────────────────────
    fig.write_html(OUTPUT_PATH, include_plotlyjs='cdn')
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nHTML written: {OUTPUT_PATH}")
    print(f"File size: {file_size / 1024:.1f} KB  ({file_size:,} bytes)")
    print(f"\nSummary:")
    print(f"  Sample index   : {idx}")
    print(f"  Frame 0 pts    : {n0:,}")
    print(f"  Frame 1 pts    : {n1:,}")
    print(f"  Dynamic insts  : {len(common_ids)}")
    print(f"  Valid voxels   : {n_valid}")
    print(f"  Occupied voxels: {n_occupied}")
    print(f"  Drawn voxels   : {n_drawn}")
    print(f"  HTML file size : {file_size / 1024:.1f} KB")


if __name__ == '__main__':
    main()
