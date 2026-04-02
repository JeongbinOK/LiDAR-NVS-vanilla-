"""Interactive 3D visualization of neural Gaussian clustering results.

Supports both 2D surfel (disc) and 3D Gaussian (ellipsoid) primitives.
Mode is auto-detected from the checkpoint's config.json.

Buttons:
  [Both]           — points + Gaussian shapes
  [Points only]    — point cloud colored by cluster
  [Gaussians only] — Gaussian shapes + axes/normals
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from config import NeuralClusteringConfig
sys.path.insert(0, os.path.join(os.path.expanduser(NeuralClusteringConfig.data_root), "loader"))
from dataset import NuScenesNVSDataset

from nn.model import NeuralClusteringModel
from nn.gaussian_head import quaternion_to_rotation_matrix


def _load_cfg_from_ckpt(checkpoint_path: str) -> NeuralClusteringConfig:
    ckpt_dir = os.path.dirname(checkpoint_path)
    run_dir  = os.path.dirname(ckpt_dir)
    config_path = os.path.join(run_dir, "configs", "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            d = json.load(f)
        cfg = NeuralClusteringConfig(**{
            k: v for k, v in d.items()
            if k in NeuralClusteringConfig.__dataclass_fields__
        })
        print(f"Loaded config from {config_path}")
        return cfg
    print("config.json not found, using NeuralClusteringConfig defaults")
    return NeuralClusteringConfig()


def _q_to_axes(q_np: np.ndarray):
    """Compute principal axes from [K, 4] quaternions.

    Uses the same row convention as gaussian_head.py:
      u = R[:, 0, :]  (first row)
      v = R[:, 1, :]  (second row)
      n = cross(u, v)

    Ellipsoid formula:  pts_world = (sphere * s) @ R + mu
    where R[k] rows are the world-space principal axis directions.

    Returns u, v, n each [K, 3], and R_all [K, 3, 3].
    """
    q_t = torch.tensor(q_np, dtype=torch.float32)
    R_t = quaternion_to_rotation_matrix(q_t)   # [K, 3, 3]
    R_np = R_t.numpy()
    u = R_np[:, 0, :]                          # [K, 3]
    v = R_np[:, 1, :]                          # [K, 3]
    n = np.cross(u, v)                         # [K, 3]
    return u, v, n, R_np


# ── 2D surfel ─────────────────────────────────────────────────────────────────

def _build_surfel_mesh(mu, u, v, s, palette, n_theta=32, min_vis_scale=1.5):
    """Batched Mesh3d for all K surfel discs (2D mode).

    Each disc is a fan-triangulated ellipse at 1-sigma in the u-v plane.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    vertex_colors = []

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    offset = 0
    for k_idx in range(len(mu)):
        color = palette[k_idx % len(palette)]
        sk = np.maximum(s[k_idx], min_vis_scale)  # [2]

        circle = (mu[k_idx]
                  + sk[0] * cos_t[:, None] * u[k_idx]
                  + sk[1] * sin_t[:, None] * v[k_idx])  # [T, 3]
        verts = np.vstack([mu[k_idx], circle])           # [T+1, 3]

        all_x.extend(verts[:, 0])
        all_y.extend(verts[:, 1])
        all_z.extend(verts[:, 2])
        vertex_colors.extend([color] * (n_theta + 1))

        for t in range(n_theta):
            all_i.append(offset)
            all_j.append(offset + 1 + t)
            all_k.append(offset + 1 + (t + 1) % n_theta)

        offset += n_theta + 1

    return (np.array(all_x), np.array(all_y), np.array(all_z),
            all_i, all_j, all_k, vertex_colors)


def _build_normal_lines(mu, n, scale=0.5):
    xs, ys, zs = [], [], []
    ends = mu + n * scale
    for k in range(len(mu)):
        xs += [mu[k, 0], ends[k, 0], None]
        ys += [mu[k, 1], ends[k, 1], None]
        zs += [mu[k, 2], ends[k, 2], None]
    return xs, ys, zs


# ── 3D ellipsoid ──────────────────────────────────────────────────────────────

def _build_ellipsoid_mesh(mu, q, s, palette, n_lat=10, n_lon=16, min_vis_scale=0.3):
    """Batched Mesh3d for all K ellipsoids (3D mode).

    Parametric construction:
      pts_world = (unit_sphere * sk) @ R + mu

    where R[k] rows are the world-space principal axis directions,
    consistent with _q_to_axes and gaussian_head.py's row convention.
    """
    all_x, all_y, all_z = [], [], []
    all_i, all_j, all_k = [], [], []
    vertex_colors = []

    # Unit sphere vertices [n_lat * n_lon, 3]
    lat = np.linspace(0, np.pi, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    sphere = np.stack([
        np.outer(np.sin(lat), np.cos(lon)),
        np.outer(np.sin(lat), np.sin(lon)),
        np.tile(np.cos(lat)[:, None], (1, n_lon)),
    ], axis=-1).reshape(-1, 3)  # [n_lat*n_lon, 3]
    n_verts = sphere.shape[0]

    # Quad → 2 triangles, precomputed indices (same for every ellipsoid)
    ti, tj, tk = [], [], []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            v00 = i * n_lon + j;       v01 = i * n_lon + j1
            v10 = (i + 1) * n_lon + j; v11 = (i + 1) * n_lon + j1
            ti += [v00, v00]; tj += [v10, v01]; tk += [v11, v11]

    # Rotation matrices from quaternions (rows = world-space principal axes)
    q_t = torch.tensor(q, dtype=torch.float32)
    R_all = quaternion_to_rotation_matrix(q_t).numpy()  # [K, 3, 3]

    offset = 0
    for k_idx in range(len(mu)):
        color = palette[k_idx % len(palette)]
        sk = np.maximum(s[k_idx], min_vis_scale)   # [3]

        pts = (sphere * sk) @ R_all[k_idx] + mu[k_idx]  # [V, 3]

        all_x.extend(pts[:, 0])
        all_y.extend(pts[:, 1])
        all_z.extend(pts[:, 2])
        vertex_colors.extend([color] * n_verts)

        all_i.extend(v + offset for v in ti)
        all_j.extend(v + offset for v in tj)
        all_k.extend(v + offset for v in tk)
        offset += n_verts

    return (np.array(all_x), np.array(all_y), np.array(all_z),
            all_i, all_j, all_k, vertex_colors)


def _build_axis_lines(mu, R_all, s, scale_factor=1.0):
    """Show the smallest-scale (thinnest) principal axis per Gaussian.

    This axis is the most 'normal-like' direction for flat/thin Gaussians.
    axis_dir = R_all[k, argmin(s[k]), :] (row = world-space direction).
    """
    xs, ys, zs = [], [], []
    for k in range(len(mu)):
        min_ax = int(np.argmin(s[k]))
        axis_dir = R_all[k, min_ax, :]
        length = float(s[k, min_ax]) * scale_factor
        end = mu[k] + axis_dir * length
        xs += [float(mu[k, 0]), float(end[0]), None]
        ys += [float(mu[k, 1]), float(end[1]), None]
        zs += [float(mu[k, 2]), float(end[2]), None]
    return xs, ys, zs


# ── main visualize ─────────────────────────────────────────────────────────────

def visualize(checkpoint: str, scene_num: int, split: str,
              data_root: str, device: str, tau: float, out: str):
    import plotly.graph_objects as go

    # ── 1. Model ──────────────────────────────────────────────────────────
    cfg = _load_cfg_from_ckpt(checkpoint)
    cfg.device = device
    primitive = cfg.primitive_type  # "2d" or "3d"

    model = NeuralClusteringModel(cfg).to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    epoch      = ckpt.get("epoch", "?")
    train_loss = ckpt.get("loss", float("nan"))
    print(f"Checkpoint: epoch={epoch}, train_loss={train_loss:.4f}, primitive={primitive}")

    # ── 2. Data ───────────────────────────────────────────────────────────
    dataset = NuScenesNVSDataset(
        dataroot=os.path.expanduser(data_root),
        version="v1.0-trainval", split=split,
    )
    if scene_num >= len(dataset):
        raise ValueError(f"--scene-num {scene_num} out of range (size={len(dataset)})")

    pts = dataset[scene_num]["input_0"]  # [N, 4]

    # ── 3. Inference ──────────────────────────────────────────────────────
    xyz_raw   = pts[:, :3].to(device)
    intensity = pts[:, 3].to(device)
    mask      = torch.norm(xyz_raw, dim=1) > cfg.ego_radius
    xyz       = xyz_raw[mask]
    intensity = intensity[mask]

    with torch.no_grad():
        output = model(xyz, intensity, tau=tau)

    gaussians = output["gaussians"]
    assign    = output["assign"]

    mu     = gaussians["mu"].cpu().numpy()  # [K, 3]
    q      = gaussians["q"].cpu().numpy()   # [K, 4]
    s      = gaussians["s"].cpu().numpy()   # [K, 2] or [K, 3]
    xyz_np = xyz.cpu().numpy()              # [N, 3]
    hard   = assign.argmax(dim=1).cpu().numpy()  # [N]

    K = mu.shape[0]
    N = xyz_np.shape[0]
    print(f"Scene {scene_num}: N={N} points, K={K} Gaussians")
    print(f"Scale stats: min={s.min():.3f}  mean={s.mean():.3f}  max={s.max():.3f}")

    # Principal axes from q (used by both modes for normal/axis display)
    u_vec, v_vec, n_vec, R_all = _q_to_axes(q)

    # ── 4. Colors ─────────────────────────────────────────────────────────
    palette = [
        "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
        "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
        "#469990","#dcbeff","#9A6324","#fffac8","#800000",
        "#aaffc3","#808000","#ffd8b1","#000075","#a9a9a9",
    ]
    pt_colors  = [palette[int(c) % len(palette)] for c in hard]
    sur_colors = [palette[k % len(palette)] for k in range(K)]

    # ── 5. Build traces ───────────────────────────────────────────────────
    # Trace 0: point cloud colored by cluster assignment
    trace_pts = go.Scatter3d(
        x=xyz_np[:, 0], y=xyz_np[:, 1], z=xyz_np[:, 2],
        mode="markers",
        marker=dict(size=1.2, color=pt_colors, opacity=0.6),
        name="Points",
        hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
    )

    # Trace 1: Gaussian shapes
    if primitive == "2d":
        # Prefer model-output u/v if available, else derive from q
        u_vis = gaussians["u"].cpu().numpy() if "u" in gaussians else u_vec
        v_vis = gaussians["v"].cpu().numpy() if "v" in gaussians else v_vec
        vx, vy, vz, fi, fj, fk, vcol = _build_surfel_mesh(
            mu, u_vis, v_vis, s, palette)
        shape_name = "Surfels"
    else:
        vx, vy, vz, fi, fj, fk, vcol = _build_ellipsoid_mesh(
            mu, q, s, palette)
        shape_name = "Ellipsoids"

    trace_shapes = go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=fi, j=fj, k=fk,
        vertexcolor=vcol,
        opacity=0.55,
        name=shape_name,
        hoverinfo="skip",
        showlegend=True,
        showscale=False,
    )

    # Trace 2: Gaussian centres
    trace_centers = go.Scatter3d(
        x=mu[:, 0], y=mu[:, 1], z=mu[:, 2],
        mode="markers",
        marker=dict(size=3, color=sur_colors, symbol="x"),
        name="Centers",
        hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
    )

    # Trace 3: surface normals (2D) or thinnest-axis arrows (3D)
    if primitive == "2d":
        n_vis = gaussians["n"].cpu().numpy() if "n" in gaussians else n_vec
        ax_x, ax_y, ax_z = _build_normal_lines(mu, n_vis, scale=0.5)
        axis_name = "Normals"
    else:
        ax_x, ax_y, ax_z = _build_axis_lines(mu, R_all, s, scale_factor=1.0)
        axis_name = "Principal axes"

    trace_axes = go.Scatter3d(
        x=ax_x, y=ax_y, z=ax_z,
        mode="lines",
        line=dict(width=2, color="rgba(255,160,0,0.7)"),
        name=axis_name,
        hoverinfo="skip",
    )

    fig = go.Figure(data=[trace_pts, trace_shapes, trace_centers, trace_axes])

    # ── 6. Toggle buttons ─────────────────────────────────────────────────
    mode_str = "2D surfel" if primitive == "2d" else "3D Gaussian"
    btn_both = dict(
        label="Both", method="update",
        args=[{"visible": [True, True, True, True]},
              {"title.text": f"Scene {scene_num} | epoch={epoch} | Both"}],
    )
    btn_pts = dict(
        label="Points only", method="update",
        args=[{"visible": [True, False, False, False]},
              {"title.text": f"Scene {scene_num} | epoch={epoch} | Points only"}],
    )
    btn_gauss = dict(
        label=f"{shape_name} only", method="update",
        args=[{"visible": [False, True, True, True]},
              {"title.text": f"Scene {scene_num} | epoch={epoch} | {shape_name} only"}],
    )

    # ── 7. Layout ─────────────────────────────────────────────────────────
    dark_bg = "rgb(15,15,20)"
    fig.update_layout(
        title=dict(
            text=(f"Scene {scene_num} | epoch={epoch} | N={N} pts | K={K} "
                  f"| {mode_str} | Both"),
            font=dict(color="white", size=13),
        ),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            bgcolor=dark_bg,
            xaxis=dict(backgroundcolor=dark_bg, gridcolor="rgb(40,40,50)", color="white"),
            yaxis=dict(backgroundcolor=dark_bg, gridcolor="rgb(40,40,50)", color="white"),
            zaxis=dict(backgroundcolor=dark_bg, gridcolor="rgb(40,40,50)", color="white"),
        ),
        paper_bgcolor=dark_bg,
        font=dict(color="white"),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(30,30,30,0.85)",
            bordercolor="rgba(255,255,255,0.15)",
            font=dict(size=11),
        ),
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.5, xanchor="center",
            y=1.06, yanchor="top",
            pad={"r": 6, "t": 6},
            showactive=True,
            bgcolor="rgba(40,40,50,0.9)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(color="white", size=12),
            buttons=[btn_both, btn_pts, btn_gauss],
        )],
        margin=dict(l=0, r=0, t=70, b=0),
    )

    # ── 8. Save ───────────────────────────────────────────────────────────
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(out)
    out_final = f"{base}_ep{epoch}{ext}.html"
    fig.write_html(out_final, include_plotlyjs="cdn")
    print(f"Saved: {out_final}")


def main():
    _defaults = NeuralClusteringConfig()
    parser = argparse.ArgumentParser(
        description="Interactive 3D visualization of Gaussian clustering")
    parser.add_argument("--checkpoint", default="outputs/train_001/ckpt/best_model.pt")
    parser.add_argument("--scene-num", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", default=_defaults.data_root)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--out", default=None,
                        help="Output HTML path (default: outputs/viz_scene_NNN.html)")
    args = parser.parse_args()

    if args.out is None:
        args.out = f"outputs/viz_scene_{args.scene_num:03d}.html"

    visualize(
        checkpoint=args.checkpoint,
        scene_num=args.scene_num,
        split=args.split,
        data_root=args.data_root,
        device=args.device,
        tau=args.tau,
        out=args.out,
    )


if __name__ == "__main__":
    main()
