"""Interactive 3D visualization of neural 2D Gaussian clustering results.

Loads a checkpoint, runs inference on a val scene, and saves an interactive
HTML file viewable in any browser (rotate, zoom, pan).

Buttons:
  [Both]          — points + Gaussian surfels
  [Points only]   — point cloud colored by cluster
  [Gaussians only]— filled surfel discs + normals
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/data/nuScenes/loader"))
from dataset import NuScenesNVSDataset

from config import NeuralClusteringConfig
from nn.model import NeuralClusteringModel


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


def _build_surfel_mesh(mu, u, v, s, palette, n_theta=32, min_vis_scale=1.5):
    """
    Build a single batched Mesh3d for all K surfel discs using vertexcolor.

    Each surfel is a fan-triangulated ellipse disc:
      vertex 0      = centre (mu[k])
      vertices 1..T = ellipse boundary at 1-sigma (using actual u, v, s)

    min_vis_scale: minimum radius (metres) so small surfels remain visible.
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

        # Clamp scale for visual clarity (actual shape preserved, just floored)
        sk = np.maximum(s[k_idx], min_vis_scale)  # [2]

        # Ellipse boundary: mu + sk[0]*cos(t)*u + sk[1]*sin(t)*v
        circle = (mu[k_idx]
                  + sk[0] * cos_t[:, None] * u[k_idx]
                  + sk[1] * sin_t[:, None] * v[k_idx])  # [T, 3]

        verts = np.vstack([mu[k_idx], circle])   # [T+1, 3]
        all_x.extend(verts[:, 0])
        all_y.extend(verts[:, 1])
        all_z.extend(verts[:, 2])
        vertex_colors.extend([color] * (n_theta + 1))

        # Fan triangles from centre
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


def visualize(checkpoint: str, scene_num: int, split: str,
              data_root: str, device: str, tau: float, out: str):
    import plotly.graph_objects as go

    # ── 1. Model ──────────────────────────────────────────────────────────
    cfg = _load_cfg_from_ckpt(checkpoint)
    cfg.device = device

    model = NeuralClusteringModel(cfg).to(device)
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    epoch      = ckpt.get("epoch", "?")
    train_loss = ckpt.get("loss", float("nan"))
    print(f"Checkpoint: epoch={epoch}, train_loss={train_loss:.4f}")

    # ── 2. Data ───────────────────────────────────────────────────────────
    dataset = NuScenesNVSDataset(
        dataroot=os.path.expanduser(data_root),
        version="v1.0-trainval", split=split,
    )
    if scene_num >= len(dataset):
        raise ValueError(f"--scene-num {scene_num} out of range (size={len(dataset)})")

    pts = dataset[scene_num]["input_0"]   # [N, 4]

    # ── 3. Inference ──────────────────────────────────────────────────────
    xyz_raw   = pts[:, :3].to(device)
    intensity = pts[:, 3].to(device)
    mask      = torch.norm(xyz_raw, dim=1) > cfg.ego_radius
    xyz       = xyz_raw[mask]
    intensity = intensity[mask]

    with torch.no_grad():
        output = model(xyz, intensity, tau=tau)

    gaussians      = output["gaussians"]
    assign         = output["assign"]

    mu     = gaussians["mu"].cpu().numpy()   # [K, 3]
    n_vec  = gaussians["n"].cpu().numpy()    # [K, 3]
    u_vec  = gaussians["u"].cpu().numpy()    # [K, 3]
    v_vec  = gaussians["v"].cpu().numpy()    # [K, 3]
    s      = gaussians["s"].cpu().numpy()    # [K, 2]
    xyz_np = xyz.cpu().numpy()               # [N, 3]

    hard = assign.argmax(dim=1).cpu().numpy()  # [N]

    K = mu.shape[0]
    N = xyz_np.shape[0]
    print(f"Scene {scene_num}: N={N} points, K={K} surfels")
    print(f"Scale stats: min={s.min():.3f}  mean={s.mean():.3f}  max={s.max():.3f}")

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
    # Trace 0: point cloud (cluster-colored)
    trace_pts = go.Scatter3d(
        x=xyz_np[:, 0], y=xyz_np[:, 1], z=xyz_np[:, 2],
        mode="markers",
        marker=dict(size=1.2, color=pt_colors, opacity=0.6),
        name="Points",
        hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
    )

    # Trace 1: surfel filled discs (Mesh3d, vertexcolor)
    vx, vy, vz, fi, fj, fk, vcolors = _build_surfel_mesh(mu, u_vec, v_vec, s, palette)
    trace_discs = go.Mesh3d(
        x=vx, y=vy, z=vz,
        i=fi, j=fj, k=fk,
        vertexcolor=vcolors,
        opacity=0.65,
        name="Surfels",
        hoverinfo="skip",
        showlegend=True,
        showscale=False,
    )

    # Trace 2: surfel centres
    trace_centers = go.Scatter3d(
        x=mu[:, 0], y=mu[:, 1], z=mu[:, 2],
        mode="markers",
        marker=dict(size=3, color=sur_colors, symbol="x"),
        name="Surfel centres",
        hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
    )

    # Trace 3: normal vectors
    nx, ny, nz = _build_normal_lines(mu, n_vec, scale=0.5)
    trace_normals = go.Scatter3d(
        x=nx, y=ny, z=nz,
        mode="lines",
        line=dict(width=2, color="rgba(255,160,0,0.7)"),
        name="Normals",
        hoverinfo="skip",
    )

    fig = go.Figure(data=[trace_pts, trace_discs, trace_centers, trace_normals])

    # ── 6. Toggle buttons ─────────────────────────────────────────────────
    # Trace order: 0=pts  1=discs  2=centers  3=normals
    btn_both = dict(
        label="Both",
        method="update",
        args=[{"visible": [True, True, True, True]},
              {"title.text": f"Scene {scene_num} | epoch={epoch} | Both"}],
    )
    btn_pts = dict(
        label="Points only",
        method="update",
        args=[{"visible": [True, False, False, False]},
              {"title.text": f"Scene {scene_num} | epoch={epoch} | Points only"}],
    )
    btn_gauss = dict(
        label="Gaussians only",
        method="update",
        args=[{"visible": [False, True, True, True]},
              {"title.text": f"Scene {scene_num} | epoch={epoch} | Gaussians only"}],
    )

    # ── 7. Layout ─────────────────────────────────────────────────────────
    dark_bg = "rgb(15,15,20)"
    fig.update_layout(
        title=dict(
            text=f"Scene {scene_num} | epoch={epoch} | N={N} pts | K={K} surfels | Both",
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
    # Insert epoch into filename: foo.html → foo_ep16.html
    base, ext = os.path.splitext(out)
    out_final = f"{base}_ep{epoch}{ext}.html"
    fig.write_html(out_final, include_plotlyjs="cdn")
    print(f"Saved: {out_final}")


def main():
    parser = argparse.ArgumentParser(description="Interactive 3D visualization of clustering")
    parser.add_argument("--checkpoint", default="outputs/train_001/ckpt/best_model.pt")
    parser.add_argument("--scene-num", type=int, default=0)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", default=os.path.expanduser("~/data/nuScenes"))
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
