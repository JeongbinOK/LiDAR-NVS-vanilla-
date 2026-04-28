"""Visualize predicted Quadratic Gaussian surface patches from a QGS checkpoint."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from config import QGSConfig
from models.geometry import decompose_scene
from nn.eval_utils import (
    _box_to_pose,
    _ensure_2d_pose,
    concat_primitives,
    filter_frame_points,
    load_cfg_from_checkpoint,
    resolve_bbox_json,
    transform_primitives,
)
from nn.model import QGSModel
from nn.render_utils import quat_to_rotmat


def _latest_checkpoint() -> str:
    candidates = list(Path("outputs").glob("train_*/ckpt/best_model.pt"))
    candidates.extend(Path("outputs").glob("train_*/ckpt/epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError("no checkpoints found under outputs/train_*/ckpt")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def _load_dataset(cfg: QGSConfig, split: str):
    target_split = cfg.train_split if split == "train" else cfg.eval_split
    sys.path.insert(0, os.path.join(os.path.expanduser(cfg.data_root), "loader"))
    from dataset import NuScenesNVSDataset  # noqa: WPS433

    bbox_json = resolve_bbox_json(cfg, target_split)
    return NuScenesNVSDataset(
        dataroot=os.path.expanduser(cfg.data_root),
        version=cfg.nuscenes_version,
        split=target_split,
        frame_gap=cfg.frame_gap,
        mode=cfg.dataset_mode,
        bbox_json_path=bbox_json if (cfg.dataset_mode == "bbox" and bbox_json) else None,
    )


def _build_context(
    model: QGSModel,
    cfg: QGSConfig,
    xyz: torch.Tensor,
    intensity: torch.Tensor,
    *,
    context_type: str,
    time_scalar: torch.Tensor,
    ego_motion: torch.Tensor,
    is_dynamic_flag: torch.Tensor,
) -> dict | None:
    if xyz.shape[0] < cfg.knn_k_min:
        return None
    return model.forward_context(
        xyz,
        intensity,
        context_type=context_type,
        time_scalar=time_scalar,
        ego_motion=ego_motion,
        is_dynamic_flag=is_dynamic_flag,
        neighbor_xyz=xyz,
        return_diagnostics=False,
    )


def _predict_frame0_primitives(model: QGSModel, cfg: QGSConfig, sample: dict, device: str):
    p0 = sample["input_0"].to(device)
    p1 = sample["input_1"].to(device)
    rel_pose = _ensure_2d_pose(sample["input_1_pose"]).to(device)

    xyz0 = p0[:, :3].float()
    xyz1 = p1[:, :3].float()
    i0 = (p0[:, 3].float() / 255.0).clamp(0.0, 1.0)
    i1 = (p1[:, 3].float() / 255.0).clamp(0.0, 1.0)
    xyz0, i0 = filter_frame_points(xyz0, i0, cfg)
    xyz1, i1 = filter_frame_points(xyz1, i1, cfg)

    boxes_0 = sample.get("boxes_0", torch.empty(0, 7)).to(device)
    boxes_1 = sample.get("boxes_1", torch.empty(0, 7)).to(device)
    ids0 = sample.get("instance_ids_0", torch.empty(0, dtype=torch.long)).to(device)
    ids1 = sample.get("instance_ids_1", torch.empty(0, dtype=torch.long)).to(device)

    if boxes_0.numel() and boxes_1.numel():
        scene = decompose_scene(xyz0, xyz1, i0, i1, boxes_0, boxes_1, ids0, ids1, rel_pose)
    else:
        p1_in_frame0 = (rel_pose[:3, :3] @ xyz1.T).T + rel_pose[:3, 3]
        scene = {
            "static_xyz": torch.cat([xyz0, p1_in_frame0], dim=0),
            "static_intensity": torch.cat([i0, i1], dim=0),
            "static_time": torch.cat([
                torch.zeros(xyz0.shape[0], device=device, dtype=xyz0.dtype),
                torch.ones(xyz1.shape[0], device=device, dtype=xyz1.dtype),
            ], dim=0),
            "dynamic": [],
        }

    e_dir = rel_pose[:3, 3].to(dtype=xyz0.dtype, device=device)
    frame0_contexts: list[dict] = []

    static = _build_context(
        model,
        cfg,
        scene["static_xyz"].to(device),
        scene["static_intensity"].to(device),
        context_type="static",
        time_scalar=scene["static_time"].to(device),
        ego_motion=e_dir,
        is_dynamic_flag=torch.zeros(scene["static_xyz"].shape[0], device=device, dtype=xyz0.dtype),
    )
    if static is not None:
        frame0_contexts.append(static)

    for dyn in scene["dynamic"]:
        dyn_prims = _build_context(
            model,
            cfg,
            dyn["canonical_xyz"].to(device),
            dyn["canonical_intensity"].to(device),
            context_type="dynamic",
            time_scalar=dyn["canonical_time"].to(device),
            ego_motion=e_dir,
            is_dynamic_flag=torch.ones(dyn["canonical_xyz"].shape[0], device=device, dtype=xyz0.dtype),
        )
        if dyn_prims is not None and dyn.get("box_0") is not None:
            frame0_contexts.append(transform_primitives(dyn_prims, _box_to_pose(dyn["box_0"].to(device))))

    if not frame0_contexts:
        raise RuntimeError("no frame0 primitives produced")

    points_frame0 = torch.cat([xyz0, (rel_pose[:3, :3] @ xyz1.T).T + rel_pose[:3, 3]], dim=0)
    intensity_frame0 = torch.cat([i0, i1], dim=0)
    return concat_primitives(frame0_contexts), points_frame0, intensity_frame0, scene


def _select_indices(primitives: dict, max_surfaces: int, min_alpha: float) -> torch.Tensor:
    alpha = primitives["opacities"].squeeze(-1).detach()
    keep = torch.where(alpha >= min_alpha)[0]
    if keep.numel() == 0:
        keep = torch.arange(alpha.numel(), device=alpha.device)
    if keep.numel() <= max_surfaces:
        return keep
    values = alpha[keep]
    top = torch.topk(values, k=max_surfaces, largest=True).indices
    return keep[top]


def _surface_color(s1: float, s2: float, kappa_abs_max: float, eps: float) -> str:
    if kappa_abs_max < eps:
        return "rgba(80,150,255,0.58)"
    if s1 * s2 < 0:
        return "rgba(255,175,55,0.62)"
    if s1 > 0 and s2 > 0:
        return "rgba(70,210,125,0.60)"
    return "rgba(255,85,95,0.60)"


def _build_surface_mesh(
    primitives: dict,
    indices: torch.Tensor,
    *,
    grid: int,
    sigma: float,
    z_clip: float,
    eps_s: float,
    eps_kappa: float,
):
    means = primitives["means3D"][indices].detach().cpu().float().numpy()
    scales = primitives["scales"][indices].detach().cpu().float().numpy()
    rotations = primitives["rotations"][indices].detach().cpu().float()
    R = quat_to_rotmat(rotations).cpu().numpy()

    gx = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    gy = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    uu, vv = np.meshgrid(gx, gy, indexing="ij")
    base_i, base_j, base_k = [], [], []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a = i * grid + j
            b = (i + 1) * grid + j
            c = i * grid + (j + 1)
            d = (i + 1) * grid + (j + 1)
            base_i.extend([a, b])
            base_j.extend([b, d])
            base_k.extend([c, c])

    xs, ys, zs = [], [], []
    ii, jj, kk = [], [], []
    colors = []
    centers = []
    normal_lines = [[], [], []]
    offset = 0
    for n, s, r in zip(means, scales, R):
        s1, s2, s3 = float(s[0]), float(s[1]), float(s[2])
        a1 = max(abs(s1), eps_s)
        a2 = max(abs(s2), eps_s)
        k1 = 2.0 * abs(s3) * (1.0 if s1 >= 0 else -1.0) / (a1 * a1)
        k2 = 2.0 * abs(s3) * (1.0 if s2 >= 0 else -1.0) / (a2 * a2)
        x = uu * sigma * a1
        y = vv * sigma * a2
        z = 0.5 * (k1 * x * x + k2 * y * y)
        z = np.clip(z, -z_clip, z_clip)

        local = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        world = local @ r + n
        xs.extend(world[:, 0].tolist())
        ys.extend(world[:, 1].tolist())
        zs.extend(world[:, 2].tolist())
        ii.extend((np.asarray(base_i) + offset).tolist())
        jj.extend((np.asarray(base_j) + offset).tolist())
        kk.extend((np.asarray(base_k) + offset).tolist())
        colors.extend([_surface_color(s1, s2, max(abs(k1), abs(k2)), eps_kappa)] * world.shape[0])
        centers.append(n)

        normal = r[2]
        end = n + normal * max(0.25, min(2.0, sigma * abs(s3) * 5.0))
        normal_lines[0].extend([float(n[0]), float(end[0]), None])
        normal_lines[1].extend([float(n[1]), float(end[1]), None])
        normal_lines[2].extend([float(n[2]), float(end[2]), None])
        offset += grid * grid

    return (xs, ys, zs, ii, jj, kk, colors, np.asarray(centers), normal_lines)


def _sample_points(xyz: torch.Tensor, intensity: torch.Tensor, max_points: int):
    if xyz.shape[0] <= max_points:
        return xyz.detach().cpu().numpy(), intensity.detach().cpu().numpy()
    idx = torch.linspace(0, xyz.shape[0] - 1, max_points, device=xyz.device).long()
    return xyz[idx].detach().cpu().numpy(), intensity[idx].detach().cpu().numpy()


def visualize(args: argparse.Namespace) -> Path:
    import plotly.graph_objects as go

    checkpoint = _latest_checkpoint() if args.checkpoint == "latest" else args.checkpoint
    cfg = load_cfg_from_checkpoint(checkpoint)
    cfg.device = args.device
    if args.data_root:
        cfg.data_root = args.data_root

    dataset = _load_dataset(cfg, args.split)
    if args.pair_idx >= len(dataset):
        raise ValueError(f"--pair-idx {args.pair_idx} out of range for split size {len(dataset)}")

    model = QGSModel(cfg).to(args.device)
    ckpt = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    with torch.no_grad():
        primitives, points, point_i, scene = _predict_frame0_primitives(
            model, cfg, dataset[args.pair_idx], args.device
        )

    selected = _select_indices(primitives, args.max_surfaces, args.min_alpha)
    mesh = _build_surface_mesh(
        primitives,
        selected,
        grid=args.grid,
        sigma=args.surface_sigma,
        z_clip=args.z_clip,
        eps_s=cfg.quadric_eps_s,
        eps_kappa=cfg.quadric_eps_kappa,
    )
    mx, my, mz, mi, mj, mk, mcolors, centers, normal_lines = mesh
    pts_np, pts_i = _sample_points(points, point_i, args.max_points)

    trace_points = go.Scatter3d(
        x=pts_np[:, 0],
        y=pts_np[:, 1],
        z=pts_np[:, 2],
        mode="markers",
        marker=dict(size=1.2, color=pts_i, colorscale="Viridis", opacity=0.45),
        name="input points frame0",
        hoverinfo="skip",
    )
    trace_surfaces = go.Mesh3d(
        x=mx,
        y=my,
        z=mz,
        i=mi,
        j=mj,
        k=mk,
        vertexcolor=mcolors,
        opacity=0.72,
        name="QGS surface patches",
        hoverinfo="skip",
        showscale=False,
    )
    trace_centers = go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode="markers",
        marker=dict(size=2.5, color="white", symbol="x"),
        name="QGS centers",
        hovertemplate="x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}<extra></extra>",
    )
    trace_normals = go.Scatter3d(
        x=normal_lines[0],
        y=normal_lines[1],
        z=normal_lines[2],
        mode="lines",
        line=dict(width=2, color="rgba(255,255,255,0.65)"),
        name="surface normals",
        hoverinfo="skip",
    )

    scales = primitives["scales"].detach()
    signs = scales[:, 0] * scales[:, 1]
    title = (
        f"QGS surfaces | pair={args.pair_idx} | epoch={ckpt.get('epoch', '?')} "
        f"| selected={selected.numel()}/{scales.shape[0]} | "
        f"saddle={(signs < 0).sum().item()} | dyn={len(scene['dynamic'])}"
    )
    bg = "rgb(15,17,22)"
    fig = go.Figure([trace_points, trace_surfaces, trace_centers, trace_normals])
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=14)),
        scene=dict(
            aspectmode="data",
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            bgcolor=bg,
            xaxis=dict(backgroundcolor=bg, gridcolor="rgb(50,55,65)", color="white"),
            yaxis=dict(backgroundcolor=bg, gridcolor="rgb(50,55,65)", color="white"),
            zaxis=dict(backgroundcolor=bg, gridcolor="rgb(50,55,65)", color="white"),
        ),
        paper_bgcolor=bg,
        font=dict(color="white"),
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(bgcolor="rgba(25,28,34,0.78)"),
    )

    out = Path(args.out) if args.out else Path(checkpoint).parents[1] / f"qgs_surfaces_pair_{args.pair_idx:04d}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"checkpoint: {checkpoint}")
    print(f"generated primitives: {scales.shape[0]}")
    print(f"selected surfaces: {selected.numel()}")
    print(f"input points shown: {pts_np.shape[0]}/{points.shape[0]}")
    print(f"saved: {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize QGS quadratic surface patches")
    parser.add_argument("--checkpoint", default="latest", help="'latest' or path to checkpoint")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--pair-idx", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--max-surfaces", type=int, default=220)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--min-alpha", type=float, default=0.05)
    parser.add_argument("--grid", type=int, default=9)
    parser.add_argument("--surface-sigma", type=float, default=1.0)
    parser.add_argument("--z-clip", type=float, default=1.5)
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
