"""Visualize scene decomposition before/after on a real dataset pair."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import QGSConfig
from models.geometry.decomposition import _points_in_box, _transform_points, _yaw_to_rotmat, decompose_scene


def _load_dataset(cfg: QGSConfig, split: str):
    target_split = cfg.train_split if split == "train" else cfg.eval_split
    sys.path.insert(0, os.path.join(os.path.expanduser(cfg.data_root), "loader"))
    from dataset import NuScenesNVSDataset  # noqa: WPS433

    bbox_json = cfg.bbox_json_path.format(split=target_split) if cfg.bbox_json_path else ""
    bbox_json = bbox_json if (cfg.dataset_mode == "bbox" and bbox_json) else None
    return NuScenesNVSDataset(
        dataroot=os.path.expanduser(cfg.data_root),
        version=cfg.nuscenes_version,
        split=target_split,
        frame_gap=cfg.frame_gap,
        mode=cfg.dataset_mode,
        bbox_json_path=bbox_json,
    )


def _filter_frame_points(xyz: torch.Tensor, intensity: torch.Tensor, cfg: QGSConfig):
    keep = xyz.norm(dim=-1) > cfg.ego_radius
    return xyz[keep], intensity[keep]


def _transform_box_to_frame0(box: torch.Tensor, rel_pose: torch.Tensor) -> torch.Tensor:
    center0 = _transform_points(box[:3].unsqueeze(0).float(), rel_pose.float())[0].to(box)
    yaw_delta = torch.atan2(rel_pose[1, 0], rel_pose[0, 0]).to(box)
    return torch.cat([center0, box[3:6], (box[6] + yaw_delta).unsqueeze(0)], dim=0)


def _from_canonical(pts_canon: torch.Tensor, box0: torch.Tensor) -> torch.Tensor:
    rot = _yaw_to_rotmat(box0[6].to(pts_canon))
    return (rot @ pts_canon.T).T + box0[:3].to(pts_canon)


def _box_wireframe_segments(box: torch.Tensor) -> tuple[list[float], list[float], list[float]]:
    center = box[:3]
    w, l, h = box[3], box[4], box[5]
    yaw = box[6]
    hx = l * 0.5
    hy = w * 0.5
    hz = h * 0.5
    corners_local = torch.tensor(
        [
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ],
        dtype=box.dtype,
        device=box.device,
    )
    rot = _yaw_to_rotmat(yaw)
    corners = (rot @ corners_local.T).T + center
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for a, b in edges:
        xs.extend([float(corners[a, 0]), float(corners[b, 0]), None])
        ys.extend([float(corners[a, 1]), float(corners[b, 1]), None])
        zs.extend([float(corners[a, 2]), float(corners[b, 2]), None])
    return xs, ys, zs


def _downsample(xyz: torch.Tensor, max_points: int) -> torch.Tensor:
    if xyz.shape[0] <= max_points:
        return xyz
    idx = torch.linspace(0, xyz.shape[0] - 1, max_points, device=xyz.device).long()
    return xyz.index_select(0, idx)


def _pick_pair(dataset, pair_idx: int | None, min_dynamic: int) -> int:
    if pair_idx is not None:
        return pair_idx
    for idx in range(len(dataset)):
        item = dataset[idx]
        ids0 = set(item.get("instance_ids_0", torch.empty(0, dtype=torch.long)).tolist())
        ids1 = set(item.get("instance_ids_1", torch.empty(0, dtype=torch.long)).tolist())
        if len(ids0 & ids1) >= min_dynamic:
            return idx
    raise RuntimeError(f"no pair found with at least {min_dynamic} tracked dynamic instances")


def _build_plot(
    before_f0: torch.Tensor,
    before_f1: torch.Tensor,
    after_f0: torch.Tensor,
    after_f1: torch.Tensor,
    boxes0: torch.Tensor,
    boxes1_in0: torch.Tensor,
    tracked_ids: list[int],
    boxes0_ids: list[int],
    boxes1_ids: list[int],
    title: str,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Before decomposition", "After decomposition"),
        horizontal_spacing=0.03,
    )

    traces = [
        (before_f0, 1, "frame0 points", "rgba(60,140,255,0.70)"),
        (before_f1, 1, "frame1 points -> frame0", "rgba(255,90,90,0.70)"),
        (after_f0, 2, "frame0 points", "rgba(60,140,255,0.70)"),
        (after_f1, 2, "frame1 points -> canonical -> box0", "rgba(255,90,90,0.70)"),
    ]
    for pts, col, name, color in traces:
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0].cpu().tolist(),
                y=pts[:, 1].cpu().tolist(),
                z=pts[:, 2].cpu().tolist(),
                mode="markers",
                marker=dict(size=1.6, color=color),
                name=name,
                legendgroup=name,
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

    for box, iid in zip(boxes0, boxes0_ids):
        xs, ys, zs = _box_wireframe_segments(box)
        label = f"id {iid} box0"
        center = box[:3]
        show_label = iid in tracked_ids
        for col in (1, 2):
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color="rgba(30,200,90,0.95)", width=4),
                    name=label,
                    legendgroup=label,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
            if show_label:
                fig.add_trace(
                    go.Scatter3d(
                        x=[float(center[0])],
                        y=[float(center[1])],
                        z=[float(center[2] + box[5] * 0.7)],
                        mode="text",
                        text=[str(iid)],
                        textfont=dict(color="rgb(30,200,90)", size=10),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )

    for box, iid in zip(boxes1_in0, boxes1_ids):
        xs, ys, zs = _box_wireframe_segments(box)
        label = f"id {iid} box1->f0"
        center = box[:3]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="rgba(255,180,40,0.9)", width=3, dash="dash"),
                name=label,
                legendgroup=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        if iid in tracked_ids:
            fig.add_trace(
                go.Scatter3d(
                    x=[float(center[0])],
                    y=[float(center[1])],
                    z=[float(center[2] + box[5] * 0.7)],
                    mode="text",
                    text=[str(iid)],
                    textfont=dict(color="rgb(255,180,40)", size=10),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    scene_layout = dict(
        aspectmode="data",
        xaxis=dict(title="X (m)", backgroundcolor="rgb(248,248,250)"),
        yaxis=dict(title="Y (m)", backgroundcolor="rgb(248,248,250)"),
        zaxis=dict(title="Z (m)", backgroundcolor="rgb(248,248,250)"),
    )
    fig.update_layout(
        title=title,
        scene=scene_layout,
        scene2=scene_layout,
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize scene decomposition before/after")
    parser.add_argument("--pair-idx", type=int, default=None, help="dataset pair index; auto-pick if omitted")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--max-points-per-group", type=int, default=25000)
    parser.add_argument("--min-dynamic", type=int, default=1)
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/decomposition_viz/decomposition_pair_auto.html",
    )
    args = parser.parse_args()

    cfg = QGSConfig()
    dataset = _load_dataset(cfg, args.split)
    pair_idx = _pick_pair(dataset, args.pair_idx, args.min_dynamic)
    item = dataset[pair_idx]

    p0 = item["input_0"].float()
    p1 = item["input_1"].float()
    rel_pose = item["input_1_pose"].float()
    xyz0 = p0[:, :3]
    xyz1 = p1[:, :3]
    i0 = p0[:, 3]
    i1 = p1[:, 3]
    xyz0, i0 = _filter_frame_points(xyz0, i0, cfg)
    xyz1, i1 = _filter_frame_points(xyz1, i1, cfg)
    i0_norm = (i0 / 255.0).clamp(0.0, 1.0)
    i1_norm = (i1 / 255.0).clamp(0.0, 1.0)

    boxes0 = item.get("boxes_0", torch.empty(0, 7)).float()
    boxes1 = item.get("boxes_1", torch.empty(0, 7)).float()
    ids0 = item.get("instance_ids_0", torch.empty(0, dtype=torch.long))
    ids1 = item.get("instance_ids_1", torch.empty(0, dtype=torch.long))
    boxes1_in0 = torch.stack([_transform_box_to_frame0(box, rel_pose) for box in boxes1], dim=0) if boxes1.numel() else boxes1

    scene = decompose_scene(
        xyz0,
        xyz1,
        i0_norm,
        i1_norm,
        boxes0,
        boxes1,
        ids0,
        ids1,
        rel_pose,
    )

    p1_in0 = _transform_points(xyz1, rel_pose)
    before_f0 = xyz0
    before_f1 = p1_in0

    after_f0_parts = [scene["static_xyz"][scene["static_time"] == 0]]
    after_f1_parts = [scene["static_xyz"][scene["static_time"] == 1]]
    tracked_ids = sorted(int(d["instance_id"]) for d in scene["dynamic"])
    summary_rows: list[str] = []

    for dyn in sorted(scene["dynamic"], key=lambda d: int(d["instance_id"])):
        box0 = dyn["box_0"]
        canon = dyn["canonical_xyz"]
        time = dyn["canonical_time"]
        frame0_world = _from_canonical(canon, box0)
        part0 = frame0_world[time == 0]
        part1 = frame0_world[time == 1]
        after_f0_parts.append(part0)
        after_f1_parts.append(part1)

        inside = _points_in_box(frame0_world, box0)
        center_mean = frame0_world.mean(dim=0) if frame0_world.shape[0] > 0 else box0[:3]
        center_err = (center_mean - box0[:3]).norm().item()
        summary_rows.append(
            "id={iid} n0={n0} n1={n1} after_inside_box0={inside}/{total} ({ratio:.3f}) "
            "mean_center_err={err:.3f}m".format(
                iid=int(dyn["instance_id"]),
                n0=int((time == 0).sum().item()),
                n1=int((time == 1).sum().item()),
                inside=int(inside.sum().item()),
                total=int(inside.numel()),
                ratio=float(inside.float().mean().item()) if inside.numel() else 0.0,
                err=center_err,
            )
        )

    after_f0 = torch.cat([p for p in after_f0_parts if p.numel() > 0], dim=0)
    after_f1 = torch.cat([p for p in after_f1_parts if p.numel() > 0], dim=0)

    before_f0 = _downsample(before_f0, args.max_points_per_group)
    before_f1 = _downsample(before_f1, args.max_points_per_group)
    after_f0 = _downsample(after_f0, args.max_points_per_group)
    after_f1 = _downsample(after_f1, args.max_points_per_group)

    title = (
        f"Scene decomposition pair={pair_idx} | tracked_dynamic={len(scene['dynamic'])} | "
        f"raw_intensity p0=[{float(i0.min()):.1f},{float(i0.max()):.1f}] "
        f"p1=[{float(i1.min()):.1f},{float(i1.max()):.1f}]"
    )
    fig = _build_plot(
        before_f0,
        before_f1,
        after_f0,
        after_f1,
        boxes0,
        boxes1_in0,
        tracked_ids,
        ids0.tolist(),
        ids1.tolist(),
        title,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")

    print(f"pair_idx={pair_idx}")
    print(f"raw_intensity_frame0=[{float(i0.min()):.6f}, {float(i0.max()):.6f}]")
    print(f"raw_intensity_frame1=[{float(i1.min()):.6f}, {float(i1.max()):.6f}]")
    print(f"normalized_intensity_frame0=[{float(i0_norm.min()):.6f}, {float(i0_norm.max()):.6f}]")
    print(f"normalized_intensity_frame1=[{float(i1_norm.min()):.6f}, {float(i1_norm.max()):.6f}]")
    print(f"tracked_dynamic_ids={tracked_ids}")
    print(f"n_static={int(scene['static_xyz'].shape[0])}")
    print(f"n_dynamic_instances={len(scene['dynamic'])}")
    for row in summary_rows:
        print(row)
    print(f"html={out_path}")


if __name__ == "__main__":
    main()
