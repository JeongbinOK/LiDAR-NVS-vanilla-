"""Visualization smoke tests for QGS-form local quadric initialisation."""

from __future__ import annotations

import math
from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from models.geometry.quadric_fit import fit_local_quadrics


OUTPUT_HTML = Path("outputs/quadric_fit_visualization/qgs_local_patch_cases.html")


def _rot_z(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _rot_y(angle: float) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=torch.float32,
    )


def _make_local_patch(
    xy: torch.Tensor,
    a: float,
    b: float,
    *,
    rotation: torch.Tensor | None = None,
    translation: torch.Tensor | None = None,
    noise_z: float = 0.0,
) -> torch.Tensor:
    z = a * xy[:, 0].square() + b * xy[:, 1].square()
    if noise_z > 0.0:
        z = z + noise_z * torch.randn_like(z)
    local = torch.stack([xy[:, 0], xy[:, 1], z], dim=-1)
    if rotation is None:
        rotation = torch.eye(3, dtype=local.dtype)
    if translation is None:
        translation = torch.zeros(3, dtype=local.dtype)
    return local @ rotation.T + translation


def _case_points() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    grid_1d = torch.linspace(-0.6, 0.6, 4)
    gx, gy = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

    angles = torch.linspace(0.0, 2.0 * math.pi, 16 + 1)[:-1]
    ring = torch.stack([0.62 * torch.cos(angles), 0.28 * torch.sin(angles)], dim=-1)

    rand = torch.rand(16, 2) - 0.5
    rand[:, 0] *= 1.4
    rand[:, 1] *= 0.7

    tilted = _rot_z(0.55) @ _rot_y(-0.35)
    return {
        "convex_grid_16": _make_local_patch(grid, 0.42, 0.18),
        "saddle_grid_16": _make_local_patch(grid, 0.38, -0.24),
        "tilted_convex_16": _make_local_patch(
            grid,
            0.32,
            0.12,
            rotation=tilted,
            translation=torch.tensor([1.0, -0.5, 0.4]),
        ),
        "anisotropic_ring_16": _make_local_patch(ring, 0.22, 0.55, rotation=_rot_z(0.4)),
        "random_noisy_16": _make_local_patch(rand, 0.30, -0.15, noise_z=0.01),
        "near_flat_16": _make_local_patch(grid, 0.0, 0.0, rotation=_rot_y(0.25), noise_z=0.001),
    }


def _qgs_coefficients(s: torch.Tensor) -> tuple[float, float]:
    s1 = s[0]
    s2 = s[1]
    s3 = s[2].abs()
    a = s3 * torch.sign(s1) / s1.abs().clamp(min=1e-8).square()
    b = s3 * torch.sign(s2) / s2.abs().clamp(min=1e-8).square()
    return float(a.item()), float(b.item())


def _surface_trace(name: str, points: torch.Tensor, result: dict) -> tuple[go.Scatter3d, go.Surface, float]:
    center = result["c_init"][0, 0]
    R = result["R_init"][0, 0]
    s = result["s_init"][0, 0]
    a, b = _qgs_coefficients(s)

    local = (points - center) @ R
    u = local[:, 0]
    v = local[:, 1]
    w = local[:, 2]
    w_pred = a * u.square() + b * v.square()
    mse = float((w_pred - w).square().mean().item())

    u_grid = torch.linspace(float(u.min().item()) - 0.05, float(u.max().item()) + 0.05, 24)
    v_grid = torch.linspace(float(v.min().item()) - 0.05, float(v.max().item()) + 0.05, 24)
    ug, vg = torch.meshgrid(u_grid, v_grid, indexing="ij")
    wg = a * ug.square() + b * vg.square()
    local_surf = torch.stack([ug.reshape(-1), vg.reshape(-1), wg.reshape(-1)], dim=-1)
    world_surf = local_surf @ R.T + center
    xs = world_surf[:, 0].reshape_as(ug)
    ys = world_surf[:, 1].reshape_as(ug)
    zs = world_surf[:, 2].reshape_as(ug)

    point_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={"size": 4, "color": "black"},
        name=f"{name} points",
        showlegend=False,
    )
    surface_trace = go.Surface(
        x=xs,
        y=ys,
        z=zs,
        opacity=0.65,
        colorscale="Viridis",
        showscale=False,
        name=f"{name} QGS",
        showlegend=False,
    )
    return point_trace, surface_trace, mse


def test_visualize_qgs_local_patch_initialisation_html() -> None:
    cases = _case_points()
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"} for _ in range(3)] for _ in range(2)],
        subplot_titles=list(cases.keys()),
    )

    diagnostics: list[str] = []
    for idx, (name, pts) in enumerate(cases.items()):
        query = pts.mean(dim=0).view(1, 1, 3)
        neighbors = pts.view(1, 1, 16, 3)
        k_eff = torch.full((1, 1), 16, dtype=torch.long)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=8, k_target=16)
        assert result["use_geom_init"][0, 0].item(), name
        assert torch.isfinite(result["s_init"]).all(), name

        points_trace, surface_trace, mse = _surface_trace(name, pts, result)
        diagnostics.append(f"{name}: local_qgs_mse={mse:.6e}")
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig.add_trace(points_trace, row=row, col=col)
        fig.add_trace(surface_trace, row=row, col=col)

    fig.update_layout(
        title="QGS local patch initialisation from 16 LiDAR samples<br>"
        + "<br>".join(diagnostics),
        height=900,
        width=1300,
        margin={"l": 0, "r": 0, "t": 120, "b": 0},
    )
    for scene_idx in range(1, len(cases) + 1):
        scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
        fig.layout[scene_name].update(aspectmode="data")

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")

    html = OUTPUT_HTML.read_text(encoding="utf-8")
    assert OUTPUT_HTML.is_file()
    assert "QGS local patch initialisation" in html
    for name in cases:
        assert name in html
