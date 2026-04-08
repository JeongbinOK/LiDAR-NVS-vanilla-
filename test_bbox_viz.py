#!/usr/bin/env python3
"""
Test script for NuScenesNVSDataset with bbox mode.
Randomly samples 5 pairs, visualizes LiDAR points + GT BBoxes (with instance IDs)
in an interactive HTML file.

All points and boxes are expressed in Frame 0's LiDAR sensor coordinate system.

Usage:
    python test_bbox_viz.py [--split val] [--n 5] [--out test_bbox_viz.html]
"""

import sys
import os
import random
import argparse
import numpy as np
import torch

sys.path.insert(0, '/data1/nuScenes/loader')
from dataset import NuScenesNVSDataset

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)


# ─── geometry helpers ──────────────────────────────────────────────────────

def transform_points(pts_np, T_torch):
    """
    Apply 4x4 torch transform T to (N, 4) numpy point cloud.
    Returns (N, 3) in the new frame.
    """
    T = T_torch.numpy()
    xyz  = pts_np[:, :3]
    ones = np.ones((len(xyz), 1), dtype=np.float32)
    h    = np.hstack([xyz, ones])        # (N, 4)
    return (T @ h.T).T[:, :3]           # (N, 3)


def transform_boxes(boxes_torch, T_torch):
    """
    Transform (B, 7) boxes [x,y,z,w,l,h,yaw] from frame A to frame B via T (A→B).
    Yaw is adjusted by extracting the Z-rotation component of T's rotation matrix.
    Returns numpy (B, 7).
    """
    boxes = boxes_torch.numpy().copy()
    if len(boxes) == 0:
        return boxes
    T = T_torch.numpy()
    R = T[:3, :3]

    # Transform centers
    centers  = boxes[:, :3]
    ones     = np.ones((len(centers), 1), dtype=np.float32)
    centers_h = np.hstack([centers, ones])
    boxes[:, :3] = (T @ centers_h.T).T[:, :3]

    # Adjust yaw: add the Z-rotation from R
    yaw_delta   = np.arctan2(R[1, 0], R[0, 0])
    boxes[:, 6] = boxes[:, 6] + yaw_delta
    return boxes


# ─── box wireframe ─────────────────────────────────────────────────────────

# 12 edges of a box, indexed into the 8-corner array
_BOX_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),   # front face (+x)
    (4, 5), (4, 6), (5, 7), (6, 7),   # back face  (-x)
    (0, 4), (1, 5), (2, 6), (3, 7),   # lateral edges
]


def box_corners_8(cx, cy, cz, w, l, h, yaw):
    """
    Compute 8 corners of a 3D box.
    Storage format: [x, y, z, w, l, h, yaw]
      l = length (along heading / local-x)
      w = width  (lateral      / local-y)
      h = height (vertical     / local-z)
    """
    hl, hw, hh = l / 2, w / 2, h / 2
    # Local corners (x=forward, y=left, z=up)
    local = np.array([
        [ hl,  hw,  hh], [ hl,  hw, -hh],
        [ hl, -hw,  hh], [ hl, -hw, -hh],
        [-hl,  hw,  hh], [-hl,  hw, -hh],
        [-hl, -hw,  hh], [-hl, -hw, -hh],
    ], dtype=np.float32)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R = np.array([[ cos_y, -sin_y, 0],
                  [ sin_y,  cos_y, 0],
                  [ 0,      0,     1]], dtype=np.float32)
    return (R @ local.T).T + np.array([cx, cy, cz], dtype=np.float32)  # (8, 3)


def make_box_traces(boxes_np, instance_ids_np, line_color, text_color, legend_name):
    """
    Build Plotly traces for box wireframes + instance ID text labels.
    boxes_np: (B, 7), instance_ids_np: (B,) int
    Returns list of go.Scatter3d traces.
    """
    traces = []
    if len(boxes_np) == 0:
        return traces

    xs, ys, zs = [], [], []
    for box in boxes_np:
        cx, cy, cz, w, l, h, yaw = box
        corners = box_corners_8(cx, cy, cz, w, l, h, yaw)
        for a, b in _BOX_EDGES:
            xs += [corners[a, 0], corners[b, 0], None]
            ys += [corners[a, 1], corners[b, 1], None]
            zs += [corners[a, 2], corners[b, 2], None]

    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='lines',
        line=dict(color=line_color, width=2),
        name=legend_name,
        showlegend=True,
    ))

    # Instance ID labels at box centers
    traces.append(go.Scatter3d(
        x=boxes_np[:, 0],
        y=boxes_np[:, 1],
        z=boxes_np[:, 2] + boxes_np[:, 5] / 2 + 0.3,   # slightly above top
        mode='text',
        text=[f'<b>ID:{int(iid)}</b>' for iid in instance_ids_np],
        textfont=dict(color=text_color, size=11),
        showlegend=False,
        hoverinfo='skip',
    ))

    return traces


def subsample(pts, max_pts=25000):
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        return pts[idx]
    return pts


# ─── per-pair figure ────────────────────────────────────────────────────────

def make_figure(item, pair_idx, dataset_idx):
    pts0 = item['input_0'].numpy()          # (N0, 4)  LiDAR_0 frame
    pts1 = item['input_1'].numpy()          # (N1, 4)  LiDAR_1 frame
    T_1_to_0 = item['input_1_pose']         # (4, 4)   LiDAR_1 → LiDAR_0

    boxes0   = item['boxes_0'].numpy()      # (B0, 7)  LiDAR_0 frame
    ids0     = item['instance_ids_0'].numpy()
    boxes1   = item['boxes_1'].numpy()      # (B1, 7)  LiDAR_1 frame
    ids1     = item['instance_ids_1'].numpy()

    # ── Transform frame-1 data into frame-0 ──
    pts1_in_0    = transform_points(pts1, T_1_to_0)   # (N1, 3)
    boxes1_in_0  = transform_boxes(item['boxes_1'], T_1_to_0)

    # ── Subsample points ──
    pts0_xyz = subsample(pts0[:, :3])
    pts1_xyz = subsample(pts1_in_0)

    # ── Build traces ──
    traces = [
        go.Scatter3d(
            x=pts0_xyz[:, 0], y=pts0_xyz[:, 1], z=pts0_xyz[:, 2],
            mode='markers',
            marker=dict(size=1.2, color='#4FC3F7', opacity=0.55),
            name='Frame 0 LiDAR',
        ),
        go.Scatter3d(
            x=pts1_xyz[:, 0], y=pts1_xyz[:, 1], z=pts1_xyz[:, 2],
            mode='markers',
            marker=dict(size=1.2, color='#FF8A65', opacity=0.55),
            name='Frame 1 LiDAR (→ F0)',
        ),
    ]

    traces += make_box_traces(boxes0, ids0, '#00E5FF', '#00E5FF', 'BBox Frame 0')
    traces += make_box_traces(boxes1_in_0, ids1, '#FFAB40', '#FFAB40', 'BBox Frame 1 (→ F0)')

    n_common = len(set(ids0.tolist()) & set(ids1.tolist()))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f'Pair {pair_idx}  |  dataset idx = {dataset_idx}  |  '
                  f'boxes F0={len(boxes0)}  F1={len(boxes1)}  '
                  f'common IDs={n_common}'),
            font=dict(size=14, color='#E0E0E0'),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title='X (forward)', range=[-52, 52],
                       backgroundcolor='#0d0d1a', gridcolor='#333'),
            yaxis=dict(title='Y (left)',    range=[-52, 52],
                       backgroundcolor='#0d0d1a', gridcolor='#333'),
            zaxis=dict(title='Z (up)',      range=[-5, 12],
                       backgroundcolor='#0d0d1a', gridcolor='#333'),
            bgcolor='#0d0d1a',
            aspectmode='manual',
            aspectratio=dict(x=2, y=2, z=0.3),
            camera=dict(eye=dict(x=0, y=-1.5, z=1.2)),
        ),
        paper_bgcolor='#12122a',
        font=dict(color='#E0E0E0'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='#444',
            borderwidth=1,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )
    return fig


# ─── HTML assembly ──────────────────────────────────────────────────────────

HTML_HEAD = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NuScenes BBox Visualization</title>
<style>
  * { box-sizing: border-box; }
  body {
    background: #12122a;
    color: #E0E0E0;
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 0; padding: 0;
  }
  h1 {
    text-align: center;
    padding: 22px 0 4px;
    color: #4FC3F7;
    font-size: 22px;
    letter-spacing: 1px;
    margin: 0;
  }
  .subtitle {
    text-align: center;
    font-size: 12px;
    color: #888;
    padding-bottom: 14px;
  }
  .legend-bar {
    display: flex;
    justify-content: center;
    gap: 24px;
    padding: 0 0 12px;
    font-size: 13px;
  }
  .legend-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
  }
  .tab-bar {
    display: flex;
    justify-content: center;
    gap: 8px;
    padding: 0 0 16px;
    flex-wrap: wrap;
  }
  .tab-btn {
    background: #1e1e3a;
    color: #aaa;
    border: 1px solid #444;
    padding: 7px 18px;
    cursor: pointer;
    border-radius: 6px;
    font-size: 13px;
    transition: all 0.15s;
  }
  .tab-btn:hover  { background: #2a2a55; color: #ddd; }
  .tab-btn.active { background: #4FC3F7; color: #12122a; font-weight: bold; border-color: #4FC3F7; }
  .scene-panel { display: none; }
  .scene-panel.active { display: block; }
</style>
</head>
<body>
<h1>NuScenes GT BBox + LiDAR — bbox mode test</h1>
<p class="subtitle">All data expressed in Frame 0 LiDAR coordinate system &nbsp;|&nbsp;
  Drag to rotate &nbsp;·&nbsp; Scroll to zoom &nbsp;·&nbsp; Double-click to reset</p>
<div class="legend-bar">
  <span><span class="legend-dot" style="background:#4FC3F7"></span>Frame 0 LiDAR</span>
  <span><span class="legend-dot" style="background:#FF8A65"></span>Frame 1 LiDAR (→ F0)</span>
  <span><span class="legend-dot" style="background:#00E5FF; border-radius:0"></span>BBox Frame 0</span>
  <span><span class="legend-dot" style="background:#FFAB40; border-radius:0"></span>BBox Frame 1 (→ F0)</span>
</div>
<div class="tab-bar">
"""

HTML_TAB_BUTTON = '<button class="tab-btn{active}" onclick="showScene({i})">{label}</button>\n'

HTML_JS = """
<script>
function showScene(idx) {
  document.querySelectorAll('.scene-panel').forEach(function(el, i) {
    el.classList.toggle('active', i === idx);
  });
  document.querySelectorAll('.tab-btn').forEach(function(el, i) {
    el.classList.toggle('active', i === idx);
  });
}
</script>
</body>
</html>
"""


# ─── entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',  default='val',
                        help='nuScenes split (train/val/mini_train/mini_val)')
    parser.add_argument('--n',      type=int, default=5,
                        help='Number of random pairs to visualize')
    parser.add_argument('--seed',   type=int, default=42)
    parser.add_argument('--out',    default='test_bbox_viz.html',
                        help='Output HTML file path')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading NuScenesNVSDataset  split={args.split}  mode=bbox ...")
    dataset = NuScenesNVSDataset(
        dataroot='/data1/nuScenes',
        version='v1.0-trainval',
        split=args.split,
        frame_gap=1,
        mode='bbox',
    )
    print(f"Dataset size: {len(dataset)}")

    n_pairs = min(args.n, len(dataset))
    indices = random.sample(range(len(dataset)), n_pairs)
    print(f"Sampled indices: {indices}\n")

    # ── build figures ──
    figures = []
    labels  = []
    for pair_idx, ds_idx in enumerate(indices):
        print(f"[{pair_idx+1}/{n_pairs}] loading idx={ds_idx} ...")
        item = dataset[ds_idx]
        fig  = make_figure(item, pair_idx, ds_idx)
        figures.append(fig)
        b0 = len(item['boxes_0'])
        b1 = len(item['boxes_1'])
        labels.append(f'Pair {pair_idx}  ({b0}+{b1} boxes)')
        print(f"         boxes_0={b0}  boxes_1={b1}  "
              f"pts_0={len(item['input_0'])}  pts_1={len(item['input_1'])}")

    # ── assemble HTML ──
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    parts = [HTML_HEAD]

    for i, label in enumerate(labels):
        active = ' active' if i == 0 else ''
        parts.append(HTML_TAB_BUTTON.format(active=active, i=i, label=label))
    parts.append("</div>\n")

    for i, fig in enumerate(figures):
        active = ' active' if i == 0 else ''
        # Embed plotly.js only once (first figure)
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=(i == 0),
            config={'scrollZoom': True, 'displayModeBar': True},
        )
        parts.append(f'<div id="scene-{i}" class="scene-panel{active}">{fig_html}</div>\n')

    parts.append(HTML_JS)

    with open(out_path, 'w') as f:
        f.write(''.join(parts))

    print(f"\nSaved → {out_path}")
    print("Open in a browser (or run:  python -m http.server  in the directory)")


if __name__ == '__main__':
    main()
