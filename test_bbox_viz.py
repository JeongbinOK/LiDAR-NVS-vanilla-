#!/usr/bin/env python3
"""
GT BBox vs Pred BBox 시각화.
train 3 scenes + val 3 scenes, 각 scene에서 연속 2 frames.

Usage:
    python test_bbox_viz.py [--seed 42] [--out test_bbox_viz.html]
"""

import sys
import os
import random
import argparse
import numpy as np

sys.path.insert(0, '/data1/nuScenes/loader')
from dataset import NuScenesNVSDataset

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)


DATAROOT     = '/data1/nuScenes'
BBOX_JSON    = 'bbox/tracking.json'
N_SCENES     = 3
FRAME_GAP    = 1


# ─── geometry helpers ──────────────────────────────────────────────────────

def transform_points(pts_np, T_torch):
    T    = T_torch.numpy()
    xyz  = pts_np[:, :3]
    ones = np.ones((len(xyz), 1), dtype=np.float32)
    return (T @ np.hstack([xyz, ones]).T).T[:, :3]


def transform_boxes(boxes_torch, T_torch):
    boxes = boxes_torch.numpy().copy()
    if len(boxes) == 0:
        return boxes
    T, R  = T_torch.numpy(), T_torch.numpy()[:3, :3]
    centers_h = np.hstack([boxes[:, :3], np.ones((len(boxes), 1), dtype=np.float32)])
    boxes[:, :3]  = (T @ centers_h.T).T[:, :3]
    boxes[:, 6]  += np.arctan2(R[1, 0], R[0, 0])
    return boxes


# ─── box wireframe ─────────────────────────────────────────────────────────

_BOX_EDGES = [
    (0,1),(0,2),(1,3),(2,3),
    (4,5),(4,6),(5,7),(6,7),
    (0,4),(1,5),(2,6),(3,7),
]


def box_corners_8(cx, cy, cz, w, l, h, yaw):
    hl, hw, hh = l/2, w/2, h/2
    local = np.array([
        [ hl, hw, hh],[ hl, hw,-hh],
        [ hl,-hw, hh],[ hl,-hw,-hh],
        [-hl, hw, hh],[-hl, hw,-hh],
        [-hl,-hw, hh],[-hl,-hw,-hh],
    ], dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
    return (R @ local.T).T + np.array([cx,cy,cz], dtype=np.float32)


def make_box_traces(boxes_np, instance_ids_np, line_color, legend_name, show_ids=True):
    traces = []
    if len(boxes_np) == 0:
        return traces
    xs, ys, zs = [], [], []
    for box in boxes_np:
        corners = box_corners_8(*box)
        for a, b in _BOX_EDGES:
            xs += [corners[a,0], corners[b,0], None]
            ys += [corners[a,1], corners[b,1], None]
            zs += [corners[a,2], corners[b,2], None]
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs, mode='lines',
        line=dict(color=line_color, width=2),
        name=legend_name, showlegend=True,
    ))
    if show_ids and len(instance_ids_np):
        traces.append(go.Scatter3d(
            x=boxes_np[:,0], y=boxes_np[:,1],
            z=boxes_np[:,2] + boxes_np[:,5]/2 + 0.3,
            mode='text',
            text=[f'<b>{int(i)}</b>' for i in instance_ids_np],
            textfont=dict(color=line_color, size=10),
            showlegend=False, hoverinfo='skip',
        ))
    return traces


def filter_range(pts, x_range=(-54, 54), y_range=(-54, 54), z_range=(-5, 3)):
    """OpenPCDet과 동일한 point cloud range 필터링."""
    mask = ((pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1]) &
            (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) &
            (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1]))
    return pts[mask]


def subsample(pts, max_pts=20000):
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        return pts[idx]
    return pts


# ─── figure ────────────────────────────────────────────────────────────────

def make_figure(gt_item, pred_item, title):
    pts0    = gt_item['input_0'].numpy()
    pts1    = gt_item['input_1'].numpy()
    T_1to0  = gt_item['input_1_pose']

    pts1_in_0 = transform_points(pts1, T_1to0)

    # Range 필터 (OpenPCDet 기준: x/y ±54m, z -5~3m)
    pts0      = filter_range(pts0)
    pts1_in_0 = filter_range(pts1_in_0)

    gt_b0   = gt_item['boxes_0'].numpy()
    gt_ids0 = gt_item['instance_ids_0'].numpy()
    gt_b1   = transform_boxes(gt_item['boxes_1'], T_1to0)
    gt_ids1 = gt_item['instance_ids_1'].numpy()

    pr_b0   = pred_item['boxes_0'].numpy()
    pr_ids0 = pred_item['instance_ids_0'].numpy()
    pr_b1   = transform_boxes(pred_item['boxes_1'], T_1to0)
    pr_ids1 = pred_item['instance_ids_1'].numpy()

    n_gt_common   = len(set(gt_ids0.tolist()) & set(gt_ids1.tolist()))
    n_pred_common = len(set(pr_ids0.tolist()) & set(pr_ids1.tolist()))

    traces = [
        go.Scatter3d(
            x=subsample(pts0[:,:3])[:,0],
            y=subsample(pts0[:,:3])[:,1],
            z=subsample(pts0[:,:3])[:,2],
            mode='markers', marker=dict(size=1.2, color='#4FC3F7', opacity=0.5),
            name='LiDAR F0',
        ),
        go.Scatter3d(
            x=subsample(pts1_in_0)[:,0],
            y=subsample(pts1_in_0)[:,1],
            z=subsample(pts1_in_0)[:,2],
            mode='markers', marker=dict(size=1.2, color='#FF8A65', opacity=0.5),
            name='LiDAR F1 (→F0)',
        ),
    ]
    # GT boxes: green shades
    traces += make_box_traces(gt_b0,  gt_ids0,  '#69F0AE', 'GT F0')
    traces += make_box_traces(gt_b1,  gt_ids1,  '#B9F6CA', 'GT F1 (→F0)')
    # Pred boxes: red/yellow shades
    traces += make_box_traces(pr_b0,  pr_ids0,  '#FF5252', 'Pred F0')
    traces += make_box_traces(pr_b1,  pr_ids1,  '#FFD740', 'Pred F1 (→F0)')

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=(f'{title}  |  '
                  f'GT: F0={len(gt_b0)} F1={len(gt_b1)} common={n_gt_common}  |  '
                  f'Pred: F0={len(pr_b0)} F1={len(pr_b1)} common={n_pred_common}'),
            font=dict(size=13, color='#E0E0E0'), x=0.5,
        ),
        scene=dict(
            xaxis=dict(title='X (right)', range=[-52,52], backgroundcolor='#0d0d1a', gridcolor='#333'),
            yaxis=dict(title='Y (forward)', range=[-52,52], backgroundcolor='#0d0d1a', gridcolor='#333'),
            zaxis=dict(title='Z (up)', range=[-5,5],   backgroundcolor='#0d0d1a', gridcolor='#333'),
            bgcolor='#0d0d1a', aspectmode='manual',
            aspectratio=dict(x=2, y=2, z=0.3),
            camera=dict(eye=dict(x=0, y=-1.5, z=1.2)),
        ),
        paper_bgcolor='#12122a',
        font=dict(color='#E0E0E0'),
        legend=dict(bgcolor='rgba(0,0,0,0.6)', bordercolor='#444', borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )
    return fig


# ─── scene sampling ─────────────────────────────────────────────────────────

def sample_scene_indices(dataset, n_scenes, rng):
    """dataset의 data_infos를 scene별로 그룹핑해서 n_scenes개 선택, 각 scene에서 1쌍 반환."""
    # sample_token_0의 앞 글자(또는 직접 scene 이름)로 그룹핑하면 오래 걸림 →
    # nusc scene 순서와 data_infos 순서가 같으므로, data_infos를 slicing으로 그룹핑
    infos = dataset.data_infos

    # 연속된 인덱스에서 scene 경계 찾기: sample_token_1 != 다음 idx의 sample_token_0
    scene_groups = []
    start = 0
    for i in range(1, len(infos)):
        if infos[i]['sample_token_0'] != infos[i-1]['sample_token_1']:
            scene_groups.append(list(range(start, i)))
            start = i
    scene_groups.append(list(range(start, len(infos))))

    chosen_scenes = rng.sample(scene_groups, min(n_scenes, len(scene_groups)))
    # 각 scene에서 중간 idx 1개 선택 (첫/마지막 제외)
    indices = []
    for grp in chosen_scenes:
        safe = grp[1:-1] if len(grp) > 2 else grp
        indices.append(rng.choice(safe))
    return indices


# ─── HTML ──────────────────────────────────────────────────────────────────

HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>GT vs Pred BBox</title>
<style>
*{box-sizing:border-box}
body{background:#12122a;color:#E0E0E0;font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:0}
h1{text-align:center;padding:18px 0 4px;color:#4FC3F7;font-size:20px;margin:0}
.subtitle{text-align:center;font-size:12px;color:#888;padding-bottom:10px}
.legend-bar{display:flex;justify-content:center;gap:20px;padding:0 0 10px;font-size:12px;flex-wrap:wrap}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
.section-label{text-align:center;font-size:13px;color:#aaa;margin:6px 0 2px;letter-spacing:1px}
.tab-bar{display:flex;justify-content:center;gap:6px;padding:0 0 12px;flex-wrap:wrap}
.tab-btn{background:#1e1e3a;color:#aaa;border:1px solid #444;padding:6px 16px;cursor:pointer;
  border-radius:6px;font-size:12px;transition:all 0.15s}
.tab-btn:hover{background:#2a2a55;color:#ddd}
.tab-btn.active{background:#4FC3F7;color:#12122a;font-weight:bold;border-color:#4FC3F7}
.tab-btn.val-active{background:#FF8A65;color:#12122a;font-weight:bold;border-color:#FF8A65}
.scene-panel{display:none}
.scene-panel.active{display:block}
</style></head><body>
<h1>GT BBox vs Pred BBox — nuScenes</h1>
<p class="subtitle">Frame 0 기준 좌표계 | 드래그:회전 · 스크롤:줌</p>
<div class="legend-bar">
  <span><span class="dot" style="background:#4FC3F7"></span>LiDAR F0</span>
  <span><span class="dot" style="background:#FF8A65"></span>LiDAR F1</span>
  <span><span class="dot" style="background:#69F0AE;border-radius:0"></span>GT F0</span>
  <span><span class="dot" style="background:#B9F6CA;border-radius:0"></span>GT F1</span>
  <span><span class="dot" style="background:#FF5252;border-radius:0"></span>Pred F0</span>
  <span><span class="dot" style="background:#FFD740;border-radius:0"></span>Pred F1</span>
</div>
"""

HTML_JS = """
<script>
var current = 0;
function showScene(idx) {
  document.querySelectorAll('.scene-panel').forEach(function(el,i){
    el.classList.toggle('active', i===idx);
  });
  document.querySelectorAll('.tab-btn').forEach(function(el,i){
    el.classList.remove('active','val-active');
    if(i===idx) el.classList.add(el.dataset.split==='val'?'val-active':'active');
  });
}
</script></body></html>
"""


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',      type=int,   default=42)
    ap.add_argument('--min-score', type=float, default=0.3,
                    help='Pred box score threshold (default 0.3). Lower = more boxes.')
    ap.add_argument('--out',  default='test_bbox_viz.html')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    all_figures = []
    all_labels  = []
    all_splits  = []

    for split in ('train', 'val'):
        print(f"\n{'='*50}")
        print(f"Loading [{split}] datasets ...")

        ds_gt = NuScenesNVSDataset(
            dataroot=DATAROOT, version='v1.0-trainval',
            split=split, frame_gap=FRAME_GAP, mode='bbox',
            bbox_json_path=None,
        )
        ds_pred = NuScenesNVSDataset(
            dataroot=DATAROOT, version='v1.0-trainval',
            split=split, frame_gap=FRAME_GAP, mode='bbox',
            bbox_json_path=BBOX_JSON,
        )
        # score threshold 필터링: tracking_score < min_score 제거
        ds_pred.bbox_data = {
            tok: [e for e in boxes if e.get('tracking_score', 1.0) >= args.min_score]
            for tok, boxes in ds_pred.bbox_data.items()
        }
        print(f"[{split}] pred score threshold: {args.min_score}")

        indices = sample_scene_indices(ds_gt, N_SCENES, rng)
        print(f"[{split}] sampled indices: {indices}")

        for i, idx in enumerate(indices):
            print(f"  [{split} scene {i+1}] idx={idx} loading ...")
            gt_item   = ds_gt[idx]
            pred_item = ds_pred[idx]
            title = f'[{split.upper()}] Scene {i+1}  idx={idx}'
            fig   = make_figure(gt_item, pred_item, title)
            all_figures.append(fig)
            all_labels.append(f'{split.upper()} S{i+1}')
            all_splits.append(split)
            print(f"    GT:   F0={len(gt_item['boxes_0'])} F1={len(gt_item['boxes_1'])}")
            print(f"    Pred: F0={len(pred_item['boxes_0'])} F1={len(pred_item['boxes_1'])}")

    # ── assemble HTML ──
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    parts = [HTML_HEAD]

    # section label + tab buttons
    parts.append('<p class="section-label">▶ TRAIN &nbsp;&nbsp; | &nbsp;&nbsp; VAL</p>\n')
    parts.append('<div class="tab-bar">\n')
    for i, (label, sp) in enumerate(zip(all_labels, all_splits)):
        active = ' active' if i == 0 else ''
        parts.append(
            f'<button class="tab-btn{active}" data-split="{sp}" '
            f'onclick="showScene({i})">{label}</button>\n'
        )
    parts.append('</div>\n')

    for i, fig in enumerate(all_figures):
        active = ' active' if i == 0 else ''
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=(i == 0),
            config={'scrollZoom': True, 'displayModeBar': True},
        )
        parts.append(f'<div class="scene-panel{active}">{fig_html}</div>\n')

    parts.append(HTML_JS)

    with open(out_path, 'w') as f:
        f.write(''.join(parts))

    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
