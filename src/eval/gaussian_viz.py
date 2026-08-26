"""Interactive 3D HTML viewer for the Gaussians produced during eval.

One self-contained HTML per sequence (Plotly.js from CDN, no python dep). The
1/2/3/4 buttons toggle the Gaussian point cloud of each target second
(T=1,2,3,4s). Points are Gaussian *centers* (means3D) at that window's target
time, in the window's ref frame (frame-0 sensor frame, z-up). Bbox-transport
models show static/dynamic centers. Velocity-transport models color target-time
centers by their source endpoint frame and draw each actual source-to-target
transport as an arrow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def extract_frame(means_np: np.ndarray, is_dyn_np: np.ndarray,
                  max_static=None, seed: int = 0):
    """Split into static / dynamic. `max_static=None` keeps ALL Gaussians."""
    is_dyn_np = is_dyn_np.astype(bool)
    static = means_np[~is_dyn_np]
    dynamic = means_np[is_dyn_np]
    if max_static is not None and static.shape[0] > max_static:
        rng = np.random.default_rng(seed)
        static = static[rng.choice(static.shape[0], max_static, replace=False)]
    return static, dynamic


def gaussians_from_output(g2p_model, b_gs, t_target: float, max_static=None):
    """means3D at the target time + dynamic mask -> (static_xyz, dynamic_xyz)."""
    means = g2p_model.get_means3D(b_gs, t_target).detach().cpu().numpy()
    is_dyn = b_gs.get("is_dynamic")
    if torch.is_tensor(is_dyn):
        is_dyn = is_dyn.detach().cpu().numpy()
    else:
        is_dyn = np.zeros((means.shape[0],), dtype=bool)
    return extract_frame(means, is_dyn, max_static=max_static)


def velocity_transport_from_output(g2p_model, b_gs, t_target: float):
    """Return the exact source-to-target transport used by DynamicGausRender.

    For every Gaussian, ``delta_t_sec = target_time - source_time_sec`` and the
    rendered target center is ``source_center + velocity * delta_t_sec``. The
    sign matters: when the target lies between two endpoint frames, Gaussians
    from the later endpoint travel backward to the target with a negative delta.
    All positions and velocities are already in the window reference frame.
    """
    position = b_gs.get("position")
    velocity = b_gs.get("velocity")
    source_time = b_gs.get("source_time_sec")
    if not all(torch.is_tensor(value) for value in (
        position, velocity, source_time,
    )):
        raise KeyError(
            "Velocity visualization requires position, velocity, and source_time_sec"
        )
    means = g2p_model.get_means3D(b_gs, t_target).detach()
    position = position.detach().to(device=means.device, dtype=means.dtype)
    velocity = velocity.detach().to(device=means.device, dtype=means.dtype)
    source_time = source_time.detach().to(device=means.device, dtype=means.dtype)
    if position.shape != means.shape or velocity.shape != means.shape:
        raise ValueError(
            "Positions and velocity vectors must align with rendered centers"
        )
    if source_time.shape != (means.shape[0],):
        raise ValueError("source_time_sec must provide one value per Gaussian")
    target_time = torch.as_tensor(
        float(t_target), device=means.device, dtype=means.dtype
    )
    delta_t = target_time - source_time
    expected_means = position + velocity * delta_t.unsqueeze(-1)
    if not torch.allclose(means, expected_means, rtol=1.0e-5, atol=1.0e-5):
        raise RuntimeError(
            "Velocity visualization does not match renderer target centers"
        )
    if not bool(all(torch.isfinite(value).all() for value in (
        position, means, velocity, source_time, delta_t,
    ))):
        raise ValueError("Velocity visualization received non-finite transport data")

    source_times, source_frame_index = torch.unique(
        source_time, sorted=True, return_inverse=True
    )
    target_centers_by_source_frame = [
        means[source_frame_index == frame].cpu().numpy()
        for frame in range(int(source_times.numel()))
    ]
    return {
        "source_centers": position.cpu().numpy(),
        "target_centers": means.cpu().numpy(),
        "target_centers_by_source_frame": target_centers_by_source_frame,
        "velocity_mps": velocity.cpu().numpy(),
        "source_time_sec": source_time.cpu().numpy(),
        "source_times_sec": source_times.cpu().numpy(),
        "source_frame_index": source_frame_index.cpu().numpy(),
        "delta_t_sec": delta_t.cpu().numpy(),
    }


_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{font-family:sans-serif;margin:0;background:#111;color:#eee}
#bar{padding:8px 12px;background:#1c1c1c;border-bottom:1px solid #333;font-size:14px}
button{font-size:14px;margin:3px;padding:5px 14px;cursor:pointer;border:1px solid #555;
       background:#333;color:#eee;border-radius:4px}
button.active{background:#2a9d8f;color:#000;font-weight:bold;border-color:#2a9d8f}
#plot{width:100vw;height:89vh}
.lg{color:#888;font-size:13px}
.sep{color:#666;margin:0 8px}
</style></head>
<body>
<div id="bar"><b>__TITLE__</b> <span class="sep">|</span> ref frame: <span id="frames"></span>
  <span class="sep">|</span> layers: <span id="layers"></span>
  <span id="info" class="lg"></span></div>
<div id="plot"></div>
<script>
var TRACES = __TRACES__;
var LABELS = __LABELS__;
var TAG = __TAG__;          // [frame_index, layer_name] per trace
var LAYERS = __LAYERS__;
var OFF = __OFF__;          // layers that start hidden
var cur = 0, on = {};
LAYERS.forEach(function(l){on[l] = OFF.indexOf(l) < 0;});
var layout = {paper_bgcolor:'#111',showlegend:true,margin:{l:0,r:0,t:0,b:0},
  legend:{font:{color:'#ccc'}},
  scene:{aspectmode:'data',bgcolor:'#111',
    xaxis:{color:'#777',title:'x'},yaxis:{color:'#777',title:'y'},zaxis:{color:'#777',title:'z'}}};
Plotly.newPlot('plot',TRACES,layout,{responsive:true});
function refresh(){
  Plotly.restyle('plot','visible',TAG.map(function(t){return t[0]===cur && on[t[1]];}));
  document.querySelectorAll('.fbtn').forEach(function(b,i){
    b.className='fbtn'+(i===cur?' active':'');});
  document.querySelectorAll('.lbtn').forEach(function(b){
    b.className='lbtn'+(on[b.dataset.l]?' active':'');});
  document.getElementById('info').textContent='   '+LABELS[cur];
}
LABELS.forEach(function(lb,i){
  var e=document.createElement('button'); e.className='fbtn'; e.textContent=(i+1);
  e.onclick=function(){cur=i;refresh();}; document.getElementById('frames').appendChild(e);});
LAYERS.forEach(function(l){
  var e=document.createElement('button'); e.className='lbtn'; e.dataset.l=l; e.textContent=l;
  e.onclick=function(){on[l]=!on[l];refresh();}; document.getElementById('layers').appendChild(e);});
refresh();
</script></body></html>"""

# layer names shared by both viewers
L_STATIC = "gaussians (static)"
L_DYN = "gaussians (dynamic)"
L_BOX = "bbox"
L_INPUT = "input LiDAR (2 endpoints)"
L_GT = "GT LiDAR @ target t"
L_PRED = "predicted LiDAR (raydrop kept)"
SOURCE_FRAME_COLORS = (
    "#00cc96", "#ab63fa", "#ffa15a", "#19d3f3", "#ff6692", "#b6e880",
)


def _source_frame_layer(frame_index: int) -> str:
    return f"gaussians from source frame {int(frame_index)}"


def _source_frame_velocity_layer(frame_index: int) -> str:
    return (
        f"velocity from source frame {int(frame_index)} "
        "× actual Δt (source → target)"
    )


def _point_trace(xyz, color, size, name, sub=120000, seed=0):
    """Raw-LiDAR scatter trace (subsampled so a 4-frame page stays loadable)."""
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    if xyz.shape[0] > sub:
        xyz = xyz[np.random.default_rng(seed).choice(xyz.shape[0], sub, replace=False)]
    xyz = np.round(xyz, 2)
    return {"type": "scatter3d", "mode": "markers",
            "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(), "z": xyz[:, 2].tolist(),
            "marker": {"size": size, "color": color, "opacity": 0.75},
            "name": name, "visible": False, "hoverinfo": "skip"}


def _velocity_arrow_trace(origins, velocity_mps, delta_t_sec, name, color):
    """Build exact source-to-target 3D transport arrows as one line trace."""
    origins = np.asarray(origins, np.float32).reshape(-1, 3)
    velocity_mps = np.asarray(velocity_mps, np.float32).reshape(-1, 3)
    if origins.shape != velocity_mps.shape:
        raise ValueError("velocity origins and vectors must have matching (N,3) shape")
    delta_t_sec = np.asarray(delta_t_sec, np.float32).reshape(-1)
    if delta_t_sec.shape != (origins.shape[0],):
        raise ValueError("delta_t_sec must provide one value per velocity vector")

    displacement = velocity_mps * delta_t_sec[:, None]
    length = np.linalg.norm(displacement, axis=1)
    keep = np.isfinite(origins).all(axis=1) & np.isfinite(velocity_mps).all(axis=1)
    keep &= np.isfinite(delta_t_sec) & (length > 1.0e-6)
    origins = origins[keep]
    displacement = displacement[keep]
    length = length[keep]

    if origins.shape[0] == 0:
        xyz = np.zeros((0, 3), np.float32)
    else:
        direction = displacement / length[:, None]
        tips = origins + displacement

        # Choose a stable vector that is not parallel to the arrow direction,
        # then form one camera-independent 3D side direction with a cross product.
        reference = np.zeros_like(direction)
        reference[:, 2] = 1.0
        near_vertical = np.abs(direction[:, 2]) > 0.9
        reference[near_vertical] = np.array([0.0, 1.0, 0.0], np.float32)
        side = np.cross(direction, reference)
        side /= np.linalg.norm(side, axis=1, keepdims=True).clip(min=1.0e-12)

        head_length = 0.22 * length
        head_width = 0.45 * head_length
        head_base = tips - head_length[:, None] * direction
        left = head_base + head_width[:, None] * side
        right = head_base - head_width[:, None] * side

        # shaft, left fin, right fin; NaNs separate independent line segments.
        nan = np.full_like(origins, np.nan)
        xyz = np.stack(
            [origins, tips, nan, tips, left, nan, tips, right, nan], axis=1
        ).reshape(-1, 3)

    xyz = np.round(xyz, 3)
    return {
        "type": "scatter3d", "mode": "lines",
        "x": xyz[:, 0].tolist(), "y": xyz[:, 1].tolist(),
        "z": xyz[:, 2].tolist(),
        "line": {"color": color, "width": 2},
        "name": name, "visible": False, "hoverinfo": "skip",
    }


def _render(path, title, labels, traces, tags, layers, off):
    html = (_HTML
            .replace("__TITLE__", str(title))
            .replace("__TRACES__", json.dumps(traces))
            .replace("__TAG__", json.dumps(tags))
            .replace("__LABELS__", json.dumps(labels))
            .replace("__LAYERS__", json.dumps(layers))
            .replace("__OFF__", json.dumps(off)))
    Path(path).write_text(html)
    return path


def save_sequence_html(path, title: str, frames: list, point_size: float = 1.6):
    """Write bbox-mode or velocity-mode Gaussian center frames.

    Bbox frames contain ``static`` / ``dynamic`` centers. Velocity frames contain
    ``source_frame_centers`` plus aligned ``velocity_origins``, ``velocity_mps``,
    ``velocity_delta_t_sec``, and ``velocity_source_frame_index``. All positions
    use the window reference frame.

    The raw-LiDAR layers are what make the Gaussian centres readable: without
    them you cannot tell a misplaced centre from a correctly placed one.
    """
    source_frame_flags = [
        "source_frame_centers" in fr for fr in frames
    ]
    velocity_mode = any(source_frame_flags)
    if velocity_mode and not all(source_frame_flags):
        raise ValueError("All sequence frames must use the same center color mode")
    if velocity_mode and not all(
        all(key in fr for key in (
            "velocity_origins", "velocity_mps", "velocity_delta_t_sec",
            "velocity_source_frame_index",
        ))
        for fr in frames
    ):
        raise ValueError(
            "Velocity frames require origins, vectors, actual delta_t_sec, "
            "and source frame indices"
        )
    source_frame_count = (
        len(frames[0]["source_frame_centers"]) if velocity_mode and frames else 0
    )
    if velocity_mode and any(
        len(fr["source_frame_centers"]) != source_frame_count for fr in frames
    ):
        raise ValueError("Source frame count must stay constant across a sequence")

    traces, tags = [], []
    for fi, fr in enumerate(frames):
        if velocity_mode:
            origins = np.asarray(fr["velocity_origins"], np.float32).reshape(-1, 3)
            velocity = np.asarray(fr["velocity_mps"], np.float32).reshape(-1, 3)
            delta_t = np.asarray(
                fr["velocity_delta_t_sec"], np.float32
            ).reshape(-1)
            source_frame_index = np.asarray(
                fr["velocity_source_frame_index"]
            ).reshape(-1)
            if (
                velocity.shape != origins.shape
                or delta_t.shape != (origins.shape[0],)
                or source_frame_index.shape != (origins.shape[0],)
            ):
                raise ValueError(
                    "Velocity vectors, times, and source frame indices must "
                    "align with origins"
                )
            if not np.issubdtype(source_frame_index.dtype, np.integer):
                raise ValueError("Velocity source frame indices must be integers")
            if source_frame_index.size and (
                source_frame_index.min() < 0
                or source_frame_index.max() >= source_frame_count
            ):
                raise ValueError("Velocity source frame index is out of range")

            for source_frame, centers in enumerate(fr["source_frame_centers"]):
                centers = np.round(np.asarray(centers, np.float32), 2)
                layer = _source_frame_layer(source_frame)
                velocity_layer = _source_frame_velocity_layer(source_frame)
                color = SOURCE_FRAME_COLORS[
                    source_frame % len(SOURCE_FRAME_COLORS)
                ]
                source_mask = source_frame_index == source_frame
                if int(source_mask.sum()) != centers.shape[0]:
                    raise ValueError(
                        "Source-frame centers must align with velocity source "
                        "frame indices"
                    )
                traces.append({
                    "type": "scatter3d", "mode": "markers",
                    "x": centers[:, 0].tolist(),
                    "y": centers[:, 1].tolist(),
                    "z": centers[:, 2].tolist(),
                    "marker": {
                        "size": point_size + 1.0,
                        "color": color,
                        "opacity": 0.9,
                    },
                    "name": layer, "visible": False, "hoverinfo": "skip",
                })
                tags.append([fi, layer])
                traces.append(_velocity_arrow_trace(
                    origins[source_mask], velocity[source_mask],
                    delta_t[source_mask], velocity_layer, color,
                ))
                tags.append([fi, velocity_layer])
        else:
            s = np.round(np.asarray(fr["static"], np.float32), 2)
            d = np.round(np.asarray(fr["dynamic"], np.float32), 2)
            traces.append({
                "type": "scatter3d", "mode": "markers",
                "x": s[:, 0].tolist(), "y": s[:, 1].tolist(), "z": s[:, 2].tolist(),
                "marker": {"size": point_size, "color": s[:, 2].tolist(),
                           "colorscale": "Viridis", "opacity": 0.8},
                "name": L_STATIC, "visible": False, "hoverinfo": "skip"})
            tags.append([fi, L_STATIC])
            traces.append({
                "type": "scatter3d", "mode": "markers",
                "x": d[:, 0].tolist(), "y": d[:, 1].tolist(), "z": d[:, 2].tolist(),
                "marker": {"size": point_size + 1.8, "color": "#e63946", "opacity": 0.95},
                "name": L_DYN, "visible": False, "hoverinfo": "skip"})
            tags.append([fi, L_DYN])
        traces.append(_point_trace(fr.get("input_points", np.zeros((0, 3))),
                                   "#9aa0a6", 1.2, L_INPUT))
        tags.append([fi, L_INPUT])
        traces.append(_point_trace(fr.get("gt_points", np.zeros((0, 3))),
                                   "#f4a261", 1.3, L_GT))
        tags.append([fi, L_GT])

    layers = (
        [
            layer
            for i in range(source_frame_count)
            for layer in (
                _source_frame_layer(i), _source_frame_velocity_layer(i),
            )
        ]
        if velocity_mode else [L_STATIC, L_DYN]
    )
    layers.extend([L_INPUT, L_GT])
    return _render(path, title, [fr["label"] for fr in frames], traces, tags,
                   layers, off=[L_INPUT])


# ===========================================================================
# 2D-Gaussian SURFEL viewer (each Gaussian drawn as its 1-sigma oriented disk)
# ===========================================================================
def _build_rotation_np(quat: np.ndarray) -> np.ndarray:
    """[N,4] (w,x,y,z) -> [N,3,3] rotation matrices (quat normalized first)."""
    q = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _ellipse_mesh(centers, quats, su, sv, k=8, sigma=1.0):
    """Triangulated 1-sigma disks. Returns (verts[V,3], faces[F,3]) with a
    center+fan per surfel. Tangents = first two columns of R (2DGS convention)."""
    m = centers.shape[0]
    if m == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    R = _build_rotation_np(quats)
    tu, tv = R[:, :, 0], R[:, :, 1]                          # [m,3] disk tangents
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)  # [k]
    # boundary [m,k,3] = center + sigma*(su*cos*tu + sv*sin*tv)
    bd = (centers[:, None, :]
          + sigma * (su[:, None, None] * cos[None, :, None] * tu[:, None, :]
                     + sv[:, None, None] * sin[None, :, None] * tv[:, None, :]))
    verts = np.concatenate([centers[:, None, :], bd], axis=1).reshape(-1, 3)   # [m*(k+1),3]
    base = (np.arange(m) * (k + 1)).astype(np.int64)
    ci = base
    faces = []
    for kk in range(k):
        a = base + 1 + kk
        b = base + 1 + ((kk + 1) % k)
        faces.append(np.stack([ci, a, b], axis=1))
    faces = np.concatenate(faces, axis=0)                    # [m*k,3]
    return verts.astype(np.float32), faces


def dyn_boxes_from_output(g2p_model, b_gs, t_target: float) -> np.ndarray:
    """Per-instance dynamic bbox [cx,cy,cz,dx,dy,dz,yaw] in the ref frame,
    interpolated to the target time (same path the renderer uses). [M,7]."""
    trajs = b_gs.get("object_trajectories", {})
    if not trajs:
        return np.zeros((0, 7), np.float32)
    dev = b_gs["position"].device
    boxes = []
    for _, traj in trajs.items():
        box = g2p_model.interpolate_box_ref(traj, t_target, dev, torch.float32)
        boxes.append(box.detach().cpu().numpy())
    return np.stack(boxes).astype(np.float32) if boxes else np.zeros((0, 7), np.float32)


def surfels_from_output(g2p_model, b_gs, t_target: float,
                        max_static=None, k: int = 8, sigma: float = 1.0,
                        seed: int = 0):
    """Build 1-sigma surfel meshes (static + dynamic) for one window.

    Uses the SAME means/rotations the renderer uses (`get_means3D` /
    `get_rotations` at the target time) and 2D scales = softplus(scaling[:, :2]).
    Scales are shown AS-IS (no clamp) and `max_static=None` keeps ALL Gaussians,
    so degenerate large-scale Gaussians are fully visible for diagnosis. Also
    returns the dynamic bboxes at the target time for overlay."""
    means = g2p_model.get_means3D(b_gs, t_target).detach()
    rots = g2p_model.get_rotations(b_gs, t_target).detach()
    sca = b_gs["scaling"].detach()
    su = torch.nn.functional.softplus(sca[:, 0])
    sv = torch.nn.functional.softplus(sca[:, 1])
    is_dyn = b_gs.get("is_dynamic")
    is_dyn = (is_dyn.detach().cpu().numpy().astype(bool)
              if torch.is_tensor(is_dyn) else np.zeros((means.shape[0],), bool))

    means = means.cpu().numpy(); rots = rots.cpu().numpy()
    su = su.cpu().numpy(); sv = sv.cpu().numpy()

    s_idx = np.where(~is_dyn)[0]
    if max_static is not None and s_idx.shape[0] > max_static:
        s_idx = np.random.default_rng(seed).choice(s_idx, max_static, replace=False)
    d_idx = np.where(is_dyn)[0]

    static = _ellipse_mesh(means[s_idx], rots[s_idx], su[s_idx], sv[s_idx], k, sigma)
    dynamic = _ellipse_mesh(means[d_idx], rots[d_idx], su[d_idx], sv[d_idx], k, sigma)
    boxes = dyn_boxes_from_output(g2p_model, b_gs, t_target)
    return {"static": static, "dynamic": dynamic, "boxes": boxes}


def _box_lines(boxes: np.ndarray):
    """[M,7] oriented boxes -> (x,y,z) line lists with NaN edge separators for a
    single scatter3d line trace (12 edges per box)."""
    xs, ys, zs = [], [], []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),       # bottom loop
             (4, 5), (5, 6), (6, 7), (7, 4),       # top loop
             (0, 4), (1, 5), (2, 6), (3, 7)]       # verticals
    sgn = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32)
    for box in boxes:
        c, d, yaw = box[:3], box[3:6], float(box[6])
        # d is nuScenes (w, l, h); yaw rotates the LENGTH onto local +x, so the
        # local extents are (l, w, h) -- using (w, l, h) draws every non-square
        # box rotated by 90 deg (see boxes.point_in_box).
        extent = np.array([d[1], d[0], d[2]], np.float32)
        corners = sgn * (0.5 * extent)[None, :]                    # [8,3] local
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        rx = cos_y * corners[:, 0] - sin_y * corners[:, 1]
        ry = sin_y * corners[:, 0] + cos_y * corners[:, 1]
        world = np.stack([rx, ry, corners[:, 2]], axis=1) + c[None, :]
        for a, bb in edges:
            xs += [world[a, 0], world[bb, 0], np.nan]
            ys += [world[a, 1], world[bb, 1], np.nan]
            zs += [world[a, 2], world[bb, 2], np.nan]
    return xs, ys, zs


def _mesh_trace(mesh, color_by_height, visible, name, opacity, red=False):
    v, f = mesh
    v = np.round(v, 1)
    tr = {"type": "mesh3d",
          "x": v[:, 0].tolist(), "y": v[:, 1].tolist(), "z": v[:, 2].tolist(),
          "i": f[:, 0].tolist(), "j": f[:, 1].tolist(), "k": f[:, 2].tolist(),
          "opacity": opacity, "flatshading": True, "name": name,
          "visible": visible, "hoverinfo": "skip"}
    if red:
        tr["color"] = "#e63946"
    else:
        tr["intensity"] = v[:, 2].tolist()
        tr["colorscale"] = "Viridis"
        tr["showscale"] = False
    return tr


def _box_trace(boxes, visible, name):
    xs, ys, zs = _box_lines(np.asarray(boxes, np.float32).reshape(-1, 7))
    return {"type": "scatter3d", "mode": "lines",
            "x": [None if np.isnan(v) else round(float(v), 2) for v in xs],
            "y": [None if np.isnan(v) else round(float(v), 2) for v in ys],
            "z": [None if np.isnan(v) else round(float(v), 2) for v in zs],
            "line": {"color": "#ffd000", "width": 3},
            "name": name, "visible": visible, "hoverinfo": "skip"}


def save_sequence_surfel_html(path, title: str, frames: list):
    """frames: list of {"label", "static": (V,F), "dynamic": (V,F), "boxes": [M,7],
    optional "input_points" [P,3], "gt_points" [Q,3], and
    "pred_points" [R,3]} (all point sets in the Gaussian ref frame)."""
    traces, tags = [], []
    for fi, fr in enumerate(frames):
        traces.append(_mesh_trace(fr["static"], True, False, L_STATIC, 0.65))
        tags.append([fi, L_STATIC])
        traces.append(_mesh_trace(fr["dynamic"], False, False, L_DYN, 0.9, red=True))
        tags.append([fi, L_DYN])
        traces.append(_box_trace(fr.get("boxes", np.zeros((0, 7))), False, L_BOX))
        tags.append([fi, L_BOX])
        traces.append(_point_trace(fr.get("input_points", np.zeros((0, 3))),
                                   "#9aa0a6", 1.2, L_INPUT))
        tags.append([fi, L_INPUT])
        traces.append(_point_trace(fr.get("gt_points", np.zeros((0, 3))),
                                   "#f4a261", 1.3, L_GT))
        tags.append([fi, L_GT])
        traces.append(_point_trace(fr.get("pred_points", np.zeros((0, 3))),
                                   "#00d4ff", 1.3, L_PRED))
        tags.append([fi, L_PRED])

    return _render(path, str(title) + " — 1σ surfels",
                   [fr["label"] for fr in frames], traces, tags,
                   [L_STATIC, L_DYN, L_BOX, L_INPUT, L_GT, L_PRED], off=[L_INPUT])
