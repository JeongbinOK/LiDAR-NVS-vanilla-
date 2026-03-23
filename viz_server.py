"""Interactive 3D viewer: point cloud + Gaussian surfels via Three.js."""

import argparse
import http.server
import json
import os
import sys
import threading
import webbrowser

import numpy as np
import torch

from gaustering.config import ClusteringConfig
from gaustering.data_loader import list_lidar_files
from gaustering.pipeline import cluster_frame


def build_viewer_data(result: dict) -> dict:
    """Extract visualization data from pipeline result."""
    xyz = result["source_xyz"].cpu().numpy()
    assignments = result["point_assignments"].cpu().numpy()
    point_mask = result["point_mask"].cpu().numpy()
    K = result["num_gaussians"]

    # Per-point cluster colors
    rng = np.random.RandomState(42)
    palette = rng.rand(K + 1, 3).tolist()
    palette[-1] = [0.3, 0.3, 0.3]

    valid = point_mask & (assignments >= 0)
    color_idx = np.full(xyz.shape[0], K, dtype=np.int64)
    color_idx[valid] = assignments[valid] % K

    # Gaussian surfel data
    centers = result["xyz"].cpu().numpy()
    normals = result["normal"].cpu().numpy()
    tangent_u = result["tangent_u"].cpu().numpy()
    tangent_v = result["tangent_v"].cpu().numpy()
    scaling = result["scaling"].cpu().numpy()
    gamma_rms = result["_gamma_rms"].cpu().numpy()

    return {
        "points": xyz.tolist(),
        "color_idx": color_idx.tolist(),
        "palette": palette,
        "gaussians": {
            "centers": centers.tolist(),
            "normals": normals.tolist(),
            "tangent_u": tangent_u.tolist(),
            "tangent_v": tangent_v.tolist(),
            "scaling": scaling.tolist(),
            "gamma_rms": gamma_rms.tolist(),
        },
        "stats": {
            "num_points": int(xyz.shape[0]),
            "num_gaussians": int(K),
            "coverage": float(point_mask.mean()),
            "gamma_rms_median": float(np.median(gamma_rms)),
        },
    }


VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Gaustering 3D Viewer</title>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; font-family: monospace; }
  canvas { display: block; }
  #info {
    position: absolute; top: 10px; left: 10px;
    color: #eee; background: rgba(0,0,0,0.7); padding: 12px 16px;
    border-radius: 6px; font-size: 13px; line-height: 1.6;
    pointer-events: none;
  }
  #controls {
    position: absolute; top: 10px; right: 10px;
    color: #eee; background: rgba(0,0,0,0.7); padding: 12px 16px;
    border-radius: 6px; font-size: 13px; line-height: 2;
  }
  #controls label { cursor: pointer; }
  #controls input[type=range] { width: 120px; vertical-align: middle; }
</style>
</head>
<body>
<div id="info">Loading...</div>
<div id="controls">
  <label><input type="checkbox" id="showPoints" checked> Points</label><br>
  <label><input type="checkbox" id="showSurfels" checked> Surfels</label><br>
  <label><input type="checkbox" id="showNormals"> Normals</label><br>
  <label>Point size: <input type="range" id="pointSize" min="1" max="8" value="2" step="0.5"></label><br>
  <label>Surfel opacity: <input type="range" id="surfelOpacity" min="0" max="1" value="0.4" step="0.05"></label><br>
  <label>Surfel scale: <input type="range" id="surfelScale" min="0.5" max="5" value="2" step="0.25"></label>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(0, -40, 30);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0, 0, 0);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(10, -10, 20);
scene.add(dirLight);

// Grid
const grid = new THREE.GridHelper(100, 50, 0x444466, 0x333355);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

// Load data
const data = await fetch('/data.json').then(r => r.json());
const { points, color_idx, palette, gaussians, stats } = data;

document.getElementById('info').innerHTML =
  `<b>Gaustering 3D Viewer</b><br>` +
  `Points: ${stats.num_points.toLocaleString()}<br>` +
  `Gaussians: ${stats.num_gaussians}<br>` +
  `Coverage: ${(stats.coverage * 100).toFixed(1)}%<br>` +
  `gamma_rms median: ${stats.gamma_rms_median.toFixed(4)}<br>` +
  `<span style="color:#888">Drag: rotate | Scroll: zoom | Right-drag: pan</span>`;

// === Point Cloud ===
const N = points.length;
const positions = new Float32Array(N * 3);
const colors = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  positions[i*3]   = points[i][0];
  positions[i*3+1] = points[i][1];
  positions[i*3+2] = points[i][2];
  const c = palette[color_idx[i]];
  colors[i*3]   = c[0];
  colors[i*3+1] = c[1];
  colors[i*3+2] = c[2];
}
const pcGeom = new THREE.BufferGeometry();
pcGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
pcGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
const pcMat = new THREE.PointsMaterial({ size: 0.15, vertexColors: true, sizeAttenuation: true });
const pointCloud = new THREE.Points(pcGeom, pcMat);
scene.add(pointCloud);

// === Surfel Ellipses ===
const surfelGroup = new THREE.Group();
const G = gaussians.centers.length;
const circleGeom = new THREE.CircleGeometry(1, 24);

for (let i = 0; i < G; i++) {
  const c = gaussians.centers[i];
  const n = gaussians.normals[i];
  const u = gaussians.tangent_u[i];
  const v = gaussians.tangent_v[i];
  const s = gaussians.scaling[i];
  const col = palette[i % (palette.length - 1)];

  const mat = new THREE.MeshBasicMaterial({
    color: new THREE.Color(col[0], col[1], col[2]),
    transparent: true, opacity: 0.4, side: THREE.DoubleSide,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(circleGeom, mat);

  // Build rotation matrix from tangent frame
  const m = new THREE.Matrix4();
  const sc = 2.0;
  m.set(
    u[0]*s[0]*sc, v[0]*s[1]*sc, n[0], c[0],
    u[1]*s[0]*sc, v[1]*s[1]*sc, n[1], c[1],
    u[2]*s[0]*sc, v[2]*s[1]*sc, n[2], c[2],
    0, 0, 0, 1
  );
  mesh.matrixAutoUpdate = false;
  mesh.matrix.copy(m);

  surfelGroup.add(mesh);
}
scene.add(surfelGroup);

// === Normal Lines ===
const normalGroup = new THREE.Group();
const normalPositions = new Float32Array(G * 6);
const normalColors = new Float32Array(G * 6);
for (let i = 0; i < G; i++) {
  const c = gaussians.centers[i];
  const n = gaussians.normals[i];
  const len = 0.5;
  normalPositions[i*6]   = c[0];
  normalPositions[i*6+1] = c[1];
  normalPositions[i*6+2] = c[2];
  normalPositions[i*6+3] = c[0] + n[0]*len;
  normalPositions[i*6+4] = c[1] + n[1]*len;
  normalPositions[i*6+5] = c[2] + n[2]*len;
  // Green lines
  normalColors[i*6] = 0; normalColors[i*6+1] = 1; normalColors[i*6+2] = 0;
  normalColors[i*6+3] = 0; normalColors[i*6+4] = 1; normalColors[i*6+5] = 0;
}
const nlGeom = new THREE.BufferGeometry();
nlGeom.setAttribute('position', new THREE.BufferAttribute(normalPositions, 3));
nlGeom.setAttribute('color', new THREE.BufferAttribute(normalColors, 3));
const nlMat = new THREE.LineBasicMaterial({ vertexColors: true });
const normalLines = new THREE.LineSegments(nlGeom, nlMat);
normalLines.visible = false;
normalGroup.add(normalLines);
scene.add(normalGroup);

// === Controls ===
document.getElementById('showPoints').addEventListener('change', e => {
  pointCloud.visible = e.target.checked;
});
document.getElementById('showSurfels').addEventListener('change', e => {
  surfelGroup.visible = e.target.checked;
});
document.getElementById('showNormals').addEventListener('change', e => {
  normalLines.visible = e.target.checked;
});
document.getElementById('pointSize').addEventListener('input', e => {
  pcMat.size = parseFloat(e.target.value) * 0.075;
});
document.getElementById('surfelOpacity').addEventListener('input', e => {
  const val = parseFloat(e.target.value);
  surfelGroup.children.forEach(m => { m.material.opacity = val; });
});
document.getElementById('surfelScale').addEventListener('input', e => {
  const sc = parseFloat(e.target.value);
  for (let i = 0; i < G; i++) {
    const c = gaussians.centers[i];
    const n = gaussians.normals[i];
    const u = gaussians.tangent_u[i];
    const v = gaussians.tangent_v[i];
    const s = gaussians.scaling[i];
    const m = new THREE.Matrix4();
    m.set(
      u[0]*s[0]*sc, v[0]*s[1]*sc, n[0], c[0],
      u[1]*s[0]*sc, v[1]*s[1]*sc, n[1], c[1],
      u[2]*s[0]*sc, v[2]*s[1]*sc, n[2], c[2],
      0, 0, 0, 1
    );
    surfelGroup.children[i].matrix.copy(m);
  }
});

// === Animate ===
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, data_json=None, **kwargs):
        self._data_json = data_json
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(VIEWER_HTML.encode())
        elif self.path == '/data.json':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(self._data_json)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress logs


def main():
    parser = argparse.ArgumentParser(description="3D Gaussian surfel viewer")
    parser.add_argument("--data-root", default=os.path.expanduser("~/data/nuScenes"))
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--port", type=int, default=8890)
    args = parser.parse_args()

    files = list_lidar_files(args.data_root)
    if not files:
        print(f"No .pcd.bin files found under {args.data_root}")
        sys.exit(1)

    path = files[min(args.frame_idx, len(files) - 1)]
    print(f"Processing: {os.path.basename(path)}")

    cfg = ClusteringConfig()
    result = cluster_frame(path, cfg)
    print(f"Done: {result['num_gaussians']} gaussians from {result['num_filtered_points']} points")

    viewer_data = build_viewer_data(result)
    data_json = json.dumps(viewer_data).encode()

    def handler_factory(*args, **kwargs):
        return Handler(*args, data_json=data_json, **kwargs)

    server = http.server.HTTPServer(('0.0.0.0', args.port), handler_factory)
    print(f"\nViewer ready: http://localhost:{args.port}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
