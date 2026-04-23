# Source attribution

This directory is a **snapshot copy** (not a fork, not a submodule) of the upstream
`diff-quadratic-rasterization` CUDA rasterizer from the QGS (Quadratic Gaussian
Surfels) project. It is included directly in this repository so we can refactor it
freely for LiDAR rendering without tracking upstream changes.

## Provenance

- **Origin**: <https://github.com/will-zzy/QGS>
- **Source path in upstream**: `submodules/diff-quadratic-rasterization/`
- **Upstream commit (parent repo HEAD)**: `74d05c945e99fcaef7afe5a8831903be71ad9b55`
- **Snapshot date**: 2026-04-18
- **Snapshot taken by**: `git clone --depth 1 https://github.com/will-zzy/QGS /tmp/QGS-snapshot`
- **Original `.git` directories removed** — this directory is plain source.

## License

Inria / Max Planck Institut für Informatik **Gaussian Splatting License**
(non-commercial / research use). Full text: see `LICENSE.md` next to this file
(unmodified copy of the upstream license).

If we ever distribute or publish anything derived from this code we must keep the
license text and credit the original authors:

- 3D Gaussian Splatting (Inria GRAPHDECO group)
- 2D Gaussian Splatting (Huang, Yu et al.)
- Quadratic Gaussian Splatting / QGS (Will Zhang et al., ICCV 2025)

## Planned modifications (Phase A3 of the QGS-Flow plan)

The upstream rasterizer is camera-centric (perspective projection, RGB / normal /
curvature channel layout). We will incrementally rework it for LiDAR rendering:

- **A3.1** (this commit) — bring it in untouched and verify it builds.
- **A3.2** — replace camera projection with **spherical (azimuth × elevation)**
  projection; reference: `GS-LiDAR/diff-gaussian-rasterization-2d/` panoramic kernel.
  Handle azimuth wraparound.
- **A3.3** — re-design the rendered-image channel layout for LiDAR:
  `range, intensity, drop_logit, alpha_accum, normal(3), curvature, feat_agg(D_f)`.
  (Upstream uses `rendered_image[3:6]=normal, [11:12]=curvature` etc., hard-coded.)
- **A3.4** — add the **curvature-aware drop MLP** (Plan §5.5, S-1 novelty):
  `feat_agg + κ_render + cos θ_inc + log(1+r) + ray_dir → p_drop_phys`,
  with the analytic decomposition `p_drop = (1 − α_accum) + α_accum · p_drop_phys`.
- **A3.5** — single static-frame self-rendering test against a 2DGS baseline.

The exact ray–paraboloid intersection (`A·t² + B·t + C = 0`) from the upstream
kernel is **kept as-is** — that is the whole reason we picked QGS as our base
instead of `gsplat` (which is Gaussian-density-centric).
