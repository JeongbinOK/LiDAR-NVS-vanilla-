"""
Tests for models/geometry/quadric_fit.py

Run:
    conda activate lnvs && cd /data/jeongbin/qgs && python -m pytest tests/test_quadric_fit.py -v
"""

import math

import pytest
import torch

# Add repo root to path for imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.geometry.quadric_fit import fit_local_quadrics
from models.geometry.knn_radius import hybrid_radius_knn, gather_neighbors, default_r_max


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paraboloid_patch(
    a: float = 0.5,
    b: float = 0.3,
    grid_n: int = 5,
    noise: float = 0.0,
    device="cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate points on z = a*x^2 + b*y^2 on a grid.

    Returns:
        points [1, 1, 3]  — centre at origin
        neighbors [1, 1, K, 3]
        k_eff [1, 1]
    """
    xs = torch.linspace(-1.0, 1.0, grid_n, device=device)
    ys = torch.linspace(-1.0, 1.0, grid_n, device=device)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    x_flat = xg.reshape(-1)   # [K]
    y_flat = yg.reshape(-1)
    z_flat = a * x_flat ** 2 + b * y_flat ** 2

    if noise > 0:
        z_flat = z_flat + noise * torch.randn_like(z_flat)

    pts = torch.stack([x_flat, y_flat, z_flat], dim=-1)  # [K, 3]
    K = pts.shape[0]

    # Query point = origin (0, 0, 0)
    query = torch.zeros(1, 1, 3, device=device)
    neighbors = pts.unsqueeze(0).unsqueeze(0)   # [1, 1, K, 3]
    k_eff = torch.full((1, 1), K, dtype=torch.long, device=device)

    return query, neighbors, k_eff


def _qgs_coeff_from_scale(s: torch.Tensor) -> torch.Tensor:
    s1 = s[..., 0]
    s2 = s[..., 1]
    s3 = s[..., 2].abs()
    a = s3 * s1.sign() / s1.abs().clamp(min=1e-8).square()
    b = s3 * s2.sign() / s2.abs().clamp(min=1e-8).square()
    return torch.stack([a, b], dim=-1)


# ---------------------------------------------------------------------------
# Test 1: Synthetic paraboloid — recover curvature and frame
# ---------------------------------------------------------------------------

class TestParaboloid:
    def test_extra_diagnostics_are_finite(self):
        query, neighbors, k_eff = _make_paraboloid_patch(a=0.5, b=0.3, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)

        assert torch.isfinite(result["kappa1_init"]).all()
        assert torch.isfinite(result["kappa2_init"]).all()
        assert torch.isfinite(result["tangent_aniso"]).all()
        assert torch.isfinite(result["curvature_aniso"]).all()

    def test_s3_solves_signed_curvature_least_squares(self):
        """s_init should reproduce the directly fitted QGS coefficients."""
        a, b = 0.5, 0.3
        query, neighbors, k_eff = _make_paraboloid_patch(a=a, b=b, grid_n=7)

        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)

        s = result["s_init"][0, 0]
        use_geom = result["use_geom_init"][0, 0].item()

        assert use_geom, "use_geom_init should be True for dense patch"
        assert (s[:2] > 0).all(), f"convex patch should use same positive signature, got {s}"
        assert s[2].item() > 0.0

        coeff = _qgs_coeff_from_scale(s)
        expected = torch.tensor([a, b], dtype=coeff.dtype)
        assert torch.allclose(
            coeff.sort().values,
            expected.sort().values,
            rtol=0.08,
            atol=0.03,
        ), f"QGS coefficients {coeff.tolist()} should match direct patch coeffs {expected.tolist()}"

    def test_center_is_surface_point_under_pbar(self):
        a, b = 0.5, 0.3
        query, neighbors, k_eff = _make_paraboloid_patch(a=a, b=b, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)

        center = result["c_init"][0, 0]
        pbar = neighbors[0, 0].mean(dim=0)
        expected_z = a * center[0].item() ** 2 + b * center[1].item() ** 2

        assert torch.allclose(center[:2], pbar[:2], atol=0.05)
        assert center[2].item() == pytest.approx(expected_z, abs=0.05)

    def test_fit_residual_low_for_clean_paraboloid(self):
        """fit_residual should be very small for noise-free paraboloid."""
        query, neighbors, k_eff = _make_paraboloid_patch(a=0.5, b=0.3, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)

        fq = result["fit_quality"][0, 0]   # [4]
        fit_res = fq[0].item()
        assert fit_res < 0.05, f"fit_residual={fit_res:.4f} should be low for clean paraboloid"

    def test_normal_approximately_z(self):
        """
        For z = a*x^2 + b*y^2 the normal at origin is (0,0,1).
        R_init[:,:,:,2] should be close to (0,0,1) or (0,0,-1).
        """
        query, neighbors, k_eff = _make_paraboloid_patch(a=0.5, b=0.3, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)

        R = result["R_init"][0, 0]   # [3, 3]  columns are e1, e2, e3
        normal = R[:, 2]              # third column = normal
        assert abs(abs(normal[2].item()) - 1.0) < 0.1, (
            f"Normal z-component should be ≈±1, got {normal.tolist()}"
        )

    def test_canonical_ordering_guarantees_s1_ge_s2(self):
        query, neighbors, k_eff = _make_paraboloid_patch(a=0.5, b=0.3, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        s_init = result["s_init"][0, 0]
        assert s_init[0].abs().item() >= s_init[1].abs().item()

    def test_near_isotropic_patch_has_low_anisotropy(self):
        query, neighbors, k_eff = _make_paraboloid_patch(a=0.4, b=0.4, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)

        tangent_aniso = result["tangent_aniso"][0, 0].item()
        curvature_aniso = result["curvature_aniso"][0, 0].item()

        assert tangent_aniso < 0.1, f"tangent_aniso={tangent_aniso:.4f} should be low"
        assert curvature_aniso < 0.1, f"curvature_aniso={curvature_aniso:.4f} should be low"


# ---------------------------------------------------------------------------
# Test 1b: Saddle and sloped plane signed-QGS semantics
# ---------------------------------------------------------------------------

class TestSignedLocalTangent:
    def test_saddle_uses_opposite_s1_s2_signs(self):
        query, neighbors, k_eff = _make_paraboloid_patch(a=0.4, b=-0.3, grid_n=7)
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        s = result["s_init"][0, 0]

        assert result["use_geom_init"][0, 0].item()
        assert s[0].abs().item() >= s[1].abs().item()
        assert s[0].item() * s[1].item() < 0.0, f"saddle signature should be mixed, got {s}"
        assert s[2].item() > 0.0

    def test_sloped_plane_normal_and_linear_free_local_frame(self):
        xs = torch.linspace(-1.0, 1.0, 7)
        ys = torch.linspace(-1.0, 1.0, 7)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        z = 0.2 * xg - 0.1 * yg + 0.3
        pts = torch.stack([xg.reshape(-1), yg.reshape(-1), z.reshape(-1)], dim=-1)
        query = torch.tensor([[[0.0, 0.0, 0.3]]])
        neighbors = pts.unsqueeze(0).unsqueeze(0)
        k_eff = torch.full((1, 1), pts.shape[0], dtype=torch.long)

        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        R = result["R_init"][0, 0]
        center = result["c_init"][0, 0]
        normal = R[:, 2]
        plane_normal = torch.tensor([-0.2, 0.1, 1.0])
        plane_normal = plane_normal / plane_normal.norm()

        assert abs(torch.dot(normal, plane_normal).item()) > 0.99
        assert center[2].item() == pytest.approx(
            0.2 * center[0].item() - 0.1 * center[1].item() + 0.3,
            abs=1e-4,
        )

        local = (neighbors[0, 0] - center) @ R
        assert local[:, 2].abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# Test 2: Flat plane — planarity ≈ 0, s3 ≈ 0
# ---------------------------------------------------------------------------

class TestFlatPlane:
    def _make_flat_patch(self, grid_n=7, device="cpu"):
        """Points on z = 0 plane."""
        xs = torch.linspace(-1.0, 1.0, grid_n, device=device)
        ys = torch.linspace(-1.0, 1.0, grid_n, device=device)
        xg, yg = torch.meshgrid(xs, ys, indexing="ij")
        z = torch.zeros_like(xg)
        pts = torch.stack([xg.reshape(-1), yg.reshape(-1), z.reshape(-1)], dim=-1)
        K = pts.shape[0]
        query = torch.zeros(1, 1, 3, device=device)
        neighbors = pts.unsqueeze(0).unsqueeze(0)
        k_eff = torch.full((1, 1), K, dtype=torch.long, device=device)
        return query, neighbors, k_eff

    def test_planarity_near_zero(self):
        """Planarity λ3/(λ1+λ2+λ3) should be near 0 for flat patch."""
        query, neighbors, k_eff = self._make_flat_patch()
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        planarity = result["fit_quality"][0, 0, 1].item()
        assert planarity < 0.05, f"planarity={planarity:.4f} should be low for flat plane"

    def test_s3_near_zero(self):
        """Near-flat patches should keep a tiny positive curvature magnitude."""
        query, neighbors, k_eff = self._make_flat_patch()
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        s3 = result["s_init"][0, 0, 2].item()
        assert s3 == pytest.approx(1e-4, abs=5e-5)

    def test_center_near_pbar_for_flat_plane(self):
        query, neighbors, k_eff = self._make_flat_patch()
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        center = result["c_init"][0, 0]
        pbar = neighbors[0, 0].mean(dim=0)
        assert torch.allclose(center, pbar, atol=1e-4)

    def test_fit_residual_near_zero(self):
        """fit_residual should be ≈ 0 for perfectly flat plane (all z = 0, var(z) = 0 -> 0/eps)."""
        query, neighbors, k_eff = self._make_flat_patch()
        result = fit_local_quadrics(query, neighbors, k_eff, k_min=4, k_target=49)
        fit_res = result["fit_quality"][0, 0, 0].item()
        # var(z) = 0 -> ratio = 0/eps = 0
        assert fit_res < 1.0, f"fit_residual={fit_res:.4f} should be finite/low for flat plane"


# ---------------------------------------------------------------------------
# Test 3: k_eff < k_min -> use_geom_init = False, worst-case fit_quality
# ---------------------------------------------------------------------------

class TestSubFloorFallback:
    def test_use_geom_init_false(self):
        """Points with k_eff < k_min should have use_geom_init = False."""
        # Create a patch with K=4 valid, but k_min=8
        K = 4
        pts = torch.randn(1, 1, K, 3)
        query = torch.zeros(1, 1, 3)
        k_eff = torch.full((1, 1), K, dtype=torch.long)

        result = fit_local_quadrics(query, pts, k_eff, k_min=8)
        assert not result["use_geom_init"][0, 0].item(), "use_geom_init should be False"

    def test_fallback_fit_quality(self):
        """
        For k_eff < k_min, fit_quality should be injected worst-case:
        [r_max_clip, 1/3, 0, 1].
        """
        K = 4
        r_max_clip = 1.0
        pts = torch.randn(1, 1, K, 3)
        query = torch.zeros(1, 1, 3)
        k_eff = torch.full((1, 1), K, dtype=torch.long)

        result = fit_local_quadrics(query, pts, k_eff, k_min=8, r_max_clip=r_max_clip)
        fq = result["fit_quality"][0, 0]

        assert abs(fq[0].item() - r_max_clip) < 1e-5, f"fq[0]={fq[0].item()}, expected {r_max_clip}"
        assert abs(fq[1].item() - 1.0 / 3.0) < 1e-5, f"fq[1]={fq[1].item()}, expected 1/3"
        assert abs(fq[2].item() - 0.0) < 1e-5,       f"fq[2]={fq[2].item()}, expected 0"
        assert abs(fq[3].item() - 1.0) < 1e-5,       f"fq[3]={fq[3].item()}, expected 1"

    def test_fallback_R_init_is_identity(self):
        """For k_eff < k_min, R_init should be identity."""
        K = 4
        pts = torch.randn(1, 1, K, 3)
        query = torch.zeros(1, 1, 3)
        k_eff = torch.full((1, 1), K, dtype=torch.long)

        result = fit_local_quadrics(query, pts, k_eff, k_min=8)
        R = result["R_init"][0, 0]
        eye = torch.eye(3)
        assert torch.allclose(R, eye, atol=1e-5), f"R_init should be I for fallback, got {R}"

    def test_fallback_c_init_equals_query(self):
        """c_init should equal the query point even in fallback."""
        K = 3
        pts = torch.randn(1, 1, K, 3)
        query = torch.tensor([[[1.0, 2.0, 3.0]]])
        k_eff = torch.full((1, 1), K, dtype=torch.long)

        result = fit_local_quadrics(query, pts, k_eff, k_min=8)
        c = result["c_init"][0, 0]
        assert torch.allclose(c, query[0, 0], atol=1e-5), f"c_init={c}, expected {query[0,0]}"


# ---------------------------------------------------------------------------
# Test 4: Gradient check
# ---------------------------------------------------------------------------

class TestGradients:
    def test_backward_no_nan(self):
        """loss.backward() should run without NaN on random small batch."""
        B, N, K = 2, 8, 16
        torch.manual_seed(42)
        points = torch.randn(B, N, 3, requires_grad=False)
        # neighbours slightly perturbed from query
        neighbors = points.unsqueeze(2).expand(B, N, K, 3).clone()
        neighbors = neighbors + 0.1 * torch.randn(B, N, K, 3)
        neighbors.requires_grad_(False)
        k_eff = torch.full((B, N), K, dtype=torch.long)

        result = fit_local_quadrics(points, neighbors, k_eff, k_min=4, k_target=K)

        # Pick a scalar loss: sum of s_init values (all are downstream of quadric fit)
        s_init = result["s_init"]   # [B, N, 3]
        fq = result["fit_quality"]  # [B, N, 4]

        # We need at least one tensor that requires grad for backward test
        # Re-create with grad-tracking input
        points_g = points.detach().requires_grad_(True)
        neighbors_g = neighbors.detach().requires_grad_(True)

        result_g = fit_local_quadrics(points_g, neighbors_g, k_eff, k_min=4, k_target=K)
        loss = result_g["s_init"].sum() + result_g["fit_quality"].sum()
        loss.backward()

        assert not torch.isnan(loss), f"Loss is NaN: {loss}"
        assert points_g.grad is not None, "points grad should exist"
        assert not torch.isnan(points_g.grad).any(), "points grad contains NaN"
        assert neighbors_g.grad is not None, "neighbors grad should exist"
        assert not torch.isnan(neighbors_g.grad).any(), "neighbors grad contains NaN"

    def test_gradcheck_small(self):
        """
        torch.autograd.gradcheck on a tiny input to verify autograd correctness.
        Uses float64 for numerical stability.
        """
        B, N, K = 1, 2, 10
        torch.manual_seed(7)

        points = torch.randn(B, N, 3, dtype=torch.float64, requires_grad=True)
        neighbors_base = torch.randn(B, N, K, 3, dtype=torch.float64)
        neighbors = neighbors_base.clone().requires_grad_(True)
        k_eff = torch.full((B, N), K, dtype=torch.long)

        def func(pts, nbrs):
            result = fit_local_quadrics(
                pts, nbrs, k_eff, k_min=4, k_target=K, ls_reg=1e-3
            )
            return torch.cat([
                result["s_init"].reshape(-1),
                result["fit_quality"].reshape(-1),
            ])

        passed = torch.autograd.gradcheck(
            func,
            (points, neighbors),
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
            raise_exception=True,
        )
        assert passed, "gradcheck failed"


# ---------------------------------------------------------------------------
# Test 5: knn_radius module
# ---------------------------------------------------------------------------

class TestKnnRadius:
    def test_basic_knn(self):
        """Verify that k-NN finds the correct nearest neighbours for simple input."""
        # 5 candidates at known positions
        candidates = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 0.0],   # far
            [6.0, 0.0, 0.0],   # far
        ])
        # Query: origin
        points = torch.tensor([[0.0, 0.0, 0.0]])

        result = hybrid_radius_knn(
            points, candidates,
            k_target=3,
            r_max_fn=lambda p: torch.full((p.shape[0],), 1.0),
            k_min=1,
        )
        idx = result["idx"][0]   # [3]
        k_eff = result["k_eff"][0].item()

        assert k_eff == 3, f"Expected 3 neighbors within radius 1.0, got {k_eff}"
        # Nearest 3 should be indices 0, 1, 2 (within r=1.0)
        valid_idx = idx[result["mask"][0]]
        assert set(valid_idx.tolist()) == {0, 1, 2}, f"Unexpected indices: {valid_idx.tolist()}"

    def test_radius_exclusion(self):
        """Points outside radius should be excluded."""
        candidates = torch.tensor([
            [0.1, 0.0, 0.0],
            [3.0, 0.0, 0.0],   # outside r=0.5
        ])
        points = torch.tensor([[0.0, 0.0, 0.0]])

        result = hybrid_radius_knn(
            points, candidates,
            k_target=4,
            r_max_fn=lambda p: torch.full((p.shape[0],), 0.5),
        )
        k_eff = result["k_eff"][0].item()
        assert k_eff == 1, f"Expected 1 neighbor, got {k_eff}"

    def test_voxel_local_matches_bruteforce_knn(self):
        """Voxel-local search should preserve exact radius k-NN results."""
        gen = torch.Generator().manual_seed(7)
        candidates = torch.rand((80, 3), generator=gen) * 6.0 - 3.0
        points = torch.rand((12, 3), generator=gen) * 4.0 - 2.0

        def r_max_fn(p):
            return torch.full((p.shape[0],), 1.25)

        brute = hybrid_radius_knn(
            points,
            candidates,
            k_target=5,
            r_max_fn=r_max_fn,
            method="bruteforce",
        )
        local = hybrid_radius_knn(
            points,
            candidates,
            k_target=5,
            r_max_fn=r_max_fn,
            method="voxel_local",
            voxel_size=0.75,
        )

        assert torch.equal(local["k_eff"], brute["k_eff"])
        assert torch.equal(local["mask"], brute["mask"])
        assert torch.allclose(local["dist"], brute["dist"], atol=1e-6)

    def test_voxel_chunk_matches_bruteforce_knn(self):
        """Chunked voxel search should preserve exact radius k-NN results."""
        gen = torch.Generator().manual_seed(11)
        candidates = torch.rand((96, 3), generator=gen) * 8.0 - 4.0
        points = torch.rand((16, 3), generator=gen) * 5.0 - 2.5

        def r_max_fn(p):
            return torch.full((p.shape[0],), 1.5)

        brute = hybrid_radius_knn(
            points,
            candidates,
            k_target=6,
            r_max_fn=r_max_fn,
            method="bruteforce",
        )
        chunked = hybrid_radius_knn(
            points,
            candidates,
            k_target=6,
            r_max_fn=r_max_fn,
            method="voxel_chunk",
            voxel_size=0.8,
            chunk_size=4,
        )

        assert torch.equal(chunked["k_eff"], brute["k_eff"])
        assert torch.equal(chunked["mask"], brute["mask"])
        assert torch.allclose(chunked["dist"], brute["dist"], atol=1e-6)

    def test_voxel_methods_use_points_inside_neighbor_voxels_not_voxel_centres(self):
        """Adjacent voxel points close to the query must remain valid candidates."""
        points = torch.tensor([[0.99, 0.0, 0.0]])
        candidates = torch.tensor([
            [1.01, 0.0, 0.0],  # adjacent voxel but raw point is very close
            [1.20, 0.0, 0.0],
        ])

        for method in ("voxel_local", "voxel_chunk"):
            result = hybrid_radius_knn(
                points,
                candidates,
                k_target=2,
                r_max_fn=lambda p: torch.full((p.shape[0],), 0.05),
                method=method,
                voxel_size=1.0,
                chunk_size=1,
            )

            assert result["k_eff"].tolist() == [1]
            assert result["idx"][0, 0].item() == 0

    def test_gather_neighbors(self):
        """gather_neighbors should fill padded slots with zeros."""
        candidates = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        idx = torch.tensor([[0, -1]])   # 1 valid, 1 padded
        mask = torch.tensor([[True, False]])

        gathered = gather_neighbors(candidates, idx, mask)
        assert gathered.shape == (1, 2, 3)
        assert torch.allclose(gathered[0, 0], candidates[0]), "Valid slot mismatch"
        assert torch.allclose(gathered[0, 1], torch.zeros(3)), "Padded slot should be zero"

    def test_default_r_max_range(self):
        """default_r_max should respect [0.3, 2.0] clamp."""
        pts = torch.tensor([
            [0.0, 0.0, 0.0],   # ||p|| = 0 -> r = 0.3
            [100.0, 0.0, 0.0], # ||p|| = 100 -> r = clip(1.3, 0.3, 2.0) = 1.3... actually 2.0
        ])
        r = default_r_max(pts)
        assert r[0].item() == pytest.approx(0.3, abs=1e-5)
        # 0.3 + 0.01 * 100 = 1.3 < 2.0
        assert r[1].item() == pytest.approx(1.3, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 6: Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_output_shapes(self):
        """All output tensors should have correct shapes."""
        B, N, K = 2, 10, 16
        points = torch.randn(B, N, 3)
        neighbors = torch.randn(B, N, K, 3)
        k_eff = torch.full((B, N), K, dtype=torch.long)

        result = fit_local_quadrics(points, neighbors, k_eff, k_min=8)

        assert result["c_init"].shape == (B, N, 3),       f"c_init shape: {result['c_init'].shape}"
        assert result["R_init"].shape == (B, N, 3, 3),    f"R_init shape: {result['R_init'].shape}"
        assert result["s_init"].shape == (B, N, 3),       f"s_init shape: {result['s_init'].shape}"
        assert result["fit_quality"].shape == (B, N, 4),  f"fit_quality shape: {result['fit_quality'].shape}"
        assert result["use_geom_init"].shape == (B, N),   f"use_geom_init shape: {result['use_geom_init'].shape}"
        assert result["kappa1_init"].shape == (B, N),     f"kappa1_init shape: {result['kappa1_init'].shape}"
        assert result["kappa2_init"].shape == (B, N),     f"kappa2_init shape: {result['kappa2_init'].shape}"
        assert result["tangent_aniso"].shape == (B, N),   f"tangent_aniso shape: {result['tangent_aniso'].shape}"
        assert result["curvature_aniso"].shape == (B, N), f"curvature_aniso shape: {result['curvature_aniso'].shape}"

    def test_r_init_is_rotation(self):
        """R_init columns should be orthonormal (det ≈ ±1)."""
        B, N, K = 1, 5, 20
        points = torch.randn(B, N, 3)
        neighbors = (
            points.unsqueeze(2).expand(B, N, K, 3).clone()
            + 0.5 * torch.randn(B, N, K, 3)
        )
        k_eff = torch.full((B, N), K, dtype=torch.long)

        result = fit_local_quadrics(points, neighbors, k_eff, k_min=4)

        R = result["R_init"]   # [B, N, 3, 3]
        use_geom = result["use_geom_init"]

        for b in range(B):
            for n in range(N):
                if not use_geom[b, n]:
                    continue
                Rbn = R[b, n]
                # Columns orthonormal: R^T R ≈ I
                RtR = Rbn.T @ Rbn
                eye = torch.eye(3, dtype=Rbn.dtype)
                assert torch.allclose(RtR, eye, atol=1e-4), (
                    f"R[{b},{n}] not orthonormal: R^T R = {RtR}"
                )
                # Determinant ≈ +1 for a proper right-handed frame
                det = torch.linalg.det(Rbn).item()
                assert abs(det - 1.0) < 1e-4, f"det(R[{b},{n}]) = {det}"
