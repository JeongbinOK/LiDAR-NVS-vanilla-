"""Full neural clustering model: A -> B -> C -> D."""

import torch
import torch.nn as nn

from geometry import _knn_points
from nn.backbone import PointFeatureBackbone
from nn.seed_gen import SeedGenerator
from nn.soft_assign import SoftAssignment
from nn.gaussian_head import GaussianParameterHead


class NeuralClusteringModel(nn.Module):
    """Neural 2D Gaussian clustering in a single forward pass.

    Pipeline: Single Frame (N x 4) -> [A] Feature Backbone -> [B] Seed Generation
              -> [C] Soft Assignment -> [D] Gaussian Prediction -> Surfels + Assignments
    """

    def __init__(self, cfg):
        super().__init__()
        D = cfg.feature_dim

        self.backbone = PointFeatureBackbone(
            dim=D, num_blocks=cfg.num_blocks,
            window_size=cfg.window_size, num_heads=cfg.num_heads,
        )
        self.seed_gen = SeedGenerator(
            dim=D, k_max=cfg.k_max,
            nms_radius_factor=cfg.nms_radius_factor,
            target_cluster_size=cfg.target_cluster_size,
        )
        self.soft_assign = SoftAssignment(
            dim=D, top_k=cfg.top_k_seeds,
            lambda_normal=cfg.geo_lambda_normal,
        )
        self.gaussian_head = GaussianParameterHead(dim=D)

        self.knn_k = cfg.knn_k

    def forward(self, xyz: torch.Tensor, intensity: torch.Tensor,
                tau: float = 1.0) -> dict:
        """
        Args:
            xyz: [N, 3] point positions (ego-masked)
            intensity: [N] per-point intensity
            tau: Gumbel temperature (annealed 1.0 -> 0.1)

        Returns:
            dict with gaussians, assignments, seed info, normals
        """
        # PCA normals for geometric bias in Module C (non-learned, detached)
        normals = self._estimate_normals(xyz)

        # A: Feature backbone
        features = self.backbone(xyz, intensity)

        # B: Seed generation
        seed_indices, seed_scores, all_scores = self.seed_gen(features, xyz, tau)

        # C: Soft assignment
        assign_indices, assign_weights = self.soft_assign(
            features, xyz, normals, seed_indices, tau,
        )

        # D: Gaussian parameter prediction
        gaussians = self.gaussian_head(
            features, xyz, assign_indices, assign_weights, seed_indices,
        )

        return {
            "gaussians": gaussians,
            "assign_indices": assign_indices,
            "assign_weights": assign_weights,
            "seed_scores": all_scores,
            "seed_indices": seed_indices,
            "normals": normals,
        }

    def _estimate_normals(self, xyz: torch.Tensor) -> torch.Tensor:
        """Per-point normals via local PCA (non-learned, for geometric bias)."""
        K = self.knn_k

        _, indices = _knn_points(xyz.unsqueeze(0), xyz.unsqueeze(0), k=K)
        indices = indices.squeeze(0)

        neighbors = xyz[indices]
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        cov = torch.bmm(centered.transpose(1, 2), centered) / (K - 1)

        _, _, Vh = torch.linalg.svd(cov)
        normals = Vh[:, 2, :]

        flip = (normals * (-xyz)).sum(dim=1) < 0
        normals[flip] *= -1

        return normals.detach()
