from __future__ import annotations

import json
import math
import unittest
import tempfile
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.dataloader.nuscene import relative_time_coordinates
from src.eval.gaussian_viz import (
    SOURCE_FRAME_COLORS,
    _source_frame_velocity_layer,
    _velocity_arrow_trace,
    save_sequence_html,
    velocity_transport_from_output,
)
from src.models_new.module.dynamic_gaussian import (
    AttentionInitializedVelocityGaussianBackend,
    ConsensusAttentionMotionMatcher,
    ConsensusAttentionVelocityGaussianBackend,
    EmbeddedInitDurationVelocityHead,
    EmbeddedInitVelocityHead,
    FeatureOnlyVelocityOffsetHead,
    GaussianAttributeHead,
    InitConditionedVelocityHead,
    LayerWeightedAttentionVelocityGaussianBackend,
    LayerWeightedDistanceBiasCrossAttention,
    MaxSpeedBarrierLayerWeightedCrossAttention,
    MaxSpeedBarrierVelocityGaussianBackend,
    GroupedHeadRMSNorm,
    GroupedGainBarrierCrossAttention,
    SingleGaussianBarrierVelocityGaussianBackend,
    PhysicalVelocityGaussianBackend,
    PostAttentionProposalVelocityGaussianBackend,
    ProjectedDenseMotionProposal,
    ProposalInitializedVelocityGaussianBackend,
    SparseMotionProposal,
    StraightThroughProposalVelocityGaussianBackend,
    StraightThroughTop4MotionProposal,
    TimeConditionedParallelCrossAttention,
    WarpedProposalVelocityGaussianBackend,
)
from src.models_new.module.feature_fusion import build_feature_fusion
from src.models_new.module.m3_g2p import DynamicGausRender, GausRender
from src.models_new.module.token_refiner import SparseLocalTokenRefiner
from src.models_new.utils.attention import Rotary3D
from src.models_new.utils.loss import Loss
from src.models_new.utonia.model import Point3DRoPE
from src.model_wrapper import ModelWrapper


def temporal_cfg(dim=24, heads=4, layers=2, motion_head_count=0):
    values = {
        "implementation": "auto",
        "num_heads": heads,
        "layers": layers,
        "mlp_ratio": 2,
        "time_embedding_dim": 16,
        "time_frequencies": 4,
        "time_reference_sec": 1.0,
        "rope_base": 10.0,
        "rope_position_scale": 0.2,
        "layer_scale_init": 0.1,
    }
    if motion_head_count:
        values["motion_head_count"] = motion_head_count
    return OmegaConf.create(values)


def v5_temporal_cfg(dim=24, heads=4, layers=2, motion_head_count=2,
                    qk_norm=True, motion_rope=True):
    cfg = temporal_cfg(dim=dim, heads=heads, layers=layers,
                       motion_head_count=motion_head_count)
    if qk_norm:
        cfg.qk_norm = True
    if motion_rope:
        cfg.motion_rope_base = 200.0
        cfg.motion_rope_position_scale = 4.0
    return cfg


def v6_temporal_cfg(dim=24, heads=4, layers=2):
    values = OmegaConf.to_container(
        temporal_cfg(dim=dim, heads=heads, layers=layers), resolve=True
    )
    values["time_hidden_dim"] = values.pop("time_embedding_dim")
    return OmegaConf.create(values)


def v7_temporal_cfg(dim=24, heads=4, layers=2):
    del dim
    return OmegaConf.create({
        "implementation": "auto",
        "num_heads": heads,
        "layers": layers,
        "mlp_ratio": 2,
        "use_time_embedding": False,
        "rope_base": 100.0,
        "rope_position_scale": 2.0 * torch.pi / 5.0,
        "layer_scale_init": 0.1,
    })


def _gs_params():
    return OmegaConf.create({
        "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
    })


def backend_cfg(dim=24):
    return OmegaConf.create({
        "temporal": OmegaConf.to_container(temporal_cfg(dim=dim), resolve=True),
        "gaussian_head": {
            "initial_opacity": 0.2,
            "initial_scale_m": 0.3,
        },
        "motion": {
            "zero_init": True,
        },
    })


def attention_backend_cfg(dim=24, heads=4, motion_head_count=4):
    cfg = backend_cfg(dim=dim)
    cfg.temporal.num_heads = heads
    cfg.temporal.motion_head_count = motion_head_count
    return cfg


def proposal_cfg(**overrides):
    values = {
        "adapter_hidden_dim": 4,
        "descriptor_dim": 8,
        "candidate_count": 4,
        "match_count": 2,
        "temperature": 0.1,
        "score_chunk_size": 2,
        "dustbin_similarity_init": 0.70,
        "search_speed_min_mps": 1.0,
        "search_speed_init_mps": 5.0,
        "search_speed_max_mps": 20.0,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def v7_proposal_cfg(**overrides):
    values = {
        "descriptor_mode": "direct_l2",
        "dustbin_mode": "evidence_mlp",
        "dustbin_hidden_dim": 16,
        "dustbin_prior_probability": 0.05,
        "unmatched_gate_mode": "ste_hard",
        "unmatched_hard_threshold": 0.9,
        "candidate_count": 4,
        "match_count": 2,
        "temperature": 0.1,
        "score_chunk_size": 2,
        "search_speed_min_mps": 1.0,
        "search_speed_init_mps": 5.0,
        "search_speed_max_mps": 20.0,
    }
    values.update(overrides)
    return OmegaConf.create(values)


def v7_1_proposal_cfg(**overrides):
    values = OmegaConf.to_container(v7_proposal_cfg(), resolve=True)
    values.update({
        "dustbin_evidence_mode": "mean_spread_reciprocal_m4",
        "mean_displacement_scale_m": 4.0,
        "spread_scale_m": 0.5,
        "unmatched_hard_threshold": 0.5,
        "candidate_count": 4,
        "match_count": 4,
    })
    values.update(overrides)
    return OmegaConf.create(values)


def proposal_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v6_temporal_cfg(dim=dim)
    cfg.motion_proposal = proposal_cfg()
    cfg.motion.init_conditioned_residual = True
    cfg.motion.init_condition_scale_mps = 10.0
    cfg.motion.detach_init_condition = True
    return cfg


def v7_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v7_temporal_cfg(dim=dim)
    cfg.motion_proposal = v7_proposal_cfg()
    cfg.motion.detach_init_condition = True
    cfg.motion.velocity_embedding_dim = 32
    cfg.motion.duration_embedding_dim = 16
    cfg.motion.duration_frequencies = 4
    cfg.motion.duration_reference_sec = 1.0
    cfg.motion.residual_hidden_dim = 8
    return cfg


def v7_2_proposal_cfg(**overrides):
    values = {
        "descriptor_mode": "direct_l2",
        "match_count": 4,
        "temperature": 0.1,
        "score_chunk_size": 2,
        "distance_prior_speed_mps": 30.0,
        "ste_surrogate": "dense_softmax",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def v7_2_backend_cfg(dim=24):
    cfg = v7_backend_cfg(dim=dim)
    cfg.temporal.use_time_embedding = True
    cfg.temporal.time_embedding_dim = 8
    cfg.temporal.time_frequencies = 4
    cfg.temporal.time_reference_sec = 1.0
    cfg.motion_proposal = v7_2_proposal_cfg()
    cfg.motion = OmegaConf.create({
        "zero_init": True,
        "residual_hidden_dim": 8,
    })
    return cfg


def v9_proposal_cfg(**overrides):
    values = {
        "descriptor_mode": "projected_l2",
        "descriptor_dim": 6,
        "temperature": 0.1,
        "score_chunk_size": 2,
        "distance_prior_speed_mps": 5.0,
        "readout": "dense_expectation",
    }
    values.update(overrides)
    return OmegaConf.create(values)


def v9_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal.use_time_embedding = True
    cfg.temporal.rope_base = 100.0
    cfg.temporal.rope_position_scale = 2.0 * torch.pi / 5.0
    cfg.motion_proposal = v9_proposal_cfg(descriptor_dim=6)
    cfg.motion = OmegaConf.create({
        "zero_init": True,
        "residual_hidden_dim": 8,
    })
    return cfg


def v10_temporal_cfg(dim=24, heads=4, layers=3):
    del dim
    return OmegaConf.create({
        "implementation": "auto",
        "layers": layers,
        "num_heads": heads,
        "mlp_ratio": 2,
        "use_time_embedding": True,
        "time_embedding_dim": 8,
        "time_frequencies": 4,
        "time_reference_sec": 1.0,
        "position_encoding": "distance_bias",
        "distance_bias_speed_mps": 30.0,
        "qk_norm": True,
        "layer_weight_hidden_dim": 8,
        "layer_scale_init": 0.1,
    })


def v10_gaussian_count_cfg():
    return OmegaConf.create({
        "count_mode": "learned_gumbel",
        "grad_balance": "sqrt_k",
        "learned_count": {
            "K_max": 3,
            "tau": 1.0,
            "seed_mode": "range_quantile",
            "grad_balance_scope": "token",
            "budget": {"enable": False},
        },
    })


def v10_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v10_temporal_cfg(dim=dim)
    cfg.motion = OmegaConf.create({
        "zero_init": True,
        "residual_hidden_dim": 8,
    })
    return cfg


def v11_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    del dim
    return OmegaConf.create({
        "implementation": "auto",
        "layers": layers,
        "num_heads": heads,
        "mlp_ratio": 2,
        "use_time_embedding": True,
        "time_embedding_dim": 8,
        "time_frequencies": 4,
        "time_reference_sec": 1.0,
        "position_encoding": "barrier_rope_split",
        "barrier_speed_mps": 30.0,
        "barrier_weight": 4.0,
        "match_chunk_size": match_chunk_size,
        "rope_base": 100.0,
        "rope_position_scale": 2.0 * math.pi / 5.0,
        "qk_norm": True,
        "layer_scale_init": 0.1,
    })


def v11_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v11_temporal_cfg(dim=dim)
    cfg.motion = OmegaConf.create({
        "zero_init": True,
        "residual_hidden_dim": 8,
    })
    return cfg


def v11_1_temporal_cfg(dim=24, heads=4, layers=3, match_chunk_size=2):
    # V11.1's temporal config is V11's; only the QK gain's shape differs, and
    # that is a module decision rather than a configured one.
    return v11_temporal_cfg(
        dim=dim, heads=heads, layers=layers,
        match_chunk_size=match_chunk_size,
    )


def v11_1_backend_cfg(dim=24):
    cfg = v11_backend_cfg(dim=dim)
    cfg.temporal = v11_1_temporal_cfg(dim=dim)
    cfg.motion.velocity_embedding_dim = 8
    cfg.motion.detach_init_condition = True
    return cfg


def v8_backend_cfg(dim=24):
    cfg = attention_backend_cfg(
        dim=dim, heads=4, motion_head_count=4
    )
    cfg.temporal.layers = 2
    cfg.temporal.use_time_embedding = True
    cfg.temporal.qk_norm = True
    cfg.temporal.tie_motion_qk_init = True
    cfg.temporal.rope_base = 100.0
    cfg.temporal.rope_position_scale = 2.0 * torch.pi / 5.0
    cfg.motion_matching = OmegaConf.create({
        "candidate_count": 5,
        "match_count": 4,
        "score_chunk_size": 2,
    })
    cfg.motion.detach_init_condition = True
    cfg.motion.velocity_embedding_dim = 8
    cfg.motion.residual_hidden_dim = 8
    return cfg


class DynamicGaussianTest(unittest.TestCase):
    def test_training_loss_skips_disabled_metrics_and_chamfer(self):
        loss = Loss(OmegaConf.create({
            "w_chamfer": 0.0,
            "w_depth": 0.5,
            "w_depth_median": 0.5,
            "w_intensity": 0.2,
            "w_raydrop": 0.5,
            "w_scale": 0.0,
            "enable_lpips": False,
        }))
        maps = {
            "depth": torch.ones(1, 1, 2, 2),
            "depth_median": torch.ones(1, 1, 2, 2),
            "intensity_sh": torch.full((1, 1, 2, 2), 0.5),
            "raydrop": torch.full((1, 1, 2, 2), 0.25),
            "gt_depth": torch.ones(1, 1, 2, 2),
            "gt_intensity_sh": torch.full((1, 1, 2, 2), 0.5),
            "gt_raydrop": torch.zeros(1, 1, 2, 2),
            "render_points": [],
            "gt_points": [],
        }
        output = loss(
            maps,
            metric_mode="train",
            compute_valid_metrics=False,
            compute_raydrop_metrics=False,
        )
        self.assertEqual(set(output), {
            "loss_depth", "loss_depth_median", "loss_intensity",
            "loss_raydrop", "loss_scale", "total",
        })

    @staticmethod
    def _apply_velocity_l2(
        velocities, *, offsets=None, prefix="train", weight=0.005, step=0,
        mode="final_l2",
    ):
        wrapper = SimpleNamespace(
            _velocity_l2_cfg=OmegaConf.create({
                "enabled": True,
                "weight": weight,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }),
            _velocity_l2_mode=mode,
            global_step=step,
            _active_motion_items=ModelWrapper._active_motion_items,
        )
        losses = {"total": torch.zeros(())}
        items = [{"velocity": item} for item in velocities]
        if offsets is not None:
            if len(offsets) != len(items):
                raise ValueError("offsets must align with velocities")
            for item, offset in zip(items, offsets):
                item["velocity_offset"] = offset
        ModelWrapper._add_velocity_l2_prior(
            wrapper,
            losses,
            items,
            prefix=prefix,
        )
        return losses

    @staticmethod
    def _apply_split_velocity_l2(
        inits, offsets, *, prefix="train", init_weight=0.01,
        offset_weight=0.01, step=0,
    ):
        wrapper = SimpleNamespace(
            _velocity_l2_cfg=OmegaConf.create({
                "enabled": True,
                "mode": "split_group_l2",
                "init_weight": init_weight,
                "offset_weight": offset_weight,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }),
            _velocity_l2_mode="split_group_l2",
            global_step=step,
            _active_motion_items=ModelWrapper._active_motion_items,
        )
        wrapper._add_split_velocity_l2_prior = MethodType(
            ModelWrapper._add_split_velocity_l2_prior, wrapper
        )
        losses = {"total": torch.zeros(())}
        items = [
            {
                "velocity": init + offset,
                "velocity_init": init,
                "velocity_offset": offset,
            }
            for init, offset in zip(inits, offsets)
        ]
        ModelWrapper._add_velocity_l2_prior(
            wrapper, losses, items, prefix=prefix
        )
        return losses

    def test_velocity_l2_averages_over_components_like_storm(self):
        # STORM's prior is mse_loss(v, 0).mean() = ||v||^2 / 3, not ||v||^2.
        # The reference weight 0.005 is calibrated to that normalization.
        losses = self._apply_velocity_l2([torch.tensor([[3.0, 4.0, 0.0]])])
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]), 25.0 / 3.0, places=5
        )
        self.assertAlmostEqual(
            float(losses["total"]), 0.005 * 25.0 / 3.0, places=7
        )

    def test_velocity_l2_pools_every_gaussian_in_the_batch(self):
        # A batch mean, not a mean of per-sample means: sparse fast motion has
        # to stay cheap while a scene-wide escape is expensive.
        losses = self._apply_velocity_l2([
            torch.zeros(9, 3),
            torch.full((1, 3), 2.0),
        ])
        self.assertAlmostEqual(float(losses["loss_velocity_l2"]), 0.4, places=5)

    def test_velocity_l2_reports_but_does_not_optimize_validation(self):
        velocities = [torch.tensor([[3.0, 4.0, 0.0]])]
        losses = self._apply_velocity_l2(velocities, prefix="val")
        self.assertIn("loss_velocity_l2", losses)
        self.assertNotIn("wc_velocity_l2", losses)
        self.assertEqual(float(losses["total"]), 0.0)

    def test_velocity_l2_backpropagates_into_velocity(self):
        velocity = torch.tensor([[6.0, 0.0, 0.0]], requires_grad=True)
        losses = self._apply_velocity_l2([velocity])
        losses["total"].backward()
        # d/dv [w * v^2/3] = 2*w*v/3 for the single non-zero component.
        self.assertAlmostEqual(
            float(velocity.grad[0, 0]), 2.0 * 0.005 * 6.0 / 3.0, places=7
        )

    def test_velocity_group_l2_is_three_times_the_component_mean(self):
        # final_l2 is mean(||v||^2)/3; final_group_l2 keeps the same numerator
        # over one count per Gaussian, so at an equal weight it is exactly 3x.
        velocity = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        losses = self._apply_velocity_l2(
            [velocity], weight=0.05, mode="final_group_l2"
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]), 25.0, places=5
        )
        component = self._apply_velocity_l2(
            [torch.tensor([[3.0, 4.0, 0.0]])], weight=0.05, mode="final_l2"
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]),
            3.0 * float(component["loss_velocity_l2"]),
            places=5,
        )
        losses["total"].backward()
        # d/dv [w * ||v||^2] = 2*w*v, with no 1/3 component averaging.
        torch.testing.assert_close(
            velocity.grad, torch.tensor([[0.3, 0.4, 0.0]])
        )

    def test_velocity_group_l2_pools_gaussians_not_components(self):
        losses = self._apply_velocity_l2(
            [torch.zeros(9, 3), torch.full((1, 3), 2.0)],
            mode="final_group_l2",
        )
        # 12 / 10 Gaussians, where final_l2 would divide by 30 components.
        self.assertAlmostEqual(float(losses["loss_velocity_l2"]), 1.2, places=5)

    def test_velocity_group_l1_uses_vector_norm_and_constant_slope(self):
        velocity = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        losses = self._apply_velocity_l2(
            [velocity], weight=0.05, mode="final_group_l1"
        )
        self.assertAlmostEqual(float(losses["loss_velocity_l2"]), 5.0, places=6)
        losses["total"].backward()
        self.assertTrue(torch.allclose(
            velocity.grad,
            torch.tensor([[0.03, 0.04, 0.0]]),
            atol=1.0e-7,
        ))

    def test_offset_l2_regularizes_only_residual_components(self):
        final_velocity = torch.tensor(
            [[30.0, 0.0, 0.0]], requires_grad=True
        )
        velocity_offset = torch.tensor(
            [[3.0, 4.0, 0.0]], requires_grad=True
        )
        losses = self._apply_velocity_l2(
            [final_velocity],
            offsets=[velocity_offset],
            weight=0.05,
            mode="offset_l2",
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]), 25.0 / 3.0, places=6
        )
        losses["total"].backward()
        self.assertIsNone(final_velocity.grad)
        self.assertTrue(torch.allclose(
            velocity_offset.grad,
            torch.tensor([[0.1, 0.13333334, 0.0]]),
            atol=1.0e-7,
        ))

    def test_offset_l2_requires_velocity_offset_field(self):
        with self.assertRaisesRegex(ValueError, "requires every active"):
            self._apply_velocity_l2(
                [torch.zeros(1, 3)], mode="offset_l2"
            )

    def test_offset_group_l2_is_the_squared_norm_of_the_residual_only(self):
        final_velocity = torch.tensor([[30.0, 0.0, 0.0]], requires_grad=True)
        velocity_offset = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        losses = self._apply_velocity_l2(
            [final_velocity],
            offsets=[velocity_offset],
            weight=0.01,
            mode="offset_group_l2",
        )
        # mean(||v_offset||^2), i.e. exactly 3x offset_l2 at the same weight.
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]), 25.0, places=5
        )
        component = self._apply_velocity_l2(
            [torch.tensor([[30.0, 0.0, 0.0]])],
            offsets=[torch.tensor([[3.0, 4.0, 0.0]])],
            weight=0.01,
            mode="offset_l2",
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]),
            3.0 * float(component["loss_velocity_l2"]),
            places=5,
        )
        losses["total"].backward()
        # v_init reaches the prior through no path at all.
        self.assertIsNone(final_velocity.grad)
        torch.testing.assert_close(
            velocity_offset.grad, torch.tensor([[0.06, 0.08, 0.0]])
        )

    def test_offset_group_l2_pools_gaussians_not_components(self):
        losses = self._apply_velocity_l2(
            [torch.zeros(9, 3), torch.zeros(1, 3)],
            offsets=[torch.zeros(9, 3), torch.full((1, 3), 2.0)],
            mode="offset_group_l2",
        )
        self.assertAlmostEqual(float(losses["loss_velocity_l2"]), 1.2, places=5)

    def test_split_group_l2_charges_init_and_offset_independently(self):
        velocity_init = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        velocity_offset = torch.tensor([[0.0, 0.0, 2.0]], requires_grad=True)
        losses = self._apply_split_velocity_l2(
            [velocity_init], [velocity_offset],
            init_weight=0.01, offset_weight=0.02,
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2_init"]), 25.0, places=5
        )
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2_offset"]), 4.0, places=5
        )
        # loss_velocity_l2 stays raw and unweighted, as in every other mode;
        # only `total` sees init_weight/offset_weight.
        self.assertAlmostEqual(
            float(losses["loss_velocity_l2"]), 25.0 + 4.0, places=5
        )
        self.assertAlmostEqual(
            float(losses["total"]), 0.01 * 25.0 + 0.02 * 4.0, places=6
        )
        losses["total"].backward()
        torch.testing.assert_close(
            velocity_init.grad, torch.tensor([[0.06, 0.08, 0.0]])
        )
        torch.testing.assert_close(
            velocity_offset.grad, torch.tensor([[0.0, 0.0, 0.08]])
        )

    def test_split_group_l2_has_no_cancellation_cross_term(self):
        """An offset that merely negates v_init must not be free."""
        opposed = torch.tensor([[-3.0, -4.0, 0.0]])
        aligned = torch.tensor([[3.0, 4.0, 0.0]])
        init = torch.tensor([[3.0, 4.0, 0.0]])
        split_opposed = self._apply_split_velocity_l2(
            [init], [opposed], init_weight=0.01, offset_weight=0.01
        )
        split_aligned = self._apply_split_velocity_l2(
            [init], [aligned], init_weight=0.01, offset_weight=0.01
        )
        # Both offsets have the same magnitude, so the split prior is blind to
        # the sign; only the render loss decides direction.
        self.assertAlmostEqual(
            float(split_opposed["total"]), float(split_aligned["total"]),
            places=6,
        )
        # final_group_l2 on the sum instead rewards the cancelling offset.
        total_opposed = self._apply_velocity_l2(
            [init + opposed], weight=0.01, mode="final_group_l2"
        )
        total_aligned = self._apply_velocity_l2(
            [init + aligned], weight=0.01, mode="final_group_l2"
        )
        self.assertLess(
            float(total_opposed["loss_velocity_l2"]),
            float(total_aligned["loss_velocity_l2"]),
        )
        self.assertAlmostEqual(
            float(total_opposed["loss_velocity_l2"]), 0.0, places=6
        )

    def test_split_group_l2_reports_but_does_not_optimize_validation(self):
        losses = self._apply_split_velocity_l2(
            [torch.tensor([[3.0, 4.0, 0.0]])],
            [torch.tensor([[1.0, 0.0, 0.0]])],
            prefix="val",
        )
        self.assertIn("loss_velocity_l2_init", losses)
        self.assertIn("loss_velocity_l2_offset", losses)
        self.assertNotIn("wc_velocity_l2", losses)
        self.assertEqual(float(losses["total"]), 0.0)

    def test_split_group_l2_requires_both_velocity_fields(self):
        wrapper = SimpleNamespace(
            _velocity_l2_cfg=OmegaConf.create({
                "enabled": True, "mode": "split_group_l2",
                "init_weight": 0.01, "offset_weight": 0.01,
                "warmup_steps": 0, "ramp_steps": 0,
            }),
            _velocity_l2_mode="split_group_l2",
            global_step=0,
            _active_motion_items=ModelWrapper._active_motion_items,
        )
        wrapper._add_split_velocity_l2_prior = MethodType(
            ModelWrapper._add_split_velocity_l2_prior, wrapper
        )
        losses = {"total": torch.zeros(())}
        with self.assertRaisesRegex(ValueError, "velocity_init"):
            ModelWrapper._add_velocity_l2_prior(
                wrapper,
                losses,
                [{"velocity": torch.zeros(1, 3)}],
                prefix="train",
            )

    def test_offset_group_l2_requires_velocity_offset_field(self):
        with self.assertRaisesRegex(ValueError, "requires every active"):
            self._apply_velocity_l2(
                [torch.zeros(1, 3)], mode="offset_group_l2"
            )

    def test_irregular_pair_keeps_normalized_and_physical_time(self):
        normalized, seconds, duration = relative_time_coordinates(
            [1_000_000, 1_400_000, 1_870_000],
            1_000_000,
            1_870_000,
        )
        self.assertTrue(torch.allclose(seconds, torch.tensor([0.0, 0.4, 0.87])))
        self.assertAlmostEqual(float(duration), 0.87, places=6)
        self.assertTrue(torch.allclose(
            normalized,
            torch.tensor([0.0, 0.4 / 0.87, 1.0]),
            atol=1.0e-6,
        ))

    def test_normalized_time_is_bit_exact_with_legacy_formula(self):
        timestamps_us = [1_000_003, 1_317_219, 1_998_641]
        normalized, _seconds, _duration = relative_time_coordinates(
            timestamps_us,
            timestamps_us[0],
            timestamps_us[-1],
        )
        span_us = timestamps_us[-1] - timestamps_us[0]
        legacy = torch.tensor(
            [(timestamp - timestamps_us[0]) / span_us for timestamp in timestamps_us],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(normalized, legacy))

    def test_parallel_cross_attention_is_frame_swap_equivariant(self):
        torch.manual_seed(7)
        module = TimeConditionedParallelCrossAttention(
            temporal_cfg(), dim=24
        ).eval()
        self.assertEqual(module.time_encoder.out_dim, 16)
        self.assertEqual(module.time_to_feature.in_features, 16)
        self.assertEqual(module.time_to_feature.out_features, 24)
        feat0 = torch.randn(3, 24)
        feat1 = torch.randn(2, 24)
        pos0 = torch.randn(3, 3)
        pos1 = torch.randn(2, 3)
        out = module(
            torch.cat([feat0, feat1]),
            torch.cat([pos0, pos1]),
            torch.tensor([3, 5]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.0, 0.0, 0.87, 0.87]),
        )
        swapped = module(
            torch.cat([feat1, feat0]),
            torch.cat([pos1, pos0]),
            torch.tensor([2, 5]),
            torch.tensor([0, 0]),
            torch.tensor([0.87, 0.87, 0.0, 0.0, 0.0]),
        )
        restored = torch.cat([swapped[2:], swapped[:2]])
        self.assertTrue(torch.allclose(out, restored, atol=2.0e-5, rtol=2.0e-5))

    def test_v6_time_mlp_maps_fourier_features_directly_to_token_width(self):
        module = TimeConditionedParallelCrossAttention(
            v6_temporal_cfg(), dim=24
        ).eval()
        self.assertEqual(module.time_encoder.num_frequencies, 4)
        self.assertEqual(module.time_encoder.hidden_dim, 16)
        self.assertEqual(module.time_encoder.mlp[0].in_features, 9)
        self.assertEqual(module.time_encoder.mlp[0].out_features, 16)
        self.assertEqual(module.time_encoder.mlp[2].in_features, 16)
        self.assertEqual(module.time_encoder.mlp[2].out_features, 24)
        self.assertIsInstance(module.time_to_feature, torch.nn.Identity)
        encoded = module.time_encoder(torch.tensor([0.0, 0.8]))
        self.assertEqual(encoded.shape, (2, 24))

    def test_v7_attention_has_no_time_path_and_uses_warped_query_rope(self):
        torch.manual_seed(37)
        module = TimeConditionedParallelCrossAttention(
            v7_temporal_cfg(dim=72, heads=2, layers=1), dim=72
        ).eval()
        self.assertFalse(module.use_time_embedding)
        self.assertIsNone(module.time_encoder)
        self.assertIsNone(module.time_to_feature)
        self.assertAlmostEqual(
            2.0 * torch.pi / module.rope.position_scale, 5.0, places=6
        )
        wavelengths = (
            2.0 * torch.pi
            / (module.rope.position_scale * module.rope.inv_freq)
        )
        torch.testing.assert_close(
            wavelengths,
            torch.tensor([
                5.0,
                10.772173,
                23.207945,
                50.0,
                107.721733,
                232.079437,
            ]),
            rtol=1.0e-5,
            atol=1.0e-5,
        )

        feature = torch.randn(4, 72)
        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        layout = (torch.tensor([2, 4]), torch.tensor([0, 0]))
        output_a = module(
            feature, position, *layout, torch.zeros(4)
        )
        # Endpoint time values cannot affect a V7 token because the entire time
        # encoder/projection path is absent, rather than merely zero-gated.
        output_b = module(
            feature, position, *layout, torch.tensor([7.0, -3.0, 1.5, 9.0])
        )
        torch.testing.assert_close(output_a, output_b)

        warped = position.clone()
        warped[:, 0] += torch.tensor([2.0, 2.0, -2.0, -2.0])
        output_warped = module(
            feature,
            position,
            *layout,
            torch.zeros(4),
            query_position_ref=warped,
        )
        self.assertGreater(float((output_warped - output_a).abs().max()), 1.0e-6)

    def test_motion_heads_compute_soft_opposite_frame_displacement(self):
        module = TimeConditionedParallelCrossAttention(
            temporal_cfg(layers=1, motion_head_count=2), dim=24
        ).eval()
        with torch.no_grad():
            module.q_proj[0].weight.zero_()
            module.q_proj[0].bias.zero_()
            module.k_proj[0].weight.zero_()
            module.k_proj[0].bias.zero_()

        position = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
        ])
        _refined, delta_p = module(
            torch.zeros(5, 24),
            position,
            torch.tensor([2, 5]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]),
            return_motion_displacement=True,
        )
        expected = torch.tensor([
            [14.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [-9.0, 0.0, 0.0],
            [-13.0, 0.0, 0.0],
            [-17.0, 0.0, 0.0],
        ])
        self.assertTrue(torch.allclose(delta_p, expected, atol=1.0e-6))

    def test_motion_displacement_gradients_touch_only_first_four_heads(self):
        torch.manual_seed(19)
        module = TimeConditionedParallelCrossAttention(
            temporal_cfg(
                dim=72,
                heads=12,
                layers=1,
                motion_head_count=4,
            ),
            dim=72,
        )
        _refined, delta_p = module(
            torch.randn(5, 72),
            torch.randn(5, 3),
            torch.tensor([3, 5]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.0, 0.0, 0.8, 0.8]),
            return_motion_displacement=True,
        )
        upstream = torch.tensor([
            [0.7, -0.2, 0.3],
            [-0.1, 0.4, 0.9],
            [0.2, 0.5, -0.6],
            [0.8, -0.7, 0.1],
            [-0.4, 0.6, 0.2],
        ])
        (delta_p * upstream).sum().backward()

        for projection in (module.q_proj[0], module.k_proj[0]):
            grad_by_head = projection.weight.grad.reshape(12, 6, 72)
            self.assertGreater(float(grad_by_head[:4].abs().sum()), 0.0)
            self.assertEqual(float(grad_by_head[4:].abs().sum()), 0.0)

    def test_fixed_reference_time_preserves_irregular_pair_duration(self):
        gs_params = OmegaConf.create({
            "shs": 32,
            "opacity": 1,
            "scaling": 2,
            "rotation": 4,
            "offset": 3,
        })
        backend = PhysicalVelocityGaussianBackend(
            backend_cfg(), gs_params, dim=24, offset_bound=0.8
        )
        geometry = {
            "time_sec": torch.tensor([0.0, 0.8, 0.0, 1.0]),
        }
        coordinate = backend._temporal_coordinate(geometry)
        self.assertTrue(torch.equal(
            coordinate,
            torch.tensor([0.0, 0.8, 0.0, 1.0]),
        ))

    def test_dynamic_rope_is_numerically_equal_to_active_utonia_rope(self):
        torch.manual_seed(13)
        metric_position = torch.randn(7, 3) * 20.0
        q = torch.randn(7, 2, 24)
        k = torch.randn(7, 2, 24)

        utonia = Point3DRoPE(head_dim=24, base=10.0)
        expected_q, expected_k = utonia(q, k, metric_position * 0.2)

        dynamic = Rotary3D(head_dim=24, base=10.0, position_scale=0.2)
        angles = dynamic.angles(metric_position)
        actual_q = dynamic.rotate(q, angles[:, None])
        actual_k = dynamic.rotate(k, angles[:, None])
        self.assertTrue(torch.allclose(actual_q, expected_q, atol=1.0e-6))
        self.assertTrue(torch.allclose(actual_k, expected_k, atol=1.0e-6))

    def _backend_output(self, gaussians_per_token=1):
        torch.manual_seed(11)
        gs_params = OmegaConf.create({
            "shs": 32,
            "opacity": 1,
            "scaling": 2,
            "rotation": 4,
            "offset": 3,
        })
        backend = PhysicalVelocityGaussianBackend(
            backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussians_per_token=gaussians_per_token,
        )
        feature = torch.randn(5, 24)
        token = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ])
        seed = (token + torch.tensor([0.02, 0.01, 0.0])).unsqueeze(1).expand(
            -1, gaussians_per_token, -1
        ).contiguous()
        pose = [torch.eye(4).repeat(2, 1, 1)]
        output = backend(
            feature,
            token,
            seed,
            torch.tensor([3, 5]),
            torch.tensor([0, 0]),
            pose,
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.87])],
            torch.tensor([0.87]),
        )
        return backend, output

    def test_backend_emits_one_zero_velocity_gaussian_per_token(self):
        backend, output = self._backend_output()
        item = output["batch_gaussians"][0]
        self.assertFalse(hasattr(backend.gaussian_head, "geometry_encoder"))
        self.assertFalse(hasattr(backend.velocity_head, "position_encoder"))
        self.assertEqual(item["position"].shape, (5, 3))
        self.assertTrue(torch.allclose(item["position"], torch.tensor([
            [0.02, 0.01, 0.0],
            [1.02, 0.01, 0.0],
            [2.02, 0.01, 0.0],
            [0.12, 0.01, 0.0],
            [1.12, 0.01, 0.0],
        ])))
        self.assertEqual(item["shs"].shape, (5, 32))
        self.assertEqual(item["rotation"].shape, (5, 4))
        self.assertTrue(torch.allclose(
            item["source_time_sec"], torch.tensor([0.0, 0.0, 0.0, 0.87, 0.87])
        ))
        self.assertTrue(torch.equal(item["velocity"], torch.zeros_like(item["velocity"])))
        self.assertEqual(int(output["batch"].numel()), 5)

    def test_config_sets_how_many_gaussians_a_token_becomes(self):
        """p2g.grid_query.K_max is the per-token Gaussian count, not a cap."""
        backend, output = self._backend_output(gaussians_per_token=3)
        item = output["batch_gaussians"][0]
        self.assertEqual(backend.gaussians_per_token, 3)
        self.assertEqual(item["position"].shape, (15, 3))
        self.assertEqual(item["shs"].shape, (15, 32))
        self.assertEqual(item["rotation"].shape, (15, 4))
        self.assertEqual(int(output["batch"].numel()), 15)
        # One head per attribute, three slots wide.
        self.assertEqual(backend.gaussian_head.heads["shs"].out_features, 96)

    def test_slot_rows_stay_token_major_so_per_token_fields_line_up(self):
        _backend, output = self._backend_output(gaussians_per_token=3)
        item = output["batch_gaussians"][0]
        # Token 0-2 come from frame 0 at t=0; tokens 3-4 from frame 1 at 0.87.
        self.assertTrue(torch.allclose(
            item["source_time_sec"],
            torch.tensor([0.0] * 9 + [0.87] * 6),
        ))
        velocity = item["velocity"].reshape(5, 3, 3)
        self.assertTrue(torch.allclose(
            velocity, velocity[:, :1].expand_as(velocity)
        ))

    def test_identical_seeds_separate_through_independent_slot_offsets(self):
        """Every slot starts at the same seed but owns its own offset rows."""
        backend, output = self._backend_output(gaussians_per_token=3)
        offset_head = backend.gaussian_head.heads["offset"]
        self.assertEqual(offset_head.out_features, 9)
        position = output["batch_gaussians"][0]["position"].reshape(5, 3, 3)
        # Zero-initialized offsets leave all three slots on the shared seed.
        self.assertTrue(torch.allclose(
            position, position[:, :1].expand_as(position)
        ))
        # Once a slot's own rows move, only that slot's Gaussian moves.
        with torch.no_grad():
            offset_head.bias.copy_(torch.tensor(
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
            ))
        _raw, offset, _feature = backend.gaussian_head(torch.zeros(2, 24))
        offset = offset.reshape(2, 3, 3)
        self.assertTrue(torch.allclose(offset[:, 0], torch.zeros(2, 3)))
        self.assertFalse(torch.allclose(offset[:, 1], offset[:, 2]))

    def test_every_slot_starts_at_the_identity_quaternion(self):
        gs_params = OmegaConf.create({
            "shs": 4, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        head = GaussianAttributeHead(
            None, gs_params, dim=8, offset_bound=0.8, gaussians_per_token=3
        )
        self.assertTrue(torch.equal(
            head.heads["rotation"].bias,
            torch.tensor([1.0, 0.0, 0.0, 0.0] * 3),
        ))

    def test_v4_signed_displacement_gives_both_frames_forward_velocity(self):
        gs_params = OmegaConf.create({
            "shs": 32,
            "opacity": 1,
            "scaling": 2,
            "rotation": 4,
            "offset": 3,
        })
        backend = AttentionInitializedVelocityGaussianBackend(
            attention_backend_cfg(), gs_params, dim=24, offset_bound=0.8
        )
        output = backend(
            torch.zeros(2, 24),
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            torch.tensor([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]]),
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 1.0])],
            torch.tensor([1.0]),
        )
        item = output["batch_gaussians"][0]
        self.assertTrue(torch.equal(
            item["delta_p_init"],
            torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
        ))
        self.assertTrue(torch.equal(
            item["pair_delta_t_sec"], torch.tensor([1.0, -1.0])
        ))
        self.assertTrue(torch.equal(
            item["velocity_init"],
            torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        ))
        self.assertTrue(torch.equal(item["velocity"], item["velocity_init"]))
        self.assertTrue(torch.equal(
            item["velocity_residual"], torch.zeros(2, 3)
        ))
        midpoint = DynamicGausRender(OmegaConf.create({})).get_means3D(item, 0.5)
        self.assertTrue(torch.equal(
            midpoint,
            torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ))

    def test_v4_reuses_v3_parameter_structure(self):
        gs_params = OmegaConf.create({
            "shs": 32,
            "opacity": 1,
            "scaling": 2,
            "rotation": 4,
            "offset": 3,
        })
        v3 = PhysicalVelocityGaussianBackend(
            backend_cfg(), gs_params, dim=24, offset_bound=0.8
        )
        v4 = AttentionInitializedVelocityGaussianBackend(
            attention_backend_cfg(), gs_params, dim=24, offset_bound=0.8
        )
        self.assertEqual(set(v3.state_dict()), set(v4.state_dict()))

    def test_physical_seconds_transport_and_velocity_gradient(self):
        backend, output = self._backend_output()
        item = output["batch_gaussians"][0]
        renderer = DynamicGausRender(OmegaConf.create({}))
        means = renderer.get_means3D(item, 0.4)
        self.assertTrue(torch.allclose(means, item["position"]))

        target = item["position"].detach() + torch.tensor([0.4, 0.0, 0.0])
        loss = (renderer.get_means3D(item, 0.4) - target).square().mean()
        loss.backward()
        grad = backend.velocity_head.velocity.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_renderers_declare_transport_contract(self):
        self.assertEqual(GausRender.transport_mode, "bbox")
        self.assertEqual(DynamicGausRender.transport_mode, "velocity")

    def test_velocity_viz_uses_actual_signed_source_to_target_time(self):
        renderer = DynamicGausRender(OmegaConf.create({}))
        item = {
            "position": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "velocity": torch.tensor([[2.0, 0.0, 0.0], [0.0, -1.0, 0.5]]),
            "source_time_sec": torch.tensor([0.0, 1.0]),
        }
        transport = velocity_transport_from_output(renderer, item, 0.5)
        np.testing.assert_allclose(
            transport["target_centers"],
            np.array([[2.0, 2.0, 3.0], [4.0, 5.5, 5.75]], np.float32),
        )
        np.testing.assert_allclose(
            transport["source_centers"], item["position"].numpy()
        )
        np.testing.assert_allclose(
            transport["velocity_mps"], item["velocity"].numpy()
        )
        np.testing.assert_allclose(transport["delta_t_sec"], [0.5, -0.5])
        np.testing.assert_allclose(transport["source_times_sec"], [0.0, 1.0])
        self.assertEqual(len(transport["target_centers_by_source_frame"]), 2)

    def test_velocity_viz_arrow_encodes_actual_signed_travel_time(self):
        trace = _velocity_arrow_trace(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32),
            np.array([[4.0, -2.0, 0.0], [0.0, -1.0, 0.5]], np.float32),
            delta_t_sec=np.array([0.5, -0.5], np.float32),
            name="velocity",
            color="#123456",
        )
        # Each arrow occupies nine vertices; indices 0/1 and 9/10 are shafts.
        np.testing.assert_allclose(
            np.array([trace[axis][0] for axis in "xyz"]), [1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(
            np.array([trace[axis][1] for axis in "xyz"]), [3.0, 1.0, 3.0]
        )
        np.testing.assert_allclose(
            np.array([trace[axis][9] for axis in "xyz"]), [4.0, 5.0, 6.0]
        )
        np.testing.assert_allclose(
            np.array([trace[axis][10] for axis in "xyz"]), [4.0, 5.5, 5.75]
        )

    def test_center_html_adds_velocity_layer_only_when_vectors_exist(self):
        base = {
            "label": "T=1s",
            "static": np.zeros((0, 3), np.float32),
            "dynamic": np.array([[1.0, 2.0, 3.0]], np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            without_path = Path(tmp) / "without.html"
            with_path = Path(tmp) / "with.html"
            save_sequence_html(without_path, "bbox", [base])
            save_sequence_html(with_path, "velocity", [{
                "label": "T=1s",
                "source_frame_centers": [
                    np.array([[1.5, 2.0, 3.0]], np.float32),
                    np.array([[4.0, 5.5, 5.75]], np.float32),
                ],
                "velocity_origins": np.array(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32
                ),
                "velocity_mps": np.array(
                    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.5]], np.float32
                ),
                "velocity_delta_t_sec": np.array([0.5, -0.5], np.float32),
                "velocity_source_frame_index": np.array([0, 1]),
            }])

            without = without_path.read_text()
            with_velocity = with_path.read_text()
            self.assertNotIn("gaussians from source frame", without)
            self.assertIn('"gaussians from source frame 0"', with_velocity)
            self.assertIn('"gaussians from source frame 1"', with_velocity)
            self.assertIn(
                json.dumps(_source_frame_velocity_layer(0)), with_velocity
            )
            self.assertIn(
                json.dumps(_source_frame_velocity_layer(1)), with_velocity
            )
            for color in SOURCE_FRAME_COLORS[:2]:
                self.assertIn(
                    json.dumps({"color": color, "width": 2}), with_velocity
                )
                self.assertIn(
                    json.dumps({
                        "size": 2.6, "color": color, "opacity": 0.9,
                    }),
                    with_velocity,
                )
            self.assertNotIn('"gaussians (static)"', with_velocity)
            self.assertNotIn('"gaussians (dynamic)"', with_velocity)

    def test_direct_velocity_does_not_depend_on_sample_duration(self):
        backend, _output = self._backend_output()
        head = backend.velocity_head
        with torch.no_grad():
            head.velocity.bias.fill_(0.1)
        feature = torch.zeros(2, 24)
        velocity = head(feature)
        self.assertTrue(torch.equal(velocity, torch.full((2, 3), 0.1)))

    def test_v3_fusion_is_576_to_1024_to_1024_to_768(self):
        cfg = OmegaConf.create({
            "agg_mlp": {
                "hidden_dim": 1024,
                "hidden_layers": 2,
                "out_dim": 576,
            },
            "utonia_adapter": {"bottleneck": 108},
            "joint_refiner": {"enable": False},
        })
        _adapter, fusion, refiner = build_feature_fusion(
            cfg, utonia_dim=432, intensity_dim=144, coord_scale=0.2
        )
        self.assertIsNone(refiner)
        self.assertEqual(len(fusion), 5)
        self.assertEqual(fusion[0].in_features, 576)
        self.assertEqual(fusion[0].out_features, 1024)
        self.assertEqual(fusion[2].in_features, 1024)
        self.assertEqual(fusion[2].out_features, 1024)
        self.assertEqual(fusion[4].in_features, 1024)
        self.assertEqual(fusion[4].out_features, 576)
        output = fusion(torch.randn(2, 576))
        self.assertEqual(output.shape, (2, 576))


class SparseMotionProposalTest(unittest.TestCase):
    def test_shared_score_is_endpoint_swap_symmetric(self):
        torch.manual_seed(3)
        module = SparseMotionProposal(proposal_cfg(), dim=8)
        descriptor0 = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        descriptor1 = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
        position0 = torch.randn(3, 3)
        position1 = torch.randn(5, 3)
        radius0 = torch.rand(3) + 1.0
        radius1 = torch.rand(5) + 1.0
        score01 = module._pair_score(
            descriptor0, descriptor1, position0, position1, radius0, radius1
        )
        score10 = module._pair_score(
            descriptor1, descriptor0, position1, position0, radius1, radius0
        )
        self.assertTrue(torch.allclose(score01, score10.t(), atol=1.0e-6))
        self.assertFalse(module.input_norm.elementwise_affine)
        self.assertTrue(torch.count_nonzero(
            module.descriptor_adapter[-1].weight
        ).item() == 0)

        feature = torch.randn(4, 8)
        normalized = module.input_norm(feature)
        descriptor, _speed, _radius = module._encode(
            feature, torch.ones(4)
        )
        expected = torch.nn.functional.normalize(
            module.descriptor(normalized).float(), dim=-1, eps=1.0e-6
        )
        self.assertTrue(torch.equal(descriptor, expected))

    def test_joint_topk_is_sparse_without_a_distance_cutoff(self):
        module = SparseMotionProposal(
            proposal_cfg(candidate_count=2, temperature=0.1), dim=8
        )
        query_descriptor = torch.tensor([[1.0, 0.0]])
        key_descriptor = torch.tensor([
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ])
        query_position = torch.zeros(1, 3)
        key_position = torch.tensor([
            [0.1, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])
        result = module._topk_direction(
            query_descriptor,
            key_descriptor,
            query_position,
            key_position,
            torch.tensor([30.0]),
            torch.full((3,), 30.0),
        )
        self.assertEqual(result["candidate_index"].shape, (1, 2))
        # The 10 m semantic match survives because geometry is a bias, not a cut.
        self.assertEqual(int(result["candidate_index"][0, 0]), 1)

    def test_zero_init_adapter_preserves_then_learns_bottleneck(self):
        torch.manual_seed(29)
        module = SparseMotionProposal(proposal_cfg(), dim=8)
        feature = torch.randn(6, 8)
        target = torch.randn(6, module.descriptor_dim)
        optimizer = torch.optim.SGD(
            module.descriptor_adapter.parameters(), lr=0.1
        )

        descriptor, _speed, _radius = module._encode(feature, torch.ones(6))
        (descriptor * target).sum().backward()
        down_grad = module.descriptor_adapter[0].weight.grad
        up_grad = module.descriptor_adapter[-1].weight.grad
        self.assertEqual(float(down_grad.abs().sum()), 0.0)
        self.assertGreater(float(up_grad.abs().sum()), 0.0)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        descriptor, _speed, _radius = module._encode(feature, torch.ones(6))
        (descriptor * target).sum().backward()
        down_grad = module.descriptor_adapter[0].weight.grad
        self.assertIsNotNone(down_grad)
        self.assertGreater(float(down_grad.abs().sum()), 0.0)

    def test_dustbin_uses_absolute_cosine_evidence(self):
        module = SparseMotionProposal(
            proposal_cfg(
                candidate_count=1, match_count=1, temperature=0.1
            ), dim=8
        )
        result, _reverse = module._match_pair(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0]]),
            torch.zeros(2, 3),
            torch.zeros(1, 3),
            torch.full((2,), 100.0),
            torch.full((1,), 100.0),
        )
        self.assertLess(float(result["p_unmatched"][0]), 0.1)
        self.assertGreater(float(result["p_unmatched"][1]), 0.99)

    def test_token_conditioned_dustbin_starts_global_then_learns_per_token(self):
        torch.manual_seed(31)
        module = SparseMotionProposal(
            proposal_cfg(
                dustbin_similarity_init=0.50,
                dustbin_token_conditioned=True,
            ),
            dim=8,
        )
        feature = torch.randn(4, 8, requires_grad=True)

        initial = module._token_dustbin_similarity(feature)
        torch.testing.assert_close(initial, torch.full((4,), 0.50))
        self.assertEqual(
            int(torch.count_nonzero(module.dustbin_token_residual.weight)), 0
        )

        weights = torch.tensor([1.0, -0.5, 0.25, -0.75])
        (initial * weights).sum().backward()
        grad = module.dustbin_token_residual.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

        with torch.no_grad():
            module.dustbin_token_residual.weight.copy_(0.1 * grad.sign())
        learned = module._token_dustbin_similarity(feature.detach())
        self.assertGreater(float(learned.max() - learned.min()), 0.0)

    def test_v7_descriptor_is_exact_raw_feature_l2_normalization(self):
        module = SparseMotionProposal(v7_proposal_cfg(), dim=8)
        feature = torch.tensor([
            [10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ])
        descriptor, _speed, _radius = module._encode(
            feature, torch.ones(2)
        )
        self.assertIsNone(module.descriptor_adapter)
        self.assertIsNone(module.descriptor)
        self.assertEqual(module.descriptor_dim, 8)
        torch.testing.assert_close(
            descriptor,
            torch.nn.functional.normalize(feature.float(), dim=-1),
        )
        self.assertFalse(torch.equal(
            descriptor,
            torch.nn.functional.normalize(
                module.input_norm(feature).float(), dim=-1
            ),
        ))

    def test_v7_dustbin_uses_normalized_displacement_magnitude_and_ste(self):
        module = SparseMotionProposal(v7_proposal_cfg(), dim=8)
        self.assertEqual(module.dustbin_predictor[0].in_features, 3)
        direction = {
            "candidate_index": torch.tensor([[0, 1], [0, 1]]),
            "score": torch.tensor([[10.0, 0.0], [10.0, 0.0]]),
            "conditional_probability": torch.tensor([
                [0.75, 0.25], [0.60, 0.40],
            ]),
            "support": 2,
        }
        reverse = {
            "candidate_index": torch.tensor([[0, 1], [1, 0]]),
            "score": torch.zeros(2, 2),
            "conditional_probability": torch.tensor([
                [0.80, 0.20], [0.70, 0.30],
            ]),
            "support": 2,
        }
        query_position = torch.tensor([
            [10.0, 20.0, 30.0], [16.0, 16.0, 38.0],
        ])
        key_position = torch.tensor([
            [13.0, 18.0, 34.0], [14.0, 20.0, 30.0],
        ])
        duration_sec = torch.tensor([0.5, 1.0])
        captured = []
        handle = module.dustbin_predictor[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        try:
            result = module._finish_direction(
                direction, reverse, query_position, key_position,
                duration_sec=duration_sec,
            )
        finally:
            handle.remove()

        expected_entropy = -(direction["conditional_probability"] * (
            direction["conditional_probability"].log()
        )).sum(dim=-1)
        expected_displacement = torch.tensor([
            [3.0, -2.0, 4.0], [-3.0, 2.0, -4.0],
        ])
        expected_evidence = torch.cat([
            (expected_entropy / math.log(2.0)).unsqueeze(-1),
            torch.tanh(
                expected_displacement.norm(dim=-1)
                / torch.tensor([10.0, 20.0])
            ).unsqueeze(-1),
            (
                torch.log1p(torch.tensor([1.10, 0.90])) / math.log(3.0)
            ).unsqueeze(-1),
        ], dim=-1)
        torch.testing.assert_close(captured[0], expected_evidence)
        self.assertTrue(torch.all((captured[0] >= 0.0) & (captured[0] <= 1.0)))
        torch.testing.assert_close(
            result["motion_best_candidate_displacement_m"],
            expected_displacement,
        )
        torch.testing.assert_close(
            result["p_unmatched"], torch.full((2,), 0.05)
        )
        # The dustbin is not a (M+1)-th softmax class: a conservative initial
        # decision passes the complete conditional match with no 0.95 shrink.
        torch.testing.assert_close(
            result["delta_p_init"], result["delta_p_match"]
        )
        torch.testing.assert_close(
            result["match_probability"], torch.ones(2)
        )
        torch.testing.assert_close(
            result["motion_hard_reject"], torch.zeros(2)
        )

        with torch.no_grad():
            module.dustbin_predictor[-1].bias.fill_(
                torch.logit(torch.tensor(0.95))
            )
        rejected = module._finish_direction(
            direction, reverse, query_position, key_position,
            duration_sec=duration_sec,
        )
        torch.testing.assert_close(
            rejected["delta_p_init"], torch.zeros(2, 3)
        )
        self.assertTrue(torch.equal(
            rejected["motion_hard_reject"], torch.ones(2)
        ))
        # Forward is hard-zero, but STE keeps a non-zero gradient into the
        # unmatched predictor.
        rejected["delta_p_init"].sum().backward()
        self.assertGreater(
            float(module.dustbin_predictor[-1].bias.grad.abs()), 0.0
        )

    def test_v7_1_dustbin_uses_mean_spread_and_weighted_m4_reciprocal(self):
        module = SparseMotionProposal(v7_1_proposal_cfg(), dim=8)
        direction = {
            "candidate_index": torch.tensor([[0, 1, 2, 3]]),
            "score": torch.zeros(1, 4),
            "conditional_probability": torch.full((1, 4), 0.25),
            "support": 4,
        }
        reverse = {
            "candidate_index": torch.tensor([
                [0, 1, 2, 3],
                [1, 0, 2, 3],
                [1, 2, 0, 3],
                [1, 2, 3, 0],
            ]),
            "score": torch.zeros(4, 4),
            "conditional_probability": torch.tensor([
                [0.10, 0.30, 0.30, 0.30],
                [0.40, 0.20, 0.20, 0.20],
                [0.30, 0.25, 0.30, 0.15],
                [0.40, 0.20, 0.00, 0.40],
            ]),
            "support": 4,
        }
        query_position = torch.zeros(1, 3)
        key_position = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        captured = []
        handle = module.dustbin_predictor[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        try:
            result_short = module._finish_direction(
                direction,
                reverse,
                query_position,
                key_position,
                duration_sec=torch.tensor([0.5]),
            )
            result_long = module._finish_direction(
                direction,
                reverse,
                query_position,
                key_position,
                duration_sec=torch.tensor([2.0]),
            )
        finally:
            handle.remove()

        reciprocal = torch.tensor([0.10, 0.20, 0.30, 0.40])
        weight = torch.softmax(torch.log(reciprocal + 0.25), dim=0)
        displacement = key_position
        expected_mean = (weight[:, None] * displacement).sum(dim=0)
        centered = displacement - expected_mean
        expected_spread = torch.sqrt(
            (weight * centered.square().sum(dim=-1)).sum()
        )
        expected_reciprocal = (weight * reciprocal).sum()
        expected_evidence = torch.stack([
            expected_mean.norm() / (expected_mean.norm() + 4.0),
            expected_spread / (expected_spread + 0.5),
            expected_reciprocal,
        ])[None]

        torch.testing.assert_close(captured[0], expected_evidence)
        # Dustbin geometry is measured in metres and is independent of delta_t.
        torch.testing.assert_close(captured[1], captured[0])
        torch.testing.assert_close(
            result_short["p_unmatched"], torch.full((1,), 0.05)
        )
        torch.testing.assert_close(
            result_short["motion_mean_displacement_m"], expected_mean[None]
        )
        torch.testing.assert_close(
            result_short["motion_candidate_spread_m"], expected_spread[None]
        )
        torch.testing.assert_close(
            result_short["motion_reciprocal_probability"],
            expected_reciprocal[None],
        )
        self.assertNotIn("motion_candidate_entropy", result_short)
        self.assertNotIn(
            "motion_best_candidate_displacement_m", result_short
        )
        torch.testing.assert_close(
            result_short["delta_p_init"], result_short["delta_p_match"]
        )
        torch.testing.assert_close(
            result_long["delta_p_init"], result_short["delta_p_init"]
        )

        with torch.no_grad():
            module.dustbin_predictor[-1].bias.fill_(
                torch.logit(torch.tensor(0.6))
            )
        rejected = module._finish_direction(
            direction, reverse, query_position, key_position
        )
        torch.testing.assert_close(
            rejected["delta_p_init"], torch.zeros(1, 3)
        )
        torch.testing.assert_close(
            rejected["motion_hard_reject"], torch.ones(1)
        )

    def test_soft_reciprocal_evidence_reweights_many_to_one_winner(self):
        direction = {
            "candidate_index": torch.tensor([[0, 1], [0, 1]]),
            "score": torch.tensor([[0.9, 0.1], [0.9, 0.1]]).log() + 12.0,
            "conditional_probability": torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
            "support": 2,
        }
        reverse = {
            "candidate_index": torch.tensor([[0, 1], [1, 0]]),
            "score": (
                torch.tensor([[1.0, 1.0e-6], [1.0, 1.0e-6]]).log()
                + 12.0
            ),
            "conditional_probability": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "support": 2,
        }
        position0 = torch.zeros(2, 3)
        position1 = torch.tensor([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        module = SparseMotionProposal(proposal_cfg(match_count=2), dim=8)
        result = module._finish_direction(
            direction, reverse, position0, position1
        )
        # Query 1's non-reciprocal key-0 winner drops from 0.9 to 0.75 and the
        # reciprocal alternative rises from 0.1 to 0.25. It is not hard-cut.
        self.assertAlmostEqual(float(result["delta_p_match"][1, 0]), 7.75, places=5)
        self.assertTrue(torch.all((result["p_unmatched"] >= 0.0)
                                  & (result["p_unmatched"] <= 1.0)))

    def test_forward_is_differentiable_through_selected_candidates(self):
        torch.manual_seed(17)
        module = SparseMotionProposal(proposal_cfg(), dim=8)
        feature = torch.randn(7, 8, requires_grad=True)
        position = torch.randn(7, 3)
        result = module(
            feature,
            position,
            torch.tensor([3, 7]),
            torch.tensor([0, 0]),
            torch.full((7,), 0.8),
        )
        self.assertTrue(torch.all(
            result["motion_effective_support"] <= module.match_count + 1.0e-5
        ))
        objective = (
            result["delta_p_init"].square().sum()
            + result["p_unmatched"].sum()
        )
        objective.backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(module.descriptor.weight.grad).all())
        adapter_grad = module.descriptor_adapter[-1].weight.grad
        self.assertIsNotNone(adapter_grad)
        self.assertTrue(torch.isfinite(adapter_grad).all())
        self.assertGreater(float(adapter_grad.abs().sum()), 0.0)

    def test_v7_2_forward_is_exactly_renormalized_direct_top4(self):
        module = StraightThroughTop4MotionProposal(
            v7_2_proposal_cfg(score_chunk_size=1), dim=4
        ).eval()
        query = F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
        ]), dim=-1)
        key = F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.4, 0.0, 0.916515, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]), dim=-1)
        query_position = torch.zeros(1, 3)
        # The best feature-only match is intentionally 100 m away. The weak
        # fixed 30 m/s envelope should demote it without any learned/tokenwise
        # radius.
        key_position = torch.tensor([
            [100.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [-50.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        key = torch.cat([key, F.normalize(torch.tensor([
            [0.2, 0.979796, 0.0, 0.0],
        ]), dim=-1)], dim=0)
        result = module._match_direction(
            query, key, query_position, key_position
        )

        squared_distance = (key_position - query_position).square().sum(dim=-1)
        penalty = squared_distance / 30.0 ** 2
        dense = torch.softmax((query @ key.T) / 0.1 - penalty[None], dim=-1)
        top_probability, top_index = torch.topk(dense, k=4, dim=-1)
        normalized = top_probability / top_probability.sum(dim=-1, keepdim=True)
        expected_position = (
            normalized.unsqueeze(-1) * key_position[top_index]
        ).sum(dim=1)
        torch.testing.assert_close(result["matched_position"], expected_position)
        torch.testing.assert_close(
            result["delta_p_init"], result["delta_p_match"]
        )
        self.assertEqual(int(result["motion_selected_index"][0, 0]), 1)
        self.assertNotIn(0, result["motion_selected_index"][0].tolist())
        torch.testing.assert_close(
            result["motion_top1_probability"], normalized[:, 0]
        )
        torch.testing.assert_close(
            result["motion_candidate_probability_mass"],
            top_probability.sum(dim=-1),
        )
        expected_selected_penalty = (
            normalized * penalty[top_index]
        ).sum(dim=-1)
        torch.testing.assert_close(
            result["motion_selected_distance_prior_penalty"],
            expected_selected_penalty,
        )

    def test_v7_2_dense_surrogate_grad_reaches_key_outside_top4(self):
        module = StraightThroughTop4MotionProposal(
            v7_2_proposal_cfg(score_chunk_size=1), dim=4
        ).eval()
        query_raw = torch.tensor([
            [1.0, 0.2, -0.1, 0.3],
        ], requires_grad=True)
        key_raw = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.4, 0.0, 0.9, 0.1],
            [-0.7, 0.4, 0.2, -0.1],
        ], requires_grad=True)
        query = F.normalize(query_raw, dim=-1)
        key = F.normalize(key_raw, dim=-1)
        position = torch.tensor([
            [1.0, 2.0, 0.0],
            [-3.0, 1.0, 2.0],
            [2.0, -4.0, 1.0],
            [5.0, 3.0, -2.0],
            [-6.0, 7.0, 4.0],
        ])
        result = module._match_direction(
            query, key, torch.zeros(1, 3), position
        )
        excluded = ({0, 1, 2, 3, 4}
                    - set(result["motion_selected_index"][0].tolist())).pop()
        result["delta_p_init"].square().sum().backward()

        self.assertGreater(float(query_raw.grad.abs().sum()), 0.0)
        # A plain Top-4 graph would give this row exactly zero. The non-zero
        # value proves that backward followed the dense all-key softmax.
        self.assertGreater(float(key_raw.grad[excluded].abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(key_raw.grad).all())

    def test_v7_2_ste_jacobian_equals_dense_softmax_jacobian(self):
        module = StraightThroughTop4MotionProposal(
            v7_2_proposal_cfg(score_chunk_size=1), dim=4
        ).eval()
        query_raw = torch.tensor([
            [0.9, 0.2, -0.3, 0.1],
        ], requires_grad=True)
        key_raw = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.4, 0.0, 0.9, 0.1],
            [-0.7, 0.4, 0.2, -0.1],
        ], requires_grad=True)
        position = torch.tensor([
            [1.0, 2.0, 0.0], [-3.0, 1.0, 2.0],
            [2.0, -4.0, 1.0], [5.0, 3.0, -2.0],
            [-6.0, 7.0, 4.0],
        ])
        upstream = torch.tensor([[0.7, -1.3, 0.4]])
        result = module._match_direction(
            F.normalize(query_raw, dim=-1),
            F.normalize(key_raw, dim=-1),
            torch.zeros(1, 3),
            position,
        )
        ste_grad = torch.autograd.grad(
            result["matched_position"],
            (query_raw, key_raw),
            grad_outputs=upstream,
        )

        dense_query_raw = query_raw.detach().clone().requires_grad_()
        dense_key_raw = key_raw.detach().clone().requires_grad_()
        dense_query = F.normalize(dense_query_raw, dim=-1)
        dense_key = F.normalize(dense_key_raw, dim=-1)
        dense_probability = torch.softmax(
            (dense_query @ dense_key.T) / 0.1
            - position.square().sum(dim=-1)[None] / 30.0 ** 2,
            dim=-1,
        )
        dense_matched = dense_probability @ position
        dense_grad = torch.autograd.grad(
            dense_matched,
            (dense_query_raw, dense_key_raw),
            grad_outputs=upstream,
        )
        torch.testing.assert_close(ste_grad[0], dense_grad[0])
        torch.testing.assert_close(ste_grad[1], dense_grad[1])

    def test_v7_2_backend_uses_time_conditioned_q_warp_and_feature_offset(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = StraightThroughProposalVelocityGaussianBackend(
            v7_2_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            proposal_dim=8,
        )
        self.assertIsInstance(
            backend.motion_proposal, StraightThroughTop4MotionProposal
        )
        self.assertIsInstance(
            backend.velocity_head, FeatureOnlyVelocityOffsetHead
        )
        self.assertTrue(backend.temporal.use_time_embedding)
        self.assertEqual(sum(
            parameter.numel() for parameter in backend.motion_proposal.parameters()
        ), 0)

        class FixedProposal(torch.nn.Module):
            def forward(self, *_args, **_kwargs):
                delta = torch.tensor([
                    [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
                ])
                return {"delta_p_match": delta, "delta_p_init": delta}

        class CaptureTemporal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query_position_ref = None
                self.token_time_coordinate = None

            def forward(self, feature, _position, _offset, _batch_idx,
                        token_time_coordinate,
                        query_position_ref=None, **_kwargs):
                self.query_position_ref = query_position_ref
                self.token_time_coordinate = token_time_coordinate
                return feature

        backend.motion_proposal = FixedProposal()
        backend.temporal = CaptureTemporal()
        geometry = {
            "token_ref": torch.tensor([
                [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            ]),
            "local_frame": torch.tensor([0, 1]),
            "duration_sec": torch.ones(2),
            "time_sec": torch.tensor([0.0, 1.0]),
        }
        _, fields = backend._temporal_refine(
            torch.zeros(2, 24),
            geometry,
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            motion_proposal_feature=torch.randn(2, 8),
        )
        torch.testing.assert_close(
            backend.temporal.query_position_ref,
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            backend.temporal.token_time_coordinate,
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            fields["velocity_match"],
            torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )

    def test_v7_2_feature_only_offset_has_one_activation(self):
        head = FeatureOnlyVelocityOffsetHead(
            v7_2_backend_cfg().motion, dim=24
        )
        self.assertEqual(head.motion_refiner[0].in_features, 24)
        self.assertEqual(head.motion_refiner[0].out_features, 8)
        self.assertEqual(tuple(head.velocity.weight.shape), (3, 8))
        self.assertEqual(
            sum(isinstance(module, torch.nn.SiLU) for module in head.modules()),
            1,
        )
        self.assertFalse(hasattr(head, "velocity_encoder"))
        self.assertFalse(hasattr(head, "duration_encoder"))
        output = head(torch.randn(2, 24))
        self.assertTrue(torch.equal(output, torch.zeros(2, 3)))

    def test_init_conditioned_head_receives_only_feature_and_scaled_init(self):
        head = InitConditionedVelocityHead(
            OmegaConf.create({
                "zero_init": True,
                "init_condition_scale_mps": 10.0,
                "detach_init_condition": True,
            }),
            dim=24,
        )
        captured = []
        handle = head.motion_refiner[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0])
        )
        velocity_init = torch.tensor(
            [[10.0, -5.0, 0.0], [0.0, 0.0, 20.0]],
            requires_grad=True,
        )
        try:
            output = head(torch.zeros(2, 24), velocity_init)
        finally:
            handle.remove()

        self.assertEqual(head.motion_refiner[0].in_features, 27)
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.equal(
            captured[0][:, -3:],
            torch.tensor([[1.0, -0.5, 0.0], [0.0, 0.0, 2.0]]),
        ))
        condition_grad = torch.autograd.grad(
            output.sum(), velocity_init, allow_unused=True
        )[0]
        self.assertIsNone(condition_grad)

    def test_residual_hidden_dim_builds_one_bottleneck_layer(self):
        cfg = {
            "zero_init": True,
            "init_condition_scale_mps": 10.0,
            "detach_init_condition": True,
        }
        wide = InitConditionedVelocityHead(OmegaConf.create(cfg), dim=24)
        narrow = InitConditionedVelocityHead(
            OmegaConf.create({**cfg, "residual_hidden_dim": 8}), dim=24
        )

        # Omitting the key must reproduce the historical two dim-wide layers so
        # every pre-existing V6 checkpoint rebuilds with identical shapes.
        self.assertEqual(
            [tuple(p.shape) for p in wide.motion_refiner.parameters()],
            [(24, 27), (24,), (24, 24), (24,)],
        )
        self.assertEqual(tuple(wide.velocity.weight.shape), (3, 24))

        self.assertEqual(
            [tuple(p.shape) for p in narrow.motion_refiner.parameters()],
            [(8, 27), (8,)],
        )
        self.assertEqual(tuple(narrow.velocity.weight.shape), (3, 8))
        self.assertLess(
            sum(p.numel() for p in narrow.parameters()),
            sum(p.numel() for p in wide.parameters()) // 4,
        )

        # Zero init still starts a fresh model at v_final = v_init, and the
        # conditioning copy stays detached in both forms.
        velocity_init = torch.tensor(
            [[10.0, -5.0, 0.0], [0.0, 0.0, 20.0]], requires_grad=True
        )
        output = narrow(torch.randn(2, 24), velocity_init)
        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.equal(output, torch.zeros(2, 3)))
        self.assertIsNone(torch.autograd.grad(
            narrow(torch.randn(2, 24), velocity_init).sum(),
            velocity_init,
            allow_unused=True,
        )[0])

    def test_residual_hidden_dim_rejects_a_non_positive_width(self):
        with self.assertRaises(ValueError):
            InitConditionedVelocityHead(
                OmegaConf.create({"residual_hidden_dim": 0}), dim=24
            )

    def test_v7_offset_head_has_exact_embedding_and_concat_contract(self):
        head = EmbeddedInitDurationVelocityHead(
            v7_backend_cfg().motion, dim=24
        )
        self.assertEqual(
            [tuple(parameter.shape)
             for parameter in head.velocity_encoder.parameters()],
            [(32, 3), (32,), (32, 32), (32,)],
        )
        self.assertEqual(
            [tuple(parameter.shape)
             for parameter in head.duration_encoder.parameters()],
            [(16, 8), (16,), (16, 16), (16,)],
        )
        self.assertEqual(head.motion_refiner[0].in_features, 24 + 32 + 16)
        self.assertEqual(head.motion_refiner[0].out_features, 8)
        self.assertEqual(tuple(head.velocity.weight.shape), (3, 8))
        duration = torch.tensor([0.5, 1.0])
        expected_phase = duration[:, None] * (
            torch.pi * 2.0 ** torch.arange(4)
        )[None]
        torch.testing.assert_close(
            head._duration_fourier(duration),
            torch.cat([expected_phase.sin(), expected_phase.cos()], dim=-1),
        )

        velocity_init = torch.randn(2, 3, requires_grad=True)
        with torch.no_grad():
            head.velocity.weight.fill_(1.0)
        output = head(torch.randn(2, 24), velocity_init, duration)
        self.assertEqual(output.shape, (2, 3))
        self.assertIsNone(torch.autograd.grad(
            output.sum(), velocity_init, allow_unused=True
        )[0])

    def test_v7_q_warp_uses_detached_hard_gated_init_without_feature_gate(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = WarpedProposalVelocityGaussianBackend(
            v7_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            proposal_dim=8,
        )

        class FixedProposal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta = torch.nn.Parameter(torch.tensor([
                    [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
                ]))

            def forward(self, *_args, **_kwargs):
                hard_match = self.delta.new_tensor([[1.0], [0.0]])
                return {
                    "delta_p_match": self.delta,
                    "delta_p_init": self.delta * hard_match,
                    # The second query is confidently unmatched, but this must
                    # not multiply the cross-attention feature update.
                    "p_unmatched": torch.tensor([0.05, 0.95]),
                }

        class CaptureTemporal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query_position_ref = None

            def forward(self, feature, _position, *_args,
                        query_position_ref=None, **_kwargs):
                self.query_position_ref = query_position_ref
                return feature + 3.0

        proposal = FixedProposal()
        temporal = CaptureTemporal()
        backend.motion_proposal = proposal
        backend.temporal = temporal
        feature = torch.zeros(2, 24)
        geometry = {
            "token_ref": torch.tensor([
                [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            ]),
            "local_frame": torch.tensor([0, 1]),
            "duration_sec": torch.ones(2),
            "time_sec": torch.tensor([0.0, 1.0]),
        }
        refined, fields = backend._temporal_refine(
            feature,
            geometry,
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            motion_proposal_feature=torch.randn(2, 8),
        )
        torch.testing.assert_close(refined, torch.full((2, 24), 3.0))
        torch.testing.assert_close(
            temporal.query_position_ref,
            # Matched frame-0 token moves to its proposal; rejected frame-1
            # token remains at its own observed x=2 position.
            torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )
        self.assertFalse(temporal.query_position_ref.requires_grad)
        torch.testing.assert_close(
            fields["velocity_match"],
            torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )

    def test_backend_adds_init_conditioned_residual_without_dustbin_gate(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = ProposalInitializedVelocityGaussianBackend(
            proposal_backend_cfg(), gs_params, dim=24, offset_bound=0.8
        )
        with torch.no_grad():
            backend.velocity_head.velocity.bias.fill_(0.5)
        temporal_fields = {
            "delta_p_match": torch.tensor([
                [2.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
            ]),
            "delta_p_init": torch.tensor([
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]),
            "match_probability": torch.tensor([1.0, 0.0]),
            "p_unmatched": torch.tensor([0.0, 1.0]),
            "matched_position": torch.zeros(2, 3),
            "motion_top1_probability": torch.ones(2),
            "motion_effective_support": torch.ones(2),
            "motion_reciprocal_probability": torch.ones(2),
            "motion_dustbin_similarity": torch.full((2,), 0.5),
            "motion_search_speed_mps": torch.full((2,), 5.0),
        }
        velocity, fields = backend._predict_motion(
            torch.zeros(2, 24),
            {
                "local_frame": torch.tensor([0, 1]),
                "duration_sec": torch.ones(2),
            },
            torch.zeros(2, 3),
            temporal_fields,
        )
        self.assertTrue(torch.equal(
            velocity,
            torch.tensor([[2.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
        ))
        self.assertTrue(backend.init_conditioned_residual)
        self.assertIsInstance(backend.velocity_head, InitConditionedVelocityHead)
        self.assertIn("velocity_offset", fields)
        self.assertTrue(torch.equal(
            fields["velocity_init"],
            torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ))
        self.assertIn("velocity_match", fields)
        self.assertNotIn("velocity_residual", fields)

    def test_legacy_v6_config_keeps_feature_only_dustbin_gated_fallback(self):
        cfg = proposal_backend_cfg()
        del cfg.motion.init_conditioned_residual
        del cfg.motion.init_condition_scale_mps
        del cfg.motion.detach_init_condition
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = ProposalInitializedVelocityGaussianBackend(
            cfg, gs_params, dim=24, offset_bound=0.8
        )
        with torch.no_grad():
            backend.velocity_head.velocity.bias.fill_(0.5)
        velocity, _fields = backend._predict_motion(
            torch.zeros(2, 24),
            {
                "local_frame": torch.tensor([0, 1]),
                "duration_sec": torch.ones(2),
            },
            torch.zeros(2, 3),
            {
                "delta_p_match": torch.tensor([
                    [2.0, 0.0, 0.0],
                    [-2.0, 0.0, 0.0],
                ]),
                "delta_p_init": torch.tensor([
                    [2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]),
                "p_unmatched": torch.tensor([0.0, 1.0]),
                "matched_position": torch.zeros(2, 3),
                "motion_top1_probability": torch.ones(2),
                "motion_effective_support": torch.ones(2),
                "motion_reciprocal_probability": torch.ones(2),
                "motion_dustbin_similarity": torch.full((2,), 0.5),
                "motion_search_speed_mps": torch.full((2,), 5.0),
                "match_probability": torch.tensor([1.0, 0.0]),
            },
        )
        self.assertFalse(backend.init_conditioned_residual)
        self.assertEqual(backend.velocity_head.motion_refiner[0].in_features, 24)
        self.assertTrue(torch.equal(
            velocity,
            torch.tensor([[2.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        ))

    def test_v6_backend_uses_independent_raw_utonia_feature_and_dustbin(self):
        torch.manual_seed(23)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = ProposalInitializedVelocityGaussianBackend(
            proposal_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            proposal_dim=10,
        )
        feature = torch.randn(5, 24, requires_grad=True)
        utonia_feature = torch.randn(5, 10, requires_grad=True)
        token = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
        ])
        output = backend(
            feature,
            token,
            token.unsqueeze(1),
            torch.tensor([3, 5]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            motion_proposal_feature=utonia_feature,
        )
        item = output["batch_gaussians"][0]
        expected = item["velocity_init"] + item["velocity_offset"]
        self.assertTrue(torch.allclose(item["velocity"], expected))
        self.assertTrue(torch.allclose(
            item["p_unmatched"], 1.0 - item["match_probability"]
        ))
        item["velocity"].square().sum().backward()
        self.assertIsNotNone(utonia_feature.grad)
        self.assertEqual(backend.motion_proposal.descriptor.in_features, 10)
        self.assertEqual(
            backend.motion_proposal.descriptor_adapter[0].in_features, 10
        )
        self.assertIsNotNone(backend.motion_proposal.descriptor.weight.grad)
        self.assertGreater(
            float(backend.motion_proposal.descriptor.weight.grad.abs().sum()), 0.0
        )

    def test_sparse_local_rope_has_an_independent_metric_scale(self):
        refiner = SparseLocalTokenRefiner(
            dim=24,
            depth=1,
            num_heads=4,
            window_size=3,
            coord_scale=0.2,
            rope_base=10.0,
            rope_position_scale=4.0,
        )
        self.assertEqual(refiner.coord_scale, 0.2)
        self.assertEqual(refiner.rope_position_scale, 4.0)
        self.assertEqual(refiner.blocks[0].attn.rope.base, 10.0)



class V5StabilizedAttentionTest(unittest.TestCase):
    """QK-Norm and the motion-head RoPE band added in V5.

    Both target the V4 collapse measured in
    docs/dynamic_v4_attention_velocity_diagnosis.md: the motion softmax became a
    hard argmax (logit max 2e6) because nothing bounded |q||k| and the shared
    RoPE band could not separate keys at the 0.39 m anchor spacing.
    """

    @staticmethod
    def _run(module, position, feature):
        return module(
            feature,
            position,
            torch.tensor([2, 4]),
            torch.tensor([0, 0]),
            torch.zeros(4),
        )

    def test_v4_config_leaves_both_features_off(self):
        module = TimeConditionedParallelCrossAttention(
            temporal_cfg(layers=1, motion_head_count=2), dim=24
        )
        self.assertFalse(module.qk_norm)
        self.assertIsNone(module.q_norm)
        self.assertIsNone(module.motion_rope)
        self.assertEqual(module.attention_temperature_stats(), {})

    def test_qk_norm_makes_attention_invariant_to_projection_gain(self):
        """The V4 failure was q_proj growing 10x and inflating logits as its
        square. Under QK-Norm the projection's scale cannot reach the logit."""
        torch.manual_seed(5)
        position = torch.randn(4, 3) * 5.0
        feature = torch.randn(4, 24)
        for qk_norm, expect_invariant in ((True, True), (False, False)):
            module = TimeConditionedParallelCrossAttention(
                v5_temporal_cfg(layers=1, qk_norm=qk_norm, motion_rope=False),
                dim=24,
            ).eval()
            with torch.no_grad():
                before = self._run(module, position, feature)
                for layer in range(module.n_layers):
                    module.q_proj[layer].weight.mul_(100.0)
                    module.q_proj[layer].bias.mul_(100.0)
                after = self._run(module, position, feature)
            same = torch.allclose(before, after, atol=1.0e-5)
            self.assertEqual(same, expect_invariant, f"qk_norm={qk_norm}")

    def test_qk_norm_bound_matches_the_logged_temperature(self):
        module = TimeConditionedParallelCrossAttention(
            v5_temporal_cfg(layers=1), dim=24
        )
        stats = module.attention_temperature_stats()
        # gamma initialises to 1, so |q| = |k| = sqrt(head_dim) exactly and the
        # bound collapses to head_dim * head_dim**-0.5.
        self.assertAlmostEqual(stats["qk_gamma_q_layer0"], 1.0, places=6)
        self.assertAlmostEqual(
            stats["qk_max_logit_bound"], module.head_dim ** 0.5, places=4
        )
        with torch.no_grad():
            module.q_norm[0].weight.mul_(3.0)
        grown = module.attention_temperature_stats()
        self.assertAlmostEqual(
            grown["qk_max_logit_bound"], 3.0 * module.head_dim ** 0.5, places=4
        )

    def test_motion_band_resolves_anchor_spacing_the_feature_band_cannot(self):
        """1 - K(d)/K(0) is the whole positional margin the softmax gets."""
        module = TimeConditionedParallelCrossAttention(
            v5_temporal_cfg(layers=1), dim=24
        )

        def separation(rope, metres):
            offset = torch.tensor([[metres, 0.0, 0.0]])
            angles = rope.angles(offset)
            return float(1.0 - torch.cos(angles).mean())

        # The ratio is the dimension-independent claim. At the production
        # head_dim of 48 the absolute figures are 8.6e-4 vs 1.8e-1 at 0.39 m.
        spacing = 0.39
        feature_band = separation(module.rope, spacing)
        motion_band = separation(module.motion_rope, spacing)
        self.assertGreater(motion_band, 100.0 * feature_band)

    def test_motion_rope_applies_only_to_the_motion_heads(self):
        torch.manual_seed(7)
        module = TimeConditionedParallelCrossAttention(
            v5_temporal_cfg(layers=1, heads=4, motion_head_count=2), dim=24
        )
        position = torch.randn(4, 3) * 5.0
        x = torch.randn(4, module.num_heads, module.head_dim)
        rotated = module._rotate_heads(
            x, module.rope.angles(position), module.motion_rope.angles(position)
        )
        split = module.motion_head_count
        self.assertTrue(torch.allclose(
            rotated[:, split:],
            module.rope.rotate(x[:, split:], module.rope.angles(position)[:, None]),
            atol=1.0e-6,
        ))
        self.assertTrue(torch.allclose(
            rotated[:, :split],
            module.motion_rope.rotate(
                x[:, :split], module.motion_rope.angles(position)[:, None]
            ),
            atol=1.0e-6,
        ))
        # The two bands must actually differ, otherwise the test above is vacuous.
        self.assertFalse(torch.allclose(
            rotated[:, :split],
            module.rope.rotate(x[:, :split], module.rope.angles(position)[:, None]),
            atol=1.0e-4,
        ))

    def test_motion_rope_requires_motion_heads_and_both_keys(self):
        with self.assertRaises(ValueError):
            TimeConditionedParallelCrossAttention(
                v5_temporal_cfg(layers=1, motion_head_count=0), dim=24
            )
        partial = temporal_cfg(layers=1, motion_head_count=2)
        partial.motion_rope_base = 200.0
        with self.assertRaises(ValueError):
            TimeConditionedParallelCrossAttention(partial, dim=24)

    def test_v5_backend_keeps_the_v4_motion_contract(self):
        torch.manual_seed(11)
        cfg = attention_backend_cfg(dim=24, heads=4, motion_head_count=2)
        cfg.temporal.qk_norm = True
        cfg.temporal.motion_rope_base = 200.0
        cfg.temporal.motion_rope_position_scale = 4.0
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = AttentionInitializedVelocityGaussianBackend(
            cfg, gs_params, dim=24, offset_bound=0.5
        )
        self.assertTrue(backend.temporal.qk_norm)
        self.assertIsNotNone(backend.temporal.motion_rope)


class V9DenseProposalTest(unittest.TestCase):
    @staticmethod
    def _reference_dense(module, query, key, query_position, key_position,
                         duration=1.0):
        """Independent all-key softmax expectation, written out by hand."""
        squared_distance = (
            (query_position.unsqueeze(1) - key_position.unsqueeze(0))
            .square().sum(dim=-1)
        )
        penalty = squared_distance / (
            module.distance_prior_speed_mps * duration
        ) ** 2
        score = (query @ key.T) / module.temperature - penalty
        probability = torch.softmax(score, dim=-1)
        return probability, probability @ key_position

    def test_descriptor_is_projected_layernorm_then_l2_normalized(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(), dim=24
        ).eval()
        self.assertIsInstance(module.descriptor_norm, torch.nn.LayerNorm)
        self.assertEqual(module.descriptor_proj.in_features, 24)
        self.assertEqual(module.descriptor_proj.out_features, 6)

        feature = torch.randn(5, 24)
        descriptor = module._encode(feature)
        self.assertEqual(tuple(descriptor.shape), (5, 6))
        torch.testing.assert_close(
            descriptor.norm(dim=-1), torch.ones(5), atol=1e-6, rtol=1e-6
        )
        # Not the parameter-free V7.2 path: the projection is learned.
        expected = F.normalize(
            module.descriptor_proj(module.descriptor_norm(feature)), dim=-1
        )
        torch.testing.assert_close(descriptor, expected)
        self.assertGreater(
            sum(p.numel() for p in module.parameters()), 0
        )

    def test_forward_is_the_exact_all_key_softmax_expectation(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(score_chunk_size=1), dim=4
        ).eval()
        query = F.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), dim=-1)
        key = F.normalize(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [0.4, 0.0, 0.916515, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.2, 0.979796, 0.0, 0.0],
        ]), dim=-1)
        query_position = torch.zeros(1, 3)
        key_position = torch.tensor([
            [100.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [-50.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ])
        result = module._match_direction(
            query, key, query_position, key_position
        )

        probability, expected_position = self._reference_dense(
            module, query, key, query_position, key_position
        )
        torch.testing.assert_close(result["matched_position"], expected_position)
        torch.testing.assert_close(
            result["delta_p_init"], expected_position - query_position
        )
        torch.testing.assert_close(
            result["delta_p_init"], result["delta_p_match"]
        )
        # No Top-K selection survives in a dense readout.
        self.assertNotIn("motion_selected_index", result)
        torch.testing.assert_close(
            result["motion_top1_probability"], probability.max(dim=-1).values
        )
        expected_support = (
            -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
        ).exp()
        torch.testing.assert_close(
            result["motion_effective_support"], expected_support
        )

    def test_chunking_does_not_change_the_expectation(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(score_chunk_size=1), dim=8
        ).eval()
        query = F.normalize(torch.randn(7, 8), dim=-1)
        key = F.normalize(torch.randn(5, 8), dim=-1)
        query_position = torch.randn(7, 3) * 4.0
        key_position = torch.randn(5, 3) * 4.0

        chunked = module._match_direction(
            query, key, query_position, key_position
        )
        module.score_chunk_size = 64
        whole = module._match_direction(
            query, key, query_position, key_position
        )
        for name in ("matched_position", "motion_effective_support"):
            torch.testing.assert_close(
                chunked[name], whole[name], atol=1e-5, rtol=1e-5
            )

    def test_distance_prior_uses_a_five_meter_per_second_envelope(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(temperature=1.0), dim=3
        ).eval()
        self.assertEqual(module.distance_prior_speed_mps, 5.0)
        # Identical feature scores, so the readout is the distance prior alone.
        query = torch.tensor([[1.0, 0.0, 0.0]])
        key = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        query_position = torch.zeros(1, 3)
        key_position = torch.tensor([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        result = module._match_direction(
            query, key, query_position, key_position,
            duration_sec=torch.ones(1),
        )
        # (5/5)^2 = 1 and (10/5)^2 = 4 logits at a one-second interval; the
        # V7.2 30 m/s envelope would have charged 0.028 and 0.111.
        weight = torch.softmax(torch.tensor([-1.0, -4.0]), dim=-1)
        torch.testing.assert_close(
            result["matched_position"],
            (weight.unsqueeze(-1) * key_position).sum(dim=0, keepdim=True),
        )

        # Halving the interval halves the radius and quadruples the penalty.
        halved = module._match_direction(
            query, key, query_position, key_position,
            duration_sec=torch.full((1,), 0.5),
        )
        weight_half = torch.softmax(torch.tensor([-4.0, -16.0]), dim=-1)
        torch.testing.assert_close(
            halved["matched_position"],
            (weight_half.unsqueeze(-1) * key_position).sum(dim=0, keepdim=True),
        )

    def test_gradient_reaches_every_key_including_low_probability_ones(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(score_chunk_size=2), dim=4
        ).eval()
        query_raw = torch.randn(2, 4, requires_grad=True)
        key_raw = torch.randn(6, 4, requires_grad=True)
        query = F.normalize(query_raw, dim=-1)
        key = F.normalize(key_raw, dim=-1)
        query_position = torch.zeros(2, 3)
        key_position = torch.randn(6, 3)

        result = module._match_direction(
            query, key, query_position, key_position
        )
        result["delta_p_init"].sum().backward()

        probability, _ = self._reference_dense(
            module, query.detach(), key.detach(), query_position, key_position
        )
        weakest = int(probability.sum(dim=0).argmin())
        self.assertGreater(float(key_raw.grad[weakest].abs().sum()), 0.0)
        self.assertTrue(bool((key_raw.grad.abs().sum(dim=-1) > 0).all()))

    def test_pairing_is_endpoint_symmetric_and_covers_both_frames(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(score_chunk_size=2), dim=8
        ).eval()
        feature = torch.randn(7, 8)
        position = torch.randn(7, 3) * 3.0
        token_offset = torch.tensor([3, 7])
        frame_batch_idx = torch.tensor([0, 0])
        duration = torch.full((7,), 0.8)

        fields = module(
            feature, position, token_offset, frame_batch_idx, duration
        )
        self.assertEqual(tuple(fields["delta_p_init"].shape), (7, 3))

        descriptor = module._encode(feature)
        forward = module._match_direction(
            descriptor[:3], descriptor[3:], position[:3], position[3:],
            duration[:3],
        )
        reverse = module._match_direction(
            descriptor[3:], descriptor[:3], position[3:], position[:3],
            duration[3:],
        )
        torch.testing.assert_close(
            fields["delta_p_init"],
            torch.cat([forward["delta_p_init"], reverse["delta_p_init"]]),
        )

    def test_an_empty_endpoint_frame_reports_a_clear_error(self):
        module = ProjectedDenseMotionProposal(v9_proposal_cfg(), dim=8).eval()
        descriptor = module._encode(torch.randn(4, 8))
        for label, args in (
            ("empty keys", (descriptor, descriptor[:0],
                            torch.randn(4, 3), torch.zeros(0, 3),
                            torch.full((4,), 0.8))),
            ("empty queries", (descriptor[:0], descriptor,
                               torch.zeros(0, 3), torch.randn(4, 3),
                               torch.zeros(0))),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "tokens in both endpoints"
            ):
                module._match_direction(*args)

    def test_wrapper_gate_only_touches_matchers_that_opt_in(self):
        proposal = ProjectedDenseMotionProposal(v9_proposal_cfg(), dim=8)
        wrapper = SimpleNamespace(
            p2g_model=SimpleNamespace(
                dynamic_backend=SimpleNamespace(motion_proposal=proposal)
            )
        )
        ModelWrapper._set_proposal_diagnostics(wrapper, False)
        self.assertFalse(proposal.collect_diagnostics)
        ModelWrapper._set_proposal_diagnostics(wrapper, True)
        self.assertTrue(proposal.collect_diagnostics)

        # V6/V7/V7.2/V8 matchers expose no such flag and must be left exactly
        # as their checkpoints ran: statistics on every step.
        legacy = StraightThroughTop4MotionProposal(v7_2_proposal_cfg(), dim=8)
        legacy_wrapper = SimpleNamespace(
            p2g_model=SimpleNamespace(
                dynamic_backend=SimpleNamespace(motion_proposal=legacy)
            )
        )
        ModelWrapper._set_proposal_diagnostics(legacy_wrapper, False)
        self.assertFalse(hasattr(legacy, "collect_diagnostics"))

        # Non-dynamic variants have no dynamic_backend at all.
        ModelWrapper._set_proposal_diagnostics(
            SimpleNamespace(p2g_model=SimpleNamespace()), False
        )

    def test_diagnostics_can_be_skipped_without_changing_the_readout(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(score_chunk_size=2), dim=8
        ).eval()
        feature = torch.randn(7, 8)
        position = torch.randn(7, 3) * 3.0
        token_offset = torch.tensor([3, 7])
        frame_batch_idx = torch.tensor([0, 0])
        duration = torch.full((7,), 0.8)

        self.assertTrue(module.collect_diagnostics)
        full = module(
            feature, position, token_offset, frame_batch_idx, duration
        )
        module.collect_diagnostics = False
        gated = module(
            feature, position, token_offset, frame_batch_idx, duration
        )

        # The rendered quantities are bit-identical; only the statistics go.
        for name in ("delta_p_init", "delta_p_match", "matched_position"):
            torch.testing.assert_close(gated[name], full[name], rtol=0, atol=0)
        self.assertEqual(
            set(gated), {"delta_p_match", "delta_p_init", "matched_position"}
        )
        for name in (
            "motion_top1_probability",
            "motion_effective_support",
            "motion_candidate_probability_mass",
            "motion_selected_distance_prior_penalty",
        ):
            self.assertIn(name, full)
            self.assertNotIn(name, gated)

    def test_gated_diagnostics_still_train_the_descriptor(self):
        module = ProjectedDenseMotionProposal(
            v9_proposal_cfg(score_chunk_size=2), dim=8
        ).train()
        module.collect_diagnostics = False
        feature = torch.randn(7, 8, requires_grad=True)
        fields = module(
            feature,
            torch.randn(7, 3) * 3.0,
            torch.tensor([3, 7]),
            torch.tensor([0, 0]),
            torch.full((7,), 0.8),
        )
        fields["delta_p_init"].square().sum().backward()
        self.assertGreater(
            float(module.descriptor_proj.weight.grad.abs().sum()), 0.0
        )

    def test_backend_matches_on_the_refined_feature_without_a_query_warp(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = PostAttentionProposalVelocityGaussianBackend(
            v9_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
        )
        self.assertIsInstance(
            backend.motion_proposal, ProjectedDenseMotionProposal
        )
        self.assertIsInstance(
            backend.velocity_head, FeatureOnlyVelocityOffsetHead
        )
        self.assertTrue(backend.temporal.use_time_embedding)
        self.assertEqual(backend.temporal.motion_head_count, 0)
        self.assertEqual(backend.motion_proposal.dim, 24)

        class CaptureTemporal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.token_time_coordinate = None
                self.query_position_ref = "unset"

            def forward(self, feature, _position, _offset, _batch_idx,
                        token_time_coordinate,
                        query_position_ref=None, **_kwargs):
                self.token_time_coordinate = token_time_coordinate
                self.query_position_ref = query_position_ref
                return feature + 1.0

        class CaptureProposal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.feature = None

            def forward(self, feature, *_args, **_kwargs):
                self.feature = feature
                delta = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]])
                return {"delta_p_match": delta, "delta_p_init": delta}

        backend.temporal = CaptureTemporal()
        backend.motion_proposal = CaptureProposal()
        geometry = {
            "token_ref": torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            "local_frame": torch.tensor([0, 1]),
            "duration_sec": torch.ones(2),
            "time_sec": torch.tensor([0.0, 1.0]),
        }
        fused = torch.zeros(2, 24)
        refined, fields = backend._temporal_refine(
            fused,
            geometry,
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            motion_proposal_feature=torch.randn(2, 8),
        )

        # The matcher consumes the refined feature, not the fused input and not
        # the raw pre-attention Utonia stream V6/V7 pass in.
        torch.testing.assert_close(backend.motion_proposal.feature, refined)
        self.assertFalse(torch.equal(backend.motion_proposal.feature, fused))
        # No detached initializer warp: queries stay at observed coordinates.
        self.assertIsNone(backend.temporal.query_position_ref)
        torch.testing.assert_close(
            backend.temporal.token_time_coordinate,
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            fields["velocity_match"],
            torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )

        velocity, motion_fields = backend._predict_motion(
            refined, geometry, None, fields
        )
        # Zero-initialized offset head, so a fresh model renders exactly v_init.
        torch.testing.assert_close(
            motion_fields["velocity_offset"], torch.zeros(2, 3)
        )
        torch.testing.assert_close(velocity, motion_fields["velocity_init"])
        torch.testing.assert_close(
            velocity, torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        )

    def test_backend_rejects_reserved_motion_heads_and_time_free_attention(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        cfg = v9_backend_cfg()
        cfg.temporal.motion_head_count = 2
        with self.assertRaisesRegex(ValueError, "motion_head_count"):
            PostAttentionProposalVelocityGaussianBackend(
                cfg, gs_params, dim=24, offset_bound=0.8
            )

        cfg = v9_backend_cfg()
        cfg.temporal.use_time_embedding = False
        with self.assertRaisesRegex(ValueError, "time embedding"):
            PostAttentionProposalVelocityGaussianBackend(
                cfg, gs_params, dim=24, offset_bound=0.8
            )

    def test_backend_emits_one_velocity_per_token_end_to_end(self):
        torch.manual_seed(0)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = PostAttentionProposalVelocityGaussianBackend(
            v9_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
        )
        n_per_frame = 5
        fused = torch.randn(2 * n_per_frame, 24)
        token_position = torch.randn(2 * n_per_frame, 3) * 3.0
        output = backend(
            fused,
            token_position,
            token_position.clone().unsqueeze(1),
            torch.tensor([n_per_frame, 2 * n_per_frame]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
        )
        item = output["batch_gaussians"][0]
        self.assertEqual(
            tuple(item["velocity"].shape), (2 * n_per_frame, 3)
        )
        self.assertTrue(bool(torch.isfinite(item["velocity"]).all()))
        # A fresh run starts at v_final = v_init, and v_init comes from the
        # dense correspondence rather than a zero head.
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros(2 * n_per_frame, 3)
        )
        torch.testing.assert_close(item["velocity"], item["velocity_init"])
        for name in (
            "motion_top1_probability",
            "motion_effective_support",
            "motion_selected_distance_prior_penalty",
        ):
            self.assertIn(name, item)

        item["velocity"].sum().backward()
        self.assertIsNotNone(
            backend.motion_proposal.descriptor_proj.weight.grad
        )
        self.assertGreater(
            float(
                backend.motion_proposal.descriptor_proj.weight.grad.abs().sum()
            ),
            0.0,
        )


class V10LayerWeightedAttentionTest(unittest.TestCase):
    def test_four_qk_channels_equal_the_quadratic_distance_bias(self):
        module = LayerWeightedDistanceBiasCrossAttention(
            v10_temporal_cfg(layers=2), dim=24
        )
        query = torch.tensor([
            [1.0, -2.0, 0.5],
            [-3.0, 0.5, 2.0],
        ])
        key = torch.tensor([
            [2.0, 1.0, -0.5],
            [-1.0, -2.0, 1.0],
            [4.0, 0.0, 3.0],
        ])
        radius = torch.full((2,), 15.0)
        key_radius = torch.full((3,), 15.0)
        query_feature = module._distance_features(query, radius)
        key_feature = module._distance_key_features(key, key_radius)
        augmented_probability = torch.softmax(
            (query_feature @ key_feature.T) * module.scale,
            dim=-1,
        )
        expected_probability = torch.softmax(
            -torch.cdist(query, key).square() / 15.0 ** 2,
            dim=-1,
        )
        torch.testing.assert_close(
            augmented_probability, expected_probability, atol=1.0e-6, rtol=1.0e-6
        )

        # A coordinate-free global query gets no distance preference.
        global_feature = module._distance_features(
            torch.zeros(1, 3), torch.ones(1), is_global=True
        )
        torch.testing.assert_close(
            torch.softmax((global_feature @ key_feature.T) * module.scale, dim=-1),
            torch.full((1, 3), 1.0 / 3.0),
        )

    def test_layer_softmax_is_exactly_the_weighted_map_readout(self):
        layer_matches = torch.tensor([
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0], [6.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 4.0, 0.0], [0.0, 8.0, 0.0]],
        ])
        frame_logits = torch.log(torch.tensor([
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 1.0],
        ]))
        position = torch.tensor([
            [0.5, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        matched, displacement, logits, weights = (
            LayerWeightedDistanceBiasCrossAttention._combine_layer_matches(
                layer_matches,
                frame_logits,
                torch.tensor([2, 1]),
                position,
            )
        )
        expected_weights = torch.tensor([
            [0.25, 0.5, 0.25],
            [0.25, 0.5, 0.25],
            [0.5, 0.25, 0.25],
        ])
        torch.testing.assert_close(weights, expected_weights)
        torch.testing.assert_close(
            matched,
            (expected_weights.unsqueeze(-1) * layer_matches).sum(dim=1),
        )
        torch.testing.assert_close(displacement, matched - position)
        torch.testing.assert_close(logits[:2], frame_logits[0].expand(2, -1))
        torch.testing.assert_close(logits[2], frame_logits[1])

    def test_global_token_is_query_only_and_every_layer_receives_gradient(self):
        torch.manual_seed(41)
        module = LayerWeightedDistanceBiasCrossAttention(
            v10_temporal_cfg(layers=3), dim=24
        )
        self.assertFalse(hasattr(module, "rope"))
        self.assertEqual(tuple(module.global_token.shape), (1, 24))
        self.assertEqual(
            sum(isinstance(layer, torch.nn.Linear)
                for layer in module.layer_score_mlp),
            2,
        )
        self.assertEqual(
            sum(isinstance(layer, torch.nn.SiLU)
                for layer in module.layer_score_mlp),
            1,
        )

        captured_counts = []
        original_sdpa = module._sdpa_attention

        def capture_sdpa(q, k, v, q_counts, k_counts):
            captured_counts.append((q_counts.clone(), k_counts.clone()))
            return original_sdpa(q, k, v, q_counts, k_counts)

        module._sdpa_attention = capture_sdpa
        feature = torch.randn(7, 24, requires_grad=True)
        position = torch.randn(7, 3)
        refined, fields = module(
            feature,
            position,
            torch.tensor([3, 7]),
            torch.tensor([0, 0]),
            torch.tensor([0.0, 0.0, 0.0, 0.8, 0.8, 0.8, 0.8]),
            torch.full((7,), 0.8),
        )
        self.assertEqual(tuple(refined.shape), (7, 24))
        self.assertEqual(tuple(fields["motion_layer_weights"].shape), (7, 3))
        torch.testing.assert_close(
            fields["motion_layer_weights"],
            torch.full((7, 3), 1.0 / 3.0),
        )
        for q_counts, k_counts in captured_counts:
            # One learned query is appended to each frame; K/V remain ordinary
            # coordinate-bearing endpoint tokens.
            self.assertTrue(torch.equal(q_counts, torch.tensor([4, 5])))
            self.assertTrue(torch.equal(k_counts, torch.tensor([4, 3])))

        loss = fields["matched_position"].square().sum() + refined.square().mean()
        loss.backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        for layer in range(module.n_layers):
            self.assertGreater(
                float(module.q_proj[layer].weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(module.k_proj[layer].weight.grad.abs().sum()), 0.0
            )
        self.assertGreater(
            float(module.layer_score_mlp[-1].weight.grad.abs().sum()), 0.0
        )

    @staticmethod
    def _adaptive_seed_bank(position):
        bank = position.new_zeros(position.shape[0], 3, 3, 3)
        delta = torch.zeros_like(bank)
        choices = (
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            torch.tensor([
                [-0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]),
        )
        for index, offset in enumerate(choices):
            k = index + 1
            offset = offset.to(device=position.device, dtype=position.dtype)
            bank[:, index, :k] = position[:, None, :] + offset[None]
            delta[:, index, :k] = offset[None]
        return bank, delta

    def test_backend_routes_k3_and_shares_one_token_velocity(self):
        torch.manual_seed(43)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = LayerWeightedAttentionVelocityGaussianBackend(
            v10_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=v10_gaussian_count_cfg(),
        ).eval()
        self.assertIsInstance(
            backend.temporal, LayerWeightedDistanceBiasCrossAttention
        )
        self.assertIsInstance(backend.gaussian_output_norm, torch.nn.LayerNorm)
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertEqual(backend.gaussian_head.k_max, 3)
        self.assertEqual(len(backend.gaussian_head.k_heads), 3)
        self.assertIsInstance(backend.velocity_head, FeatureOnlyVelocityOffsetHead)
        self.assertIsInstance(
            backend.velocity_head.feature_norm, torch.nn.LayerNorm
        )
        with torch.no_grad():
            backend.gaussian_head.count_predictor[-1].bias.copy_(
                torch.tensor([-10.0, -10.0, 10.0])
            )

        n_per_frame = 4
        feature = torch.randn(2 * n_per_frame, 24)
        position = torch.randn(2 * n_per_frame, 3) * 2.0
        seed, seed_delta = self._adaptive_seed_bank(position)
        output = backend(
            feature,
            position,
            seed,
            torch.tensor([n_per_frame, 2 * n_per_frame]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        item = output["batch_gaussians"][0]
        self.assertEqual(tuple(item["motion_layer_weights"].shape), (24, 3))
        torch.testing.assert_close(
            item["motion_layer_weights"].sum(dim=-1), torch.ones(24)
        )
        self.assertTrue(torch.equal(item["selected_k"], torch.full((24,), 3)))
        self.assertTrue(torch.equal(
            item["source_token_index"], torch.arange(8).repeat_interleave(3)
        ))
        shared_velocity = item["velocity"].reshape(8, 3, 3)
        torch.testing.assert_close(
            shared_velocity, shared_velocity[:, :1].expand_as(shared_velocity)
        )
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        torch.testing.assert_close(item["velocity"], item["velocity_init"])
        torch.testing.assert_close(
            item["velocity"],
            item["velocity_init"] + item["velocity_offset"],
        )
        stats = ModelWrapper._motion_proposal_statistics([item])
        self.assertIn("motion_layer_weight_entropy_mean", stats)
        self.assertIn("motion_layer_weight_max_mean", stats)
        for layer in range(3):
            self.assertIn(f"motion_layer_weight_l{layer}_mean", stats)

    def test_seed_delta_bank_rotates_without_pose_translation(self):
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = LayerWeightedAttentionVelocityGaussianBackend(
            v10_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=v10_gaussian_count_cfg(),
        )
        position = torch.tensor([
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        seed, seed_delta = self._adaptive_seed_bank(position)
        poses = torch.eye(4).repeat(2, 1, 1)
        poses[1, :3, :3] = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        poses[1, :3, 3] = torch.tensor([10.0, -4.0, 2.0])
        geometry = backend._reference_geometry_and_time(
            position,
            seed,
            torch.tensor([1, 2]),
            torch.tensor([0, 0]),
            [poses],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        # Translation cancels from a delta; +x rotates to +y in frame 1.
        torch.testing.assert_close(
            geometry["seed_delta_ref"][1, 2, 2],
            torch.tensor([0.0, 0.2, 0.0]),
        )
        # Padding slots remain zero rather than becoming a pose translation.
        torch.testing.assert_close(
            geometry["seed_delta_ref"][1, 0, 1], torch.zeros(3)
        )

    def test_render_loss_trains_k_router_without_a_budget_loss(self):
        torch.manual_seed(47)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = LayerWeightedAttentionVelocityGaussianBackend(
            v10_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=v10_gaussian_count_cfg(),
        ).train()

        def deterministic_ste(logits):
            soft = torch.softmax(logits / backend.gaussian_head.gumbel_tau, dim=-1)
            selected = torch.arange(logits.shape[0], device=logits.device) % 3
            hard = F.one_hot(selected, num_classes=3).to(logits.dtype)
            return soft + (hard - soft).detach()

        backend.gaussian_head._gumbel_selection = deterministic_ste
        n_per_frame = 3
        feature = torch.randn(2 * n_per_frame, 24, requires_grad=True)
        position = torch.randn(2 * n_per_frame, 3)
        seed, seed_delta = self._adaptive_seed_bank(position)
        output = backend(
            feature,
            position,
            seed,
            torch.tensor([n_per_frame, 2 * n_per_frame]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        self.assertTrue(torch.equal(
            output["routing_stats"]["selected_k"],
            torch.tensor([1, 2, 3, 1, 2, 3]),
        ))
        self.assertFalse(output["routing_stats"]["k_logits"].requires_grad)
        self.assertNotIn("routing_budget_logits", output)
        item = output["batch_gaussians"][0]
        self.assertEqual(item["opacity"].shape[0], 12)
        torch.sigmoid(item["opacity"]).sum().backward()
        router_grad = backend.gaussian_head.count_predictor[-1].weight.grad
        self.assertIsNotNone(router_grad)
        self.assertGreater(float(router_grad.abs().sum()), 0.0)
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)


class V11MaxSpeedBarrierAttentionTest(unittest.TestCase):
    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        total = feature.shape[0]
        time = torch.cat([
            torch.zeros(first_frame_tokens),
            torch.full((total - first_frame_tokens,), duration),
        ])
        return (
            feature,
            position,
            torch.tensor([first_frame_tokens, total]),
            torch.tensor([0, 0]),
            time,
            torch.full((total,), duration),
        )

    def test_barrier_is_free_inside_the_reachable_radius(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        query = torch.zeros(1, 3)
        # 0.5R, R, 1.5R, 2R and 3R at a 30 m radius (30 m/s over 1.0 s).
        key = torch.tensor([
            [15.0, 0.0, 0.0], [30.0, 0.0, 0.0], [45.0, 0.0, 0.0],
            [60.0, 0.0, 0.0], [90.0, 0.0, 0.0],
        ])
        bias = module._barrier_bias(query, key, 30.0)
        torch.testing.assert_close(
            bias, torch.tensor([[0.0, 0.0, 1.0, 4.0, 16.0]])
        )
        # The radius follows the physical endpoint interval, not the distance.
        halved = module._barrier_bias(query, key, 15.0)
        torch.testing.assert_close(
            halved, torch.tensor([[0.0, 4.0, 16.0, 36.0, 100.0]])
        )

    def test_fused_score_matches_the_reference_barrier_form(self):
        """`(a/R^2) relu(d-R)^2` in the GEMM must equal `a relu(d/R-1)^2`."""
        torch.manual_seed(13)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        query = torch.randn(5, module.head_dim)
        key = torch.randn(7, module.head_dim)
        value = torch.randn(7, module.head_dim + 3)
        query_position = torch.randn(5, 3) * 40.0
        key_position = torch.randn(7, 3) * 40.0
        radius = 21.0

        feature, matched = module._barrier_chunk(
            query, key, value, query_position, key_position, radius
        )
        reference_score = (query @ key.transpose(0, 1)) * module.scale
        reference_score = reference_score - module._barrier_bias(
            query_position, key_position, radius
        )
        expected = torch.softmax(reference_score, dim=-1) @ value
        torch.testing.assert_close(
            feature, expected[..., :module.head_dim], atol=1.0e-5, rtol=1.0e-5
        )
        torch.testing.assert_close(
            matched, expected[..., module.head_dim:], atol=1.0e-5, rtol=1.0e-5
        )

    def test_mixed_durations_within_a_frame_are_rejected(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        )
        feature = torch.randn(6, 24)
        position = torch.randn(6, 3)
        duration = torch.tensor([0.8, 0.8, 0.5, 0.8, 0.8, 0.8])
        with self.assertRaisesRegex(ValueError, "one endpoint interval"):
            module(
                feature,
                position,
                torch.tensor([3, 6]),
                torch.tensor([0, 0]),
                torch.tensor([0.0, 0.0, 0.0, 0.8, 0.8, 0.8]),
                duration,
            )

    def test_match_head_readout_equals_an_explicit_biased_softmax(self):
        torch.manual_seed(3)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1, match_chunk_size=4), dim=24
        ).eval()
        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 30.0

        captured = {}
        original = module._barrier_attention

        def capture(query, key, value, query_position, key_position,
                    q_count_values, k_count_values, frame_radius_values):
            captured.update(
                query=query, key=key, query_position=query_position,
                key_position=key_position, q_counts=q_count_values,
                k_counts=k_count_values, radii=frame_radius_values,
            )
            return original(
                query, key, value, query_position, key_position,
                q_count_values, k_count_values, frame_radius_values,
            )

        module._barrier_attention = capture
        with torch.no_grad():
            _refined, fields = module(
                *self._forward_args(feature, position, 4, duration=0.7)
            )

        # Only head 0 ever reaches the coordinate readout.
        self.assertEqual(tuple(captured["query"].shape), (9, module.head_dim))
        start = 0
        key_start = 0
        # 30 m/s over the 0.7 s endpoint interval.
        self.assertEqual(len(captured["radii"]), 2)
        for radius in captured["radii"]:
            self.assertAlmostEqual(radius, 21.0, places=4)
        for q_count, k_count, radius in zip(
            captured["q_counts"], captured["k_counts"], captured["radii"]
        ):
            rows = slice(start, start + q_count)
            keys = slice(key_start, key_start + k_count)
            query_position = captured["query_position"][rows]
            key_position = captured["key_position"][keys]
            score = (
                captured["query"][rows] @ captured["key"][keys].transpose(0, 1)
            ) * module.scale - 4.0 * torch.relu(
                torch.cdist(query_position, key_position) / radius - 1.0
            ).square()
            expected = torch.softmax(score, dim=-1) @ key_position
            torch.testing.assert_close(
                fields["matched_position"][rows], expected, atol=1.0e-4,
                rtol=1.0e-4,
            )
            start += q_count
            key_start += k_count
        torch.testing.assert_close(
            fields["delta_p_init"], fields["matched_position"] - position
        )

    def test_query_chunking_changes_neither_output_nor_gradient(self):
        reference = None
        for chunk in (1, 3, 1000):
            torch.manual_seed(7)
            module = MaxSpeedBarrierLayerWeightedCrossAttention(
                v11_temporal_cfg(layers=2, match_chunk_size=chunk), dim=24
            ).train()
            torch.manual_seed(11)
            feature = torch.randn(11, 24, requires_grad=True)
            position = torch.randn(11, 3) * 25.0
            refined, fields = module(
                *self._forward_args(feature, position, 5, duration=0.9)
            )
            (
                refined.square().sum()
                + fields["matched_position"].square().sum()
            ).backward()
            observed = (
                refined.detach(),
                fields["matched_position"].detach(),
                feature.grad.clone(),
                module.q_proj[0].weight.grad.clone(),
            )
            if reference is None:
                reference = observed
                continue
            for expected, actual in zip(reference, observed):
                torch.testing.assert_close(
                    actual, expected, atol=1.0e-5, rtol=1.0e-4
                )

    def test_rope_covers_every_head_but_the_match_head(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=2, heads=4), dim=24
        ).eval()
        self.assertEqual(module.rope_head_count, 3)
        rotated_shapes = []
        original_rotate = module.rope.rotate

        def capture(x, angles):
            rotated_shapes.append(tuple(x.shape))
            return original_rotate(x, angles)

        module.rope.rotate = capture
        feature = torch.randn(6, 24)
        position = torch.randn(6, 3) * 10.0
        with torch.no_grad():
            module(*self._forward_args(feature, position, 3))
        # Two layers x (Q, K), each carrying heads 1..3 only.
        self.assertEqual(len(rotated_shapes), 4)
        for shape in rotated_shapes:
            self.assertEqual(shape, (6, 3, module.head_dim))

    def test_layer_mixture_is_uniform_scalars_that_receive_gradient(self):
        torch.manual_seed(41)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=3), dim=24
        )
        self.assertFalse(hasattr(module, "global_token"))
        self.assertFalse(hasattr(module, "layer_score_mlp"))
        self.assertEqual(tuple(module.layer_logits.shape), (3,))
        torch.testing.assert_close(module.layer_logits, torch.zeros(3))

        feature = torch.randn(7, 24, requires_grad=True)
        position = torch.randn(7, 3) * 12.0
        refined, fields = module(
            *self._forward_args(feature, position, 3)
        )
        self.assertEqual(tuple(refined.shape), (7, 24))
        # One shared mixture, broadcast to every token rather than predicted.
        torch.testing.assert_close(
            fields["motion_layer_weights"], torch.full((7, 3), 1.0 / 3.0)
        )
        torch.testing.assert_close(
            fields["motion_layer_logits"], torch.zeros(7, 3)
        )

        loss = fields["matched_position"].square().sum() + refined.square().mean()
        loss.backward()
        self.assertGreater(float(feature.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.layer_logits.grad.abs().sum()), 0.0)
        for layer in range(module.n_layers):
            self.assertGreater(
                float(module.q_proj[layer].weight.grad.abs().sum()), 0.0
            )
            self.assertGreater(
                float(module.k_proj[layer].weight.grad.abs().sum()), 0.0
            )

    def test_layer_weights_select_a_single_layer_readout(self):
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=3), dim=24
        )
        with torch.no_grad():
            module.layer_logits.copy_(torch.tensor([-20.0, 20.0, -20.0]))
        layer_matches = torch.tensor([
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0], [6.0, 1.0, 0.0]],
        ])
        position = torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.0, 0.0]])
        matched, displacement, logits, weights = module._combine_layer_matches(
            layer_matches, position
        )
        torch.testing.assert_close(matched, layer_matches[:, 1], atol=1.0e-6,
                                   rtol=1.0e-6)
        torch.testing.assert_close(displacement, matched - position)
        torch.testing.assert_close(weights[:, 1], torch.ones(2), atol=1.0e-6,
                                   rtol=1.0e-6)
        torch.testing.assert_close(
            logits, module.layer_logits.expand(2, -1)
        )

    def test_config_rejects_a_single_head_and_a_missing_barrier(self):
        with self.assertRaises(ValueError):
            MaxSpeedBarrierLayerWeightedCrossAttention(
                v11_temporal_cfg(layers=1, heads=1), dim=24
            )
        no_qk_norm = v11_temporal_cfg(layers=1)
        no_qk_norm.qk_norm = False
        with self.assertRaises(ValueError):
            MaxSpeedBarrierLayerWeightedCrossAttention(no_qk_norm, dim=24)
        wrong_encoding = v11_temporal_cfg(layers=1)
        wrong_encoding.position_encoding = "distance_bias"
        with self.assertRaises(ValueError):
            MaxSpeedBarrierLayerWeightedCrossAttention(wrong_encoding, dim=24)

    def test_backend_shares_one_token_velocity_across_adaptive_gaussians(self):
        torch.manual_seed(29)
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = MaxSpeedBarrierVelocityGaussianBackend(
            v11_backend_cfg(), gs_params, dim=24, offset_bound=0.8,
            gaussian_count_cfg=v10_gaussian_count_cfg(),
        ).eval()
        self.assertIsInstance(
            backend.temporal, MaxSpeedBarrierLayerWeightedCrossAttention
        )
        self.assertEqual(backend.gaussian_head.count_mode, "learned_gumbel")
        self.assertEqual(backend.gaussian_head.k_max, 3)
        # V11 conditions the offset on the initializer it corrects.
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertEqual(backend.velocity_head.velocity_embedding_dim, 32)
        self.assertTrue(backend.velocity_head.detach_init_condition)
        encoder = backend.velocity_head.velocity_encoder
        self.assertEqual(encoder[0].in_features, 3)
        self.assertEqual(encoder[-1].out_features, 32)
        self.assertEqual(
            backend.velocity_head.motion_refiner[0].in_features, 24 + 32
        )
        with torch.no_grad():
            backend.gaussian_head.count_predictor[-1].bias.copy_(
                torch.tensor([-10.0, -10.0, 10.0])
            )

        n_per_frame = 4
        feature = torch.randn(2 * n_per_frame, 24)
        position = torch.randn(2 * n_per_frame, 3) * 2.0
        seed, seed_delta = (
            V10LayerWeightedAttentionTest._adaptive_seed_bank(position)
        )
        output = backend(
            feature,
            position,
            seed,
            torch.tensor([n_per_frame, 2 * n_per_frame]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
            seed_delta_sensor=seed_delta,
        )
        item = output["batch_gaussians"][0]
        self.assertTrue(torch.equal(item["selected_k"], torch.full((24,), 3)))
        shared_velocity = item["velocity"].reshape(8, 3, 3)
        torch.testing.assert_close(
            shared_velocity, shared_velocity[:, :1].expand_as(shared_velocity)
        )
        # zero_init keeps a fresh run at v_total = v_init.
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        torch.testing.assert_close(
            item["velocity"], item["velocity_init"] + item["velocity_offset"]
        )
        # v_init is the layer-mixed displacement over the signed interval.
        torch.testing.assert_close(
            item["velocity_init"] * item["pair_delta_t_sec"].unsqueeze(-1),
            item["delta_p_init"],
        )
        stats = ModelWrapper._motion_proposal_statistics([item])
        self.assertIn("motion_layer_weight_entropy_mean", stats)
        for layer in range(3):
            self.assertIn(f"motion_layer_weight_l{layer}_mean", stats)

    def test_barrier_caps_v_init_near_the_configured_top_speed(self):
        """A token with no reachable key still cannot outrun the barrier much."""
        torch.manual_seed(5)
        module = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=1), dim=24
        ).eval()
        duration = 1.0
        # Frame 0 holds one query at the origin; frame 1 holds keys at 20 m
        # (inside the 30 m radius) and 200 m (far outside it).
        position = torch.tensor([
            [0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [200.0, 0.0, 0.0],
        ])
        feature = torch.randn(3, 24)
        with torch.no_grad():
            _refined, fields = module(
                *self._forward_args(feature, position, 1, duration=duration)
            )
        speed = float(fields["delta_p_init"][0].norm() / duration)
        self.assertLess(speed, 30.0)


class V11_1GroupedGainSingleGaussianTest(unittest.TestCase):
    """V11.1 = V11's stack, a two-group QK gain, one Gaussian per token."""

    @staticmethod
    def _forward_args(feature, position, first_frame_tokens, duration=0.8):
        total = feature.shape[0]
        time = torch.cat([
            torch.zeros(first_frame_tokens),
            torch.full((total - first_frame_tokens,), duration),
        ])
        return (
            feature,
            position,
            torch.tensor([first_frame_tokens, total]),
            torch.tensor([0, 0]),
            time,
            torch.full((total,), duration),
        )

    def test_the_gain_carries_one_row_per_head_group(self):
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=2), dim=24
        )
        for layer in range(2):
            for norm in (module.q_norm[layer], module.k_norm[layer]):
                self.assertIsInstance(norm, GroupedHeadRMSNorm)
                self.assertEqual(norm.num_groups, 2)
                self.assertEqual(
                    tuple(norm.weight.shape), (2, module.head_dim)
                )
                # Group 0 is the match head; every RoPE head shares group 1.
                self.assertEqual(
                    norm.group_of_head.tolist(), [0, 1, 1, 1]
                )

    def test_a_uniform_gain_reproduces_the_v11_readout(self):
        torch.manual_seed(0)
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=2), dim=24
        ).eval()
        reference = MaxSpeedBarrierLayerWeightedCrossAttention(
            v11_temporal_cfg(layers=2), dim=24
        ).eval()
        state = module.state_dict()
        collapsed = {}
        for name, value in state.items():
            if ".weight" in name and (
                name.startswith("q_norm") or name.startswith("k_norm")
            ):
                # Both rows equal, so the split gain is the shared gain.
                value = value[0]
            collapsed[name] = value
        reference.load_state_dict(collapsed, strict=True)

        feature = torch.randn(9, 24)
        position = torch.randn(9, 3) * 4.0
        args = self._forward_args(feature, position, 4)
        with torch.no_grad():
            refined, fields = module(*args)
            ref_refined, ref_fields = reference(*args)
        # Not bitwise: nn.RMSNorm runs a fused kernel while the grouped gain
        # does the same arithmetic with an explicit rsqrt and an index gather,
        # so the two reduce in a different order. On a real 28,760-token batch
        # the readouts agree to a relative 2.7e-06, against a 1.07 m response
        # to actually scaling the match group's gain.
        torch.testing.assert_close(refined, ref_refined)
        self.assertEqual(sorted(fields), sorted(ref_fields))
        for key in ("delta_p_init", "delta_p_match", "matched_position"):
            torch.testing.assert_close(fields[key], ref_fields[key])

    def test_each_group_moves_under_its_own_gradient(self):
        torch.manual_seed(1)
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=1), dim=24
        )
        feature = torch.randn(8, 24, requires_grad=True)
        position = torch.randn(8, 3) * 3.0
        _refined, fields = module(*self._forward_args(feature, position, 4))
        fields["delta_p_init"].square().sum().backward()
        gain = module.q_norm[0].weight.grad
        self.assertIsNotNone(gain)
        self.assertEqual(tuple(gain.shape), (2, module.head_dim))
        # The readout only passes through head 0, so only its row is charged
        # by a displacement objective. That separation is the whole point.
        self.assertGreater(float(gain[0].abs().sum()), 0.0)
        self.assertEqual(float(gain[1].abs().sum()), 0.0)

    def test_stats_report_the_two_groups_separately(self):
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=2), dim=24
        )
        with torch.no_grad():
            module.q_norm[0].weight[0].mul_(2.0)
        stats = module.attention_temperature_stats()
        self.assertAlmostEqual(stats["qk_gamma_q_match_layer0"], 2.0, places=5)
        self.assertAlmostEqual(stats["qk_gamma_q_rope_layer0"], 1.0, places=5)
        self.assertGreater(
            stats["qk_max_logit_bound_match"],
            stats["qk_max_logit_bound_rope"],
        )
        self.assertEqual(
            stats["qk_max_logit_bound"], stats["qk_max_logit_bound_match"]
        )
        # The gate is gone; nothing should still advertise it.
        for name in stats:
            self.assertNotIn("support", name)

    def test_no_gate_field_survives_on_the_temporal_output(self):
        torch.manual_seed(2)
        module = GroupedGainBarrierCrossAttention(
            v11_1_temporal_cfg(layers=1), dim=24
        ).eval()
        feature = torch.randn(8, 24)
        position = torch.randn(8, 3) * 3.0
        with torch.no_grad():
            _refined, fields = module(*self._forward_args(feature, position, 4))
        self.assertEqual(
            set(fields),
            {
                "delta_p_match", "delta_p_init", "matched_position",
                "motion_layer_logits", "motion_layer_weights",
            },
        )
        # Nothing scales the readout any more, so the two are the same tensor.
        torch.testing.assert_close(
            fields["delta_p_init"], fields["delta_p_match"]
        )
        self.assertFalse(
            any(hasattr(module, name) for name in (
                "support_lo", "support_rho_m", "support_gate_width",
            ))
        )

    def test_the_backend_emits_exactly_one_gaussian_per_token(self):
        torch.manual_seed(3)
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(dim=24), _gs_params(), dim=24, offset_bound=0.8,
        ).eval()
        self.assertIsInstance(
            backend.temporal, GroupedGainBarrierCrossAttention
        )
        self.assertIsInstance(backend.gaussian_head, GaussianAttributeHead)
        # No router, no seed bank, no per-K experts.
        for name in ("count_predictor", "k_heads", "gaussian_output_norm"):
            self.assertFalse(hasattr(backend, name))
            self.assertFalse(hasattr(backend.gaussian_head, name))

        tokens, first = 10, 5
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 4.0
        with torch.no_grad():
            out = backend(
                feature,
                position,
                position.unsqueeze(1).clone(),
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, 0.8])],
                torch.tensor([0.8]),
            )
        item = out["batch_gaussians"][0]
        self.assertEqual(item["position"].shape, (tokens, 3))
        self.assertEqual(item["velocity"].shape, (tokens, 3))
        # One Gaussian per token means the packing indices vanish entirely.
        for name in ("selected_k", "gaussian_slot_index", "source_token_index"):
            self.assertNotIn(name, item)
        self.assertIsNone(out.get("routing_stats"))

    def test_the_backend_publishes_the_v11_motion_fields(self):
        torch.manual_seed(4)
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(dim=24), _gs_params(), dim=24, offset_bound=0.8,
        ).eval()
        tokens, first, duration = 10, 5, 0.8
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 4.0
        with torch.no_grad():
            out = backend(
                feature,
                position,
                position.unsqueeze(1).clone(),
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, duration])],
                torch.tensor([duration]),
            )
        item = out["batch_gaussians"][0]
        for name in (
            "velocity_init", "velocity_offset", "velocity_match",
            "delta_p_init", "delta_p_match", "matched_position",
            "motion_layer_weights",
        ):
            self.assertIn(name, item)
        # Nothing is gated, so match and init agree and the prior has one value
        # to charge rather than two.
        torch.testing.assert_close(
            item["velocity_match"], item["velocity_init"]
        )
        self.assertNotIn("velocity_init_ungated", item)
        torch.testing.assert_close(
            item["velocity"],
            item["velocity_init"] + item["velocity_offset"],
        )
        # Frame 0 travels forward, frame 1 backward, over one endpoint interval.
        signed = torch.where(
            torch.arange(tokens) < first,
            torch.full((tokens,), duration),
            torch.full((tokens,), -duration),
        )
        torch.testing.assert_close(
            item["velocity_init"],
            item["delta_p_init"] / signed.unsqueeze(-1),
        )

    def test_zero_initialized_offset_starts_at_the_initializer(self):
        torch.manual_seed(5)
        backend = SingleGaussianBarrierVelocityGaussianBackend(
            v11_1_backend_cfg(dim=24), _gs_params(), dim=24, offset_bound=0.8,
        ).eval()
        tokens, first = 8, 4
        feature = torch.randn(tokens, 24)
        position = torch.randn(tokens, 3) * 4.0
        with torch.no_grad():
            out = backend(
                feature,
                position,
                position.unsqueeze(1).clone(),
                torch.tensor([first, tokens]),
                torch.tensor([0, 0]),
                [torch.eye(4).repeat(2, 1, 1)],
                [torch.tensor([0.0, 1.0])],
                [torch.tensor([0.0, 0.8])],
                torch.tensor([0.8]),
            )
        item = out["batch_gaussians"][0]
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        torch.testing.assert_close(item["velocity"], item["velocity_init"])


class V8ConsensusAttentionTest(unittest.TestCase):
    def test_tied_init_copies_only_final_layers_first_four_heads(self):
        cfg = temporal_cfg(
            dim=72, heads=12, layers=2, motion_head_count=4
        )
        cfg.tie_motion_qk_init = True
        module = TimeConditionedParallelCrossAttention(cfg, dim=72)
        motion_width = 4 * module.head_dim

        torch.testing.assert_close(
            module.k_proj[-1].weight[:motion_width],
            module.q_proj[-1].weight[:motion_width],
        )
        torch.testing.assert_close(
            module.k_proj[-1].bias[:motion_width],
            module.q_proj[-1].bias[:motion_width],
        )
        self.assertFalse(torch.equal(
            module.k_proj[-1].weight[motion_width:],
            module.q_proj[-1].weight[motion_width:],
        ))
        self.assertFalse(torch.equal(
            module.k_proj[0].weight[:motion_width],
            module.q_proj[0].weight[:motion_width],
        ))
        self.assertNotEqual(
            module.k_proj[-1].weight.data_ptr(),
            module.q_proj[-1].weight.data_ptr(),
        )

    def test_offset_head_uses_feature_and_detached_init_without_duration(self):
        head = EmbeddedInitVelocityHead(OmegaConf.create({
            "zero_init": False,
            "detach_init_condition": True,
            "velocity_embedding_dim": 8,
            "residual_hidden_dim": 12,
        }), dim=24)
        feature = torch.randn(5, 24, requires_grad=True)
        velocity_init = torch.randn(5, 3, requires_grad=True)
        offset = head(feature, velocity_init)
        self.assertEqual(head.motion_refiner[0].in_features, 24 + 8)
        self.assertEqual(offset.shape, (5, 3))
        offset.square().sum().backward()
        self.assertIsNotNone(feature.grad)
        self.assertIsNone(velocity_init.grad)

    def test_head_probabilities_are_averaged_before_one_topk(self):
        torch.manual_seed(31)
        matcher = ConsensusAttentionMotionMatcher(OmegaConf.create({
            "candidate_count": 3,
            "match_count": 2,
            "score_chunk_size": 1,
        }))
        query = torch.randn(1, 4, 6)
        key = torch.randn(5, 4, 6)
        scale = 6 ** -0.5
        result = matcher._topk_direction(query, key, scale)

        logits = torch.einsum("qhd,khd->qhk", query, key) * scale
        consensus = torch.softmax(logits, dim=-1).mean(dim=1)
        expected_probability, expected_index = torch.topk(
            consensus, k=3, dim=-1
        )
        torch.testing.assert_close(
            result["probability"], expected_probability
        )
        torch.testing.assert_close(
            result["conditional_probability"],
            expected_probability / expected_probability.sum(
                dim=-1, keepdim=True
            ),
        )
        torch.testing.assert_close(
            result["conditional_probability"].sum(dim=-1),
            torch.ones(1),
        )
        self.assertTrue(torch.equal(
            result["candidate_index"], expected_index
        ))

    def test_reciprocal_probability_can_rerank_forward_attention(self):
        matcher = ConsensusAttentionMotionMatcher(OmegaConf.create({
            "candidate_count": 2,
            "match_count": 1,
            "score_chunk_size": 2,
        }))
        direction = {
            "candidate_index": torch.tensor([[0, 1], [1, 0]]),
            # Full-key Top-K mass is only 0.1/0.2; mutual matching must use
            # the separately normalized K-conditional probabilities below.
            "probability": torch.tensor([[0.06, 0.04], [0.10, 0.10]]),
            "conditional_probability": torch.tensor([
                [0.6, 0.4], [0.5, 0.5],
            ]),
            "consensus_entropy": torch.ones(2),
            "head_js_divergence": torch.zeros(2),
            "candidate_probability_mass": torch.ones(2),
            "support": 2,
        }
        reverse = {
            # For source query 0, key 0 returns with p=0.01 whereas key 1
            # returns with p=0.90. Reciprocal evidence must overturn 0.6>0.4.
            "candidate_index": torch.tensor([[1, 0], [0, 1]]),
            "probability": torch.tensor([[0.099, 0.001], [0.09, 0.01]]),
            "conditional_probability": torch.tensor([
                [0.99, 0.01], [0.90, 0.10],
            ]),
            "support": 2,
        }
        position = torch.tensor([
            [0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        ])
        key_position = torch.tensor([
            [1.0, 0.0, 0.0], [4.0, 0.0, 0.0],
        ])
        result = matcher._finish_direction(
            direction, reverse, position, key_position
        )
        torch.testing.assert_close(
            result["matched_position"][0], key_position[1]
        )
        self.assertEqual(float(result["motion_match_support"][0]), 1.0)

    def test_reciprocal_membership_is_hard_but_probability_is_not_detached(self):
        matcher = ConsensusAttentionMotionMatcher(OmegaConf.create({
            "candidate_count": 2,
            "match_count": 2,
            "score_chunk_size": 2,
        }))
        direction = {
            "candidate_index": torch.tensor([[0, 1], [0, 1], [0, 1]]),
            "probability": torch.full((3, 2), 0.1),
            "conditional_probability": torch.full((3, 2), 0.5),
            "consensus_entropy": torch.ones(3),
            "head_js_divergence": torch.zeros(3),
            "candidate_probability_mass": torch.full((3,), 0.2),
            "support": 2,
        }
        reverse_probability = torch.tensor([
            [0.7, 0.3],
            [0.6, 0.4],
        ], requires_grad=True)
        reverse = {
            # For source query 0, reverse row 0 excludes it while row 1
            # includes it in slot 0.
            "candidate_index": torch.tensor([[1, 2], [0, 2]]),
            "probability": 0.2 * reverse_probability,
            "conditional_probability": reverse_probability,
            "support": 2,
        }
        result = matcher._finish_direction(
            direction,
            reverse,
            torch.tensor([
                [0.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [12.0, 0.0, 0.0],
            ]),
            torch.tensor([
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]),
        )
        result["matched_position"][0, 0].backward()
        # No reverse candidate equals query 0 in row 0, so its hard mask is
        # zero. Row 1 contains query 0 and its probability remains trainable.
        torch.testing.assert_close(
            reverse_probability.grad[0], torch.zeros(2)
        )
        self.assertNotEqual(float(reverse_probability.grad[1, 0]), 0.0)

    def test_four_heads_produce_one_total_m4_support_with_gradients(self):
        torch.manual_seed(37)
        matcher = ConsensusAttentionMotionMatcher(OmegaConf.create({
            "candidate_count": 5,
            "match_count": 4,
            "score_chunk_size": 2,
        }))
        query = torch.randn(10, 4, 6, requires_grad=True)
        key = torch.randn(10, 4, 6, requires_grad=True)
        position = torch.randn(10, 3)
        result = matcher(
            query,
            key,
            position,
            torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4]),
            torch.tensor([5, 5]),
            torch.tensor([5, 5]),
            scale=6 ** -0.5,
        )
        self.assertTrue(torch.equal(
            result["motion_match_support"], torch.full((10,), 4.0)
        ))
        self.assertTrue(torch.all(
            result["motion_effective_support"] <= 4.0 + 1.0e-6
        ))
        torch.testing.assert_close(
            result["delta_p_init"], result["delta_p_match"]
        )

        result["delta_p_init"].square().sum().backward()
        self.assertGreater(float(query.grad.abs().sum()), 0.0)
        self.assertGreater(float(key.grad.abs().sum()), 0.0)

    def test_backend_uses_shared_rope_time_projection_and_offset_contract(self):
        cfg = v8_backend_cfg()
        gs_params = OmegaConf.create({
            "shs": 32, "opacity": 1, "scaling": 2, "rotation": 4, "offset": 3,
        })
        backend = ConsensusAttentionVelocityGaussianBackend(
            cfg, gs_params, dim=24, offset_bound=0.5
        )
        self.assertIsNone(backend.temporal.motion_rope)
        self.assertEqual(backend.temporal.rope.base, 100.0)
        self.assertAlmostEqual(
            backend.temporal.rope.position_scale,
            float(2.0 * torch.pi / 5.0),
        )
        self.assertEqual(backend.temporal.time_to_feature.in_features, 16)
        self.assertEqual(backend.temporal.time_to_feature.out_features, 24)
        self.assertTrue(backend.temporal.tie_motion_qk_init)
        final_q = backend.temporal.q_proj[-1]
        final_k = backend.temporal.k_proj[-1]
        motion_width = (
            backend.temporal.motion_head_count * backend.temporal.head_dim
        )
        torch.testing.assert_close(
            final_k.weight[:motion_width], final_q.weight[:motion_width]
        )
        torch.testing.assert_close(
            final_k.bias[:motion_width], final_q.bias[:motion_width]
        )
        # This is initialization only, not permanent parameter sharing.
        self.assertNotEqual(final_k.weight.data_ptr(), final_q.weight.data_ptr())
        self.assertNotEqual(final_k.bias.data_ptr(), final_q.bias.data_ptr())
        self.assertFalse(torch.equal(
            backend.temporal.k_proj[0].weight,
            backend.temporal.q_proj[0].weight,
        ))
        self.assertIsInstance(backend.velocity_head, EmbeddedInitVelocityHead)
        self.assertEqual(
            backend.velocity_head.motion_refiner[0].in_features, 24 + 8
        )

        feature = torch.randn(10, 24)
        token = torch.randn(10, 3)
        output = backend(
            feature,
            token,
            token.unsqueeze(1),
            torch.tensor([5, 10]),
            torch.tensor([0, 0]),
            [torch.eye(4).repeat(2, 1, 1)],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 1.0])],
            torch.tensor([1.0]),
        )
        item = output["batch_gaussians"][0]
        torch.testing.assert_close(
            item["velocity"], item["velocity_init"] + item["velocity_offset"]
        )
        torch.testing.assert_close(
            item["velocity_offset"], torch.zeros_like(item["velocity_offset"])
        )
        self.assertTrue(torch.equal(
            item["motion_match_support"], torch.full((10,), 4.0)
        ))


if __name__ == "__main__":
    unittest.main()
