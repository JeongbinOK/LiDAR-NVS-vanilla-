from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.dataloader.nuscene import relative_time_coordinates
from src.models_new.module.dynamic_gaussian import (
    AttentionInitializedVelocityGaussianBackend,
    InitConditionedVelocityHead,
    PhysicalVelocityGaussianBackend,
    ProposalInitializedVelocityGaussianBackend,
    SparseMotionProposal,
    TimeConditionedParallelCrossAttention,
)
from src.models_new.module.feature_fusion import build_feature_fusion
from src.models_new.module.m3_g2p import DynamicGausRender
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


def proposal_backend_cfg(dim=24):
    cfg = backend_cfg(dim=dim)
    cfg.temporal = v6_temporal_cfg(dim=dim)
    cfg.motion_proposal = proposal_cfg()
    cfg.motion.init_conditioned_residual = True
    cfg.motion.init_condition_scale_mps = 10.0
    cfg.motion.detach_init_condition = True
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
        velocities, *, prefix="train", weight=0.005, step=0, mode="final_l2"
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
        ModelWrapper._add_velocity_l2_prior(
            wrapper,
            losses,
            [{"velocity": item} for item in velocities],
            prefix=prefix,
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

    def _backend_output(self):
        torch.manual_seed(11)
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
        feature = torch.randn(5, 24)
        token = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
        ])
        seed = (token + torch.tensor([0.02, 0.01, 0.0])).unsqueeze(1)
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


if __name__ == "__main__":
    unittest.main()
