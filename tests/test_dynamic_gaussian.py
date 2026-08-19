from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.dataloader.nuscene import relative_time_coordinates
from src.models_new.module.dynamic_gaussian import (
    AttentionInitializedVelocityGaussianBackend,
    PhysicalVelocityGaussianBackend,
    TimeConditionedParallelCrossAttention,
)
from src.models_new.module.feature_fusion import build_feature_fusion
from src.models_new.module.m3_g2p import DynamicGausRender
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
    def _apply_velocity_l2(velocities, *, prefix="train", weight=0.005, step=0):
        wrapper = SimpleNamespace(
            _velocity_l2_cfg=OmegaConf.create({
                "enabled": True,
                "weight": weight,
                "warmup_steps": 0,
                "ramp_steps": 0,
            }),
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
