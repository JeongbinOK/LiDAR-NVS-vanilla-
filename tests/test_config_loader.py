from __future__ import annotations

import tempfile
import unittest
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.config_loader import (
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    DYNAMIC_VARIANT,
    DYNAMIC_VARIANT_V1,
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V6,
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
    DYNAMIC_VARIANT_V8,
    DYNAMIC_VARIANT_V9,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
    DYNAMIC_VARIANT_V11_1,
    LEGACY_VARIANT,
    assert_model_variant_implemented,
    compose_fresh_config,
    load_checkpoint_config,
    resolve_eval_config,
    resolve_main_config,
)


class ConfigLoaderTest(unittest.TestCase):
    def test_default_compose_is_current_dynamic(self):
        config, source = compose_fresh_config()
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT)
        self.assertIn(f"{DYNAMIC_VARIANT}.yaml", source)

    def test_dynamic_effective_config_excludes_legacy_temporal_blocks(self):
        config, _ = compose_fresh_config()
        self.assertNotIn("squery", config.p2g)
        self.assertNotIn("grid_query", config.p2g)

    def test_dynamic_overlay_and_cli_precedence(self):
        cli = OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT}",
            "train.batch_size=3",
        ])
        config, source = compose_fresh_config(cli)
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT)
        self.assertEqual(config.p2g.anchor_mode, "grid")
        self.assertNotIn("grid_query", config.p2g)
        self.assertEqual(
            config.dynamic_2dgs.temporal.time_reference_sec, 1.0
        )
        rope_scale = config.dynamic_2dgs.temporal.rope_position_scale
        self.assertEqual(config.dynamic_2dgs.temporal.rope_base, 100.0)
        self.assertAlmostEqual(rope_scale, 2.0 * math.pi / 5.0)
        self.assertAlmostEqual(2.0 * math.pi / rope_scale, 5.0)
        self.assertTrue(config.p2g.freeze_utonia)
        self.assertTrue(config.p2g.utonia_lora.enable)
        self.assertEqual(config.p2g.utonia_lora.input_mode, "xyzi")
        self.assertEqual(config.p2g.utonia_lora.rank, 16)
        self.assertEqual(config.p2g.utonia_lora.alpha, 16)
        self.assertEqual(config.p2g.utonia_feature_stage, 3)
        self.assertEqual(config.dynamic_2dgs.temporal.layers, 4)
        self.assertEqual(config.dynamic_2dgs.temporal.num_heads, 12)
        self.assertTrue(config.dynamic_2dgs.temporal.use_time_embedding)
        self.assertNotIn("time_hidden_dim", config.dynamic_2dgs.temporal)
        self.assertEqual(config.dynamic_2dgs.temporal.time_frequencies, 8)
        self.assertEqual(config.dynamic_2dgs.temporal.time_embedding_dim, 64)
        self.assertNotIn("motion_head_count", config.dynamic_2dgs.temporal)
        self.assertNotIn("motion_rope_base", config.dynamic_2dgs.temporal)
        self.assertNotIn("qk_norm", config.dynamic_2dgs.temporal)
        proposal = config.dynamic_2dgs.motion_proposal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V7_2)
        self.assertEqual(proposal.descriptor_mode, "direct_l2")
        self.assertEqual(proposal.match_count, 4)
        self.assertEqual(proposal.score_chunk_size, 128)
        self.assertEqual(proposal.ste_surrogate, "dense_softmax")
        self.assertAlmostEqual(proposal.distance_prior_speed_mps, 30.0)
        for removed in (
            "candidate_count", "dustbin_mode", "dustbin_hidden_dim",
            "dustbin_prior_probability", "dustbin_evidence_mode",
            "unmatched_gate_mode", "unmatched_hard_threshold",
            "mean_displacement_scale_m", "spread_scale_m",
            "search_speed_min_mps", "search_speed_init_mps",
            "search_speed_max_mps",
        ):
            self.assertNotIn(removed, proposal)
        self.assertNotIn("descriptor_dim", proposal)
        self.assertNotIn("adapter_hidden_dim", proposal)
        self.assertNotIn("dustbin_similarity_init", proposal)
        self.assertEqual(config.dynamic_2dgs.motion.residual_hidden_dim, 96)
        for removed in (
            "detach_init_condition", "velocity_embedding_dim",
            "duration_embedding_dim", "duration_frequencies",
            "duration_reference_sec",
        ):
            self.assertNotIn(removed, config.dynamic_2dgs.motion)
        self.assertEqual(list(config.device), [0])
        self.assertIsNone(config.data.bbox_json_path)
        self.assertNotIn("cycle_radius_m", config.dynamic_2dgs.motion_proposal)
        self.assertNotIn("geometry_dim", config.dynamic_2dgs.gaussian_head)
        self.assertNotIn("cell_size_m", config.dynamic_2dgs.gaussian_head)
        self.assertNotIn("position_frequencies", config.dynamic_2dgs.motion)
        self.assertNotIn("position_range_m", config.dynamic_2dgs.motion)
        self.assertNotIn("max_speed_mps", config.dynamic_2dgs.motion)
        self.assertEqual(config.train.batch_size, 3)
        self.assertEqual(
            OmegaConf.to_container(config, resolve=True)["exp_name"],
            f"{DYNAMIC_VARIANT}-bs-3",
        )
        self.assertIn(f"{DYNAMIC_VARIANT}.yaml", source)

    def test_utonia_lora_requires_frozen_base(self):
        with self.assertRaisesRegex(ValueError, "freeze_utonia=true"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.freeze_utonia=false",
            ]))

    def test_utonia_lora_rejects_invalid_adapter_settings(self):
        with self.assertRaisesRegex(ValueError, "linear_rank must be positive"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.utonia_lora.linear_rank=0",
            ]))

    def test_utonia_lora_stage_width_validates_temporal_attention(self):
        # Stage 2 is 216D; 216/8=27 is invalid for xyz/sin-cos 3D RoPE.
        with self.assertRaisesRegex(ValueError, "head_dim=27"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "p2g.utonia_feature_stage=2",
                "dynamic_2dgs.temporal.num_heads=8",
            ]))

    def test_v6_can_fall_back_to_previous_intensity_fusion_path(self):
        config, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V6}",
            "p2g.utonia_lora.enable=false",
        ]))
        self.assertFalse(config.p2g.utonia_lora.enable)
        self.assertEqual(config.p2g.agg_mlp.hidden_dim, 1024)
        self.assertEqual(config.p2g.agg_mlp.hidden_layers, 2)
        self.assertEqual(config.p2g.agg_mlp.out_dim, 576)
        self.assertEqual(config.p2g.joint_refiner.num_heads, 12)

    def test_direct_velocity_v1_remains_composable(self):
        config, source = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT_V1}"])
        )
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V1)
        self.assertEqual(config.dynamic_2dgs.motion.max_speed_mps, 25.0)
        self.assertIn(f"{DYNAMIC_VARIANT_V1}.yaml", source)

    def test_physical_velocity_v3_remains_composable(self):
        config, source = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT_V3}"])
        )
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V3)
        # V3 is the pre-regularization architecture and must stay that way so
        # its checkpoints keep an exact config.
        self.assertNotIn("velocity_l2", config.dynamic_2dgs.regularization)
        self.assertIn(f"{DYNAMIC_VARIANT_V3}.yaml", source)

    def test_physical_velocity_v3_1_remains_composable(self):
        config, source = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT_V3_1}"])
        )
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V3_1)
        self.assertNotIn("motion_head_count", config.dynamic_2dgs.temporal)
        self.assertTrue(config.dynamic_2dgs.regularization.velocity_l2.enabled)
        self.assertIn(f"{DYNAMIC_VARIANT_V3_1}.yaml", source)

    def test_attention_velocity_v5_remains_composable(self):
        config, source = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT_V5}"])
        )
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V5)
        self.assertEqual(config.dynamic_2dgs.temporal.motion_head_count, 4)
        self.assertIn("motion_rope_base", config.dynamic_2dgs.temporal)
        self.assertNotIn("motion_proposal", config.dynamic_2dgs)
        self.assertIn(f"{DYNAMIC_VARIANT_V5}.yaml", source)

    def test_current_dynamic_enables_velocity_l2_prior(self):
        config, _ = compose_fresh_config()
        prior = config.dynamic_2dgs.regularization.velocity_l2
        self.assertTrue(prior.enabled)
        self.assertEqual(prior.mode, "offset_l2")
        self.assertEqual(prior.weight, 0.05)
        self.assertEqual(prior.warmup_steps, 0)
        self.assertEqual(prior.ramp_steps, 500)
        # Two motion priors with different gradient shapes must not stack.
        self.assertFalse(config.dynamic_2dgs.regularization.small_motion.enabled)

    def test_v7_remains_composable_with_legacy_dustbin_evidence(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V7}",
        ]))
        proposal = config.dynamic_2dgs.motion_proposal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V7)
        self.assertAlmostEqual(proposal.unmatched_hard_threshold, 0.9)
        self.assertNotIn("dustbin_evidence_mode", proposal)
        self.assertNotIn("mean_displacement_scale_m", proposal)
        self.assertNotIn("spread_scale_m", proposal)
        self.assertIn(f"{DYNAMIC_VARIANT_V7}.yaml", source)

    def test_v7_1_remains_composable_with_metric_dustbin_evidence(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V7_1}",
        ]))
        proposal = config.dynamic_2dgs.motion_proposal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V7_1)
        self.assertEqual(proposal.candidate_count, 16)
        self.assertEqual(proposal.match_count, 4)
        self.assertEqual(
            proposal.dustbin_evidence_mode,
            "mean_spread_reciprocal_m4",
        )
        self.assertAlmostEqual(proposal.unmatched_hard_threshold, 0.5)
        self.assertAlmostEqual(proposal.search_speed_min_mps, 4.0)
        self.assertIn(f"{DYNAMIC_VARIANT_V7_1}.yaml", source)

    def test_v8_composes_final_attention_consensus_contract(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V8}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        matching = config.dynamic_2dgs.motion_matching
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V8)
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertEqual(temporal.motion_head_count, 4)
        self.assertTrue(temporal.use_time_embedding)
        self.assertEqual(temporal.time_embedding_dim, 64)
        self.assertEqual(temporal.time_frequencies, 8)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(
            temporal.rope_position_scale, 2.0 * math.pi / 5.0
        )
        self.assertTrue(temporal.qk_norm)
        self.assertTrue(temporal.tie_motion_qk_init)
        self.assertNotIn("motion_rope_base", temporal)
        self.assertNotIn("motion_rope_position_scale", temporal)
        self.assertEqual(matching.candidate_count, 16)
        self.assertEqual(matching.match_count, 4)
        self.assertNotIn("motion_proposal", config.dynamic_2dgs)
        self.assertEqual(
            config.dynamic_2dgs.motion.velocity_embedding_dim, 32
        )
        self.assertEqual(config.dynamic_2dgs.motion.residual_hidden_dim, 96)
        self.assertNotIn(
            "duration_embedding_dim", config.dynamic_2dgs.motion
        )
        self.assertNotIn("duration_frequencies", config.dynamic_2dgs.motion)
        self.assertNotIn(
            "duration_reference_sec", config.dynamic_2dgs.motion
        )
        self.assertEqual(
            config.dynamic_2dgs.regularization.velocity_l2.mode,
            "final_l2",
        )
        self.assertIn(f"{DYNAMIC_VARIANT_V8}.yaml", source)

    def test_v9_composes_post_attention_dense_proposal_contract(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V9}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        proposal = config.dynamic_2dgs.motion_proposal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V9)
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertTrue(temporal.use_time_embedding)
        self.assertEqual(temporal.time_embedding_dim, 64)
        self.assertEqual(temporal.time_frequencies, 8)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(
            temporal.rope_position_scale, 2.0 * math.pi / 5.0
        )
        # The proposal is a separate branch, so no attention head is reserved
        # for it and the 12-layer stack stays a plain V7.2-style trunk.
        self.assertNotIn("motion_head_count", temporal)
        self.assertNotIn("qk_norm", temporal)
        self.assertNotIn("tie_motion_qk_init", temporal)
        self.assertNotIn("motion_matching", config.dynamic_2dgs)
        self.assertEqual(proposal.descriptor_mode, "projected_l2")
        self.assertEqual(proposal.descriptor_dim, 128)
        self.assertEqual(proposal.readout, "dense_expectation")
        self.assertEqual(proposal.distance_prior_speed_mps, 5.0)
        self.assertAlmostEqual(proposal.temperature, 0.07)
        # V7.2's Top-4/STE knobs must not survive into a dense readout.
        self.assertNotIn("match_count", proposal)
        self.assertNotIn("ste_surrogate", proposal)
        self.assertNotIn("candidate_count", proposal)
        self.assertEqual(config.dynamic_2dgs.motion.residual_hidden_dim, 96)
        self.assertNotIn(
            "velocity_embedding_dim", config.dynamic_2dgs.motion
        )
        self.assertEqual(
            config.dynamic_2dgs.regularization.velocity_l2.mode,
            "final_group_l2",
        )
        self.assertIn(f"{DYNAMIC_VARIANT_V9}.yaml", source)

    def test_v9_requires_projected_dense_matcher_contract(self):
        with self.assertRaisesRegex(ValueError, "projected_l2"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V9}",
                "dynamic_2dgs.motion_proposal.descriptor_mode=direct_l2",
            ]))
        with self.assertRaisesRegex(ValueError, "dense_expectation"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V9}",
                "dynamic_2dgs.motion_proposal.readout=top4_ste",
            ]))
        with self.assertRaisesRegex(ValueError, "descriptor_dim"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V9}",
                "dynamic_2dgs.motion_proposal.descriptor_dim=0",
            ]))
        with self.assertRaisesRegex(ValueError, "distance_prior_speed_mps"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V9}",
                "dynamic_2dgs.motion_proposal.distance_prior_speed_mps=0",
            ]))

    def test_v10_composes_all_layer_distance_attention_contract(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V10}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V10)
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertTrue(temporal.use_time_embedding)
        self.assertEqual(temporal.time_embedding_dim, 64)
        self.assertEqual(temporal.time_frequencies, 8)
        self.assertEqual(temporal.position_encoding, "distance_bias")
        self.assertEqual(temporal.distance_bias_speed_mps, 30.0)
        self.assertTrue(temporal.qk_norm)
        self.assertEqual(temporal.layer_weight_hidden_dim, 96)
        for removed in (
            "rope_base", "rope_position_scale", "motion_head_count",
            "motion_rope_base", "motion_rope_position_scale",
            "tie_motion_qk_init",
        ):
            self.assertNotIn(removed, temporal)
        self.assertNotIn("motion_proposal", config.dynamic_2dgs)
        self.assertNotIn("motion_matching", config.dynamic_2dgs)
        self.assertTrue(config.p2g.freeze_utonia)
        self.assertTrue(config.p2g.utonia_lora.enable)
        self.assertEqual(config.p2g.utonia_lora.input_mode, "xyzi")
        self.assertEqual(config.p2g.utonia_lora.rank, 16)
        count = config.p2g.grid_query
        self.assertEqual(count.count_mode, "learned_gumbel")
        self.assertEqual(count.learned_count.K_max, 3)
        self.assertEqual(count.learned_count.tau, 1.0)
        self.assertEqual(count.learned_count.seed_mode, "range_quantile")
        self.assertFalse(count.learned_count.budget.enable)
        self.assertEqual(config.dynamic_2dgs.motion.residual_hidden_dim, 96)
        prior = config.dynamic_2dgs.regularization.velocity_l2
        self.assertEqual(prior.mode, "final_group_l2")
        self.assertEqual(prior.weight, 0.01)
        self.assertEqual(prior.warmup_steps, 0)
        self.assertEqual(prior.ramp_steps, 0)
        self.assertEqual(config.loss.w_chamfer, 0.02)
        self.assertEqual(config.loss.w_scale, 0.0)
        self.assertIn(f"{DYNAMIC_VARIANT_V10}.yaml", source)

    def test_v10_validates_distance_attention_contract(self):
        invalid = (
            ("position_encoding=rope", "position_encoding"),
            ("distance_bias_speed_mps=0", "distance_bias_speed_mps"),
            ("qk_norm=false", "qk_norm"),
            ("layer_weight_hidden_dim=0", "layer_weight_hidden_dim"),
            ("use_time_embedding=false", "use_time_embedding"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V10}",
                    f"dynamic_2dgs.temporal.{override}",
                ]))

    def test_v10_distance_bias_does_not_require_rope_head_width(self):
        # 216/8=27 cannot be split into xyz sin/cos bands, but V10 has no RoPE.
        config, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V10}",
            "p2g.utonia_feature_stage=2",
            "dynamic_2dgs.temporal.num_heads=8",
        ]))
        self.assertEqual(config.p2g.utonia_feature_stage, 2)
        self.assertEqual(config.dynamic_2dgs.temporal.num_heads, 8)

    def test_v10_requires_exact_k123_router_without_budget(self):
        invalid = (
            ("p2g.utonia_lora.enable=false", "utonia_lora.enable=true"),
            # Switching V10 to a fixed count is allowed, but the router keys
            # the overlay ships would then be read by nothing.
            (
                "p2g.grid_query.count_mode=legacy",
                "grad_balance, learned_count",
            ),
            ("p2g.grid_query.learned_count.K_max=4", "K_max=3"),
            ("p2g.grid_query.learned_count.tau=0", "tau"),
            (
                "p2g.grid_query.learned_count.seed_mode=medoid",
                "range_quantile",
            ),
            ("p2g.grid_query.learned_count.budget.enable=true", "budget"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V10}", override,
                ]))

    def test_v11_1_composes_the_single_gaussian_contract(self):
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_1}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        # Everything V11 established is inherited unchanged.
        self.assertEqual(temporal.position_encoding, "barrier_rope_split")
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 4.0)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertTrue(temporal.qk_norm)
        self.assertEqual(config.dynamic_2dgs.regularization.velocity_l2.mode,
                         "split_group_l2")
        self.assertTrue(config.p2g.utonia_lora.enable)
        # One Gaussian per token: no grid_query block is configured, and
        # V11.1's backend has no router it could be pointed at.
        self.assertIsNone(OmegaConf.select(config, "p2g.grid_query"))
        self.assertNotIn(DYNAMIC_VARIANT_V11_1, ROUTER_CAPABLE_DYNAMIC_VARIANTS)
        self.assertIn(DYNAMIC_VARIANT_V11, ROUTER_CAPABLE_DYNAMIC_VARIANTS)

    def test_v11_1_rejects_a_router_its_backend_cannot_run(self):
        """V11.1 has no count router, so asking for one is an error."""
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_1}",
                "p2g.grid_query.count_mode=learned_gumbel",
            ]))
        self.assertIn("grid_query", str(raised.exception))
        self.assertIn("V11.1", str(raised.exception))

    def test_grid_query_sets_the_per_token_gaussian_count(self):
        """K_max is how many Gaussians each token becomes; exp seeds them."""
        for variant in (DYNAMIC_VARIANT, DYNAMIC_VARIANT_V11_1):
            for k_max, exp in ((1, 1), (2, 1), (2, 2), (4, None)):
                with self.subTest(variant=variant, K_max=k_max, exp=exp):
                    config, _source = compose_fresh_config(
                        OmegaConf.from_dotlist([
                            f"model.variant={variant}",
                            f"p2g.grid_query.K_max={k_max}",
                            f"p2g.grid_query.exp={'null' if exp is None else exp}",
                        ])
                    )
                    self.assertEqual(config.p2g.grid_query.K_max, k_max)
                    self.assertEqual(config.p2g.grid_query.exp, exp)

    def test_absent_grid_query_is_one_gaussian_per_token(self):
        config, _source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11_1}",
        ]))
        self.assertIsNone(OmegaConf.select(config, "p2g.grid_query"))

    def test_fixed_gaussian_count_rejects_impossible_settings(self):
        invalid = (
            ("p2g.grid_query.K_max=0", "positive"),
            ("p2g.grid_query.K_max=1 p2g.grid_query.exp=2", "K_max >= 2"),
            ("p2g.grid_query.K_max=2 p2g.grid_query.exp=3", "null, 1, or 2"),
            (
                "p2g.grid_query.K_max=2 p2g.grid_query.points_per_gaussian=0",
                "points_per_gaussian",
            ),
            ("p2g.grid_query.K_max=2 p2g.grid_query.grad_balance=sqrt_k",
             "Unsupported keys"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11_1}",
                    *override.split(),
                ]))

    def test_neither_barrier_variant_accepts_the_removed_gate_keys(self):
        """The gate is gone from V11.1 as well, not just absent from V11."""
        for variant in (DYNAMIC_VARIANT_V11, DYNAMIC_VARIANT_V11_1):
            for key in (
                "support_rho_m", "support_gate_lo", "support_gate_width",
            ):
                with self.subTest(variant=variant, key=key):
                    with self.assertRaises(ValueError):
                        compose_fresh_config(OmegaConf.from_dotlist([
                            f"model.variant={variant}",
                            f"dynamic_2dgs.temporal.{key}=1.0",
                        ]))

    def test_v11_1_validates_the_barrier_geometry(self):
        for override, message in (
            ("dynamic_2dgs.temporal.position_encoding=distance_bias",
             "barrier_rope_split"),
            ("dynamic_2dgs.temporal.num_heads=1", "num_heads"),
            ("dynamic_2dgs.temporal.barrier_speed_mps=0.0",
             "barrier_speed_mps"),
            ("p2g.utonia_lora.enable=false", "utonia_lora"),
        ):
            with self.assertRaises(ValueError) as raised:
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11_1}", override,
                ]))
            self.assertIn(message, str(raised.exception))

    def test_barrier_variant_errors_name_the_variant_they_came_from(self):
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11_1}",
                "dynamic_2dgs.temporal.qk_norm=false",
            ]))
        self.assertIn("V11.1", str(raised.exception))
        with self.assertRaises(ValueError) as raised:
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.temporal.qk_norm=false",
            ]))
        self.assertIn("V11 ", str(raised.exception))

    def test_v11_composes_split_position_encoding_contract(self):
        config, source = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V11}",
        ]))
        temporal = config.dynamic_2dgs.temporal
        self.assertEqual(config.model.variant, DYNAMIC_VARIANT_V11)
        self.assertEqual(temporal.layers, 12)
        self.assertEqual(temporal.num_heads, 12)
        self.assertTrue(temporal.use_time_embedding)
        self.assertEqual(temporal.position_encoding, "barrier_rope_split")
        self.assertEqual(temporal.barrier_speed_mps, 30.0)
        self.assertEqual(temporal.barrier_weight, 4.0)
        self.assertGreater(temporal.match_chunk_size, 0)
        self.assertEqual(temporal.rope_base, 100.0)
        self.assertAlmostEqual(
            2.0 * math.pi / temporal.rope_position_scale, 5.0
        )
        self.assertTrue(temporal.qk_norm)
        # V10's global-token scorer and quadratic prior are both gone.
        for removed in (
            "layer_weight_hidden_dim", "distance_bias_speed_mps",
            "motion_head_count", "tie_motion_qk_init",
        ):
            self.assertNotIn(removed, temporal)
        self.assertNotIn("motion_proposal", config.dynamic_2dgs)
        self.assertTrue(config.p2g.utonia_lora.enable)
        self.assertEqual(config.p2g.utonia_lora.input_mode, "xyzi")
        self.assertEqual(config.p2g.utonia_lora.rank, 16)
        count = config.p2g.grid_query
        self.assertEqual(count.count_mode, "learned_gumbel")
        self.assertEqual(count.learned_count.K_max, 3)
        self.assertEqual(count.learned_count.seed_mode, "range_quantile")
        self.assertFalse(count.learned_count.budget.enable)
        motion = config.dynamic_2dgs.motion
        self.assertEqual(motion.residual_hidden_dim, 96)
        self.assertEqual(motion.velocity_embedding_dim, 32)
        self.assertTrue(motion.detach_init_condition)
        prior = config.dynamic_2dgs.regularization.velocity_l2
        self.assertEqual(prior.mode, "split_group_l2")
        self.assertEqual(prior.init_weight, 0.01)
        self.assertEqual(prior.offset_weight, 0.01)
        self.assertNotIn("weight", prior)
        self.assertEqual(config.loss.w_chamfer, 0.02)
        self.assertEqual(config.loss.w_scale, 0.0)
        self.assertIn(f"{DYNAMIC_VARIANT_V11}.yaml", source)

    def test_v11_validates_barrier_and_rope_contract(self):
        invalid = (
            ("dynamic_2dgs.temporal.position_encoding=distance_bias",
             "position_encoding"),
            ("dynamic_2dgs.temporal.barrier_speed_mps=0", "barrier_speed_mps"),
            ("dynamic_2dgs.temporal.barrier_weight=-1", "barrier_weight"),
            ("dynamic_2dgs.temporal.match_chunk_size=0", "match_chunk_size"),
            ("dynamic_2dgs.temporal.qk_norm=false", "qk_norm"),
            ("dynamic_2dgs.temporal.use_time_embedding=false",
             "use_time_embedding"),
            ("p2g.utonia_lora.enable=false", "utonia_lora.enable=true"),
            ("p2g.grid_query.learned_count.K_max=4", "K_max=3"),
            ("p2g.grid_query.learned_count.budget.enable=true", "budget"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, message
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11}", override,
                ]))

    def test_v11_still_requires_a_rope_capable_head_width(self):
        # Heads 1-11 keep 3D RoPE, so 432/16=27 is rejected where V10 allows it.
        with self.assertRaisesRegex(ValueError, "divisible by six"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.temporal.num_heads=16",
            ]))

    def test_v11_split_prior_requires_both_weights(self):
        for override in (
            "dynamic_2dgs.regularization.velocity_l2.init_weight=-1",
            "dynamic_2dgs.regularization.velocity_l2.offset_weight=-1",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError, "split_group_l2"
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V11}", override,
                ]))

    def test_split_prior_weights_are_rejected_by_single_term_modes(self):
        with self.assertRaisesRegex(ValueError, "split_group_l2"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.regularization.velocity_l2.mode=final_group_l2",
                "dynamic_2dgs.regularization.velocity_l2.weight=0.01",
            ]))

    def test_v11_rejects_v10_only_temporal_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V11}",
                "dynamic_2dgs.temporal.layer_weight_hidden_dim=96",
            ]))

    def test_v9_rejects_v7_2_top4_matcher_keys(self):
        for key in ("match_count=4", "ste_surrogate=dense_softmax"):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "Unsupported keys"
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V9}",
                    f"dynamic_2dgs.motion_proposal.{key}",
                ]))

    def test_v9_requires_temporal_time_embedding(self):
        with self.assertRaisesRegex(ValueError, "use_time_embedding=true"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V9}",
                "dynamic_2dgs.temporal.use_time_embedding=false",
            ]))

    def test_velocity_l2_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "dynamic_2dgs.regularization.velocity_l2.epsilon_m=0.1",
            ]))

    def test_v4_motion_head_count_must_fit_temporal_heads(self):
        with self.assertRaisesRegex(ValueError, "motion_head_count"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V4}",
                "dynamic_2dgs.temporal.motion_head_count=13",
            ]))

    def test_v6_rejects_temporal_motion_heads(self):
        with self.assertRaisesRegex(ValueError, "motion_head_count"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.temporal.motion_head_count=4",
            ]))

    def test_v6_rejects_obsolete_temporal_qk_norm(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.temporal.qk_norm=true",
            ]))

    def test_v6_rejects_legacy_two_stage_time_projection(self):
        with self.assertRaisesRegex(ValueError, "time_embedding_dim"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.temporal.time_embedding_dim=64",
            ]))

    def test_v6_rejects_nonpositive_time_hidden_width(self):
        with self.assertRaisesRegex(ValueError, "time_hidden_dim"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.temporal.time_hidden_dim=0",
            ]))

    def test_v6_rejects_unknown_motion_proposal_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.motion_proposal.absolute_position=true",
            ]))

    def test_v6_rejects_invalid_motion_proposal_search_speeds(self):
        with self.assertRaisesRegex(ValueError, "0 < min < init < max"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.motion_proposal.search_speed_init_mps=31",
            ]))

    def test_v6_rejects_nonpositive_motion_proposal_adapter_width(self):
        with self.assertRaisesRegex(ValueError, "adapter_hidden_dim"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.motion_proposal.adapter_hidden_dim=0",
            ]))

    def test_v6_rejects_nonpositive_init_condition_scale(self):
        with self.assertRaisesRegex(ValueError, "init_condition_scale_mps"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.motion.init_condition_scale_mps=0",
            ]))

    def test_v6_rejects_nonpositive_residual_hidden_dim(self):
        with self.assertRaisesRegex(ValueError, "residual_hidden_dim"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.motion.residual_hidden_dim=0",
            ]))

    def test_v6_accepts_a_null_residual_hidden_dim_for_the_wide_head(self):
        config, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V6}",
            "dynamic_2dgs.motion.residual_hidden_dim=null",
        ]))
        self.assertIsNone(config.dynamic_2dgs.motion.residual_hidden_dim)

    def test_v6_rejects_match_support_larger_than_candidate_pool(self):
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V6}",
                "dynamic_2dgs.motion_proposal.match_count=17",
            ]))

    def test_v7_rejects_invalid_unmatched_hard_threshold(self):
        with self.assertRaisesRegex(ValueError, "unmatched_hard_threshold"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7}",
                "dynamic_2dgs.motion_proposal.unmatched_hard_threshold=1.0",
            ]))

    def test_v7_1_requires_four_selected_reciprocal_candidates(self):
        with self.assertRaisesRegex(ValueError, "match_count=4"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7_1}",
                "dynamic_2dgs.motion_proposal.match_count=3",
            ]))

    def test_v7_1_rejects_nonpositive_metric_evidence_scales(self):
        with self.assertRaisesRegex(ValueError, "spread_scale_m"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7_1}",
                "dynamic_2dgs.motion_proposal.spread_scale_m=0",
            ]))

    def test_v7_2_rejects_distance_and_reciprocal_matcher_keys(self):
        for key in (
            "candidate_count=16",
            "search_speed_max_mps=30",
            "dustbin_mode=evidence_mlp",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "Unsupported keys"
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V7_2}",
                    f"dynamic_2dgs.motion_proposal.{key}",
                ]))

    def test_v7_2_requires_dense_softmax_top4_contract(self):
        with self.assertRaisesRegex(ValueError, "match_count=4"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7_2}",
                "dynamic_2dgs.motion_proposal.match_count=3",
            ]))
        with self.assertRaisesRegex(ValueError, "ste_surrogate"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7_2}",
                "dynamic_2dgs.motion_proposal.ste_surrogate=top4_only",
            ]))
        with self.assertRaisesRegex(ValueError, "distance_prior_speed_mps"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7_2}",
                "dynamic_2dgs.motion_proposal.distance_prior_speed_mps=0",
            ]))

    def test_v7_2_requires_temporal_time_embedding(self):
        with self.assertRaisesRegex(ValueError, "use_time_embedding=true"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V7_2}",
                "dynamic_2dgs.temporal.use_time_embedding=false",
            ]))

    def test_v7_2_rejects_explicit_offset_conditions(self):
        for key in (
            "detach_init_condition=true",
            "velocity_embedding_dim=32",
            "duration_embedding_dim=16",
        ):
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "Unsupported keys"
            ):
                compose_fresh_config(OmegaConf.from_dotlist([
                    f"model.variant={DYNAMIC_VARIANT_V7_2}",
                    f"dynamic_2dgs.motion.{key}",
                ]))

    def test_group_l2_velocity_prior_is_accepted_for_every_variant(self):
        # The squared-norm prior is a mode, not a V9-only rule: V7.2 must stay
        # able to A/B it without editing the loss.
        config, _ = compose_fresh_config(OmegaConf.from_dotlist([
            f"model.variant={DYNAMIC_VARIANT_V7_2}",
            "dynamic_2dgs.regularization.velocity_l2.mode=final_group_l2",
        ]))
        self.assertEqual(
            config.dynamic_2dgs.regularization.velocity_l2.mode,
            "final_group_l2",
        )

    def test_velocity_prior_rejects_obsolete_component_mode(self):
        with self.assertRaisesRegex(ValueError, "final_group_l1"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "dynamic_2dgs.regularization.velocity_l2.mode=component_group_l1",
            ]))

    def test_current_dynamic_rejects_unimplemented_semantic_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT}",
                "dynamic_2dgs.motion.deform_rotation=true",
            ]))

    def test_unknown_variant_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown model.variant"):
            compose_fresh_config(
                OmegaConf.from_dotlist(["model.variant=does_not_exist"])
            )

    def _legacy_checkpoint(self, root: Path) -> Path:
        checkpoint = root / "run" / "epoch=1.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()

        legacy, _ = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={LEGACY_VARIANT}"])
        )
        stored = OmegaConf.to_container(legacy, resolve=False)
        stored.pop("model", None)
        stored.pop("g2g", None)
        wrapped = {key: {"value": value} for key, value in stored.items()}
        config_file = checkpoint.parent / "wandb/latest-run/files/config.yaml"
        config_file.parent.mkdir(parents=True)
        OmegaConf.save(OmegaConf.create(wrapped), config_file)
        return checkpoint

    def test_old_wandb_config_recovers_structural_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            config, source = load_checkpoint_config(
                checkpoint,
                allow_legacy_wandb_fallback=True,
            )
            self.assertEqual(config.model.variant, LEGACY_VARIANT)
            self.assertIn("g2g", config)
            self.assertEqual(len(config.g2g), 0)
            self.assertIn("wandb/latest-run/files/config.yaml", source)

    def test_checkpoint_variant_cannot_be_changed_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                f"model.variant={DYNAMIC_VARIANT}",
            ])
            with self.assertRaisesRegex(ValueError, "cannot be changed by CLI"):
                resolve_eval_config(cli)

    def test_checkpoint_structural_subtree_cannot_be_changed_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "p2g.grid_query.bg_radius_m=123",
            ])
            with self.assertRaisesRegex(ValueError, "changed roots: p2g"):
                resolve_eval_config(cli)

    def test_checkpoint_data_geometry_cannot_be_changed_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "data.window_us=2000000",
            ])
            with self.assertRaisesRegex(ValueError, "paths: data.window_us"):
                resolve_eval_config(cli)

    def test_checkpoint_bbox_source_alias_cannot_be_added_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"test.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "data.bbox_json_paths.val=/other/tracking.json",
            ])
            with self.assertRaisesRegex(ValueError, "data.bbox_json_paths"):
                resolve_eval_config(cli)

    def test_legacy_wandb_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([f"test.ckpt_path={checkpoint}"])
            with self.assertRaisesRegex(RuntimeError, "opt in explicitly"):
                resolve_eval_config(cli)

    def test_embedded_config_wins_over_mutable_wandb_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            embedded, _ = compose_fresh_config(
                OmegaConf.from_dotlist(["train.batch_size=9"])
            )
            torch.save(
                {
                    "experiment_config": OmegaConf.to_container(
                        embedded,
                        resolve=True,
                    )
                },
                checkpoint,
            )
            config, source = load_checkpoint_config(checkpoint)
            self.assertEqual(config.train.batch_size, 9)
            self.assertIn("embedded checkpoint config", source)

    def test_pre_lora_checkpoint_config_stays_on_legacy_feature_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "epoch=1.ckpt"
            stored, _ = compose_fresh_config()
            stored = OmegaConf.to_container(stored, resolve=True)
            stored["p2g"].pop("utonia_lora")
            torch.save({"experiment_config": stored}, checkpoint)

            config, _ = load_checkpoint_config(checkpoint)

            self.assertNotIn("utonia_lora", config.p2g)
            self.assertEqual(config.p2g.agg_mlp.out_dim, 576)

    def test_dynamic_variant_is_executable(self):
        config, _ = compose_fresh_config(
            OmegaConf.from_dotlist([f"model.variant={DYNAMIC_VARIANT}"])
        )
        self.assertIsNone(assert_model_variant_implemented(config))
        self.assertTrue(config.dynamic_2dgs.motion.zero_init)
        self.assertNotIn("max_speed_mps", config.dynamic_2dgs.motion)

    def test_train_resume_uses_checkpoint_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._legacy_checkpoint(Path(tmp))
            cli = OmegaConf.from_dotlist([
                f"train.ckpt_path={checkpoint}",
                "allow_legacy_wandb_fallback=true",
                "train.batch_size=7",
            ])
            config, source = resolve_main_config(cli)
            self.assertEqual(config.model.variant, LEGACY_VARIANT)
            self.assertEqual(config.train.batch_size, 7)
            self.assertIn("wandb/latest-run/files/config.yaml", source)

    def test_fresh_main_and_eval_use_identical_model_config(self):
        cli = OmegaConf.from_dotlist(["train.batch_size=3"])
        main_config, _ = resolve_main_config(cli)
        eval_config, _ = resolve_eval_config(cli)
        for key in ("model", "p2g", "g2g", "g2p"):
            self.assertEqual(
                OmegaConf.to_container(main_config[key], resolve=True),
                OmegaConf.to_container(eval_config[key], resolve=True),
            )


if __name__ == "__main__":
    unittest.main()
