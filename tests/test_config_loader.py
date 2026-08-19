from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.config_loader import (
    DYNAMIC_VARIANT,
    DYNAMIC_VARIANT_V1,
    DYNAMIC_VARIANT_V3,
    DYNAMIC_VARIANT_V3_1,
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
        self.assertEqual(config.dynamic_2dgs.temporal.time_reference_sec, 1.0)
        self.assertEqual(config.dynamic_2dgs.temporal.rope_position_scale, 0.2)
        self.assertEqual(config.p2g.agg_mlp.hidden_dim, 1024)
        self.assertEqual(config.p2g.agg_mlp.hidden_layers, 2)
        self.assertEqual(config.p2g.agg_mlp.out_dim, 576)
        self.assertEqual(config.p2g.joint_refiner.num_heads, 12)
        self.assertEqual(config.dynamic_2dgs.temporal.num_heads, 12)
        self.assertEqual(config.dynamic_2dgs.temporal.motion_head_count, 4)
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

    def test_current_dynamic_enables_velocity_l2_prior(self):
        config, _ = compose_fresh_config()
        prior = config.dynamic_2dgs.regularization.velocity_l2
        self.assertTrue(prior.enabled)
        self.assertEqual(prior.weight, 0.005)
        # V4's correspondence initializer is non-zero before training, so its
        # final-velocity guard must be active immediately.
        self.assertEqual(prior.warmup_steps, 0)
        self.assertEqual(prior.ramp_steps, 0)
        # Two motion priors with different gradient shapes must not stack.
        self.assertFalse(config.dynamic_2dgs.regularization.small_motion.enabled)

    def test_velocity_l2_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, "Unsupported keys"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "dynamic_2dgs.regularization.velocity_l2.epsilon_m=0.1",
            ]))

    def test_v4_motion_head_count_must_fit_temporal_heads(self):
        with self.assertRaisesRegex(ValueError, "motion_head_count"):
            compose_fresh_config(OmegaConf.from_dotlist([
                "dynamic_2dgs.temporal.motion_head_count=13",
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
