"""Checkpoint and optimizer compatibility contracts for Dynamic 2DGS.

The digests below were captured from pre-refactor commit
``a44d941a236948b1d9248936deaf2616d12de453``. They deliberately cover ordered
names, tensor shapes, and dtypes without making the test depend on a Git
checkout or copying a large historical source file.
"""
from __future__ import annotations

import hashlib
import importlib
import pathlib
import sys
import types
import unittest

import torch
from omegaconf import OmegaConf

from src.config_loader import (
    ADAPTIVE_COUNT_DYNAMIC_VARIANTS,
    DYNAMIC_VARIANTS,
    DYNAMIC_VARIANT_V1,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    compose_fresh_config,
)


_BACKEND_CLASS = {
    "dynamic_2dgs_direct_velocity_v1": "DynamicGaussianBackend",
    "dynamic_2dgs_physical_velocity_v3": "PhysicalVelocityGaussianBackend",
    "dynamic_2dgs_physical_velocity_v3_1": "PhysicalVelocityGaussianBackend",
    "dynamic_2dgs_attention_velocity_v4": (
        "AttentionInitializedVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v5": (
        "AttentionInitializedVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v6": (
        "ProposalInitializedVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v7": "WarpedProposalVelocityGaussianBackend",
    "dynamic_2dgs_attention_velocity_v7_1": (
        "WarpedProposalVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v7_2": (
        "StraightThroughProposalVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v8": (
        "ConsensusAttentionVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v9": (
        "PostAttentionProposalVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v10": (
        "LayerWeightedAttentionVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11": (
        "MaxSpeedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_1": (
        "SingleGaussianBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_2": (
        "FinalFeatureBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_3": (
        "SeedConditionedBarrierVelocityGaussianBackend"
    ),
    "dynamic_2dgs_attention_velocity_v11_4": (
        "MaxSpeedBarrierVelocityGaussianBackend"
    ),
}

_PROPOSAL_VARIANTS = {
    "dynamic_2dgs_attention_velocity_v6",
    "dynamic_2dgs_attention_velocity_v7",
    "dynamic_2dgs_attention_velocity_v7_1",
    "dynamic_2dgs_attention_velocity_v7_2",
}

# (ordered state_dict schema, ordered named-parameter list), SHA-256 prefixes.
_PRE_REFACTOR_MANIFEST = {
    "dynamic_2dgs_direct_velocity_v1": (
        "8e882fa993e2cf8fd5c9", "8c0381e3e4e6bd56ce12"
    ),
    "dynamic_2dgs_physical_velocity_v3": (
        "4d8b6764892f49cdcbd9", "750889ed52d8c0c951a9"
    ),
    "dynamic_2dgs_physical_velocity_v3_1": (
        "4d8b6764892f49cdcbd9", "750889ed52d8c0c951a9"
    ),
    "dynamic_2dgs_attention_velocity_v4": (
        "4d8b6764892f49cdcbd9", "750889ed52d8c0c951a9"
    ),
    "dynamic_2dgs_attention_velocity_v5": (
        "fb0ac56d44e6f9232bd2", "4b07d64b61632b27a32c"
    ),
    "dynamic_2dgs_attention_velocity_v6": (
        "4078c102a780f857cd23", "a80fd0e17556f5bc4c1c"
    ),
    "dynamic_2dgs_attention_velocity_v7": (
        "b2072f0ac928deee931c", "fedf4248c19ec944a552"
    ),
    "dynamic_2dgs_attention_velocity_v7_1": (
        "b2072f0ac928deee931c", "fedf4248c19ec944a552"
    ),
    "dynamic_2dgs_attention_velocity_v7_2": (
        "96dbcc6eda924003096a", "68326c05555f1e6cb615"
    ),
    "dynamic_2dgs_attention_velocity_v8": (
        "4388375c624e2d2c7514", "b44ce23131a852b0a120"
    ),
    "dynamic_2dgs_attention_velocity_v9": (
        "7f3658407384eeec306c", "30d35db9c5689160eec7"
    ),
    "dynamic_2dgs_attention_velocity_v10": (
        "f0add34f786db8d33181", "38a17230ad2df5268028"
    ),
    "dynamic_2dgs_attention_velocity_v11": (
        "a7f420bb2b15d7023d1f", "b72527798ae7652e170c"
    ),
    "dynamic_2dgs_attention_velocity_v11_1": (
        "faa47cf242bad2189a94", "7fbbbd9820a3046f0aa3"
    ),
    # V11.2 keeps V11's layer parameters in V11's order, drops the layer
    # mixture scalar it no longer has readouts to weight, and appends one
    # correspondence readout head.
    "dynamic_2dgs_attention_velocity_v11_2": (
        "4c40f14d3df7daad233b", "00d8592912ab2b795fa1"
    ),
    # V11.3 keeps V11's stack but drops the router, so its Gaussian head
    # rebuilds what the router's decoder did around it: a trunk taking the
    # two slots' 6 seed-delta columns, then attribute projections twice as
    # wide. That trunk is what separates it from V11.1's parameter names.
    "dynamic_2dgs_attention_velocity_v11_3": (
        "27b61a7e5c78bec30446", "274e243c1107ab2fa1b0"
    ),
    # V11.4 changes only the hinge's shape, and the barrier is a constant
    # of the graph rather than a parameter, so its digests are V11's. The
    # two are weight-compatible and behaviourally different; what keeps
    # them apart is the config embedded in the checkpoint.
    "dynamic_2dgs_attention_velocity_v11_4": (
        "a7f420bb2b15d7023d1f", "b72527798ae7652e170c"
    ),
}


def _digest(lines) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:20]


def _load_dynamic_module_without_renderer():
    """Import the backend package without initializing the CUDA renderer."""
    package_name = "src.models_new._checkpoint_compat_module"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(pathlib.Path(__file__).parents[1] / "src/models_new/module")
        ]
        package.__package__ = package_name
        sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.dynamic_gaussian")


_DYNAMIC = _load_dynamic_module_without_renderer()
_SEMANTIC_MODULE_BY_EXPORT = {
    "DynamicGaussianBackend": "backends",
    "FinalFeatureBarrierCrossAttention": "temporal",
    "FinalFeatureBarrierVelocityGaussianBackend": "backends",
    "ConsensusAttentionMotionMatcher": "attention_matching",
    "GaussianAttributeHead": "heads",
    "StraightThroughTop4MotionProposal": "motion_proposals",
    "DynamicGausTemp": "trajectory",
}


def _build_backend(variant):
    config, _source = compose_fresh_config(
        OmegaConf.from_dotlist([f"model.variant={variant}"])
    )
    kwargs = {"proposal_dim": 144} if variant in _PROPOSAL_VARIANTS else {}
    count = getattr(config.p2g, "grid_query", None)
    count_mode = str(getattr(count, "count_mode", "legacy")).lower()
    if count is not None and count_mode == "learned_gumbel":
        kwargs["gaussian_count_cfg"] = count
    else:
        kwargs["gaussians_per_token"] = int(getattr(count, "K_max", 1))
    backend_cls = getattr(_DYNAMIC, _BACKEND_CLASS[variant])
    return backend_cls(
        config.dynamic_2dgs,
        config.p2g.gs_params,
        dim=144,
        offset_bound=float(config.p2g.head_offset_bound),
        **kwargs,
    )


class DynamicCheckpointCompatibilityTest(unittest.TestCase):
    def test_every_variant_keeps_checkpoint_and_optimizer_parameter_order(self):
        self.assertEqual(set(DYNAMIC_VARIANTS), set(_PRE_REFACTOR_MANIFEST))
        for variant in DYNAMIC_VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(777)
                source = _build_backend(variant)
                state_schema = [
                    f"{name}|{tuple(value.shape)}|{value.dtype}"
                    for name, value in source.state_dict().items()
                ]
                parameter_names = [name for name, _ in source.named_parameters()]
                expected_schema, expected_order = _PRE_REFACTOR_MANIFEST[variant]
                self.assertEqual(_digest(state_schema), expected_schema)
                self.assertEqual(_digest(parameter_names), expected_order)

                # This is the exact strict restore operation used by a checkpoint.
                torch.manual_seed(991)
                restored = _build_backend(variant)
                incompatible = restored.load_state_dict(
                    source.state_dict(), strict=True
                )
                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])

                # Optimizer state_dict uses positional parameter ids. Give every
                # slot a unique scalar and ensure load maps it back to the same
                # ordered named parameter.
                old_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
                for index, parameter in enumerate(source.parameters()):
                    old_optimizer.state[parameter]["compatibility_index"] = (
                        torch.tensor(index)
                    )
                new_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
                new_optimizer.load_state_dict(old_optimizer.state_dict())
                for index, (_name, parameter) in enumerate(
                    restored.named_parameters()
                ):
                    self.assertEqual(
                        new_optimizer.state[parameter]["compatibility_index"].item(),
                        index,
                    )

    def test_v1_rejects_a_fixed_count_its_backend_cannot_restore(self):
        with self.assertRaisesRegex(ValueError, "K_max=1"):
            compose_fresh_config(OmegaConf.from_dotlist([
                f"model.variant={DYNAMIC_VARIANT_V1}",
                "p2g.grid_query.K_max=2",
                "p2g.grid_query.exp=1",
            ]))

    def test_adaptive_backends_keep_the_historical_fifth_positional_argument(self):
        for variant in (
            "dynamic_2dgs_attention_velocity_v10",
            "dynamic_2dgs_attention_velocity_v11",
            "dynamic_2dgs_attention_velocity_v11_2",
        ):
            with self.subTest(variant=variant):
                config, _source = compose_fresh_config(
                    OmegaConf.from_dotlist([f"model.variant={variant}"])
                )
                backend_cls = getattr(_DYNAMIC, _BACKEND_CLASS[variant])
                backend = backend_cls(
                    config.dynamic_2dgs,
                    config.p2g.gs_params,
                    144,
                    float(config.p2g.head_offset_bound),
                    config.p2g.grid_query,
                )
                self.assertTrue(backend.adaptive_count)
                self.assertEqual(backend.gaussian_head.k_max, 3)

    def test_single_seed_slot_preserves_the_legacy_geometry_transform(self):
        """The new explicit slot axis must preserve legacy (N, 3) inputs."""
        variant = "dynamic_2dgs_physical_velocity_v3"
        torch.manual_seed(31)
        backend = _build_backend(variant).eval()
        feature = torch.randn(4, 144)
        token = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.0],
            [0.1, 0.0, 0.0],
            [1.1, 0.2, 0.0],
        ])
        legacy_seed = token + torch.tensor([0.02, 0.01, 0.0])
        pose = torch.eye(4).repeat(2, 1, 1)
        pose[0, :3, 3] = torch.tensor([4.0, -2.0, 0.5])
        pose[1, :3, 3] = torch.tensor([-1.0, 3.0, -0.25])
        arguments = (
            feature,
            token,
            torch.tensor([2, 4]),
            torch.tensor([0, 0]),
            [pose],
            [torch.tensor([0.0, 1.0])],
            [torch.tensor([0.0, 0.8])],
            torch.tensor([0.8]),
        )
        legacy = backend(arguments[0], arguments[1], legacy_seed, *arguments[2:])
        slotted = backend(
            arguments[0], arguments[1], legacy_seed.unsqueeze(1), *arguments[2:]
        )
        for name in ("position", "coord_ref", "velocity"):
            torch.testing.assert_close(
                legacy["batch_gaussians"][0][name],
                slotted["batch_gaussians"][0][name],
            )

    def test_historical_adaptive_count_import_remains_available(self):
        self.assertIs(
            ADAPTIVE_COUNT_DYNAMIC_VARIANTS,
            ROUTER_CAPABLE_DYNAMIC_VARIANTS,
        )

    def test_historical_facade_reexports_the_moved_class_objects(self):
        for name, module_name in _SEMANTIC_MODULE_BY_EXPORT.items():
            with self.subTest(name=name):
                semantic_module = importlib.import_module(
                    "src.models_new._checkpoint_compat_module.dynamic."
                    + module_name
                )
                self.assertIs(
                    getattr(_DYNAMIC, name),
                    getattr(semantic_module, name),
                )


if __name__ == "__main__":
    unittest.main()
