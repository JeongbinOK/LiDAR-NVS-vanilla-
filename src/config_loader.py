"""Compose experiment configs and restore checkpoint-local configs safely.

The project intentionally keeps OmegaConf dot-list CLI overrides (``key=value``)
instead of introducing a second configuration framework.  Fresh runs compose a
shared base with one allow-listed model-variant overlay.  Checkpoint-backed runs
prefer the immutable config embedded in new checkpoints and never mix in the
checkout's current overlay.  Historical checkpoints fall back to W&B metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "nuscene_train.yaml"

LEGACY_VARIANT = "bbox_rigid_v1"
DYNAMIC_VARIANT_V1 = "dynamic_2dgs_direct_velocity_v1"
DYNAMIC_VARIANT_V3 = "dynamic_2dgs_physical_velocity_v3"
DYNAMIC_VARIANT_V3_1 = "dynamic_2dgs_physical_velocity_v3_1"
DYNAMIC_VARIANT_V4 = "dynamic_2dgs_attention_velocity_v4"
DYNAMIC_VARIANT_V5 = "dynamic_2dgs_attention_velocity_v5"
DYNAMIC_VARIANT_V6 = "dynamic_2dgs_attention_velocity_v6"
DYNAMIC_VARIANT_V7 = "dynamic_2dgs_attention_velocity_v7"
DYNAMIC_VARIANT_V7_1 = "dynamic_2dgs_attention_velocity_v7_1"
DYNAMIC_VARIANT_V7_2 = "dynamic_2dgs_attention_velocity_v7_2"
DYNAMIC_VARIANT_V8 = "dynamic_2dgs_attention_velocity_v8"
DYNAMIC_VARIANT_V9 = "dynamic_2dgs_attention_velocity_v9"
DYNAMIC_VARIANT_V10 = "dynamic_2dgs_attention_velocity_v10"
DYNAMIC_VARIANT_V11 = "dynamic_2dgs_attention_velocity_v11"
# The unsuffixed constant always names the current fresh-run baseline. Keep an
# explicit constant for each predecessor so checkpoint semantics stay exact.
DYNAMIC_VARIANT = DYNAMIC_VARIANT_V7_2
# V4/V5 use a dense coordinate expectation from selected attention heads. V8
# reuses selected final-layer heads with a consensus readout; V10 reads every
# head at every layer and therefore has no `motion_head_count` config key. V11
# reads head 0 at every layer, so its match head is fixed rather than configured.
_SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V4,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V8,
)
ATTENTION_VELOCITY_VARIANTS = (
    *_SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
)
# Both read correspondence out of temporal attention itself and route the
# refined token through the learned_gumbel K={1,2,3} grid Gaussian head.
ADAPTIVE_COUNT_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANT_V11,
)
# Head 0 alone carries the max-speed barrier and the coordinate readout; every
# other head keeps 3D RoPE.
BARRIER_MATCH_DYNAMIC_VARIANTS = (
    DYNAMIC_VARIANT_V11,
)
DYNAMIC_VARIANTS = (
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
)

# V3 and later share the physical-time contract. V4/V5 add attention-derived
# initialization; V6 uses an independent proposal branch.
PHYSICAL_VELOCITY_VARIANTS = (
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
)

WARPED_PROPOSAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
)
DURATION_OFFSET_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V7,
    DYNAMIC_VARIANT_V7_1,
)
# These proposals run *before* temporal attention on raw LoRA Utonia tokens,
# which is what makes them need a detached query warp to reach the trunk.
PROPOSAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V6,
    *WARPED_PROPOSAL_VELOCITY_VARIANTS,
)
# V9 keeps a standalone proposal branch but reads the refined feature instead,
# so it shares the `motion_proposal` config root and nothing else with the above.
REFINED_PROPOSAL_VELOCITY_VARIANTS = (
    DYNAMIC_VARIANT_V9,
)

VARIANT_CONFIG_PATHS = {
    LEGACY_VARIANT: REPO_ROOT / "config" / "variants" / f"{LEGACY_VARIANT}.yaml",
    DYNAMIC_VARIANT_V1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V1}.yaml"
    ),
    DYNAMIC_VARIANT_V3: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V3}.yaml"
    ),
    DYNAMIC_VARIANT_V3_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V3_1}.yaml"
    ),
    DYNAMIC_VARIANT_V4: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V4}.yaml"
    ),
    DYNAMIC_VARIANT_V5: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V5}.yaml"
    ),
    DYNAMIC_VARIANT_V6: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V6}.yaml"
    ),
    DYNAMIC_VARIANT_V7: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V7}.yaml"
    ),
    DYNAMIC_VARIANT_V7_1: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V7_1}.yaml"
    ),
    DYNAMIC_VARIANT_V7_2: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V7_2}.yaml"
    ),
    DYNAMIC_VARIANT_V8: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V8}.yaml"
    ),
    DYNAMIC_VARIANT_V9: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V9}.yaml"
    ),
    DYNAMIC_VARIANT_V10: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V10}.yaml"
    ),
    DYNAMIC_VARIANT_V11: (
        REPO_ROOT / "config" / "variants" / f"{DYNAMIC_VARIANT_V11}.yaml"
    ),
}

# W&B omits empty mappings, and configs saved before model variants existed have
# no model block.  Left-merging these defaults restores the historical contract.
STRUCTURAL_DEFAULTS = {
    "model": {"variant": LEGACY_VARIANT},
    "g2g": {},
}

_REQUIRED_ROOTS = (
    "model",
    "p2g",
    "g2g",
    "g2p",
    "data",
    "loss",
    "train",
    "test",
    "logger",
)

# These roots control model construction or shape-neutral rendering semantics.
# Runtime knobs can change during resume/evaluation, but saved weights must not
# be reinterpreted by modifying one of these subtrees from the CLI.
_CHECKPOINT_STRUCTURAL_ROOTS = (
    "model",
    "p2g",
    "g2g",
    "g2p",
    "dynamic_2dgs",
)

_CHECKPOINT_PROTECTED_PATHS = (
    "data.window_us",
    "data.sample_gap_us",
    "data.gt_middle_count",
    "data.pair_mode",
    "data.pair_kf_stride",
    "data.mode",
    "data.bbox_json_path",
    "data.bbox_json_paths",
    "data.vfov",
    "data.hfov",
    "data.image_height",
    "data.image_width",
    "data.ring_to_elevation_deg",
    "data.ego_radius",
)

_IMPLEMENTED_VARIANTS = {LEGACY_VARIANT, *DYNAMIC_VARIANTS}

# Pointcept/Utonia encoder widths from config/utonia_pretrained.yaml. LoRA mode
# exits at the configured encoder stage and bypasses the legacy fusion MLP, so
# static config validation must use this width for downstream attention.
_UTONIA_ENCODER_STAGE_DIMS = (54, 108, 216, 432, 576)


def _as_config(value=None) -> DictConfig:
    if value is None:
        return OmegaConf.create()
    if OmegaConf.is_config(value):
        return value
    return OmegaConf.create(value)


def _has_path(config, dotted_path: str) -> bool:
    current = config
    for part in dotted_path.split("."):
        if not OmegaConf.is_dict(current) or part not in current:
            return False
        current = current[part]
    return True


def _variant(config, *, default=LEGACY_VARIANT) -> str:
    value = OmegaConf.select(config, "model.variant", default=default)
    return str(value)


def _reject_unknown_keys(config, dotted_path: str, allowed) -> None:
    block = OmegaConf.select(config, dotted_path, default=None)
    if block is None or not OmegaConf.is_dict(block):
        raise ValueError(f"{dotted_path} must be a config mapping")
    unknown = sorted(set(block.keys()) - set(allowed))
    if unknown:
        raise ValueError(
            f"Unsupported keys under {dotted_path}: {', '.join(unknown)}"
        )


def _validate_3d_rope_width(dim: int, num_heads: int, label: str) -> None:
    if dim % num_heads != 0:
        raise ValueError(f"{label}: dim={dim} must be divisible by heads={num_heads}")
    head_dim = dim // num_heads
    if head_dim % 6 != 0:
        raise ValueError(
            f"{label}: head_dim={head_dim} must be divisible by six for "
            "the xyz/sin-cos 3D RoPE split"
        )


def _validate_utonia_lora_config(config) -> bool:
    block = OmegaConf.select(config, "p2g.utonia_lora", default=None)
    if block is None:
        return False
    _reject_unknown_keys(
        config,
        "p2g.utonia_lora",
        {
            "enable", "input_mode", "rank", "linear_rank", "conv_rank",
            "alpha", "linear_alpha", "conv_alpha", "dropout",
        },
    )
    enabled = bool(OmegaConf.select(config, "p2g.utonia_lora.enable", default=False))
    if not enabled:
        return False
    if not bool(OmegaConf.select(config, "p2g.freeze_utonia", default=True)):
        raise ValueError(
            "p2g.utonia_lora.enable=true requires p2g.freeze_utonia=true"
        )
    input_mode = str(OmegaConf.select(
        config, "p2g.utonia_lora.input_mode", default="xyzi"
    )).lower()
    if input_mode != "xyzi":
        raise ValueError(
            "p2g.utonia_lora.input_mode currently supports only 'xyzi'"
        )
    shared_rank = int(OmegaConf.select(
        config, "p2g.utonia_lora.rank", default=16
    ))
    for name in ("linear_rank", "conv_rank"):
        value = int(OmegaConf.select(
            config, f"p2g.utonia_lora.{name}", default=shared_rank
        ))
        if value <= 0:
            raise ValueError(f"p2g.utonia_lora.{name} must be positive")
    shared_alpha = float(OmegaConf.select(
        config, "p2g.utonia_lora.alpha", default=shared_rank
    ))
    for name in ("linear_alpha", "conv_alpha"):
        value = float(OmegaConf.select(
            config, f"p2g.utonia_lora.{name}", default=shared_alpha
        ))
        if value <= 0.0:
            raise ValueError(f"p2g.utonia_lora.{name} must be positive")
    dropout = float(OmegaConf.select(
        config, "p2g.utonia_lora.dropout", default=0.0
    ))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("p2g.utonia_lora.dropout must be in [0, 1)")
    return True


def _configured_p2g_trunk_dim(config, *, lora_enabled: bool) -> int:
    if not lora_enabled:
        return int(OmegaConf.select(config, "p2g.agg_mlp.out_dim"))
    stage = int(OmegaConf.select(config, "p2g.utonia_feature_stage"))
    if not 0 <= stage < len(_UTONIA_ENCODER_STAGE_DIMS):
        raise ValueError(
            "p2g.utonia_feature_stage must be in [0, "
            f"{len(_UTONIA_ENCODER_STAGE_DIMS) - 1}], got {stage}"
        )
    return _UTONIA_ENCODER_STAGE_DIMS[stage]


def _load_variant_overlay(variant: str) -> tuple[DictConfig, Path]:
    try:
        path = VARIANT_CONFIG_PATHS[variant]
    except KeyError as exc:
        allowed = ", ".join(sorted(VARIANT_CONFIG_PATHS))
        raise ValueError(
            f"Unknown model.variant={variant!r}; expected one of: {allowed}"
        ) from exc
    if not path.is_file():
        raise FileNotFoundError(f"Variant config does not exist: {path}")
    overlay = OmegaConf.load(path)
    declared = _variant(overlay, default="")
    if declared != variant:
        raise ValueError(
            f"Variant overlay {path} declares model.variant={declared!r}, "
            f"expected {variant!r}"
        )
    return overlay, path


def _validate_adaptive_gaussian_count(config, label: str) -> None:
    """Shared learned_gumbel K={1,2,3} router contract for V10 and V11."""

    count_path = "p2g.grid_query"
    _reject_unknown_keys(
        config,
        count_path,
        {"count_mode", "grad_balance", "learned_count"},
    )
    if str(OmegaConf.select(
        config, f"{count_path}.count_mode"
    )).lower() != "learned_gumbel":
        raise ValueError(
            f"{label} requires p2g.grid_query.count_mode=learned_gumbel"
        )
    if str(OmegaConf.select(
        config, f"{count_path}.grad_balance"
    )).lower() not in ("sqrt_k", "none"):
        raise ValueError(
            f"{label} grid_query.grad_balance must be 'sqrt_k' or 'none'"
        )
    learned_path = f"{count_path}.learned_count"
    _reject_unknown_keys(
        config,
        learned_path,
        {"K_max", "tau", "seed_mode", "grad_balance_scope", "budget", "logging"},
    )
    if int(OmegaConf.select(config, f"{learned_path}.K_max")) != 3:
        raise ValueError(f"{label} requires grid_query.learned_count.K_max=3")
    if float(OmegaConf.select(config, f"{learned_path}.tau")) <= 0.0:
        raise ValueError(f"{label} grid_query.learned_count.tau must be positive")
    if str(OmegaConf.select(
        config, f"{learned_path}.seed_mode"
    )).lower() != "range_quantile":
        raise ValueError(
            f"{label} grid_query.learned_count.seed_mode must be "
            "'range_quantile'"
        )
    if str(OmegaConf.select(
        config, f"{learned_path}.grad_balance_scope"
    )).lower() not in ("output", "token"):
        raise ValueError(
            f"{label} learned_count.grad_balance_scope must be "
            "'output' or 'token'"
        )
    budget_path = f"{learned_path}.budget"
    _reject_unknown_keys(config, budget_path, {"enable"})
    if bool(OmegaConf.select(config, f"{budget_path}.enable")):
        raise ValueError(f"{label} does not use a routing budget loss")
    logging_path = f"{learned_path}.logging"
    _reject_unknown_keys(
        config, logging_path, {"enable", "interval", "range_edges_m"}
    )
    if int(OmegaConf.select(config, f"{logging_path}.interval")) <= 0:
        raise ValueError(
            f"{label} learned_count.logging.interval must be positive"
        )


def _variant_label(variant: str) -> str:
    """Short human label ("V7.2", "V10") for a registered variant identifier."""

    head, separator, tail = variant.rpartition("_v")
    if head and separator and tail.replace("_", "").isdigit():
        return "V" + tail.replace("_", ".")
    return variant


def validate_experiment_config(config) -> None:
    missing = [key for key in _REQUIRED_ROOTS if key not in config]
    if missing:
        raise ValueError("Experiment config is missing root keys: " + ", ".join(missing))

    variant = _variant(config)
    if variant not in VARIANT_CONFIG_PATHS:
        allowed = ", ".join(sorted(VARIANT_CONFIG_PATHS))
        raise ValueError(
            f"Unknown model.variant={variant!r}; expected one of: {allowed}"
        )

    lora_enabled = _validate_utonia_lora_config(config)

    if variant in DYNAMIC_VARIANTS:
        if str(OmegaConf.select(config, "p2g.anchor_mode", default="")) != "grid":
            raise ValueError(f"{variant} requires p2g.anchor_mode=grid")
        if variant in ADAPTIVE_COUNT_DYNAMIC_VARIANTS and not lora_enabled:
            raise ValueError(
                f"{_variant_label(variant)} requires p2g.utonia_lora.enable=true"
            )
        if OmegaConf.select(config, "dynamic_2dgs", default=None) is None:
            raise ValueError(f"{variant} requires a dynamic_2dgs config block")
        trunk_dim = _configured_p2g_trunk_dim(
            config, lora_enabled=lora_enabled
        )
        temporal_heads = int(OmegaConf.select(
            config, "dynamic_2dgs.temporal.num_heads"
        ))
        if variant == DYNAMIC_VARIANT_V10:
            if trunk_dim % temporal_heads != 0:
                raise ValueError(
                    "dynamic temporal attention: "
                    f"dim={trunk_dim} must be divisible by heads={temporal_heads}"
                )
        else:
            _validate_3d_rope_width(
                trunk_dim, temporal_heads, "dynamic temporal attention"
            )
        if not lora_enabled and bool(OmegaConf.select(
            config, "p2g.joint_refiner.enable", default=False
        )):
            joint_heads = int(OmegaConf.select(
                config, "p2g.joint_refiner.num_heads"
            ))
            _validate_3d_rope_width(
                trunk_dim, joint_heads, "post-fusion joint attention"
            )
        if variant in PHYSICAL_VELOCITY_VARIANTS:
            dynamic_keys = {
                "temporal", "gaussian_head", "motion", "regularization",
            }
            if variant in (
                *PROPOSAL_VELOCITY_VARIANTS,
                *REFINED_PROPOSAL_VELOCITY_VARIANTS,
            ):
                dynamic_keys.add("motion_proposal")
            if variant == DYNAMIC_VARIANT_V8:
                dynamic_keys.add("motion_matching")
            _reject_unknown_keys(
                config,
                "dynamic_2dgs",
                dynamic_keys,
            )
            if variant == DYNAMIC_VARIANT_V10:
                temporal_keys = {
                    "implementation", "layers", "num_heads", "mlp_ratio",
                    "use_time_embedding", "time_frequencies",
                    "time_embedding_dim", "time_reference_sec",
                    "position_encoding", "distance_bias_speed_mps", "qk_norm",
                    "layer_weight_hidden_dim", "layer_scale_init",
                }
            elif variant in BARRIER_MATCH_DYNAMIC_VARIANTS:
                temporal_keys = {
                    "implementation", "layers", "num_heads", "mlp_ratio",
                    "use_time_embedding", "time_frequencies",
                    "time_embedding_dim", "time_reference_sec",
                    "position_encoding", "barrier_speed_mps", "barrier_weight",
                    "match_chunk_size", "rope_base", "rope_position_scale",
                    "qk_norm", "layer_scale_init",
                }
            else:
                temporal_keys = {
                    "implementation", "layers", "num_heads", "mlp_ratio",
                    "rope_base", "rope_position_scale", "layer_scale_init",
                }
                if variant in WARPED_PROPOSAL_VELOCITY_VARIANTS:
                    temporal_keys.add("use_time_embedding")
                    if variant == DYNAMIC_VARIANT_V7_2:
                        temporal_keys.update({
                            "time_frequencies", "time_embedding_dim",
                            "time_reference_sec",
                        })
                elif variant == DYNAMIC_VARIANT_V6:
                    temporal_keys.update({
                        "time_frequencies", "time_hidden_dim",
                        "time_reference_sec",
                    })
                else:
                    # V3-V5 checkpoints keep the historical two-stage time path.
                    temporal_keys.update({
                        "time_frequencies", "time_embedding_dim",
                        "time_reference_sec",
                    })
                    if variant == DYNAMIC_VARIANT_V8:
                        temporal_keys.update({
                            "use_time_embedding", "tie_motion_qk_init",
                        })
                    elif variant == DYNAMIC_VARIANT_V9:
                        temporal_keys.add("use_time_embedding")
            if variant in _SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS:
                temporal_keys.add("motion_head_count")
            if variant in (DYNAMIC_VARIANT_V5, DYNAMIC_VARIANT_V8):
                temporal_keys.add("qk_norm")
            if variant == DYNAMIC_VARIANT_V5:
                temporal_keys.update({
                    "motion_rope_base", "motion_rope_position_scale",
                })
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.temporal",
                temporal_keys,
            )
            if variant in _SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS:
                motion_heads = int(OmegaConf.select(
                    config, "dynamic_2dgs.temporal.motion_head_count"
                ))
                if not 0 < motion_heads <= temporal_heads:
                    raise ValueError(
                        "dynamic_2dgs.temporal.motion_head_count must be in "
                        f"[1, {temporal_heads}]"
                    )
            if variant == DYNAMIC_VARIANT_V8:
                matching_path = "dynamic_2dgs.motion_matching"
                _reject_unknown_keys(
                    config,
                    matching_path,
                    {"candidate_count", "match_count", "score_chunk_size"},
                )
                for name in (
                    "candidate_count", "match_count", "score_chunk_size",
                ):
                    value = int(OmegaConf.select(
                        config, f"{matching_path}.{name}"
                    ))
                    if value <= 0:
                        raise ValueError(
                            f"{matching_path}.{name} must be positive"
                        )
                candidate_count = int(OmegaConf.select(
                    config, f"{matching_path}.candidate_count"
                ))
                match_count = int(OmegaConf.select(
                    config, f"{matching_path}.match_count"
                ))
                if match_count > candidate_count:
                    raise ValueError(
                        "motion_matching.match_count must not exceed "
                        "candidate_count"
                    )
                if match_count != 4:
                    raise ValueError("V8 requires motion_matching.match_count=4")
            if variant in WARPED_PROPOSAL_VELOCITY_VARIANTS:
                use_time_embedding = bool(OmegaConf.select(
                    config,
                    "dynamic_2dgs.temporal.use_time_embedding",
                    default=True,
                ))
                if variant == DYNAMIC_VARIANT_V7_2:
                    if not use_time_embedding:
                        raise ValueError(
                            "V7.2 requires dynamic_2dgs.temporal."
                            "use_time_embedding=true"
                        )
                elif use_time_embedding:
                    raise ValueError(
                        "V7 requires dynamic_2dgs.temporal."
                        "use_time_embedding=false"
                    )
            elif variant == DYNAMIC_VARIANT_V9:
                if not bool(OmegaConf.select(
                    config,
                    "dynamic_2dgs.temporal.use_time_embedding",
                    default=True,
                )):
                    raise ValueError(
                        "V9 requires dynamic_2dgs.temporal."
                        "use_time_embedding=true"
                    )
            elif variant == DYNAMIC_VARIANT_V10:
                temporal_path = "dynamic_2dgs.temporal"
                if not bool(OmegaConf.select(
                    config, f"{temporal_path}.use_time_embedding"
                )):
                    raise ValueError(
                        "V10 requires dynamic_2dgs.temporal."
                        "use_time_embedding=true"
                    )
                if str(OmegaConf.select(
                    config, f"{temporal_path}.position_encoding"
                )).lower() != "distance_bias":
                    raise ValueError(
                        "V10 temporal.position_encoding must be 'distance_bias'"
                    )
                if float(OmegaConf.select(
                    config, f"{temporal_path}.distance_bias_speed_mps"
                )) <= 0.0:
                    raise ValueError(
                        "V10 temporal.distance_bias_speed_mps must be positive"
                    )
                if not bool(OmegaConf.select(
                    config, f"{temporal_path}.qk_norm"
                )):
                    raise ValueError("V10 requires temporal.qk_norm=true")
                if int(OmegaConf.select(
                    config, f"{temporal_path}.layer_weight_hidden_dim"
                )) <= 0:
                    raise ValueError(
                        "V10 temporal.layer_weight_hidden_dim must be positive"
                    )
                _validate_adaptive_gaussian_count(config, "V10")
            elif variant in BARRIER_MATCH_DYNAMIC_VARIANTS:
                label = _variant_label(variant)
                temporal_path = "dynamic_2dgs.temporal"
                if not bool(OmegaConf.select(
                    config, f"{temporal_path}.use_time_embedding"
                )):
                    raise ValueError(
                        f"{label} requires dynamic_2dgs.temporal."
                        "use_time_embedding=true"
                    )
                if str(OmegaConf.select(
                    config, f"{temporal_path}.position_encoding"
                )).lower() != "barrier_rope_split":
                    raise ValueError(
                        f"{label} temporal.position_encoding must be "
                        "'barrier_rope_split'"
                    )
                if float(OmegaConf.select(
                    config, f"{temporal_path}.barrier_speed_mps"
                )) <= 0.0:
                    raise ValueError(
                        f"{label} temporal.barrier_speed_mps must be positive"
                    )
                if float(OmegaConf.select(
                    config, f"{temporal_path}.barrier_weight"
                )) < 0.0:
                    raise ValueError(
                        f"{label} temporal.barrier_weight must be non-negative"
                    )
                if int(OmegaConf.select(
                    config, f"{temporal_path}.match_chunk_size"
                )) <= 0:
                    raise ValueError(
                        f"{label} temporal.match_chunk_size must be positive"
                    )
                if not bool(OmegaConf.select(
                    config, f"{temporal_path}.qk_norm"
                )):
                    raise ValueError(f"{label} requires temporal.qk_norm=true")
                # Head 0 is the match head; every remaining head keeps 3D RoPE,
                # so V11 still needs at least one of each.
                if int(OmegaConf.select(
                    config, f"{temporal_path}.num_heads"
                )) < 2:
                    raise ValueError(
                        f"{label} requires dynamic_2dgs.temporal.num_heads >= 2"
                    )
                _validate_adaptive_gaussian_count(config, label)
            elif variant == DYNAMIC_VARIANT_V6:
                time_hidden_dim = int(OmegaConf.select(
                    config, "dynamic_2dgs.temporal.time_hidden_dim"
                ))
                if time_hidden_dim <= 0:
                    raise ValueError(
                        "dynamic_2dgs.temporal.time_hidden_dim must be positive"
                    )
            if variant in REFINED_PROPOSAL_VELOCITY_VARIANTS:
                proposal_path = "dynamic_2dgs.motion_proposal"
                _reject_unknown_keys(
                    config,
                    proposal_path,
                    {
                        "descriptor_mode", "descriptor_dim", "temperature",
                        "score_chunk_size", "readout",
                        "distance_prior_speed_mps",
                    },
                )
                for name in ("descriptor_dim", "score_chunk_size"):
                    value = int(OmegaConf.select(
                        config, f"{proposal_path}.{name}"
                    ))
                    if value <= 0:
                        raise ValueError(
                            f"{proposal_path}.{name} must be positive"
                        )
                if float(OmegaConf.select(
                    config, f"{proposal_path}.temperature"
                )) <= 0.0:
                    raise ValueError(
                        "motion_proposal.temperature must be positive"
                    )
                if float(OmegaConf.select(
                    config, f"{proposal_path}.distance_prior_speed_mps"
                )) <= 0.0:
                    raise ValueError(
                        "motion_proposal.distance_prior_speed_mps must be "
                        "positive"
                    )
                if str(OmegaConf.select(
                    config, f"{proposal_path}.descriptor_mode"
                )) != "projected_l2":
                    raise ValueError(
                        "V9 motion_proposal.descriptor_mode must be "
                        "'projected_l2'"
                    )
                if str(OmegaConf.select(
                    config, f"{proposal_path}.readout"
                )) != "dense_expectation":
                    raise ValueError(
                        "V9 motion_proposal.readout must be "
                        "'dense_expectation'"
                    )
            if variant in PROPOSAL_VELOCITY_VARIANTS:
                proposal_path = "dynamic_2dgs.motion_proposal"
                if variant == DYNAMIC_VARIANT_V7_2:
                    _reject_unknown_keys(
                        config,
                        proposal_path,
                        {
                            "descriptor_mode", "match_count", "temperature",
                            "score_chunk_size", "ste_surrogate",
                            "distance_prior_speed_mps",
                        },
                    )
                    for name in ("match_count", "score_chunk_size"):
                        value = int(OmegaConf.select(
                            config, f"{proposal_path}.{name}"
                        ))
                        if value <= 0:
                            raise ValueError(
                                f"{proposal_path}.{name} must be positive"
                            )
                    if int(OmegaConf.select(
                        config, f"{proposal_path}.match_count"
                    )) != 4:
                        raise ValueError(
                            "V7.2 requires motion_proposal.match_count=4"
                        )
                    if float(OmegaConf.select(
                        config, f"{proposal_path}.temperature"
                    )) <= 0.0:
                        raise ValueError(
                            "motion_proposal.temperature must be positive"
                        )
                    if float(OmegaConf.select(
                        config, f"{proposal_path}.distance_prior_speed_mps"
                    )) <= 0.0:
                        raise ValueError(
                            "motion_proposal.distance_prior_speed_mps must be "
                            "positive"
                        )
                    if str(OmegaConf.select(
                        config, f"{proposal_path}.descriptor_mode"
                    )) != "direct_l2":
                        raise ValueError(
                            "V7.2 motion_proposal.descriptor_mode must be "
                            "'direct_l2'"
                        )
                    if str(OmegaConf.select(
                        config, f"{proposal_path}.ste_surrogate"
                    )) != "dense_softmax":
                        raise ValueError(
                            "V7.2 motion_proposal.ste_surrogate must be "
                            "'dense_softmax'"
                        )
                else:
                    proposal_keys = {
                        "candidate_count", "match_count", "temperature",
                        "score_chunk_size", "search_speed_min_mps",
                        "search_speed_init_mps", "search_speed_max_mps",
                    }
                    if variant == DYNAMIC_VARIANT_V6:
                        proposal_keys.update({
                            "adapter_hidden_dim", "descriptor_dim",
                            "dustbin_similarity_init",
                            "dustbin_token_conditioned",
                        })
                    else:
                        proposal_keys.update({
                            "descriptor_mode", "dustbin_mode",
                            "dustbin_hidden_dim", "dustbin_prior_probability",
                            "unmatched_gate_mode", "unmatched_hard_threshold",
                        })
                        if variant == DYNAMIC_VARIANT_V7_1:
                            proposal_keys.update({
                                "dustbin_evidence_mode",
                                "mean_displacement_scale_m",
                                "spread_scale_m",
                            })
                    _reject_unknown_keys(config, proposal_path, proposal_keys)
                    integer_keys = [
                        "candidate_count", "match_count", "score_chunk_size",
                    ]
                    if variant == DYNAMIC_VARIANT_V6:
                        integer_keys.extend([
                            "adapter_hidden_dim", "descriptor_dim",
                        ])
                    else:
                        integer_keys.append("dustbin_hidden_dim")
                    for name in integer_keys:
                        value = int(OmegaConf.select(
                            config, f"{proposal_path}.{name}"
                        ))
                        if value <= 0:
                            raise ValueError(
                                f"{proposal_path}.{name} must be positive"
                            )
                    match_count = int(OmegaConf.select(
                        config, f"{proposal_path}.match_count"
                    ))
                    candidate_count = int(OmegaConf.select(
                        config, f"{proposal_path}.candidate_count"
                    ))
                    if match_count > candidate_count:
                        raise ValueError(
                            "motion_proposal.match_count must not exceed "
                            "candidate_count"
                        )
                    if float(OmegaConf.select(
                        config, f"{proposal_path}.temperature"
                    )) <= 0.0:
                        raise ValueError(
                            "motion_proposal.temperature must be positive"
                        )
                    if variant == DYNAMIC_VARIANT_V6:
                        dustbin_similarity = float(OmegaConf.select(
                            config, f"{proposal_path}.dustbin_similarity_init"
                        ))
                        if not -1.0 < dustbin_similarity < 1.0:
                            raise ValueError(
                                "motion_proposal.dustbin_similarity_init must "
                                "be in (-1, 1)"
                            )
                    else:
                        descriptor_mode = str(OmegaConf.select(
                            config, f"{proposal_path}.descriptor_mode"
                        ))
                        if descriptor_mode != "direct_l2":
                            raise ValueError(
                                "V7 motion_proposal.descriptor_mode must be "
                                "'direct_l2'"
                            )
                        dustbin_mode = str(OmegaConf.select(
                            config, f"{proposal_path}.dustbin_mode"
                        ))
                        if dustbin_mode != "evidence_mlp":
                            raise ValueError(
                                "V7 motion_proposal.dustbin_mode must be "
                                "'evidence_mlp'"
                            )
                        dustbin_prior = float(OmegaConf.select(
                            config, f"{proposal_path}.dustbin_prior_probability"
                        ))
                        if not 0.0 < dustbin_prior < 1.0:
                            raise ValueError(
                                "motion_proposal.dustbin_prior_probability must "
                                "be in (0, 1)"
                            )
                        unmatched_gate_mode = str(OmegaConf.select(
                            config, f"{proposal_path}.unmatched_gate_mode"
                        ))
                        if unmatched_gate_mode not in ("soft", "ste_hard"):
                            raise ValueError(
                                "motion_proposal.unmatched_gate_mode must be "
                                "'soft' or 'ste_hard'"
                            )
                        unmatched_threshold = float(OmegaConf.select(
                            config, f"{proposal_path}.unmatched_hard_threshold"
                        ))
                        if not 0.0 < unmatched_threshold < 1.0:
                            raise ValueError(
                                "motion_proposal.unmatched_hard_threshold must "
                                "be in (0, 1)"
                            )
                        if variant == DYNAMIC_VARIANT_V7_1:
                            evidence_mode = str(OmegaConf.select(
                                config, f"{proposal_path}.dustbin_evidence_mode"
                            ))
                            if evidence_mode != "mean_spread_reciprocal_m4":
                                raise ValueError(
                                    "V7.1 motion_proposal.dustbin_evidence_mode "
                                    "must be 'mean_spread_reciprocal_m4'"
                                )
                            if match_count != 4:
                                raise ValueError(
                                    "V7.1 requires motion_proposal.match_count=4"
                                )
                            for name in (
                                "mean_displacement_scale_m", "spread_scale_m",
                            ):
                                value = float(OmegaConf.select(
                                    config, f"{proposal_path}.{name}"
                                ))
                                if value <= 0.0:
                                    raise ValueError(
                                        f"{proposal_path}.{name} must be positive"
                                    )
                    speed_min = float(OmegaConf.select(
                        config, f"{proposal_path}.search_speed_min_mps"
                    ))
                    speed_init = float(OmegaConf.select(
                        config, f"{proposal_path}.search_speed_init_mps"
                    ))
                    speed_max = float(OmegaConf.select(
                        config, f"{proposal_path}.search_speed_max_mps"
                    ))
                    if not 0.0 < speed_min < speed_init < speed_max:
                        raise ValueError(
                            "motion_proposal search speeds must satisfy "
                            "0 < min < init < max"
                        )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.gaussian_head",
                {"initial_opacity", "initial_scale_m"},
            )
            motion_keys = {"zero_init"}
            if variant == DYNAMIC_VARIANT_V6:
                motion_keys.update({
                    "init_conditioned_residual",
                    "init_condition_scale_mps",
                    "detach_init_condition",
                    "residual_hidden_dim",
                })
            elif variant in DURATION_OFFSET_VELOCITY_VARIANTS:
                motion_keys.update({
                    "detach_init_condition", "velocity_embedding_dim",
                    "duration_embedding_dim", "duration_frequencies",
                    "duration_reference_sec", "residual_hidden_dim",
                })
            elif variant in (
                DYNAMIC_VARIANT_V7_2,
                DYNAMIC_VARIANT_V9,
                DYNAMIC_VARIANT_V10,
            ):
                motion_keys.add("residual_hidden_dim")
            elif variant in BARRIER_MATCH_DYNAMIC_VARIANTS:
                # V11's offset head embeds v_init the way V8's does.
                motion_keys.update({
                    "residual_hidden_dim", "velocity_embedding_dim",
                    "detach_init_condition",
                })
            elif variant == DYNAMIC_VARIANT_V8:
                motion_keys.update({
                    "detach_init_condition", "velocity_embedding_dim",
                    "residual_hidden_dim",
                })
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.motion",
                motion_keys,
            )
            if (
                variant == DYNAMIC_VARIANT_V6
                and _has_path(
                    config, "dynamic_2dgs.motion.init_condition_scale_mps"
                )
                and float(OmegaConf.select(
                    config,
                    "dynamic_2dgs.motion.init_condition_scale_mps",
                )) <= 0.0
            ):
                raise ValueError(
                    "dynamic_2dgs.motion.init_condition_scale_mps must be positive"
                )
            if (
                variant in (
                    *PROPOSAL_VELOCITY_VARIANTS,
                    *REFINED_PROPOSAL_VELOCITY_VARIANTS,
                    DYNAMIC_VARIANT_V8,
                    DYNAMIC_VARIANT_V10,
                    *BARRIER_MATCH_DYNAMIC_VARIANTS,
                )
                and _has_path(config, "dynamic_2dgs.motion.residual_hidden_dim")
            ):
                residual_hidden_dim = OmegaConf.select(
                    config, "dynamic_2dgs.motion.residual_hidden_dim"
                )
                if residual_hidden_dim is not None and int(
                    residual_hidden_dim
                ) <= 0:
                    raise ValueError(
                        "dynamic_2dgs.motion.residual_hidden_dim must be "
                        "positive when set"
                    )
            if variant in DURATION_OFFSET_VELOCITY_VARIANTS:
                for name in (
                    "velocity_embedding_dim", "duration_embedding_dim",
                    "duration_frequencies", "residual_hidden_dim",
                ):
                    raw_value = OmegaConf.select(
                        config, f"dynamic_2dgs.motion.{name}"
                    )
                    if raw_value is None or int(raw_value) <= 0:
                        raise ValueError(
                            f"dynamic_2dgs.motion.{name} must be positive"
                        )
                duration_reference = float(OmegaConf.select(
                    config, "dynamic_2dgs.motion.duration_reference_sec"
                ))
                if duration_reference <= 0.0:
                    raise ValueError(
                        "dynamic_2dgs.motion.duration_reference_sec must be "
                        "positive"
                    )
                if not bool(OmegaConf.select(
                    config,
                    "dynamic_2dgs.motion.detach_init_condition",
                )):
                    raise ValueError(
                        "duration-conditioned offset requires dynamic_2dgs.motion."
                        "detach_init_condition=true"
                    )
            elif variant == DYNAMIC_VARIANT_V8:
                velocity_embedding_dim = int(OmegaConf.select(
                    config,
                    "dynamic_2dgs.motion.velocity_embedding_dim",
                ))
                if velocity_embedding_dim <= 0:
                    raise ValueError(
                        "dynamic_2dgs.motion.velocity_embedding_dim must be "
                        "positive"
                    )
                if not bool(OmegaConf.select(
                    config,
                    "dynamic_2dgs.motion.detach_init_condition",
                )):
                    raise ValueError(
                        "V8 requires dynamic_2dgs.motion."
                        "detach_init_condition=true"
                    )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.regularization",
                {
                    "small_motion",
                    "velocity_l2",
                },
            )
            _reject_unknown_keys(
                config,
                "dynamic_2dgs.regularization.small_motion",
                {"enabled", "weight", "epsilon_m", "warmup_steps", "ramp_steps"},
            )
            if _has_path(config, "dynamic_2dgs.regularization.velocity_l2"):
                _reject_unknown_keys(
                    config,
                    "dynamic_2dgs.regularization.velocity_l2",
                    {
                        "enabled", "weight", "warmup_steps", "ramp_steps",
                        "mode", "init_weight", "offset_weight",
                    },
                )
                velocity_l2_path = "dynamic_2dgs.regularization.velocity_l2"
                mode = OmegaConf.select(config, f"{velocity_l2_path}.mode")
                if mode is not None and str(mode) not in (
                    "final_l2", "final_group_l2", "final_group_l1", "offset_l2",
                    "offset_group_l2", "split_group_l2",
                ):
                    raise ValueError(
                        "dynamic_2dgs.regularization.velocity_l2.mode must be "
                        "'final_l2', 'final_group_l2', 'final_group_l1', "
                        "'offset_l2', 'offset_group_l2', or 'split_group_l2'"
                    )
                if str(mode) == "split_group_l2":
                    for name in ("init_weight", "offset_weight"):
                        value = OmegaConf.select(
                            config, f"{velocity_l2_path}.{name}"
                        )
                        if value is None or float(value) < 0.0:
                            raise ValueError(
                                "velocity_l2.mode='split_group_l2' requires a "
                                f"non-negative velocity_l2.{name}"
                            )
                elif _has_path(config, f"{velocity_l2_path}.init_weight") or (
                    _has_path(config, f"{velocity_l2_path}.offset_weight")
                ):
                    raise ValueError(
                        "velocity_l2.init_weight/offset_weight apply only to "
                        "mode='split_group_l2'"
                    )


def assert_model_variant_implemented(config) -> None:
    """Fail before logger/output side effects for design-only variants."""

    variant = _variant(config)
    if variant not in _IMPLEMENTED_VARIANTS:
        raise NotImplementedError(
            f"model.variant={variant!r} is a valid design config, but its model "
            "backend is not implemented yet"
        )


def compose_fresh_config(cli=None) -> tuple[DictConfig, str]:
    """Compose defaults -> shared base -> selected variant -> CLI overrides."""

    cli = _as_config(cli)
    base = OmegaConf.load(DEFAULT_CONFIG_PATH)
    selected = _variant(cli, default=_variant(base))
    overlay, overlay_path = _load_variant_overlay(selected)
    config = OmegaConf.merge(
        OmegaConf.create(STRUCTURAL_DEFAULTS),
        base,
        overlay,
        cli,
    )
    if _variant(config) != selected:
        raise ValueError(
            "The composed model.variant changed after selecting its overlay: "
            f"selected={selected!r}, composed={_variant(config)!r}"
        )
    validate_experiment_config(config)
    source = f"base={DEFAULT_CONFIG_PATH.resolve()} overlay={overlay_path.resolve()}"
    return config, source


def _is_experiment_config(value) -> bool:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)
    return isinstance(value, Mapping) and "p2g" in value and "data" in value


def _as_experiment_config(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=False)
    if not isinstance(value, Mapping):
        return None

    for key in ("cfg", "config", "hparams"):
        nested = value.get(key)
        if _is_experiment_config(nested):
            value = nested
            break
    if not _is_experiment_config(value):
        return None

    config = dict(value)
    config.pop("_wandb", None)
    return OmegaConf.create(config)


def _load_embedded_checkpoint_config(checkpoint: Path):
    """Read config metadata embedded by ``ModelWrapper.on_save_checkpoint``."""

    try:
        import torch

        try:
            blob = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            blob = torch.load(checkpoint, map_location="cpu")
    except Exception:
        # Historical checkpoints and lightweight test fixtures may not contain
        # readable checkpoint metadata. Their W&B artifact is the fallback.
        return None

    if not isinstance(blob, Mapping) or "experiment_config" not in blob:
        return None
    config = _as_experiment_config(blob["experiment_config"])
    if config is None:
        raise RuntimeError(
            f"Checkpoint contains invalid experiment_config metadata: {checkpoint}"
        )
    return config


def load_checkpoint_config(
    checkpoint,
    *,
    allow_legacy_wandb_fallback: bool = False,
) -> tuple[DictConfig, str]:
    """Load checkpoint-bound config, with a historical W&B fallback."""

    checkpoint = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    embedded = _load_embedded_checkpoint_config(checkpoint)
    if embedded is not None:
        config = OmegaConf.merge(OmegaConf.create(STRUCTURAL_DEFAULTS), embedded)
        validate_experiment_config(config)
        return config, f"embedded checkpoint config: {checkpoint}"

    if not allow_legacy_wandb_fallback:
        raise RuntimeError(
            "Historical checkpoint has no embedded experiment config. Its only "
            "available config source is the mutable wandb/latest-run link, which "
            "may belong to a later resumed run. Inspect that artifact and pass "
            "allow_legacy_wandb_fallback=true to opt in explicitly."
        )

    config_file = checkpoint.parent / "wandb/latest-run/files/config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(
            "Checkpoint execution needs its run-local W&B config, but it was "
            f"not found: {config_file}"
        )

    raw = OmegaConf.to_container(OmegaConf.load(config_file), resolve=False)
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"Unexpected W&B config format: {config_file}")
    unwrapped = {
        key: value["value"]
        if isinstance(value, Mapping) and "value" in value
        else value
        for key, value in raw.items()
        if key != "_wandb"
    }
    stored = _as_experiment_config(unwrapped)
    if stored is None:
        raise RuntimeError(
            f"W&B config does not contain a valid experiment config: {config_file}"
        )

    config = OmegaConf.merge(OmegaConf.create(STRUCTURAL_DEFAULTS), stored)
    validate_experiment_config(config)
    return config, (
        "legacy W&B fallback (mutable latest-run): "
        f"{config_file.resolve()}"
    )


def _merge_checkpoint_cli(stored, cli) -> DictConfig:
    cli = _as_config(cli)
    config = OmegaConf.merge(stored, cli)
    changed_roots = []
    for root in _CHECKPOINT_STRUCTURAL_ROOTS:
        if root not in cli:
            continue
        before = OmegaConf.select(stored, root, default=None)
        after = OmegaConf.select(config, root, default=None)
        before_value = (
            OmegaConf.to_container(before, resolve=True)
            if OmegaConf.is_config(before)
            else before
        )
        after_value = (
            OmegaConf.to_container(after, resolve=True)
            if OmegaConf.is_config(after)
            else after
        )
        if before_value != after_value:
            changed_roots.append(root)
    changed_paths = []
    for path in _CHECKPOINT_PROTECTED_PATHS:
        if not _has_path(cli, path):
            continue
        before = OmegaConf.select(stored, path, default=None)
        after = OmegaConf.select(config, path, default=None)
        if before != after:
            changed_paths.append(path)
    if changed_roots or changed_paths:
        details = []
        if changed_roots:
            details.append("roots: " + ", ".join(changed_roots))
        if changed_paths:
            details.append("paths: " + ", ".join(changed_paths))
        raise ValueError(
            "Checkpoint structural config cannot be changed by CLI; changed "
            + "; ".join(details)
        )
    validate_experiment_config(config)
    return config


def _resolve_with_checkpoint(cli, checkpoint_key: str) -> tuple[DictConfig, str]:
    cli = _as_config(cli)
    checkpoint = OmegaConf.select(cli, checkpoint_key, default=None)
    if not checkpoint:
        return compose_fresh_config(cli)
    allow_legacy = bool(
        OmegaConf.select(cli, "allow_legacy_wandb_fallback", default=False)
    )
    stored, source = load_checkpoint_config(
        checkpoint,
        allow_legacy_wandb_fallback=allow_legacy,
    )
    return _merge_checkpoint_cli(stored, cli), source


def resolve_main_config(cli=None) -> tuple[DictConfig, str]:
    """Resolve train/test entrypoint config, including exact resume configs."""

    cli = _as_config(cli)
    mode = str(OmegaConf.select(cli, "mode", default="train")).lower()
    checkpoint_key = "train.ckpt_path" if mode == "train" else "test.ckpt_path"
    return _resolve_with_checkpoint(cli, checkpoint_key)


def resolve_eval_config(cli=None) -> tuple[DictConfig, str]:
    return _resolve_with_checkpoint(cli, "test.ckpt_path")


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ATTENTION_VELOCITY_VARIANTS",
    "DYNAMIC_VARIANT",
    "DYNAMIC_VARIANT_V1",
    "DYNAMIC_VARIANT_V3",
    "DYNAMIC_VARIANT_V3_1",
    "DYNAMIC_VARIANT_V4",
    "DYNAMIC_VARIANT_V5",
    "DYNAMIC_VARIANT_V6",
    "DYNAMIC_VARIANT_V7",
    "DYNAMIC_VARIANT_V7_1",
    "DYNAMIC_VARIANT_V7_2",
    "DYNAMIC_VARIANT_V8",
    "DYNAMIC_VARIANT_V9",
    "DYNAMIC_VARIANT_V10",
    "DYNAMIC_VARIANT_V11",
    "DYNAMIC_VARIANTS",
    "DURATION_OFFSET_VELOCITY_VARIANTS",
    "LEGACY_VARIANT",
    "PHYSICAL_VELOCITY_VARIANTS",
    "PROPOSAL_VELOCITY_VARIANTS",
    "REFINED_PROPOSAL_VELOCITY_VARIANTS",
    "WARPED_PROPOSAL_VELOCITY_VARIANTS",
    "STRUCTURAL_DEFAULTS",
    "VARIANT_CONFIG_PATHS",
    "assert_model_variant_implemented",
    "compose_fresh_config",
    "load_checkpoint_config",
    "resolve_eval_config",
    "resolve_main_config",
    "validate_experiment_config",
]
