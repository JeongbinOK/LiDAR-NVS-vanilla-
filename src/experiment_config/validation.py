"""Static validation for composed and checkpoint experiment configs."""
from __future__ import annotations

from omegaconf import OmegaConf

from .definitions import (
    BARRIER_MATCH_DYNAMIC_VARIANTS,
    DURATION_OFFSET_VELOCITY_VARIANTS,
    DYNAMIC_VARIANT_V1,
    DYNAMIC_VARIANT_V5,
    DYNAMIC_VARIANT_V6,
    DYNAMIC_VARIANT_V7_1,
    DYNAMIC_VARIANT_V7_2,
    DYNAMIC_VARIANT_V8,
    DYNAMIC_VARIANT_V9,
    DYNAMIC_VARIANT_V10,
    DYNAMIC_VARIANTS,
    LEARNED_COUNT_MODES,
    LEGACY_COUNT_MODE,
    PHYSICAL_VELOCITY_VARIANTS,
    PROPOSAL_VELOCITY_VARIANTS,
    REFINED_PROPOSAL_VELOCITY_VARIANTS,
    ROUTER_CAPABLE_DYNAMIC_VARIANTS,
    VARIANT_CONFIG_PATHS,
    WARPED_PROPOSAL_VELOCITY_VARIANTS,
    _IMPLEMENTED_VARIANTS,
    _REQUIRED_ROOTS,
    _SELECTED_HEAD_ATTENTION_VELOCITY_VARIANTS,
    _UTONIA_ENCODER_STAGE_DIMS,
)
from .helpers import (
    _has_path,
    _reject_unknown_keys,
    _require_choice,
    _require_in_open_range,
    _require_positive,
    _variant,
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


def _configured_count_mode(config) -> str:
    """The Gaussian-count contract ``p2g.grid_query`` asks for.

    An absent block is the historical fixed-count default: one Gaussian per
    occupied token, seeded at that token's observed medoid.
    """
    if not _has_path(config, "p2g.grid_query"):
        return LEGACY_COUNT_MODE
    mode = OmegaConf.select(config, "p2g.grid_query.count_mode")
    return LEGACY_COUNT_MODE if mode is None else str(mode).lower()


def _validate_fixed_gaussian_count(config, label: str) -> None:
    """The per-token Gaussian count and seed placement, when both are fixed.

    ``K_max`` is how many Gaussians every occupied token emits, and ``exp``
    says where they start: ``1`` on the token's observed medoid, ``2`` on its
    Utonia cell coordinate, and ``null`` spread over the token's own raw points
    by range quantile. Slots that coincide are separated by the independent
    parameter block each one owns in the Gaussian head.
    """
    count_path = "p2g.grid_query"
    if not _has_path(config, count_path):
        return
    _reject_unknown_keys(
        config, count_path, {"count_mode", "K_max", "points_per_gaussian", "exp"}
    )
    k_max = OmegaConf.select(config, f"{count_path}.K_max")
    if k_max is None or int(k_max) < 1:
        raise ValueError(
            f"{label} requires a positive p2g.grid_query.K_max Gaussians "
            "per token"
        )
    exp = OmegaConf.select(config, f"{count_path}.exp")
    if exp is not None and int(exp) not in (1, 2):
        raise ValueError(f"{label} grid_query.exp must be null, 1, or 2")
    if exp is not None and int(exp) > int(k_max):
        raise ValueError(
            f"{label} grid_query.exp={int(exp)} requires K_max >= {int(exp)}"
        )
    points_per_gaussian = OmegaConf.select(
        config, f"{count_path}.points_per_gaussian"
    )
    if points_per_gaussian is not None and int(points_per_gaussian) <= 0:
        raise ValueError(
            f"{label} grid_query.points_per_gaussian must be positive"
        )


def _validate_dynamic_gaussian_count(config, variant, label: str) -> None:
    """Route the config to the count contract it selected, and check it fits."""
    count_mode = _configured_count_mode(config)
    if count_mode == LEGACY_COUNT_MODE:
        _validate_fixed_gaussian_count(config, label)
        if variant == DYNAMIC_VARIANT_V1:
            k_max = OmegaConf.select(config, "p2g.grid_query.K_max", default=1)
            if int(k_max) != 1:
                raise ValueError(
                    f"{label} requires p2g.grid_query.K_max=1 because its "
                    "seed-conditioned head operates at token resolution"
                )
        return
    if count_mode not in LEARNED_COUNT_MODES:
        raise ValueError(
            f"{label} p2g.grid_query.count_mode must be 'legacy' or "
            "'learned_gumbel'"
        )
    if variant not in ROUTER_CAPABLE_DYNAMIC_VARIANTS:
        raise ValueError(
            f"{label} emits a fixed number of Gaussians per token and has no "
            f"count router; p2g.grid_query.count_mode={count_mode!r} is only "
            "available to "
            + ", ".join(
                _variant_label(name) for name in ROUTER_CAPABLE_DYNAMIC_VARIANTS
            )
        )
    _validate_adaptive_gaussian_count(config, label)


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
    _require_choice(config, f"{count_path}.grad_balance", ("sqrt_k", "none"))
    learned_path = f"{count_path}.learned_count"
    _reject_unknown_keys(
        config,
        learned_path,
        {"K_max", "tau", "seed_mode", "grad_balance_scope", "budget", "logging"},
    )
    if int(OmegaConf.select(config, f"{learned_path}.K_max")) != 3:
        raise ValueError(f"{label} requires grid_query.learned_count.K_max=3")
    _require_positive(config, f"{learned_path}.tau")
    if str(OmegaConf.select(
        config, f"{learned_path}.seed_mode"
    )).lower() != "range_quantile":
        raise ValueError(
            f"{label} grid_query.learned_count.seed_mode must be "
            "'range_quantile'"
        )
    _require_choice(
        config, f"{learned_path}.grad_balance_scope", ("output", "token")
    )
    budget_path = f"{learned_path}.budget"
    _reject_unknown_keys(config, budget_path, {"enable"})
    if bool(OmegaConf.select(config, f"{budget_path}.enable")):
        raise ValueError(f"{label} does not use a routing budget loss")
    logging_path = f"{learned_path}.logging"
    _reject_unknown_keys(
        config, logging_path, {"enable", "interval", "range_edges_m"}
    )
    _require_positive(config, f"{logging_path}.interval", kind=int)


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
        if variant in (
            *ROUTER_CAPABLE_DYNAMIC_VARIANTS, *BARRIER_MATCH_DYNAMIC_VARIANTS,
        ) and not lora_enabled:
            raise ValueError(
                f"{_variant_label(variant)} requires p2g.utonia_lora.enable=true"
            )
        if OmegaConf.select(config, "dynamic_2dgs", default=None) is None:
            raise ValueError(f"{variant} requires a dynamic_2dgs config block")
        # How many Gaussians a token becomes is a config decision, not a
        # property of the variant name: p2g.grid_query selects a fixed count
        # per token or, where the backend has a router, a predicted one.
        _validate_dynamic_gaussian_count(
            config, variant, _variant_label(variant)
        )
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
                _require_positive(
                    config,
                    *(
                        f"{matching_path}.{name}" for name in
                        ("candidate_count", "match_count", "score_chunk_size")
                    ),
                    kind=int,
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
                _require_positive(
                    config, f"{temporal_path}.distance_bias_speed_mps"
                )
                if not bool(OmegaConf.select(
                    config, f"{temporal_path}.qk_norm"
                )):
                    raise ValueError("V10 requires temporal.qk_norm=true")
                _require_positive(
                    config, f"{temporal_path}.layer_weight_hidden_dim",
                    kind=int,
                )
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
                _require_positive(config, f"{temporal_path}.barrier_speed_mps")
                if float(OmegaConf.select(
                    config, f"{temporal_path}.barrier_weight"
                )) < 0.0:
                    raise ValueError(
                        f"{label} temporal.barrier_weight must be non-negative"
                    )
                _require_positive(
                    config, f"{temporal_path}.match_chunk_size", kind=int
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
            elif variant == DYNAMIC_VARIANT_V6:
                _require_positive(
                    config, "dynamic_2dgs.temporal.time_hidden_dim", kind=int
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
                _require_positive(
                    config,
                    f"{proposal_path}.descriptor_dim",
                    f"{proposal_path}.score_chunk_size",
                    kind=int,
                )
                _require_positive(
                    config,
                    f"{proposal_path}.temperature",
                    f"{proposal_path}.distance_prior_speed_mps",
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
                    _require_positive(
                        config,
                        f"{proposal_path}.match_count",
                        f"{proposal_path}.score_chunk_size",
                        kind=int,
                    )
                    if int(OmegaConf.select(
                        config, f"{proposal_path}.match_count"
                    )) != 4:
                        raise ValueError(
                            "V7.2 requires motion_proposal.match_count=4"
                        )
                    _require_positive(
                        config,
                        f"{proposal_path}.temperature",
                        f"{proposal_path}.distance_prior_speed_mps",
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
                    _require_positive(
                        config,
                        *(f"{proposal_path}.{name}" for name in integer_keys),
                        kind=int,
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
                    _require_positive(config, f"{proposal_path}.temperature")
                    if variant == DYNAMIC_VARIANT_V6:
                        _require_in_open_range(
                            config,
                            f"{proposal_path}.dustbin_similarity_init",
                            -1.0, 1.0,
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
                        _require_in_open_range(
                            config,
                            f"{proposal_path}.dustbin_prior_probability",
                            0.0, 1.0,
                        )
                        _require_choice(
                            config,
                            f"{proposal_path}.unmatched_gate_mode",
                            ("soft", "ste_hard"),
                        )
                        _require_in_open_range(
                            config,
                            f"{proposal_path}.unmatched_hard_threshold",
                            0.0, 1.0,
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
                            _require_positive(
                                config,
                                f"{proposal_path}.mean_displacement_scale_m",
                                f"{proposal_path}.spread_scale_m",
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
                _require_positive(
                    config,
                    *(
                        f"dynamic_2dgs.motion.{name}" for name in (
                            "velocity_embedding_dim", "duration_embedding_dim",
                            "duration_frequencies", "residual_hidden_dim",
                        )
                    ),
                    kind=int,
                )
                _require_positive(
                    config, "dynamic_2dgs.motion.duration_reference_sec"
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
                _require_positive(
                    config,
                    "dynamic_2dgs.motion.velocity_embedding_dim",
                    kind=int,
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
