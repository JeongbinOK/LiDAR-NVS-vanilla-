"""Compatibility imports for the routing budget now owned by ``utils.loss``."""

from .loss import (
    budget_weight_at_step,
    distributed_token_mean,
    expected_k_sum_from_logits,
    target_budget_loss,
)


__all__ = [
    "budget_weight_at_step",
    "distributed_token_mean",
    "expected_k_sum_from_logits",
    "target_budget_loss",
]
