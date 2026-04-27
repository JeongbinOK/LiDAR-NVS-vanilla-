from __future__ import annotations

from types import SimpleNamespace

import torch

from nn.qgs_loss import QGSLoss


def test_qgs_loss_penalizes_alpha_collapse():
    loss_fn = QGSLoss(w_depth=1.0, w_intensity=1.0, w_raydrop=1.0)

    target = {
        "range_image": torch.tensor([[10.0]]),
        "intensity_image": torch.tensor([[0.25]]),
        "valid_mask": torch.tensor([[True]]),
    }

    good = SimpleNamespace(
        range=torch.tensor([[10.0]]),
        middepth=torch.tensor([[10.0]]),
        intensity=torch.tensor([[0.25]]),
        alpha_accum=torch.tensor([[1.0]]),
    )
    bad = SimpleNamespace(
        range=torch.tensor([[0.0]]),
        middepth=torch.tensor([[0.0]]),
        intensity=torch.tensor([[0.0]]),
        alpha_accum=torch.tensor([[0.0]]),
    )

    good_loss = loss_fn(good, target, drop_prob=torch.tensor([[0.0]]))
    bad_loss = loss_fn(bad, target, drop_prob=torch.tensor([[1.0]]))

    assert bad_loss["total"] > good_loss["total"]
    assert bad_loss["raydrop"] > good_loss["raydrop"]


def test_qgs_loss_uses_raw_alpha_blended_intensity():
    loss_fn = QGSLoss(w_depth=0.0, w_intensity=1.0, w_raydrop=0.0)

    target = {
        "range_image": torch.tensor([[10.0]]),
        "intensity_image": torch.tensor([[0.2]]),
        "valid_mask": torch.tensor([[True]]),
    }
    rendered = SimpleNamespace(
        range=torch.tensor([[5.0]]),
        middepth=torch.tensor([[10.0]]),
        intensity=torch.tensor([[0.2]]),
        alpha_accum=torch.tensor([[0.25]]),
    )

    loss = loss_fn(rendered, target, drop_prob=torch.tensor([[0.0]]))

    assert torch.allclose(loss["intensity"], torch.tensor(0.0))
