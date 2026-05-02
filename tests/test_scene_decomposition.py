from __future__ import annotations

import torch

from models.geometry.decomposition import decompose_scene


def test_decompose_scene_keeps_untracked_points_static():
    p0 = torch.tensor([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    p1 = torch.tensor([
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    i0 = torch.tensor([0.1, 0.2, 0.3])
    i1 = torch.tensor([0.4, 0.5, 0.6])

    boxes_0 = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    boxes_1 = torch.tensor([
        [1.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    ids_0 = torch.tensor([7, 9])
    ids_1 = torch.tensor([7, 9])
    rel = torch.eye(4)

    scene = decompose_scene(
        p0, p1, i0, i1,
        boxes_0, boxes_1,
        ids_0, ids_1,
        rel,
    )

    assert scene["untracked_stats"]["n_tracked_instances"] == 2
    assert scene["untracked_stats"]["n_dynamic_instances_kept"] == 2
    assert scene["static_xyz"].shape[0] >= 2

    dyn_ids = {d["instance_id"] for d in scene["dynamic"]}
    assert dyn_ids == {7, 9}

    tracked = next(d for d in scene["dynamic"] if d["instance_id"] == 7)
    assert tracked["present_in_0"]
    assert tracked["present_in_1"]
    assert tracked["canonical_xyz"].shape[0] >= 2
    assert torch.isfinite(tracked["canonical_xyz"]).all()


def test_decompose_scene_keeps_zero_id_dynamic_and_single_frame_static():
    p0 = torch.tensor([
        [0.5, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])
    p1 = torch.tensor([
        [10.5, 0.0, 0.0],
        [30.0, 0.0, 0.0],
    ])
    i0 = torch.tensor([0.1, 0.2, 0.3])
    i1 = torch.tensor([0.4, 0.5])

    boxes_0 = torch.tensor([
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
        [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    boxes_1 = torch.tensor([
        [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    ids_0 = torch.tensor([0, 2])
    ids_1 = torch.tensor([0])
    rel = torch.eye(4)

    scene = decompose_scene(
        p0, p1, i0, i1,
        boxes_0, boxes_1,
        ids_0, ids_1,
        rel,
    )

    dyn_ids = {d["instance_id"] for d in scene["dynamic"]}
    assert dyn_ids == {0}
    assert scene["untracked_stats"]["n_untracked_instances"] == 1
    assert torch.any(torch.all(scene["static_xyz"] == torch.tensor([5.0, 0.0, 0.0]), dim=1))

    dyn = scene["dynamic"][0]
    assert torch.allclose(
        dyn["canonical_xyz"],
        torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    assert torch.allclose(dyn["box_1"], boxes_1[0])
    assert dyn["fallback_xyz"].shape[0] == 2
