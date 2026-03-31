"""Diagnosis: verify gradient flow and loss behavior."""
import sys, os
from config import NeuralClusteringConfig
sys.path.insert(0, os.path.join(os.path.expanduser(NeuralClusteringConfig.data_root), "loader"))

import torch
from nn.model import NeuralClusteringModel
from nn.losses import ClusteringLoss


def grad_check(primitive="2d", steps=3):
    """Check gradient flow for all modules."""
    print(f"\n{'='*70}")
    print(f"GRADIENT FLOW — {primitive} (init + {steps} steps)")
    print(f"{'='*70}")

    cfg = NeuralClusteringConfig(backbone_type="ptv3", primitive_type=primitive)
    model = NeuralClusteringModel(cfg).cuda()
    loss_fn = ClusteringLoss(primitive=primitive, top_k_assign=cfg.top_k_assign)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    N = 3000
    xyz = torch.randn(N, 3, device="cuda") * 10
    intensity = torch.rand(N, 1, device="cuda")

    for step in range(steps + 1):
        model.train()
        optimizer.zero_grad()
        output = model(xyz, intensity, tau=1.0)
        loss_dict = loss_fn(xyz, output)
        loss_dict["total"].backward()

        if step == 0 or step == steps:
            label = "INIT" if step == 0 else f"STEP {step}"
            K = output["centers"].shape[0]
            print(f"\n--- {label} (K={K}) ---")
            print(f"  Loss: total={loss_dict['total'].item():.4f}")

            module_grad = {}
            for name, p in model.named_parameters():
                submod = ".".join(name.split(".")[:2])
                key = submod
                g = p.grad.norm().item() if p.grad is not None else 0.0
                if key not in module_grad:
                    module_grad[key] = {"max": 0.0, "zero": 0, "total": 0}
                module_grad[key]["max"] = max(module_grad[key]["max"], g)
                module_grad[key]["total"] += 1
                if g == 0:
                    module_grad[key]["zero"] += 1

            print(f"  {'Submodule':<35} {'Params':>6} {'Zero':>5} {'Max |g|':>10}")
            for key in sorted(module_grad.keys()):
                info = module_grad[key]
                status = "OK" if info["zero"] == 0 else f"WARN({info['zero']})"
                print(f"  {key:<35} {info['total']:>6} {info['zero']:>5} {info['max']:>10.6f}  {status}")

        optimizer.step()


def train_test(primitive="2d", epochs=20):
    """Quick training test to verify convergence."""
    print(f"\n{'='*70}")
    print(f"TRAINING TEST — {primitive}, {epochs} epochs")
    print(f"{'='*70}")

    cfg = NeuralClusteringConfig(backbone_type="ptv3", primitive_type=primitive)
    model = NeuralClusteringModel(cfg).cuda()
    loss_fn = ClusteringLoss(primitive=primitive, top_k_assign=cfg.top_k_assign)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic: two parallel planes
    N = 4000
    plane1 = torch.cat([torch.randn(N // 2, 2) * 3, torch.zeros(N // 2, 1)], dim=1)
    plane2 = torch.cat([torch.randn(N // 2, 2) * 3, 5 * torch.ones(N // 2, 1)], dim=1)
    xyz = torch.cat([plane1, plane2]).cuda()
    intensity = torch.rand(N, 1, device="cuda")

    for ep in range(epochs):
        tau = max(1.0 - ep / max(epochs - 1, 1) * 0.9, 0.1)
        model.train()
        optimizer.zero_grad()
        output = model(xyz, intensity, tau=tau)
        ld = loss_fn(xyz, output)
        ld["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ep % 5 == 0 or ep == epochs - 1:
            with torch.no_grad():
                model.eval()
                out_e = model(xyz, intensity, tau=0.1)
                mu = out_e["gaussians"]["mu"]
                K = mu.shape[0]

                # Check centers near planes
                z = mu[:, 2]
                near = ((z.abs() < 1.5).sum() + ((z - 5).abs() < 1.5).sum()).item()

            print(f"  [{ep+1:2d}] loss={ld['total'].item():.4f} "
                  f"K={K} near_plane={near}/{K}")


if __name__ == "__main__":
    torch.manual_seed(42)
    grad_check("2d", steps=3)
    grad_check("3d", steps=3)
    train_test("2d", epochs=20)
    train_test("3d", epochs=20)
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}")
