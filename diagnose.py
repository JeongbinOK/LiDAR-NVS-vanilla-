"""Post-fix diagnosis: verify gradient flow and loss behavior after refactoring."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/data/nuScenes/loader"))

import torch
from config import NeuralClusteringConfig
from nn.model import NeuralClusteringModel
from nn.losses import ClusteringLoss


def grad_check(primitive="2d", steps=3):
    """Check gradient flow for all modules, including after a few optimizer steps."""
    print(f"\n{'='*70}")
    print(f"GRADIENT FLOW — {primitive} (init + {steps} steps)")
    print(f"{'='*70}")

    cfg = NeuralClusteringConfig(backbone_type="ptv3", primitive_type=primitive)
    model = NeuralClusteringModel(cfg).cuda()
    loss_fn = ClusteringLoss(
        w_surface=cfg.w_surface, lambda_alpha=cfg.lambda_alpha,
        lambda_center=cfg.lambda_center, lambda_barrier=cfg.lambda_barrier,
        primitive=primitive, top_k_assign=cfg.top_k_assign,
    )
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
            print(f"\n--- {label} ---")
            print(f"  Loss: total={loss_dict['total'].item():.4f} "
                  f"S={loss_dict['surface'].item():.4f} "
                  f"alpha_l={loss_dict['alpha_loss'].item():.4f} "
                  f"ctr={loss_dict['centerness'].item():.4f} "
                  f"bar={loss_dict['barrier'].item():.4f}")

            module_grad = {}
            for name, p in model.named_parameters():
                mod = name.split(".")[0]
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


def loss_conflict_check():
    """Check if gradient conflict between loss terms is resolved."""
    print(f"\n{'='*70}")
    print("GRADIENT CONFLICT CHECK")
    print(f"{'='*70}")

    cfg = NeuralClusteringConfig(backbone_type="ptv3", primitive_type="2d")
    model = NeuralClusteringModel(cfg).cuda()
    loss_fn = ClusteringLoss(
        w_surface=cfg.w_surface, lambda_alpha=cfg.lambda_alpha,
        lambda_center=cfg.lambda_center, lambda_barrier=cfg.lambda_barrier,
        primitive="2d", top_k_assign=cfg.top_k_assign,
    )

    N = 3000
    xyz = torch.randn(N, 3, device="cuda") * 10
    intensity = torch.rand(N, 1, device="cuda")

    params = [p for p in model.parameters() if p.requires_grad]

    def get_grads(loss_val):
        model.zero_grad()
        loss_val.backward(retain_graph=True)
        return torch.cat([p.grad.flatten() if p.grad is not None
                          else torch.zeros(p.numel(), device="cuda")
                          for p in params])

    output = model(xyz, intensity, tau=1.0)
    ld = loss_fn(xyz, output)

    g_surface = get_grads(ld["surface"])
    g_alpha = get_grads(cfg.lambda_alpha * ld["alpha_loss"])
    g_center = get_grads(cfg.lambda_center * ld["centerness"])
    g_barrier = get_grads(cfg.lambda_barrier * ld["barrier"])

    pairs = [
        ("surface", "alpha_loss", g_surface, g_alpha),
        ("surface", "barrier", g_surface, g_barrier),
        ("surface", "centerness", g_surface, g_center),
        ("alpha_loss", "barrier", g_alpha, g_barrier),
    ]

    print(f"\n  {'Pair':<30} {'Cosine':>10}  Interpretation")
    print("  " + "-" * 60)
    for name1, name2, g1, g2 in pairs:
        norm1 = g1.norm().item()
        norm2 = g2.norm().item()
        if norm1 > 0 and norm2 > 0:
            cos = torch.nn.functional.cosine_similarity(
                g1.unsqueeze(0), g2.unsqueeze(0)
            ).item()
        else:
            cos = 0.0
        interp = "CONFLICT" if cos < -0.3 else "ALIGNED" if cos > 0.3 else "ORTHOGONAL"
        print(f"  {name1} vs {name2:<17} {cos:>10.4f}  {interp}")


def train_test(primitive="2d", epochs=20):
    """Quick training test to verify convergence."""
    print(f"\n{'='*70}")
    print(f"TRAINING TEST — {primitive}, {epochs} epochs")
    print(f"{'='*70}")

    cfg = NeuralClusteringConfig(backbone_type="ptv3", primitive_type=primitive)
    model = NeuralClusteringModel(cfg).cuda()
    loss_fn = ClusteringLoss(
        w_surface=cfg.w_surface, lambda_alpha=cfg.lambda_alpha,
        lambda_center=cfg.lambda_center, lambda_barrier=cfg.lambda_barrier,
        primitive=primitive, top_k_assign=cfg.top_k_assign,
    )
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
                al = out_e["gaussians"]["alpha"]
                active = (al.squeeze(-1) > 0.1).sum().item()
                K = mu.shape[0]

                # Check centers near planes
                if active > 0:
                    mu_a = mu[al.squeeze(-1) > 0.1]
                    z = mu_a[:, 2]
                    near = ((z.abs() < 1.5).sum() + ((z - 5).abs() < 1.5).sum()).item()
                else:
                    near = 0

            print(f"  [{ep+1:2d}] loss={ld['total'].item():.4f} "
                  f"S={ld['surface'].item():.4f} "
                  f"a_l={ld['alpha_loss'].item():.4f} "
                  f"ctr={ld['centerness'].item():.4f} "
                  f"a={active}/{K} near_plane={near}")


def eval_test():
    """Test eval.py with checkpoint."""
    print(f"\n{'='*70}")
    print("EVAL TEST")
    print(f"{'='*70}")

    import glob
    ckpts = sorted(glob.glob("outputs/train_*/ckpt/best_model.pt"))
    if not ckpts:
        print("  No checkpoint found — skipping")
        return

    ckpt_path = ckpts[-1]
    print(f"  Checkpoint: {ckpt_path}")

    cfg = NeuralClusteringConfig(backbone_type="ptv3", primitive_type="2d")
    model = NeuralClusteringModel(cfg).cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    loss_fn = ClusteringLoss(
        w_surface=cfg.w_surface, lambda_alpha=cfg.lambda_alpha,
        lambda_center=cfg.lambda_center, lambda_barrier=cfg.lambda_barrier,
        primitive="2d", top_k_assign=cfg.top_k_assign,
    )

    from dataset import NuScenesNVSDataset
    dataset = NuScenesNVSDataset(
        dataroot=os.path.expanduser("~/data/nuScenes"),
        version="v1.0-trainval", split="val",
    )

    for i in range(3):
        pts = dataset[i]["input_0"]
        xyz = pts[:, :3].cuda()
        intensity = pts[:, 3:4].cuda()
        mask = torch.norm(xyz, dim=1) > cfg.ego_radius
        xyz, intensity = xyz[mask], intensity[mask]

        with torch.no_grad():
            output = model(xyz, intensity, tau=0.1)
            ld = loss_fn(xyz, output)

        al = output["gaussians"]["alpha"]
        active = (al.squeeze(-1) > 0.1).sum().item()
        print(f"  [{i}] N={xyz.shape[0]:>5} K={output['centers'].shape[0]:>4} "
              f"active={active:>4} S={ld['surface'].item():.4f}")

    print("  Eval: OK")


if __name__ == "__main__":
    torch.manual_seed(42)
    grad_check("2d", steps=3)
    grad_check("3d", steps=3)
    loss_conflict_check()
    train_test("2d", epochs=20)
    train_test("3d", epochs=20)
    eval_test()
    print(f"\n{'='*70}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*70}")
