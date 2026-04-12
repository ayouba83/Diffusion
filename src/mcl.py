"""
Multiple Choice Learning (MCL) for Score-Based Diffusion Models
================================================================
Implements Stochastic MCL (sMCL) with Winner-Takes-All (WTA) training
on an ensemble of K expert score networks.

Core idea
---------
Instead of one network, we train K experts competitively.  For each
training example, ONLY the expert with the lowest denoising error
receives gradients.  This encourages specialisation: different experts
learn different modes of the data distribution.

Mathematical framework (sMCL)
------------------------------
For a training triple (x_0, t, ε) with x_t = x_0 + σ(t)ε:

1. Each expert k predicts:  x^k = σ(t)² s_{θ_k}(x_t, t)
2. Per-example error:       ℓ_k = ‖x^k − ε‖²
3. Winner selection:        k* = argmin_k ℓ_k     (per example!)
4. Only the winner's loss is backpropagated.

Gradient routing is achieved by `torch.gather` on the (K, B) loss
tensor — only the winning entries connect to the final scalar loss,
so PyTorch autograd naturally gives zero gradients to losing experts.
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.baseline import (
    SmallUNet,
    EMA,
    sigma_schedule,
    discrete_sigmas,
    sample_euler,
    sample_heun,
    get_inception_features,
    compute_fid,
    compute_precision_recall,
)


# ---------------------------------------------------------------------------
# 1.  MCL ENSEMBLE MODULE
# ---------------------------------------------------------------------------

class MCLDiffusion(nn.Module):
    """Ensemble of K expert score networks for Multiple Choice Learning.

    Each expert is an independent `SmallUNet` that maps (x_t, t) → score.
    The forward pass evaluates ALL K experts and stacks their outputs so
    that the WTA loss can select the best expert per example.

    Parameters
    ----------
    K              : number of expert networks
    in_channels    : image channels (1 for MNIST)
    base_channels  : base channel width of each U-Net expert
    time_emb_dim   : dimensionality of the sinusoidal time embedding
    """

    def __init__(
        self,
        K: int = 5,
        in_channels: int = 1,
        base_channels: int = 32,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.K = K
        self.experts = nn.ModuleList([
            SmallUNet(in_channels, base_channels, time_emb_dim)
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Run all K experts on the same noised input.

        Parameters
        ----------
        x : (B, C, H, W)  noised image x_t
        t : (B,)           continuous time in [0, 1]

        Returns
        -------
        scores : (K, B, C, H, W)  — stacked score predictions from all experts
        """
        # Each expert produces (B, C, H, W); stack along a new dim 0
        return torch.stack([expert(x, t) for expert in self.experts], dim=0)


# ---------------------------------------------------------------------------
# 2.  STOCHASTIC MCL LOSS  (Winner-Takes-All)
# ---------------------------------------------------------------------------

def smcl_loss(
    ensemble: MCLDiffusion,
    x0: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stochastic Multiple Choice Learning loss with per-example WTA.

    This function implements the full sMCL forward pass:
      1. Noise the clean data x_0 → x_t
      2. All K experts predict the score → noise estimate x^k
      3. Per-example, per-expert squared error ℓ_k(i) = ‖x^k(i) − ε(i)‖²
      4. Winner selection k*(i) = argmin_k ℓ_k(i)  (ELEMENT-WISE per example)
      5. Gather only the winners' losses and average → scalar

    Gradient routing
    ----------------
    The key insight is that `torch.gather` on the (K, B) loss tensor
    selects only the winning entries.  When we compute `loss.backward()`,
    PyTorch autograd only follows the computational graph of the gathered
    elements.  Losing experts' forward-pass activations exist in memory
    but are NOT connected to the final scalar — their parameter gradients
    are exactly zero (or None).

    Parameters
    ----------
    ensemble   : MCLDiffusion with K experts
    x0         : clean images (B, 1, 28, 28) in [0, 1]
    sigma_min  : minimum noise level
    sigma_max  : maximum noise level

    Returns
    -------
    loss     : scalar tensor — mean of the winning experts' losses
    k_star   : (B,) long tensor — index of the winning expert per example
    """
    B = x0.shape[0]
    device = x0.device
    K = ensemble.K

    # --- Step 1: Sample noise level and perturbation ---
    t = torch.rand(B, device=device)                            # t ~ U(0,1)
    sig = sigma_schedule(t, sigma_min, sigma_max)               # σ(t), shape (B,)
    eps = torch.randn_like(x0)                                  # ε ~ N(0, I)
    x_t = x0 + sig[:, None, None, None] * eps                  # x_t = x_0 + σ(t)ε

    # --- Step 2: Forward pass through ALL K experts ---
    # scores: (K, B, 1, 28, 28)
    scores = ensemble(x_t, t)

    # --- Step 3: Compute per-expert noise predictions (Eq. 8) ---
    # x^k = σ(t)² · s_{θ_k}(x_t, t)
    sig_sq = (sig ** 2)[:, None, None, None]                    # (B, 1, 1, 1)
    predicted_noise = sig_sq.unsqueeze(0) * scores              # (K, B, 1, 28, 28)

    # --- Step 4: Per-example, per-expert unreduced losses ---
    # ℓ_k(i) = ‖x^k(i) − ε(i)‖²  summed over spatial dimensions (C, H, W)
    # eps is (B, 1, 28, 28), broadcast to (K, B, 1, 28, 28)
    per_pixel_sq_err = (predicted_noise - eps.unsqueeze(0)) ** 2  # (K, B, 1, 28, 28)
    ell_k = per_pixel_sq_err.sum(dim=(2, 3, 4))                  # (K, B)

    # --- Step 5: Winner selection — ELEMENT-WISE per example (Eq. 9) ---
    # k*(i) = argmin_k ℓ_k(i)
    # .detach() is critical: argmin is non-differentiable, but we also want
    # to make sure the winner indices don't carry any gradient tape.
    k_star = ell_k.detach().argmin(dim=0)                        # (B,)  long tensor

    # --- Step 6: Gather only the winning experts' losses ---
    # torch.gather selects ell_k[k_star[i], i] for each example i.
    # Only these K* entries remain in the computation graph.
    # All other entries (losing experts) are disconnected from the loss.
    winner_losses = ell_k.gather(0, k_star.unsqueeze(0)).squeeze(0)  # (B,)

    # --- Step 7: Final scalar loss ---
    # L = (1/B) Σ_i ℓ_{k*(i)}(i)
    loss = winner_losses.mean()

    return loss, k_star


# ---------------------------------------------------------------------------
# 3.  MCL TRAINING LOOP
# ---------------------------------------------------------------------------

def train_mcl(
    K: int = 5,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 2e-4,
    ema_decay: float = 0.999,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_root: str = "./data",
) -> tuple[MCLDiffusion, list[EMA]]:
    """Full MCL training loop with WTA expert selection.

    Returns the trained ensemble and a list of K EMA wrappers.
    """
    print(f"Training MCL ensemble (K={K}) on device: {device}")

    # ---- Data ----
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # ---- Ensemble / Optimiser / Scheduler ----
    ensemble = MCLDiffusion(K=K, in_channels=1, base_channels=32, time_emb_dim=128).to(device)

    # Single optimiser covers ALL K experts' parameters.
    # WTA gradient routing ensures only the winner is updated each step.
    optimizer = torch.optim.AdamW(ensemble.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    # One EMA wrapper per expert — each expert's shadow tracks independently
    ema_list = [EMA(expert, decay=ema_decay) for expert in ensemble.experts]

    # ---- Training ----
    ensemble.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        win_counts = defaultdict(int)  # track per-expert win rates

        for batch_idx, (x0, _) in enumerate(loader):
            x0 = x0.to(device)

            loss, k_star = smcl_loss(ensemble, x0, sigma_min, sigma_max)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Update EMA for each expert independently
            for k, expert in enumerate(ensemble.experts):
                ema_list[k].step(expert)

            total_loss += loss.item()

            # Accumulate win statistics
            for k in range(K):
                win_counts[k] += (k_star == k).sum().item()

        avg_loss = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        total_examples = len(loader) * batch_size
        win_str = "  ".join(
            f"E{k}:{win_counts[k]/total_examples:.1%}" for k in range(K)
        )
        print(
            f"Epoch {epoch:3d}/{epochs}  |  loss = {avg_loss:.4f}  |  "
            f"lr = {current_lr:.2e}  |  wins: {win_str}"
        )

    return ensemble, ema_list


# ---------------------------------------------------------------------------
# 4.  MCL SAMPLING  (delegate to baseline samplers per expert)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler_mcl(
    ensemble: MCLDiffusion,
    initial_noise: torch.Tensor,
    expert_idx: int,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Euler ODE sampling using a specific expert from the ensemble.

    Parameters
    ----------
    ensemble     : trained MCLDiffusion
    initial_noise: x_N ~ N(0, σ_max² I)
    expert_idx   : which expert k ∈ {0, ..., K-1} to use
    N, sigma_min, sigma_max : ODE discretisation parameters

    Returns
    -------
    x_0 : denoised samples (B, 1, 28, 28)
    """
    return sample_euler(
        ensemble.experts[expert_idx], initial_noise, N, sigma_min, sigma_max
    )


@torch.no_grad()
def sample_heun_mcl(
    ensemble: MCLDiffusion,
    initial_noise: torch.Tensor,
    expert_idx: int,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Heun ODE sampling using a specific expert from the ensemble."""
    return sample_heun(
        ensemble.experts[expert_idx], initial_noise, N, sigma_min, sigma_max
    )


# ---------------------------------------------------------------------------
# 5.  MCL EVALUATION SCAFFOLDING
# ---------------------------------------------------------------------------

def evaluate_mcl(
    ensemble: MCLDiffusion,
    ema_list: list[EMA],
    n_samples_per_expert: int = 2000,
    N_steps: int = 100,
    sampler: str = "euler",
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_root: str = "./data",
) -> dict:
    """Evaluate each expert individually and the aggregate ensemble.

    For each expert k:
      - Generate `n_samples_per_expert` images
      - Compute FID, Precision, Recall vs MNIST test set

    Aggregate:
      - Pool all K × n_samples_per_expert generated images
      - Compute FID, Precision, Recall for the mixture

    Returns
    -------
    dict with keys "per_expert" (list of K dicts) and "aggregate" (dict)
    """
    K = ensemble.K
    sample_fn = sample_euler_mcl if sampler == "euler" else sample_heun_mcl

    # ---- Apply EMA weights ----
    for k in range(K):
        ema_list[k].apply(ensemble.experts[k])
    ensemble.eval()

    # ---- Generate samples per expert ----
    all_generated = []
    for k in range(K):
        print(f"Generating {n_samples_per_expert} samples from expert {k}...")
        expert_samples = []
        for start in range(0, n_samples_per_expert, batch_size):
            B = min(batch_size, n_samples_per_expert - start)
            z = torch.randn(B, 1, 28, 28, device=device) * sigma_max
            samples = sample_fn(ensemble, z, expert_idx=k, N=N_steps,
                                sigma_min=sigma_min, sigma_max=sigma_max)
            expert_samples.append(samples.clamp(0, 1).cpu())
        all_generated.append(torch.cat(expert_samples, dim=0))

    # ---- Restore original weights ----
    for k in range(K):
        ema_list[k].restore(ensemble.experts[k])

    # ---- Load real test images ----
    test_set = datasets.MNIST(data_root, train=False, download=True,
                              transform=transforms.ToTensor())
    n_real = min(n_samples_per_expert * K, len(test_set))
    real_images = torch.stack([test_set[i][0] for i in range(n_real)])

    print("Extracting features for real images...")
    real_feats = get_inception_features(real_images.to(device))

    # ---- Per-expert metrics ----
    results = {"per_expert": [], "aggregate": {}}
    for k in range(K):
        print(f"Extracting features for expert {k}...")
        gen_feats = get_inception_features(all_generated[k].to(device))
        fid = compute_fid(real_feats, gen_feats)
        prec, rec = compute_precision_recall(real_feats, gen_feats)
        results["per_expert"].append({"fid": fid, "precision": prec, "recall": rec})
        print(f"  Expert {k}:  FID={fid:.2f}  Prec={prec:.4f}  Rec={rec:.4f}")

    # ---- Aggregate metrics (pool all experts) ----
    pooled = torch.cat(all_generated, dim=0)
    print("Extracting features for aggregate (all experts pooled)...")
    pooled_feats = get_inception_features(pooled.to(device))
    fid_agg = compute_fid(real_feats, pooled_feats)
    prec_agg, rec_agg = compute_precision_recall(real_feats, pooled_feats)
    results["aggregate"] = {"fid": fid_agg, "precision": prec_agg, "recall": rec_agg}
    print(f"  Aggregate:  FID={fid_agg:.2f}  Prec={prec_agg:.4f}  Rec={rec_agg:.4f}")

    return results


# ---------------------------------------------------------------------------
# 6.  SANITY CHECK — GRADIENT ROUTING VERIFICATION
# ---------------------------------------------------------------------------

def sanity_check():
    """Verify that sMCL gradient routing works correctly.

    This test proves that:
    - Only the winning expert receives non-zero gradients for its assigned examples.
    - Losing experts have EXACTLY zero gradients.

    We use K=3 experts and B=8 examples to ensure at least one expert
    is likely to lose all examples (providing a clean zero-gradient check).
    """
    print("=" * 70)
    print("SANITY CHECK — sMCL Gradient Routing Verification")
    print("=" * 70)
    device = "cpu"
    K = 3
    B = 8

    # ---- 1. Instantiate ensemble ----
    ensemble = MCLDiffusion(K=K, in_channels=1, base_channels=32, time_emb_dim=128)
    ensemble.to(device)
    n_params = sum(p.numel() for p in ensemble.parameters())
    print(f"[OK] MCLDiffusion instantiated  (K={K}, {n_params:,} total params)")

    # ---- 2. Forward pass shape check ----
    dummy_x = torch.randn(B, 1, 28, 28, device=device)
    dummy_t = torch.rand(B, device=device)
    scores = ensemble(dummy_x, dummy_t)
    assert scores.shape == (K, B, 1, 28, 28), f"Expected ({K},{B},1,28,28), got {scores.shape}"
    print(f"[OK] Forward pass shape: {scores.shape}")

    # ---- 3. One sMCL training step ----
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    ensemble.train()

    # Zero all gradients explicitly before the test
    optimizer.zero_grad()

    loss, k_star = smcl_loss(ensemble, dummy_x, sigma_min=0.01, sigma_max=80.0)
    loss.backward()

    print(f"[OK] sMCL loss = {loss.item():.4f}")
    print(f"     Winners per example: {k_star.tolist()}")

    # ---- 4. CRITICAL: Verify gradient routing ----
    #
    # For each expert k, we check:
    #   - If k won ZERO examples → ALL of its param.grad must be exactly 0.0
    #   - If k won ≥1 examples  → at least SOME param.grad entries are non-zero.
    #
    # This is the definitive proof that torch.gather on the (K, B) loss
    # tensor correctly disconnects losing experts from the computation graph.

    print("\n--- Gradient Routing Analysis ---")
    all_checks_passed = True

    for k in range(K):
        n_wins = (k_star == k).sum().item()

        # Collect all gradient norms for this expert
        grad_norms = []
        has_nonzero_grad = False
        has_any_grad = False

        for name, param in ensemble.experts[k].named_parameters():
            if param.grad is not None:
                has_any_grad = True
                norm = param.grad.abs().sum().item()
                grad_norms.append(norm)
                if norm > 0:
                    has_nonzero_grad = True

        total_grad_norm = sum(grad_norms)

        if n_wins == 0:
            # LOSING expert: ALL gradients must be exactly zero
            if has_any_grad and total_grad_norm > 0:
                print(f"  [FAIL] Expert {k}: won {n_wins} examples but "
                      f"has non-zero gradient norm = {total_grad_norm:.6e}")
                all_checks_passed = False
            else:
                print(f"  [OK]   Expert {k}: won {n_wins} examples → "
                      f"gradient norm = {total_grad_norm:.6e} (correctly zero)")
        else:
            # WINNING expert: must have non-zero gradients
            if not has_nonzero_grad:
                print(f"  [FAIL] Expert {k}: won {n_wins} examples but "
                      f"has zero gradients")
                all_checks_passed = False
            else:
                print(f"  [OK]   Expert {k}: won {n_wins} examples → "
                      f"gradient norm = {total_grad_norm:.6e} (correctly non-zero)")

    # ---- 5. Per-example gradient isolation check ----
    #
    # Additional verification: run K separate forward passes and confirm
    # that each expert's gradient contribution matches only its won examples.
    print("\n--- Per-Example Isolation Verification ---")

    # Recompute with individual tracking
    optimizer.zero_grad()

    # Manually replicate the sMCL computation to verify isolation
    t = torch.rand(B, device=device)
    sig = sigma_schedule(t, sigma_min=0.01, sigma_max=80.0)
    eps = torch.randn(B, 1, 28, 28, device=device)
    x_t = dummy_x + sig[:, None, None, None] * eps

    scores = ensemble(x_t, t)                                     # (K, B, 1, 28, 28)
    sig_sq = (sig ** 2)[:, None, None, None]
    predicted = sig_sq.unsqueeze(0) * scores                      # (K, B, 1, 28, 28)
    ell_k = ((predicted - eps.unsqueeze(0)) ** 2).sum(dim=(2, 3, 4))  # (K, B)
    k_star2 = ell_k.detach().argmin(dim=0)                        # (B,)

    # Gather winners and backprop
    winner_losses = ell_k.gather(0, k_star2.unsqueeze(0)).squeeze(0)
    final_loss = winner_losses.mean()
    final_loss.backward()

    # For each expert, verify gradient exists iff it won at least one example
    for k in range(K):
        wins_k = (k_star2 == k).sum().item()
        grad_norm_k = sum(
            p.grad.abs().sum().item()
            for p in ensemble.experts[k].parameters()
            if p.grad is not None
        )
        is_winner = wins_k > 0
        has_grad = grad_norm_k > 1e-12

        status = "OK" if (is_winner == has_grad) else "FAIL"
        if status == "FAIL":
            all_checks_passed = False
        print(f"  [{status:4s}] Expert {k}: wins={wins_k}, "
              f"grad_norm={grad_norm_k:.6e}, "
              f"expected_grad={'yes' if is_winner else 'no'}")

    optimizer.step()
    print(f"\n[OK] Optimiser step completed")

    # ---- 6. Euler sampling from a single expert ----
    ensemble.eval()
    z = torch.randn(1, 1, 28, 28, device=device) * 80.0
    sample = sample_euler_mcl(ensemble, z, expert_idx=0, N=10,
                              sigma_min=0.01, sigma_max=80.0)
    assert sample.shape == (1, 1, 28, 28), f"Expected (1,1,28,28), got {sample.shape}"
    print(f"[OK] Euler sample from expert 0: {sample.shape}")

    print("\n" + "=" * 70)
    if all_checks_passed:
        print("ALL SANITY CHECKS PASSED — sMCL gradient routing is correct")
    else:
        print("SOME CHECKS FAILED — review gradient routing implementation")
    print("=" * 70)

    return all_checks_passed


# ---------------------------------------------------------------------------
# 7.  MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MCL Score-Based Diffusion on MNIST"
    )
    parser.add_argument("--mode", choices=["sanity", "train", "evaluate"],
                        default="sanity")
    parser.add_argument("--K", type=int, default=5, help="Number of experts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--sampler", choices=["euler", "heun"], default="euler")
    parser.add_argument("--n_samples_per_expert", type=int, default=2000)
    parser.add_argument("--checkpoint", type=str, default="mcl_diffusion.pt")
    args = parser.parse_args()

    if args.mode == "sanity":
        sanity_check()

    elif args.mode == "train":
        ensemble, ema_list = train_mcl(
            K=args.K,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            ema_decay=args.ema_decay,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        torch.save({
            "K": args.K,
            "ensemble_state_dict": ensemble.state_dict(),
            "ema_shadows": [ema.shadow for ema in ema_list],
        }, args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")

    elif args.mode == "evaluate":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.checkpoint, map_location=device)
        K = ckpt["K"]
        ensemble = MCLDiffusion(K=K).to(device)
        ensemble.load_state_dict(ckpt["ensemble_state_dict"])
        ema_list = [EMA(expert, decay=args.ema_decay) for expert in ensemble.experts]
        for k in range(K):
            ema_list[k].shadow = ckpt["ema_shadows"][k]

        evaluate_mcl(
            ensemble, ema_list,
            n_samples_per_expert=args.n_samples_per_expert,
            N_steps=args.n_steps,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            device=device,
        )
