"""
Phase 3 — Routing Strategies for MCL Diffusion Sampling
=========================================================
Implements three policies to dynamically select which expert provides
the score function s_θ during deterministic Probability Flow ODE
sampling (Euler discretisation).

All three strategies share the same ODE:
    x_i = x_{i+1} + (σ_{i+1} − σ_i) · σ_{i+1} · s_θ(x_{i+1}, t_{i+1})

They differ ONLY in how s_θ is chosen at each step i:

1. Single Expert      — one fixed expert k for the entire trajectory.
2. Heuristic Routing  — a prescribed, time-based rule (no learned params).
3. Gated Routing      — a learned GatingNetwork(x_t, t) picks the expert.

The only source of stochasticity is the initial noise x_N ~ N(0, σ(T)² I).
No random noise is injected during the ODE loop.
"""

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.baseline import (
    SinusoidalTimeEmbedding,
    sigma_schedule,
    discrete_sigmas,
)
from src.mcl import MCLDiffusion


# ---------------------------------------------------------------------------
# Helper: σ → t inversion (reused across all samplers)
# ---------------------------------------------------------------------------

def _sigma_to_time(sigma: torch.Tensor, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """Invert σ(t) = σ_min^{1-t} · σ_max^t  →  t = log(σ/σ_min) / log(σ_max/σ_min)."""
    return torch.log(sigma / sigma_min) / math.log(sigma_max / sigma_min)


# ===================================================================
# STRATEGY 1 — Single Expert
# ===================================================================

@torch.no_grad()
def sample_single_expert(
    ensemble: MCLDiffusion,
    initial_noise: torch.Tensor,
    expert_idx: int,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Deterministic Euler ODE sampling using a SINGLE fixed expert.

    The expert k = `expert_idx` is chosen once at the start and used
    for every time step of the trajectory.  No other expert is queried.

    Parameters
    ----------
    ensemble      : trained MCLDiffusion with K experts
    initial_noise : x_N ~ N(0, σ_max² I),  shape (B, 1, 28, 28)
    expert_idx    : index k ∈ {0, ..., K-1} of the expert to use
    N             : number of Euler discretisation steps
    sigma_min     : minimum σ (before appending 0)
    sigma_max     : σ(T) = σ_0

    Returns
    -------
    x_0 : denoised samples (B, 1, 28, 28)
    """
    model = ensemble.experts[expert_idx]
    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(initial_noise.device)

    x = initial_noise

    for i in range(N):
        sig_curr = sigmas[i]
        sig_next = sigmas[i + 1]

        if sig_curr == 0:
            continue

        t_batch = _sigma_to_time(sig_curr, sigma_min, sigma_max)
        t_batch = t_batch.expand(x.shape[0]).to(x.device)

        score = model(x, t_batch)
        x = x + (sig_next - sig_curr) * sig_curr * score

    return x


# ===================================================================
# STRATEGY 2 — Heuristic Routing (time-based prescribed rule)
# ===================================================================

def default_heuristic_rule(t: float, K: int) -> int:
    """Default time-based routing: partition [0, 1] into K equal bands.

    High t (large noise) → expert 0
    Low t  (fine detail)  → expert K-1

    The idea is that different experts may specialise in different
    noise regimes.  Expert 0 handles the coarse structure (high σ),
    expert K-1 handles the fine details (low σ).

    Parameters
    ----------
    t : continuous time in [0, 1]  (scalar float)
    K : number of experts

    Returns
    -------
    expert index k ∈ {0, ..., K-1}

    Modification guide
    ------------------
    Replace this function with any rule mapping (t, K) → k.
    Examples:
      - Round-robin:   k = step_index % K
      - Two-phase:     k = 0 if t > 0.5 else 1
      - Random fixed:  k = hash(step_index) % K  (still deterministic)
    """
    # Partition [0, 1] into K equal bands.
    # t=1.0 (noisiest) → band 0 → expert 0
    # t≈0.0 (cleanest) → band K-1 → expert K-1
    band = min(int(t * K), K - 1)
    return band


@torch.no_grad()
def sample_heuristic_routing(
    ensemble: MCLDiffusion,
    initial_noise: torch.Tensor,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    routing_rule: Callable[[float, int], int] = default_heuristic_rule,
) -> torch.Tensor:
    """Deterministic Euler ODE with heuristic (prescribed) expert routing.

    At each step, the expert is chosen by `routing_rule(t, K)` — a
    hard-coded function with no learned parameters.  The rule can be
    swapped out without touching the ODE loop.

    Parameters
    ----------
    ensemble      : trained MCLDiffusion
    initial_noise : x_N ~ N(0, σ_max² I)
    N             : Euler steps
    sigma_min, sigma_max : noise schedule bounds
    routing_rule  : callable (t: float, K: int) → expert index k

    Returns
    -------
    x_0 : denoised samples (B, 1, 28, 28)
    """
    K = ensemble.K
    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(initial_noise.device)

    x = initial_noise

    for i in range(N):
        sig_curr = sigmas[i]
        sig_next = sigmas[i + 1]

        if sig_curr == 0:
            continue

        t_val = _sigma_to_time(sig_curr, sigma_min, sigma_max).item()
        t_batch = torch.full((x.shape[0],), t_val, device=x.device)

        # Select expert via the prescribed rule — evaluated at each step
        k = routing_rule(t_val, K)
        score = ensemble.experts[k](x, t_batch)

        x = x + (sig_next - sig_curr) * sig_curr * score

    return x


# ===================================================================
# STRATEGY 3 — Gated Routing (learned auxiliary classifier)
# ===================================================================

# ---------------------------------------------------------------------------
# 3a.  GatingNetwork architecture
# ---------------------------------------------------------------------------

class GatingNetwork(nn.Module):
    """Small time-conditioned CNN that predicts which expert is best.

    Given a noisy image x_t and time t, outputs a probability vector
    of size K (via softmax) indicating which expert should handle
    this (x_t, t) pair.

    Architecture
    ------------
    - 3 conv layers (1→16→32→64) with stride-2 down-sampling
    - Sinusoidal time embedding concatenated with flattened spatial features
    - 2-layer MLP head → K logits

    This is intentionally much smaller than the score network (~50K params
    vs ~1.5M) since it only needs to classify, not denoise.
    """

    def __init__(self, K: int, in_channels: int = 1, time_emb_dim: int = 64):
        super().__init__()
        self.K = K

        # --- Spatial feature extractor ---
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # 28→14
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # 14→7
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 7→4
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),                              # 4→1×1
        )
        # Output: (B, 64)

        # --- Time embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # --- Classification head ---
        # Input: 64 (spatial) + time_emb_dim (time) = 64 + 64 = 128
        self.head = nn.Sequential(
            nn.Linear(64 + time_emb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, K),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict expert probabilities.

        Parameters
        ----------
        x : (B, 1, 28, 28)  noisy image x_t
        t : (B,)             continuous time in [0, 1]

        Returns
        -------
        logits : (B, K)  — raw logits (apply softmax/argmax externally)
        """
        spatial = self.conv(x).squeeze(-1).squeeze(-1)  # (B, 64)
        t_emb = self.time_embed(t)                       # (B, time_emb_dim)
        combined = torch.cat([spatial, t_emb], dim=1)    # (B, 64+64)
        return self.head(combined)                        # (B, K)


# ---------------------------------------------------------------------------
# 3b.  GatingNetwork training loop
# ---------------------------------------------------------------------------

def train_gating_network(
    ensemble: MCLDiffusion,
    K: int,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_root: str = "./data",
) -> GatingNetwork:
    """Train the GatingNetwork to predict the WTA winner k*.

    Training procedure
    ------------------
    1. Freeze all K expert weights (no gradients flow to them).
    2. For each batch of clean images x_0:
       a. Sample t ~ U(0,1) and ε ~ N(0, I), compute x_t = x_0 + σ(t)ε
       b. Forward all K frozen experts to get noise predictions x^k
       c. Compute per-expert errors ℓ_k = ‖x^k − ε‖²
       d. Label k* = argmin_k ℓ_k   (per example, the true WTA winner)
       e. Train the GatingNetwork with CrossEntropyLoss(logits, k*)
    3. The gating network learns to predict which expert would win
       for a given (x_t, t) input, without needing the ground-truth ε.

    Parameters
    ----------
    ensemble   : trained MCLDiffusion (weights are FROZEN during this training)
    K          : number of experts
    epochs     : training epochs for the gating network
    batch_size : batch size
    lr         : learning rate

    Returns
    -------
    Trained GatingNetwork
    """
    print(f"Training GatingNetwork (K={K}) on device: {device}")

    # ---- Freeze all expert parameters ----
    ensemble.eval()
    for param in ensemble.parameters():
        param.requires_grad_(False)

    # ---- Data ----
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # ---- Gating network / Optimiser ----
    gating = GatingNetwork(K=K, in_channels=1, time_emb_dim=64).to(device)
    optimizer = torch.optim.Adam(gating.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    # ---- Training ----
    gating.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x0, _ in loader:
            x0 = x0.to(device)
            B = x0.shape[0]

            # --- Generate noisy data and WTA labels (no grad for experts) ---
            with torch.no_grad():
                t = torch.rand(B, device=device)
                sig = sigma_schedule(t, sigma_min, sigma_max)           # (B,)
                eps = torch.randn_like(x0)                              # ε
                x_t = x0 + sig[:, None, None, None] * eps              # x_t

                # Forward all K experts
                scores = ensemble(x_t, t)                               # (K, B, 1, 28, 28)
                sig_sq = (sig ** 2)[:, None, None, None]                # (B, 1, 1, 1)
                predicted_noise = sig_sq.unsqueeze(0) * scores          # (K, B, 1, 28, 28)

                # Per-expert errors
                ell_k = ((predicted_noise - eps.unsqueeze(0)) ** 2).sum(dim=(2, 3, 4))  # (K, B)

                # Ground-truth WTA labels
                k_star = ell_k.argmin(dim=0)                            # (B,)

            # --- Gating network forward + cross-entropy loss ---
            logits = gating(x_t, t)                                     # (B, K)
            loss = F.cross_entropy(logits, k_star)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * B
            total_correct += (logits.argmax(dim=1) == k_star).sum().item()
            total_samples += B

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        print(f"  Epoch {epoch:2d}/{epochs}  |  CE loss = {avg_loss:.4f}  |  "
              f"accuracy = {accuracy:.1%}")

    # ---- Restore expert grad flags ----
    for param in ensemble.parameters():
        param.requires_grad_(True)

    return gating


# ---------------------------------------------------------------------------
# 3c.  Gated routing sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_gated_routing(
    ensemble: MCLDiffusion,
    gating: GatingNetwork,
    initial_noise: torch.Tensor,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Deterministic Euler ODE with learned gating-based expert routing.

    At each step i, the trained GatingNetwork receives the current
    (x_{i+1}, t_{i+1}) and outputs a probability vector over K experts.
    The expert with the highest probability (argmax) is used for the
    score computation at that step.

    Note: the gating network output is per-example, so different images
    in the same batch can be routed to different experts at the same
    time step.  For efficiency, we group examples by their assigned
    expert and run each expert only on its assigned subset.

    Parameters
    ----------
    ensemble      : trained MCLDiffusion
    gating        : trained GatingNetwork
    initial_noise : x_N ~ N(0, σ_max² I)
    N             : Euler steps
    sigma_min, sigma_max : noise schedule bounds

    Returns
    -------
    x_0 : denoised samples (B, 1, 28, 28)
    """
    K = ensemble.K
    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(initial_noise.device)

    x = initial_noise
    B = x.shape[0]

    for i in range(N):
        sig_curr = sigmas[i]
        sig_next = sigmas[i + 1]

        if sig_curr == 0:
            continue

        t_val = _sigma_to_time(sig_curr, sigma_min, sigma_max)
        t_batch = t_val.expand(B).to(x.device)

        # --- Gating: predict which expert to use per example ---
        logits = gating(x, t_batch)           # (B, K)
        expert_assignment = logits.argmax(dim=1)  # (B,)

        # --- Compute the score by routing each example to its expert ---
        score = torch.zeros_like(x)           # (B, 1, 28, 28)

        for k in range(K):
            mask = (expert_assignment == k)    # (B,) bool
            if mask.any():
                x_subset = x[mask]
                t_subset = t_batch[mask]
                score_k = ensemble.experts[k](x_subset, t_subset)
                score[mask] = score_k

        # --- Euler ODE step (purely deterministic, no noise added) ---
        x = x + (sig_next - sig_curr) * sig_curr * score

    return x


# ===================================================================
# SANITY CHECK
# ===================================================================

def sanity_check():
    """Verify all three routing strategies execute without errors.

    Uses the SAME initial noise tensor for all three strategies to
    confirm they all produce valid (B, 1, 28, 28) output tensors
    through the full Euler ODE loop from t=T down to t=0.
    """
    print("=" * 70)
    print("SANITY CHECK — Routing Strategies (Phase 3)")
    print("=" * 70)
    device = "cpu"
    K = 3
    B = 2
    N_steps = 10  # few steps for speed
    sigma_min, sigma_max = 0.01, 80.0

    # ---- Instantiate ensemble ----
    ensemble = MCLDiffusion(K=K, in_channels=1, base_channels=32, time_emb_dim=128)
    ensemble.to(device).eval()
    print(f"[OK] MCLDiffusion instantiated (K={K})")

    # ---- Fixed initial noise (shared across all strategies) ----
    torch.manual_seed(42)
    z = torch.randn(B, 1, 28, 28, device=device) * sigma_max
    print(f"[OK] Initial noise: {z.shape}, norm = {z.norm():.2f}")

    # ---- Strategy 1: Single Expert ----
    out1 = sample_single_expert(
        ensemble, z.clone(), expert_idx=0,
        N=N_steps, sigma_min=sigma_min, sigma_max=sigma_max,
    )
    assert out1.shape == (B, 1, 28, 28), f"Expected ({B},1,28,28), got {out1.shape}"
    print(f"[OK] Strategy 1 (Single Expert k=0):    shape={out1.shape}, "
          f"range=[{out1.min():.3f}, {out1.max():.3f}]")

    # ---- Strategy 2: Heuristic Routing ----
    out2 = sample_heuristic_routing(
        ensemble, z.clone(),
        N=N_steps, sigma_min=sigma_min, sigma_max=sigma_max,
        routing_rule=default_heuristic_rule,
    )
    assert out2.shape == (B, 1, 28, 28), f"Expected ({B},1,28,28), got {out2.shape}"
    print(f"[OK] Strategy 2 (Heuristic Routing):    shape={out2.shape}, "
          f"range=[{out2.min():.3f}, {out2.max():.3f}]")

    # ---- Strategy 3: Gated Routing ----
    # 3a. Instantiate a dummy (untrained) gating network
    gating = GatingNetwork(K=K, in_channels=1, time_emb_dim=64).to(device)
    gating.eval()
    n_gating_params = sum(p.numel() for p in gating.parameters())
    print(f"[OK] GatingNetwork instantiated ({n_gating_params:,} params)")

    # 3b. Verify gating forward pass shape
    dummy_t = torch.rand(B, device=device)
    logits = gating(z[:B], dummy_t)
    assert logits.shape == (B, K), f"Expected ({B},{K}), got {logits.shape}"
    print(f"[OK] GatingNetwork forward: logits shape={logits.shape}")

    # 3c. Run gated sampling
    out3 = sample_gated_routing(
        ensemble, gating, z.clone(),
        N=N_steps, sigma_min=sigma_min, sigma_max=sigma_max,
    )
    assert out3.shape == (B, 1, 28, 28), f"Expected ({B},1,28,28), got {out3.shape}"
    print(f"[OK] Strategy 3 (Gated Routing):        shape={out3.shape}, "
          f"range=[{out3.min():.3f}, {out3.max():.3f}]")

    # ---- Verify determinism: same noise + same strategy → same output ----
    out1_repeat = sample_single_expert(
        ensemble, z.clone(), expert_idx=0,
        N=N_steps, sigma_min=sigma_min, sigma_max=sigma_max,
    )
    assert torch.allclose(out1, out1_repeat, atol=1e-5), "Determinism check failed!"
    print(f"[OK] Determinism verified: same noise → same output (max diff = "
          f"{(out1 - out1_repeat).abs().max():.2e})")

    # ---- Quick gating training check (1 epoch on dummy data) ----
    print("\n--- Quick GatingNetwork Training Test (1 epoch, tiny batch) ---")
    ensemble.train()
    # Create a small dummy dataset
    dummy_x0 = torch.rand(32, 1, 28, 28, device=device)
    dummy_loader_data = TensorDataset(dummy_x0, torch.zeros(32, dtype=torch.long))
    mini_loader = DataLoader(dummy_loader_data, batch_size=16)

    # Freeze experts and train gating for 1 step
    ensemble.eval()
    for p in ensemble.parameters():
        p.requires_grad_(False)

    gating.train()
    opt = torch.optim.Adam(gating.parameters(), lr=1e-3)

    for x0_batch, _ in mini_loader:
        x0_batch = x0_batch.to(device)
        B_mini = x0_batch.shape[0]

        with torch.no_grad():
            t = torch.rand(B_mini, device=device)
            sig = sigma_schedule(t, sigma_min, sigma_max)
            eps = torch.randn_like(x0_batch)
            x_t = x0_batch + sig[:, None, None, None] * eps

            scores = ensemble(x_t, t)
            sig_sq = (sig ** 2)[:, None, None, None]
            pred = sig_sq.unsqueeze(0) * scores
            ell_k = ((pred - eps.unsqueeze(0)) ** 2).sum(dim=(2, 3, 4))
            k_star = ell_k.argmin(dim=0)

        logits = gating(x_t, t)
        loss = F.cross_entropy(logits, k_star)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[OK] Gating training step:  CE loss = {loss.item():.4f}, "
              f"labels = {k_star.tolist()}")
        break  # one step is enough

    for p in ensemble.parameters():
        p.requires_grad_(True)

    print("\n" + "=" * 70)
    print("ALL SANITY CHECKS PASSED — all 3 routing strategies work correctly")
    print("=" * 70)


# ===================================================================
# MAIN ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 — Routing strategies for MCL diffusion sampling"
    )
    parser.add_argument("--mode", choices=["sanity", "train_gating"],
                        default="sanity")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs for gating network training")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--ensemble_checkpoint", type=str, default="mcl_diffusion.pt",
                        help="Path to trained MCL ensemble checkpoint")
    parser.add_argument("--gating_checkpoint", type=str, default="gating_network.pt",
                        help="Path to save/load gating network")
    args = parser.parse_args()

    if args.mode == "sanity":
        sanity_check()

    elif args.mode == "train_gating":
        from src.mcl import EMA

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trained ensemble
        ckpt = torch.load(args.ensemble_checkpoint, map_location=device)
        K = ckpt["K"]
        ensemble = MCLDiffusion(K=K).to(device)
        ensemble.load_state_dict(ckpt["ensemble_state_dict"])

        # Optionally apply EMA weights for label generation
        ema_list = [EMA(expert, decay=0.999) for expert in ensemble.experts]
        for k in range(K):
            ema_list[k].shadow = ckpt["ema_shadows"][k]
            ema_list[k].apply(ensemble.experts[k])

        # Train gating network
        gating = train_gating_network(
            ensemble, K=K,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            device=device,
        )

        # Save gating checkpoint
        torch.save({
            "K": K,
            "gating_state_dict": gating.state_dict(),
        }, args.gating_checkpoint)
        print(f"Gating network saved to {args.gating_checkpoint}")
