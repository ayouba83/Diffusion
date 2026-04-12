"""
Phase 4 — Evaluation & Analysis for MCL Diffusion
===================================================
Generates all quantitative metrics and figures required for the final
academic report:

1. Quantitative Metrics — FID, Precision, Recall
2. Intra-class Diversity — Denoising trajectory visualisation
3. Temporal Specialisation — Expert usage frequency vs time
4. Inter-class Diversity — Per-expert digit distribution histograms

All figures are saved as .png files for direct integration into the report.
"""

import math
import os
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from src.baseline import (
    SmallUNet,
    EMA,
    sigma_schedule,
    discrete_sigmas,
    get_inception_features,
    compute_fid,
    compute_precision_recall,
)
from src.mcl import MCLDiffusion
from src.routing import (
    GatingNetwork,
    _sigma_to_time,
    sample_single_expert,
    sample_heuristic_routing,
    sample_gated_routing,
)


# ===================================================================
# 1.  QUANTITATIVE METRICS  (FID, Precision, Recall)
# ===================================================================

@torch.no_grad()
def compute_all_metrics(
    ensemble: MCLDiffusion,
    gating: Optional[GatingNetwork],
    N_steps: int = 100,
    n_samples: int = 10000,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    batch_size: int = 256,
    device: str = "cpu",
    data_root: str = "./data",
) -> dict:
    """Compute FID, Precision, and Recall for every routing strategy.

    Evaluates:
    - Each single expert k ∈ {0, ..., K-1}
    - Heuristic routing
    - Gated routing (if gating network provided)
    - Pooled ensemble (equal samples from each expert)

    All metrics are computed against the MNIST test set.

    Parameters
    ----------
    ensemble  : trained MCLDiffusion
    gating    : trained GatingNetwork (or None to skip gated routing)
    n_samples : number of images to generate per strategy
    N_steps   : Euler discretisation steps

    Returns
    -------
    dict mapping strategy name → {"fid": float, "precision": float, "recall": float}
    """
    K = ensemble.K
    ensemble.eval()
    results = {}

    # ---- Load real test images and extract features once ----
    test_set = datasets.MNIST(data_root, train=False, download=True,
                              transform=transforms.ToTensor())
    n_real = min(n_samples, len(test_set))
    real_images = torch.stack([test_set[i][0] for i in range(n_real)])
    print("Extracting features for real MNIST test images...")
    real_feats = get_inception_features(real_images.to(device))

    def _evaluate(name: str, images: torch.Tensor):
        """Helper: extract features and compute the three metrics."""
        print(f"  Extracting features for '{name}'...")
        gen_feats = get_inception_features(images.to(device))
        fid = compute_fid(real_feats, gen_feats)
        prec, rec = compute_precision_recall(real_feats, gen_feats)
        results[name] = {"fid": fid, "precision": prec, "recall": rec}
        print(f"    FID={fid:.2f}  Precision={prec:.4f}  Recall={rec:.4f}")

    def _generate_batched(sample_fn, **kwargs) -> torch.Tensor:
        """Generate n_samples images using the given sampling function."""
        samples = []
        for start in range(0, n_samples, batch_size):
            B = min(batch_size, n_samples - start)
            z = torch.randn(B, 1, 28, 28, device=device) * sigma_max
            out = sample_fn(ensemble, z, N=N_steps,
                            sigma_min=sigma_min, sigma_max=sigma_max, **kwargs)
            samples.append(out.clamp(0, 1).cpu())
        return torch.cat(samples, dim=0)

    # ---- Per-expert (single expert routing) ----
    all_expert_samples = []
    for k in range(K):
        print(f"Generating {n_samples} images with Expert {k}...")
        imgs = _generate_batched(sample_single_expert, expert_idx=k)
        all_expert_samples.append(imgs)
        _evaluate(f"expert_{k}", imgs)

    # ---- Heuristic routing ----
    print(f"Generating {n_samples} images with Heuristic routing...")
    imgs_heuristic = _generate_batched(sample_heuristic_routing)
    _evaluate("heuristic", imgs_heuristic)

    # ---- Gated routing ----
    if gating is not None:
        print(f"Generating {n_samples} images with Gated routing...")
        gating.eval()
        imgs_gated = []
        for start in range(0, n_samples, batch_size):
            B = min(batch_size, n_samples - start)
            z = torch.randn(B, 1, 28, 28, device=device) * sigma_max
            out = sample_gated_routing(ensemble, gating, z, N=N_steps,
                                       sigma_min=sigma_min, sigma_max=sigma_max)
            imgs_gated.append(out.clamp(0, 1).cpu())
        imgs_gated = torch.cat(imgs_gated, dim=0)
        _evaluate("gated", imgs_gated)

    # ---- Pooled ensemble (equal split across experts) ----
    per_expert = n_samples // K
    pooled = torch.cat([s[:per_expert] for s in all_expert_samples], dim=0)
    _evaluate("pooled_ensemble", pooled)

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print(f"{'Strategy':<20s} {'FID':>8s} {'Precision':>10s} {'Recall':>8s}")
    print("-" * 60)
    for name, m in results.items():
        print(f"{name:<20s} {m['fid']:>8.2f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
    print("=" * 60)

    return results


# ===================================================================
# 2.  INTRA-CLASS DIVERSITY — Trajectory Visualisation
# ===================================================================

@torch.no_grad()
def _sample_with_trajectory(
    ensemble: MCLDiffusion,
    initial_noise: torch.Tensor,
    expert_idx: Optional[int],
    gating: Optional[GatingNetwork],
    N: int = 100,
    n_snapshots: int = 8,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> list[torch.Tensor]:
    """Run Euler ODE and capture intermediate states for visualisation.

    Parameters
    ----------
    expert_idx : if not None, use single-expert routing with this index.
    gating     : if not None (and expert_idx is None), use gated routing.
    n_snapshots: number of intermediate states to capture (evenly spaced).

    Returns
    -------
    List of (1, 1, 28, 28) tensors at evenly-spaced steps, including the
    initial noise (index 0) and the final denoised image (index -1).
    """
    K = ensemble.K
    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(initial_noise.device)

    x = initial_noise.clone()

    # Determine which ODE steps to snapshot (evenly spaced across N steps)
    snapshot_indices = set(
        [int(j * N / (n_snapshots - 1)) for j in range(n_snapshots - 1)]
    )
    snapshot_indices.add(N - 1)  # always include the last step

    trajectory = [x.clone().cpu()]  # initial noise

    for i in range(N):
        sig_curr = sigmas[i]
        sig_next = sigmas[i + 1]

        if sig_curr == 0:
            continue

        t_val = _sigma_to_time(sig_curr, sigma_min, sigma_max)
        t_batch = t_val.expand(x.shape[0]).to(x.device)

        # Choose the score source
        if expert_idx is not None:
            score = ensemble.experts[expert_idx](x, t_batch)
        elif gating is not None:
            logits = gating(x, t_batch)
            k = logits.argmax(dim=1).item()
            score = ensemble.experts[k](x, t_batch)
        else:
            score = ensemble.experts[0](x, t_batch)

        x = x + (sig_next - sig_curr) * sig_curr * score

        if i in snapshot_indices:
            trajectory.append(x.clone().cpu())

    # Ensure the final denoised state is included
    if len(trajectory) < n_snapshots + 1:
        trajectory.append(x.clone().cpu())

    return trajectory


def plot_trajectories(
    ensemble: MCLDiffusion,
    gating: Optional[GatingNetwork] = None,
    N: int = 100,
    n_snapshots: int = 8,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cpu",
    save_path: str = "fig_trajectories.png",
):
    """Generate and plot denoising trajectories for intra-class diversity.

    Fixes ONE initial noise tensor, then denoises it through:
    - Expert 0 only
    - Expert 1 only
    - Gated routing (if gating is provided)

    Produces a grid where each row is a strategy and each column is
    an intermediate time step, visually proving that different routing
    policies yield different outputs from the same initial noise.
    """
    K = ensemble.K
    ensemble.eval()
    if gating is not None:
        gating.eval()

    # Fixed initial noise — the same for all strategies
    torch.manual_seed(0)
    z = torch.randn(1, 1, 28, 28, device=device) * sigma_max

    # Collect trajectories
    strategies = []
    labels = []

    # At least 2 single-expert rows
    for k in range(min(K, 3)):
        traj = _sample_with_trajectory(
            ensemble, z.clone(), expert_idx=k, gating=None,
            N=N, n_snapshots=n_snapshots, sigma_min=sigma_min, sigma_max=sigma_max,
        )
        strategies.append(traj)
        labels.append(f"Expert {k}")

    # Gated routing row
    if gating is not None:
        traj = _sample_with_trajectory(
            ensemble, z.clone(), expert_idx=None, gating=gating,
            N=N, n_snapshots=n_snapshots, sigma_min=sigma_min, sigma_max=sigma_max,
        )
        strategies.append(traj)
        labels.append("Gated routing")

    # ---- Plot ----
    n_rows = len(strategies)
    n_cols = min(len(strategies[0]), n_snapshots + 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, (traj, label) in enumerate(zip(strategies, labels)):
        # Evenly pick n_cols snapshots from the trajectory
        indices = np.linspace(0, len(traj) - 1, n_cols, dtype=int)
        for col_idx, snap_idx in enumerate(indices):
            img = traj[snap_idx].squeeze().clamp(0, 1).numpy()
            axes[row, col_idx].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col_idx].axis("off")
            if col_idx == 0:
                axes[row, col_idx].set_title("Noise", fontsize=9)
                axes[row, col_idx].set_ylabel(label, fontsize=10, rotation=0,
                                               labelpad=60, va="center")
            elif col_idx == n_cols - 1:
                axes[row, col_idx].set_title("Final", fontsize=9)
            else:
                step_frac = snap_idx / max(len(traj) - 1, 1)
                axes[row, col_idx].set_title(f"t≈{1 - step_frac:.2f}", fontsize=8)

    fig.suptitle("Intra-class Diversity: Same Noise, Different Routing",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Trajectory figure → {save_path}")


# ===================================================================
# 3.  TEMPORAL SPECIALISATION — Expert usage vs time
# ===================================================================

@torch.no_grad()
def record_gating_decisions(
    ensemble: MCLDiffusion,
    gating: GatingNetwork,
    n_samples: int = 64,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cpu",
) -> dict[str, list]:
    """Run gated-routing generation and record which expert is chosen at each step.

    Returns
    -------
    dict with:
      "t_values"      : list of float — continuous time at each ODE step
      "expert_counts"  : dict[int, list[int]] — per-expert count at each step
      "total_per_step" : int (= n_samples, constant for all steps)
    """
    K = ensemble.K
    ensemble.eval()
    gating.eval()

    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(device)

    z = torch.randn(n_samples, 1, 28, 28, device=device) * sigma_max
    x = z.clone()

    t_values = []
    expert_counts = defaultdict(list)  # k → [count_at_step_0, count_at_step_1, ...]

    for i in range(N):
        sig_curr = sigmas[i]
        sig_next = sigmas[i + 1]

        if sig_curr == 0:
            continue

        t_val = _sigma_to_time(sig_curr, sigma_min, sigma_max)
        t_batch = t_val.expand(n_samples).to(device)

        # Gating decision
        logits = gating(x, t_batch)
        assignment = logits.argmax(dim=1)  # (n_samples,)

        t_values.append(t_val.item())
        for k in range(K):
            expert_counts[k].append((assignment == k).sum().item())

        # Euler step (grouped by expert)
        score = torch.zeros_like(x)
        for k in range(K):
            mask = (assignment == k)
            if mask.any():
                score[mask] = ensemble.experts[k](x[mask], t_batch[mask])

        x = x + (sig_next - sig_curr) * sig_curr * score

    return {
        "t_values": t_values,
        "expert_counts": dict(expert_counts),
        "total_per_step": n_samples,
    }


def plot_temporal_specialisation(
    routing_log: dict,
    K: int,
    save_path: str = "fig_temporal_specialisation.png",
):
    """Plot a stacked area chart of expert usage frequency across the time axis.

    Parameters
    ----------
    routing_log : output of `record_gating_decisions`
    K           : number of experts
    save_path   : output .png path
    """
    t_values = np.array(routing_log["t_values"])
    total = routing_log["total_per_step"]
    expert_counts = routing_log["expert_counts"]

    # Convert counts to fractions
    fractions = np.zeros((K, len(t_values)))
    for k in range(K):
        if k in expert_counts:
            fractions[k] = np.array(expert_counts[k]) / total

    # ---- Stacked area chart ----
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    ax.stackplot(
        t_values, *fractions,
        labels=[f"Expert {k}" for k in range(K)],
        colors=colors,
        alpha=0.85,
    )
    ax.set_xlabel("Continuous time $t$ (1 = high noise, 0 = clean)", fontsize=11)
    ax.set_ylabel("Usage fraction", fontsize=11)
    ax.set_title("Temporal Specialisation: Expert Usage Across the Denoising Trajectory",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(t_values.max(), t_values.min())  # high-t on the left
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Temporal specialisation figure → {save_path}")


# ===================================================================
# 4.  INTER-CLASS DIVERSITY — Per-expert digit distributions
# ===================================================================

class LeNetClassifier(nn.Module):
    """Simple LeNet-5-style classifier for MNIST digit recognition.

    Used to classify generated images and analyse which digit modes
    each expert has specialised in.  Pre-training on real MNIST is
    handled by `train_lenet_classifier`.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),    # 28→28
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 28→14
            nn.Conv2d(6, 16, 5),               # 14→10
            nn.ReLU(),
            nn.MaxPool2d(2),                   # 10→5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_lenet_classifier(
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    data_root: str = "./data",
) -> LeNetClassifier:
    """Train a LeNet classifier on MNIST for digit-mode analysis.

    Returns the trained classifier (eval mode).
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = LeNetClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"  LeNet epoch {epoch}/{epochs}  |  acc = {correct / total:.1%}")

    model.eval()
    return model


@torch.no_grad()
def classify_generated_images(
    classifier: LeNetClassifier,
    images: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    """Classify a batch of generated images.

    Parameters
    ----------
    images : (N, 1, 28, 28) in [0, 1]

    Returns
    -------
    predictions : (N,) long tensor with digit labels 0-9
    """
    classifier.eval()
    device = next(classifier.parameters()).device
    preds = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device)
        logits = classifier(batch)
        preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(preds, dim=0)


def plot_interclass_diversity(
    ensemble: MCLDiffusion,
    classifier: LeNetClassifier,
    n_samples_per_expert: int = 1000,
    N_steps: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cpu",
    save_path: str = "fig_interclass_diversity.png",
):
    """Generate images per expert, classify them, and plot digit distributions.

    Produces a figure with K subplots (one per expert), each showing a
    histogram of the predicted digit classes (0-9).  If an expert has
    specialised in certain modes, its histogram will be concentrated on
    specific digits.
    """
    K = ensemble.K
    ensemble.eval()

    digit_dists = {}  # k → (10,) count array

    for k in range(K):
        print(f"  Generating {n_samples_per_expert} images with Expert {k}...")
        samples = []
        for start in range(0, n_samples_per_expert, 256):
            B = min(256, n_samples_per_expert - start)
            z = torch.randn(B, 1, 28, 28, device=device) * sigma_max
            out = sample_single_expert(ensemble, z, expert_idx=k, N=N_steps,
                                       sigma_min=sigma_min, sigma_max=sigma_max)
            samples.append(out.clamp(0, 1).cpu())
        samples = torch.cat(samples, dim=0)

        preds = classify_generated_images(classifier, samples)
        counts = torch.zeros(10, dtype=torch.long)
        for digit in range(10):
            counts[digit] = (preds == digit).sum().item()
        digit_dists[k] = counts.numpy()

    # ---- Plot ----
    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 3.5), sharey=True)
    if K == 1:
        axes = [axes]
    colors = plt.cm.Set2(np.linspace(0, 1, K))

    for k in range(K):
        ax = axes[k]
        counts = digit_dists[k]
        fracs = counts / counts.sum()
        ax.bar(range(10), fracs, color=colors[k], edgecolor="black", linewidth=0.5)
        ax.set_title(f"Expert {k}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Digit class")
        ax.set_xticks(range(10))
        ax.set_ylim(0, max(0.5, fracs.max() * 1.15))
        if k == 0:
            ax.set_ylabel("Fraction of generated images")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Inter-class Diversity: Digit Distribution per Expert",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Inter-class diversity figure → {save_path}")

    return digit_dists


# ===================================================================
# 5.  SANITY CHECK
# ===================================================================

def sanity_check():
    """Verify all evaluation functions work with dummy data.

    Tests
    -----
    1. FID / Precision / Recall on small dummy feature batches
    2. Trajectory visualisation plot generation
    3. Temporal specialisation plot generation
    4. Inter-class diversity plot generation
    All plots are saved to /tmp/ to confirm file I/O works.
    """
    print("=" * 70)
    print("SANITY CHECK — Evaluation & Analysis (Phase 4)")
    print("=" * 70)
    device = "cpu"
    K = 3
    B = 16
    N_steps = 10
    sigma_min, sigma_max = 0.01, 80.0

    # ---- Instantiate dummy models ----
    ensemble = MCLDiffusion(K=K, in_channels=1, base_channels=32, time_emb_dim=128)
    ensemble.to(device).eval()

    gating = GatingNetwork(K=K, in_channels=1, time_emb_dim=64).to(device)
    gating.eval()

    print(f"[OK] Dummy ensemble (K={K}) and gating network instantiated")

    # ---- Test 1: FID / Precision / Recall on dummy features ----
    print("\n--- Test 1: Metrics on dummy features ---")
    # Create dummy 2048-d feature vectors (simulating InceptionV3 output)
    real_feats = torch.randn(100, 2048)
    gen_feats = torch.randn(100, 2048)

    fid = compute_fid(real_feats, gen_feats)
    prec, rec = compute_precision_recall(real_feats, gen_feats, k=3)

    assert isinstance(fid, float), f"FID should be float, got {type(fid)}"
    assert 0.0 <= prec <= 1.0, f"Precision out of range: {prec}"
    assert 0.0 <= rec <= 1.0, f"Recall out of range: {rec}"
    print(f"[OK] FID = {fid:.2f}, Precision = {prec:.4f}, Recall = {rec:.4f}")

    # ---- Test 2: Trajectory visualisation ----
    print("\n--- Test 2: Trajectory visualisation ---")
    save_path_traj = "sanity_fig_trajectories.png"
    plot_trajectories(
        ensemble, gating=gating,
        N=N_steps, n_snapshots=6,
        sigma_min=sigma_min, sigma_max=sigma_max,
        device=device, save_path=save_path_traj,
    )
    assert os.path.isfile(save_path_traj), f"File not found: {save_path_traj}"
    size_kb = os.path.getsize(save_path_traj) / 1024
    print(f"[OK] Trajectory figure saved ({size_kb:.1f} KB)")

    # ---- Test 3: Temporal specialisation ----
    print("\n--- Test 3: Temporal specialisation ---")
    routing_log = record_gating_decisions(
        ensemble, gating,
        n_samples=8, N=N_steps,
        sigma_min=sigma_min, sigma_max=sigma_max,
        device=device,
    )
    assert len(routing_log["t_values"]) > 0, "No time steps recorded"
    assert len(routing_log["expert_counts"]) > 0, "No expert counts recorded"
    print(f"[OK] Recorded {len(routing_log['t_values'])} ODE steps, "
          f"{len(routing_log['expert_counts'])} experts tracked")

    save_path_temporal = "sanity_fig_temporal.png"
    plot_temporal_specialisation(routing_log, K=K, save_path=save_path_temporal)
    assert os.path.isfile(save_path_temporal), f"File not found: {save_path_temporal}"
    size_kb = os.path.getsize(save_path_temporal) / 1024
    print(f"[OK] Temporal specialisation figure saved ({size_kb:.1f} KB)")

    # ---- Test 4: Inter-class diversity ----
    print("\n--- Test 4: Inter-class diversity ---")

    # Train a tiny LeNet on dummy data (just check it runs)
    classifier = LeNetClassifier().to(device)
    # No real training — just verify the pipeline with random weights

    # Generate dummy images (random noise, won't look like digits)
    dummy_images = torch.rand(32, 1, 28, 28)
    preds = classify_generated_images(classifier, dummy_images)
    assert preds.shape == (32,), f"Expected (32,), got {preds.shape}"
    assert preds.min() >= 0 and preds.max() <= 9, "Predictions out of digit range"
    print(f"[OK] Classifier predictions: shape={preds.shape}, "
          f"range=[{preds.min()}, {preds.max()}]")

    save_path_inter = "sanity_fig_interclass.png"
    plot_interclass_diversity(
        ensemble, classifier,
        n_samples_per_expert=32, N_steps=N_steps,
        sigma_min=sigma_min, sigma_max=sigma_max,
        device=device, save_path=save_path_inter,
    )
    assert os.path.isfile(save_path_inter), f"File not found: {save_path_inter}"
    size_kb = os.path.getsize(save_path_inter) / 1024
    print(f"[OK] Inter-class diversity figure saved ({size_kb:.1f} KB)")

    # ---- Cleanup sanity check figures ----
    for f in [save_path_traj, save_path_temporal, save_path_inter]:
        if os.path.isfile(f):
            os.remove(f)
    print("\n[OK] Cleaned up sanity-check figures")

    print("\n" + "=" * 70)
    print("ALL SANITY CHECKS PASSED — evaluation pipeline is functional")
    print("=" * 70)


# ===================================================================
# 6.  MAIN ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4 — Evaluation & Analysis for MCL Diffusion"
    )
    parser.add_argument("--mode", choices=["sanity", "full"], default="sanity",
                        help="sanity: dummy-data checks | full: generate all figures")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Samples per strategy for metrics")
    parser.add_argument("--n_samples_per_expert", type=int, default=1000,
                        help="Samples per expert for inter-class analysis")
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--ensemble_checkpoint", type=str, default="mcl_diffusion.pt")
    parser.add_argument("--gating_checkpoint", type=str, default="gating_network.pt")
    parser.add_argument("--lenet_checkpoint", type=str, default="lenet_classifier.pt")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory for output .png figures")
    args = parser.parse_args()

    if args.mode == "sanity":
        sanity_check()

    elif args.mode == "full":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(args.output_dir, exist_ok=True)

        # ---- Load trained ensemble ----
        ckpt = torch.load(args.ensemble_checkpoint, map_location=device)
        K = ckpt["K"]
        ensemble = MCLDiffusion(K=K).to(device)
        ensemble.load_state_dict(ckpt["ensemble_state_dict"])
        # Apply EMA weights
        ema_list = [EMA(expert, decay=0.999) for expert in ensemble.experts]
        for k in range(K):
            ema_list[k].shadow = ckpt["ema_shadows"][k]
            ema_list[k].apply(ensemble.experts[k])
        ensemble.eval()
        print(f"Loaded ensemble (K={K}) from {args.ensemble_checkpoint}")

        # ---- Load gating network ----
        gating = None
        if os.path.isfile(args.gating_checkpoint):
            gating_ckpt = torch.load(args.gating_checkpoint, map_location=device)
            gating = GatingNetwork(K=K).to(device)
            gating.load_state_dict(gating_ckpt["gating_state_dict"])
            gating.eval()
            print(f"Loaded gating network from {args.gating_checkpoint}")
        else:
            print(f"[WARN] Gating checkpoint not found at {args.gating_checkpoint}; "
                  f"skipping gated routing evaluation")

        # ---- Train or load LeNet classifier ----
        if os.path.isfile(args.lenet_checkpoint):
            classifier = LeNetClassifier().to(device)
            classifier.load_state_dict(
                torch.load(args.lenet_checkpoint, map_location=device)
            )
            classifier.eval()
            print(f"Loaded LeNet classifier from {args.lenet_checkpoint}")
        else:
            print("Training LeNet classifier on MNIST...")
            classifier = train_lenet_classifier(epochs=5, device=device)
            torch.save(classifier.state_dict(), args.lenet_checkpoint)
            print(f"Saved LeNet classifier to {args.lenet_checkpoint}")

        # ---- 1. Quantitative metrics ----
        print("\n" + "=" * 60)
        print("1. QUANTITATIVE METRICS")
        print("=" * 60)
        metrics = compute_all_metrics(
            ensemble, gating,
            N_steps=args.n_steps, n_samples=args.n_samples,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            device=device,
        )

        # ---- 2. Intra-class diversity ----
        print("\n" + "=" * 60)
        print("2. INTRA-CLASS DIVERSITY (Trajectory Visualisation)")
        print("=" * 60)
        plot_trajectories(
            ensemble, gating=gating,
            N=args.n_steps, n_snapshots=8,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            device=device,
            save_path=os.path.join(args.output_dir, "fig_trajectories.png"),
        )

        # ---- 3. Temporal specialisation ----
        if gating is not None:
            print("\n" + "=" * 60)
            print("3. TEMPORAL SPECIALISATION")
            print("=" * 60)
            routing_log = record_gating_decisions(
                ensemble, gating,
                n_samples=256, N=args.n_steps,
                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                device=device,
            )
            plot_temporal_specialisation(
                routing_log, K=K,
                save_path=os.path.join(args.output_dir, "fig_temporal_specialisation.png"),
            )

        # ---- 4. Inter-class diversity ----
        print("\n" + "=" * 60)
        print("4. INTER-CLASS DIVERSITY (Digit Distributions)")
        print("=" * 60)
        plot_interclass_diversity(
            ensemble, classifier,
            n_samples_per_expert=args.n_samples_per_expert,
            N_steps=args.n_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            device=device,
            save_path=os.path.join(args.output_dir, "fig_interclass_diversity.png"),
        )

        print("\n" + "=" * 60)
        print(f"ALL FIGURES SAVED TO: {args.output_dir}/")
        print("=" * 60)
