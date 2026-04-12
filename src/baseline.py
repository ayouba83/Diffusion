"""
Score-Based Diffusion Model — Baseline on MNIST
=================================================
Implements the Tweedie-based denoising objective with deterministic
Probability Flow ODE sampling (Euler + Heun).

Mathematical framework
----------------------
Forward noising:   X_t = X_0 + ε,  ε ~ N(0, σ(t)² I_d)
Tweedie loss:      E_{t, x0, ε}[ ‖σ(t)² s_θ(x0+ε, t) − ε‖² ]
Euler ODE step:    x_i = x_{i+1} + (σ_{i+1} − σ_i) σ_{i+1} s_θ(x_{i+1}, t_{i+1})
Heun ODE step:     (2nd-order Runge-Kutta, see `sample_heun`)
"""

import math
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# 1.  NOISE SCHEDULE
# ---------------------------------------------------------------------------

def sigma_schedule(t: torch.Tensor, sigma_min: float = 0.01, sigma_max: float = 80.0) -> torch.Tensor:
    """Continuous noise level σ(t) for t ∈ [0, 1].

    Uses a geometric (log-linear) schedule:  σ(t) = σ_min^{1-t} · σ_max^{t}.
    """
    return sigma_min ** (1.0 - t) * sigma_max ** t


def discrete_sigmas(N: int, sigma_min: float = 0.01, sigma_max: float = 80.0) -> torch.Tensor:
    """Return N+1 noise levels from σ_0 = σ_max down to σ_N = 0.

    Indices follow the assignment convention: σ_0 is the noisiest level
    (corresponding to t = T) and σ_N = 0 is the clean data endpoint.
    """
    # N levels from σ_max → σ_min, then append 0
    ts = torch.linspace(1.0, 0.0, N)
    sigmas = sigma_schedule(ts, sigma_min, sigma_max)
    sigmas = torch.cat([sigmas, torch.zeros(1)])  # σ_N = 0
    return sigmas


# ---------------------------------------------------------------------------
# 2.  LIGHTWEIGHT TIME-CONDITIONED U-NET
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps a scalar time/sigma to a vector via sinusoidal positional encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with GroupNorm + time-conditioned affine modulation."""

    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # Time conditioning: project to scale & shift
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * channels),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        # Affine modulation from time embedding
        scale_shift = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return x + h


class SmallUNet(nn.Module):
    """Memory-efficient U-Net score network for 1×28×28 images.

    Architecture
    ------------
    Encoder:  28→14→7    (2 down-sampling stages)
    Decoder:  7→14→28    (2 up-sampling stages with skip connections)

    Channel widths are kept small (32→64→128) so that K copies fit in
    memory when scaling to a mixture later.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_emb_dim: int = 128):
        super().__init__()
        C = base_channels  # 32

        # --- Time embedding ---
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # --- Encoder ---
        self.enc_conv_in = nn.Conv2d(in_channels, C, 3, padding=1)       # 1 → 32
        self.enc_res1 = ResBlock(C, time_emb_dim)                        # 32 @ 28×28
        self.down1 = nn.Conv2d(C, 2 * C, 3, stride=2, padding=1)        # 32 → 64 @ 14×14
        self.enc_res2 = ResBlock(2 * C, time_emb_dim)                    # 64 @ 14×14
        self.down2 = nn.Conv2d(2 * C, 4 * C, 3, stride=2, padding=1)    # 64 → 128 @ 7×7

        # --- Bottleneck ---
        self.mid_res1 = ResBlock(4 * C, time_emb_dim)
        self.mid_res2 = ResBlock(4 * C, time_emb_dim)

        # --- Decoder (skip connections double the channel count) ---
        self.up2 = nn.ConvTranspose2d(4 * C, 2 * C, 4, stride=2, padding=1)  # 128 → 64 @ 14×14
        self.dec_res2 = ResBlock(4 * C, time_emb_dim)   # 64+64=128 after cat → processes 128
        self.reduce2 = nn.Conv2d(4 * C, 2 * C, 1)       # 128 → 64

        self.up1 = nn.ConvTranspose2d(2 * C, C, 4, stride=2, padding=1)      # 64 → 32 @ 28×28
        self.dec_res1 = ResBlock(2 * C, time_emb_dim)    # 32+32=64 after cat → processes 64
        self.reduce1 = nn.Conv2d(2 * C, C, 1)            # 64 → 32

        # --- Output projection → score has same shape as input ---
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, C),
            nn.SiLU(),
            nn.Conv2d(C, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the score s_θ(x_t, t).

        Parameters
        ----------
        x : (B, 1, 28, 28)  noised image
        t : (B,)             continuous time in [0, 1]

        Returns
        -------
        score : (B, 1, 28, 28)  — the model output s_θ(x_t, t)
        """
        t_emb = self.time_embed(t)                      # (B, time_emb_dim)

        # Encoder
        h1 = self.enc_res1(self.enc_conv_in(x), t_emb)  # (B, C, 28, 28)
        h2 = self.enc_res2(self.down1(h1), t_emb)        # (B, 2C, 14, 14)
        h = self.down2(h2)                                # (B, 4C, 7, 7)

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_res2(h, t_emb)

        # Decoder + skip connections
        h = self.up2(h)                                   # (B, 2C, 14, 14)
        h = torch.cat([h, h2], dim=1)                     # (B, 4C, 14, 14)
        h = self.reduce2(self.dec_res2(h, t_emb))         # (B, 2C, 14, 14)

        h = self.up1(h)                                   # (B, C, 28, 28)
        h = torch.cat([h, h1], dim=1)                     # (B, 2C, 28, 28)
        h = self.reduce1(self.dec_res1(h, t_emb))         # (B, C, 28, 28)

        return self.out_conv(h)


# ---------------------------------------------------------------------------
# 3.  EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage
    -----
    ema = EMA(model, decay=0.999)
    # after each optimizer step:
    ema.step(model)
    # for evaluation / sampling:
    ema.apply(model)    # copy EMA weights into model
    ema.restore(model)  # restore original weights
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: p.clone().detach() for name, p in model.named_parameters()}
        self.backup = {}

    @torch.no_grad()
    def step(self, model: nn.Module):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Replace model params with EMA params (save originals for restore)."""
        self.backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original (non-EMA) parameters."""
        for name, p in model.named_parameters():
            p.data.copy_(self.backup[name])
        self.backup = {}


# ---------------------------------------------------------------------------
# 4.  TWEEDIE TRAINING LOSS
# ---------------------------------------------------------------------------

def tweedie_loss(
    model: nn.Module,
    x0: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Compute the Tweedie denoising loss (Eq. 4 from the assignment).

    L = E_{t, x0, ε}[ ‖σ(t)² s_θ(x0 + ε, t) − ε‖² ]

    Parameters
    ----------
    model : score network s_θ
    x0    : clean images (B, 1, 28, 28) in [0, 1]

    Returns
    -------
    scalar loss (mean over batch)
    """
    B = x0.shape[0]
    device = x0.device

    # Sample t ~ Uniform(0, 1)
    t = torch.rand(B, device=device)
    sig = sigma_schedule(t, sigma_min, sigma_max)  # (B,)

    # Sample unit noise z ~ N(0, I) and form ε = σ(t) z ~ N(0, σ(t)² I)
    z = torch.randn_like(x0)
    sig_4d = sig[:, None, None, None]
    x_t = x0 + sig_4d * z                           # X_t = X_0 + σ(t) z

    # Model predicts s_θ(x_t, t)
    score = model(x_t, t)                            # (B, 1, 28, 28)

    # σ(t)² s_θ should approximate ε = σ(t) z  (the SCALED noise, not unit z)
    predicted_noise = sig_4d ** 2 * score
    target = sig_4d * z                              # ε = σ z

    # MSE loss: ‖σ(t)² s_θ(x_t, t) − ε‖²
    loss = F.mse_loss(predicted_noise, target)
    return loss


# ---------------------------------------------------------------------------
# 5.  DETERMINISTIC ODE SAMPLERS
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler(
    model: nn.Module,
    initial_noise: torch.Tensor,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Euler (1st order) Probability Flow ODE sampler.

    Iteration (from i = N-1 down to 0):
        x_i = x_{i+1} + (σ_{i+1} − σ_i) · σ_{i+1} · s_θ(x_{i+1}, t_{i+1})

    Parameters
    ----------
    model         : trained score network s_θ  (should have EMA weights applied)
    initial_noise : x_N ~ N(0, σ(T)² I),  shape (B, 1, 28, 28)
    N             : number of discretisation steps
    sigma_min     : minimum noise level (before appending 0)
    sigma_max     : maximum noise level = σ(T)

    Returns
    -------
    x_0 : denoised samples (B, 1, 28, 28)
    """
    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(initial_noise.device)
    # sigmas[0] = σ_max, sigmas[1], ..., sigmas[N-1] = σ_min, sigmas[N] = 0

    x = initial_noise  # starts at noise level σ_max

    for i in range(N):
        sig_curr = sigmas[i]       # current noise level (σ_max at start)
        sig_next = sigmas[i + 1]   # target noise level  (one step cleaner)

        # At the very last step sig_next = 0 (appended sentinel).
        # The step is still valid: it denoises from σ_min down to 0.
        # But if sig_curr = 0 somehow, skip (no-op).
        if sig_curr == 0:
            continue

        # Evaluate score at the CURRENT noise level of x
        t_val = torch.log(sig_curr / sigma_min) / math.log(sigma_max / sigma_min)
        t_batch = t_val.expand(x.shape[0]).to(x.device)

        score = model(x, t_batch)

        # Euler step: (sig_next - sig_curr) < 0, so update opposes score → toward data
        x = x + (sig_next - sig_curr) * sig_curr * score

    return x


@torch.no_grad()
def sample_heun(
    model: nn.Module,
    initial_noise: torch.Tensor,
    N: int = 100,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
) -> torch.Tensor:
    """Heun's method (2nd order Runge-Kutta) Probability Flow ODE sampler.

    For each step from i+1 → i:
        x'_i = x_{i+1} + (σ_{i+1} − σ_i) · σ_{i+1} · s_θ(x_{i+1}, t_{i+1})
        x_i  = x_{i+1} + (σ_{i+1} − σ_i) · σ_{i+1} / 2
                         · [ s_θ(x_{i+1}, t_{i+1}) + s_θ(x'_i, t_i) ]

    Parameters
    ----------
    model         : trained score network s_θ
    initial_noise : x_N ~ N(0, σ(T)² I),  shape (B, 1, 28, 28)
    N             : number of discretisation steps
    sigma_min, sigma_max : noise schedule bounds

    Returns
    -------
    x_0 : denoised samples (B, 1, 28, 28)
    """
    sigmas = discrete_sigmas(N, sigma_min, sigma_max).to(initial_noise.device)

    x = initial_noise

    for i in range(N):
        sig_curr = sigmas[i]      # current noise level
        sig_next = sigmas[i + 1]  # target noise level (cleaner)

        if sig_curr == 0:
            continue

        # Time corresponding to current noise level
        t_curr = torch.log(sig_curr / sigma_min) / math.log(sigma_max / sigma_min)
        t_curr_batch = t_curr.expand(x.shape[0]).to(x.device)

        score_curr = model(x, t_curr_batch)

        # Euler predictor (first stage)
        delta = sig_next - sig_curr  # negative (denoising)
        x_prime = x + delta * sig_curr * score_curr

        # For the last step (sig_next = 0), Heun correction is not needed
        # because t_next would be undefined (log(0)).  Fall back to Euler.
        if sig_next == 0.0:
            x = x_prime
        else:
            # Time corresponding to target noise level
            t_next = torch.log(sig_next / sigma_min) / math.log(sigma_max / sigma_min)
            t_next_batch = t_next.expand(x.shape[0]).to(x.device)

            score_next = model(x_prime, t_next_batch)

            # Heun corrector (average of two slopes)
            x = x + delta * sig_curr / 2.0 * (score_curr + score_next)

    return x


# ---------------------------------------------------------------------------
# 6.  TRAINING LOOP
# ---------------------------------------------------------------------------

def train(
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 2e-4,
    ema_decay: float = 0.999,
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_root: str = "./data",
) -> tuple[nn.Module, EMA]:
    """Full training loop for the baseline diffusion model.

    Returns the trained model and its EMA wrapper.
    """
    print(f"Training on device: {device}")

    # ---- Data ----
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1], shape (1, 28, 28)
    ])
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    # ---- Model / Optimiser / Scheduler ----
    model = SmallUNet(in_channels=1, base_channels=32, time_emb_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))
    ema = EMA(model, decay=ema_decay)

    # ---- Training ----
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (x0, _) in enumerate(loader):
            x0 = x0.to(device)

            loss = tweedie_loss(model, x0, sigma_min, sigma_max)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.step(model)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{epochs}  |  loss = {avg_loss:.4f}  |  lr = {current_lr:.2e}")

    return model, ema


# ---------------------------------------------------------------------------
# 7.  EVALUATION SCAFFOLDING  (FID, Precision, Recall)
# ---------------------------------------------------------------------------

def get_inception_features(images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Extract 2048-d features from a pre-trained InceptionV3 for metric computation.

    Parameters
    ----------
    images : (N, 1, 28, 28) in [0, 1]  — MNIST-sized, single channel

    Returns
    -------
    features : (N, 2048)
    """
    from torchvision.models import inception_v3, Inception_V3_Weights

    device = images.device
    # InceptionV3 expects 3-channel 299×299 images
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    inception.eval()
    # Remove the final classification head; use pool5 features
    inception.fc = nn.Identity()

    features_list = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        # Repeat grayscale → 3 channels, resize to 299×299
        batch_rgb = batch.repeat(1, 3, 1, 1)
        batch_rgb = F.interpolate(batch_rgb, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = inception(batch_rgb)
        features_list.append(feats.cpu())

    return torch.cat(features_list, dim=0)


def compute_fid(real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
    """Fréchet Inception Distance between two sets of Inception features.

    FID = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2})
    """
    import numpy as np
    from scipy.linalg import sqrtm

    mu_r = real_features.mean(dim=0).numpy()
    mu_g = gen_features.mean(dim=0).numpy()
    sigma_r = np.cov(real_features.numpy(), rowvar=False)
    sigma_g = np.cov(gen_features.numpy(), rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    # Numerical correction: discard imaginary parts from numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


def compute_precision_recall(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    k: int = 5,
) -> tuple[float, float]:
    """Improved Precision & Recall (Kynkäänniemi et al., 2019).

    Uses k-NN manifold estimation:
    - Precision = fraction of generated samples that fall inside the real manifold
    - Recall    = fraction of real samples that fall inside the generated manifold

    Parameters
    ----------
    real_features, gen_features : (N, D)  Inception features
    k : number of nearest neighbours for manifold estimation

    Returns
    -------
    (precision, recall)
    """
    def manifold_radius(features: torch.Tensor, k: int) -> torch.Tensor:
        """Compute the k-th nearest-neighbour distance for each sample."""
        # Pairwise L2 distances
        dists = torch.cdist(features, features, p=2)
        # Set self-distance to inf
        dists.fill_diagonal_(float("inf"))
        # k-th smallest distance (0-indexed, so index k-1)
        kth_dist, _ = dists.kthvalue(k, dim=1)
        return kth_dist

    real_radii = manifold_radius(real_features, k)   # (N_real,)
    gen_radii = manifold_radius(gen_features, k)     # (N_gen,)

    # Precision: fraction of gen samples within the real manifold
    dist_gen_to_real = torch.cdist(gen_features, real_features, p=2)  # (N_gen, N_real)
    min_dist_to_real, nearest_real_idx = dist_gen_to_real.min(dim=1)
    precision = (min_dist_to_real <= real_radii[nearest_real_idx]).float().mean().item()

    # Recall: fraction of real samples within the generated manifold
    dist_real_to_gen = torch.cdist(real_features, gen_features, p=2)  # (N_real, N_gen)
    min_dist_to_gen, nearest_gen_idx = dist_real_to_gen.min(dim=1)
    recall = (min_dist_to_gen <= gen_radii[nearest_gen_idx]).float().mean().item()

    return precision, recall


def evaluate_model(
    model: nn.Module,
    ema: EMA,
    n_samples: int = 10000,
    N_steps: int = 100,
    sampler: str = "euler",
    sigma_min: float = 0.01,
    sigma_max: float = 80.0,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    data_root: str = "./data",
) -> dict[str, float]:
    """Generate samples and compute FID, Precision, and Recall against MNIST test set.

    Parameters
    ----------
    model   : trained score network
    ema     : EMA wrapper (EMA weights will be used for sampling)
    n_samples : number of images to generate
    N_steps : ODE discretisation steps
    sampler : "euler" or "heun"

    Returns
    -------
    dict with keys "fid", "precision", "recall"
    """
    sample_fn = sample_euler if sampler == "euler" else sample_heun

    # ---- Apply EMA weights for generation ----
    ema.apply(model)
    model.eval()

    # ---- Generate samples ----
    print(f"Generating {n_samples} samples with {sampler} sampler ({N_steps} steps)...")
    generated = []
    for start in range(0, n_samples, batch_size):
        B = min(batch_size, n_samples - start)
        z = torch.randn(B, 1, 28, 28, device=device) * sigma_max
        samples = sample_fn(model, z, N=N_steps, sigma_min=sigma_min, sigma_max=sigma_max)
        generated.append(samples.clamp(0, 1).cpu())
    generated = torch.cat(generated, dim=0)

    # ---- Restore original (non-EMA) weights ----
    ema.restore(model)

    # ---- Load real test images ----
    test_set = datasets.MNIST(data_root, train=False, download=True,
                              transform=transforms.ToTensor())
    real_images = torch.stack([test_set[i][0] for i in range(min(n_samples, len(test_set)))])

    # ---- Extract Inception features ----
    print("Extracting features for real images...")
    real_feats = get_inception_features(real_images.to(device))
    print("Extracting features for generated images...")
    gen_feats = get_inception_features(generated.to(device))

    # ---- Compute metrics ----
    fid = compute_fid(real_feats, gen_feats)
    precision, recall = compute_precision_recall(real_feats, gen_feats)

    results = {"fid": fid, "precision": precision, "recall": recall}
    print(f"FID = {fid:.2f}  |  Precision = {precision:.4f}  |  Recall = {recall:.4f}")
    return results


# ---------------------------------------------------------------------------
# 8.  SANITY CHECK / TESTING BLOCK
# ---------------------------------------------------------------------------

def sanity_check():
    """Executable sanity check using dummy random tensors.

    Tests
    -----
    1. Instantiate the SmallUNet model.
    2. Forward pass — verify output shape matches input.
    3. One training step with the Tweedie loss.
    4. Run sample_euler on random noise — verify output is (1, 1, 28, 28).
    """
    print("=" * 60)
    print("SANITY CHECK — running on CPU with dummy data")
    print("=" * 60)
    device = "cpu"

    # --- 1. Instantiate model ---
    model = SmallUNet(in_channels=1, base_channels=32, time_emb_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model instantiated  ({n_params:,} parameters)")

    # --- 2. Forward pass shape check ---
    dummy_x = torch.randn(4, 1, 28, 28, device=device)
    dummy_t = torch.rand(4, device=device)
    out = model(dummy_x, dummy_t)
    assert out.shape == (4, 1, 28, 28), f"Expected (4,1,28,28), got {out.shape}"
    print(f"[OK] Forward pass shape: {out.shape}")

    # --- 3. One training step with Tweedie loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    loss = tweedie_loss(model, dummy_x, sigma_min=0.01, sigma_max=80.0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[OK] Training step completed  |  loss = {loss.item():.4f}")

    # --- 4. Euler sampling ---
    model.eval()
    z = torch.randn(1, 1, 28, 28, device=device) * 80.0  # x_N ~ N(0, σ_max² I)
    sample = sample_euler(model, z, N=10, sigma_min=0.01, sigma_max=80.0)
    assert sample.shape == (1, 1, 28, 28), f"Expected (1,1,28,28), got {sample.shape}"
    print(f"[OK] Euler sample shape: {sample.shape}")

    print("=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 9.  MAIN ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Score-Based Diffusion on MNIST")
    parser.add_argument("--mode", choices=["sanity", "train", "evaluate"], default="sanity",
                        help="sanity: run shape/loss checks | train: full training | evaluate: generate + FID")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--n_steps", type=int, default=100, help="ODE discretisation steps")
    parser.add_argument("--sampler", choices=["euler", "heun"], default="euler")
    parser.add_argument("--n_samples", type=int, default=10000, help="Samples to generate for eval")
    parser.add_argument("--checkpoint", type=str, default="baseline_diffusion.pt",
                        help="Path to save/load model checkpoint")
    args = parser.parse_args()

    if args.mode == "sanity":
        sanity_check()

    elif args.mode == "train":
        model, ema = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            ema_decay=args.ema_decay,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        # Save checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_shadow": ema.shadow,
        }, args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")

    elif args.mode == "evaluate":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SmallUNet().to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ema = EMA(model, decay=args.ema_decay)
        ema.shadow = ckpt["ema_shadow"]

        evaluate_model(
            model, ema,
            n_samples=args.n_samples,
            N_steps=args.n_steps,
            sampler=args.sampler,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            device=device,
        )
