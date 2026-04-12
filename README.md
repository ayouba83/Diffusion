# Mixture Score-Based Diffusion Models with Multiple Choice Learning

Implementation of score-based generative models (diffusion models) with **Multiple Choice Learning (MCL)** for the IASD DataLab project. The project consists of two phases:

- **Phase 1 — Baseline**: A single score network trained with the Tweedie denoising objective and deterministic Probability Flow ODE sampling.
- **Phase 2 — MCL Ensemble**: K expert score networks trained competitively using Stochastic Winner-Takes-All (sMCL), where each training example is routed to the best-performing expert.
- **Phase 3 — Routing Strategies**: Three inference-time policies (single expert, heuristic, learned gating) to dynamically select which expert provides the score during ODE sampling.
- **Phase 4 — Evaluation & Analysis**: Quantitative metrics (FID, Precision, Recall) and three visual analyses (intra-class diversity, temporal specialisation, inter-class diversity) for the final report.

---

## Mathematical Framework

### Noising Process

Data is continuously noised via Gaussian convolution:

$$X_t = X_0 + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\, \sigma(t)^2 I_d)$$

The noise schedule is geometric (log-linear):

$$\sigma(t) = \sigma_{\min}^{1-t} \cdot \sigma_{\max}^{t}, \quad t \in [0, 1]$$

with default values $\sigma_{\min} = 0.01$ and $\sigma_{\max} = 80.0$.

### Tweedie's Formula and Training Objective

From Tweedie's formula, the posterior mean satisfies:

$$\mathbb{E}[X_0 \mid X_t = x_t] = x_t + \sigma(t)^2 \nabla_x \log p_t(x_t)$$

This shows that $\sigma(t)^2 s_\theta(x_t, t)$ estimates the noise component. The network is trained to minimise the denoising objective (Eq. 4):

$$\mathcal{L} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\left\|\sigma(t)^2\, s_\theta(x_0 + \epsilon,\, t) - \epsilon\right\|^2\right]$$

### Deterministic ODE Sampling

Sampling uses the **Probability Flow ODE** (not SDE) — trajectories are fully deterministic given a fixed initial noise $x_N \sim \mathcal{N}(0, \sigma_{\max}^2 I_d)$. Two solvers are implemented:

**Euler (1st order):**

$$x_i = x_{i+1} + (\sigma_{i+1} - \sigma_i)\,\sigma_{i+1}\, s_\theta(x_{i+1},\, t_{i+1})$$

**Heun's method (2nd order Runge-Kutta):**

$$x'_i = x_{i+1} + (\sigma_{i+1} - \sigma_i)\,\sigma_{i+1}\, s_\theta(x_{i+1},\, t_{i+1})$$

$$x_i = x_{i+1} + \frac{(\sigma_{i+1} - \sigma_i)\,\sigma_{i+1}}{2}\left[s_\theta(x_{i+1},\, t_{i+1}) + s_\theta(x'_i,\, t_i)\right]$$

The discrete noise schedule runs from $\sigma_0 = \sigma_{\max}$ down to $\sigma_N = 0$.

### Multiple Choice Learning (Phase 2)

Instead of training a single network, we train an ensemble of $K$ experts using a competitive **Winner-Takes-All (WTA)** strategy.

For a training triple $(x_0, t, \epsilon)$ with $x_t = x_0 + \sigma(t)\epsilon$:

**1. Each expert predicts the noise (Eq. 8):**

$$x^k = \sigma(t)^2\, s_{\theta_k}(x_t,\, t), \qquad \ell_k = \|x^k - \epsilon\|^2$$

**2. Winner selection — element-wise per example (Eq. 9):**

$$k^\star(i) = \arg\min_{1 \le k \le K} \ell_k(i)$$

**3. Only the winner is updated:**

$$\mathcal{L}_{\text{sMCL}} = \frac{1}{B}\sum_{i=1}^{B} \ell_{k^\star(i)}(i)$$

**Gradient routing mechanism:** The `(K, B)` loss tensor is passed through `torch.gather` using the winner indices. Only the gathered (winning) entries remain in the computation graph. When `.backward()` is called, PyTorch autograd propagates gradients exclusively through the winning experts' paths — losing experts receive exactly zero gradients.

---

## Project Structure

```
DataLab/
├── run.py                  # Unified CLI entry point (all phases & modes)
├── requirements.txt        # pip dependencies
├── src/
│   ├── __init__.py         # Re-exports key classes/functions
│   ├── baseline.py         # Phase 1: single model, training, sampling, evaluation
│   ├── mcl.py              # Phase 2: MCL ensemble, sMCL training, per-expert eval
│   ├── routing.py          # Phase 3: 3 routing strategies + GatingNetwork
│   └── evaluation.py       # Phase 4: metrics, figures, and analysis scripts
├── figures/                # Output .png figures (created by evaluation)
│   └── .gitkeep
├── data/                   # MNIST data (auto-downloaded on first run)
│   └── .gitkeep
└── README.md
```

---

## Phase 1 — Baseline (`src/baseline.py`)

### What is inside `src/baseline.py`

The file is organised into 9 clearly separated sections:

| Section | Contents |
|---------|----------|
| **1. Noise Schedule** | `sigma_schedule(t)` — continuous geometric schedule. `discrete_sigmas(N)` — N+1 levels from $\sigma_{\max}$ to 0. |
| **2. Model Architecture** | `SmallUNet` — lightweight time-conditioned U-Net (~1.5M params) for 1x28x28 images. Uses sinusoidal time embeddings, GroupNorm, residual blocks with affine time modulation, and skip connections. Channel widths 32 → 64 → 128 kept small for future scaling to K parallel networks. |
| **3. EMA** | `EMA` class — exponential moving average of model parameters with `step()`, `apply()`, and `restore()` methods. |
| **4. Tweedie Loss** | `tweedie_loss(model, x0)` — implements the exact denoising objective $\|\sigma(t)^2 s_\theta(x_0+\epsilon, t) - \epsilon\|^2$. |
| **5. ODE Samplers** | `sample_euler(model, initial_noise, N)` — 1st order Euler. `sample_heun(model, initial_noise, N)` — 2nd order Heun. Both are deterministic and decorated with `@torch.no_grad()`. |
| **6. Training Loop** | `train(epochs, ...)` — full training with AdamW optimiser, cosine annealing LR scheduler, gradient clipping (max norm 1.0), and EMA updates after every step. |
| **7. Evaluation Scaffolding** | `get_inception_features()` — extracts 2048-d InceptionV3 pool5 features. `compute_fid()` — Frechet Inception Distance. `compute_precision_recall()` — Improved Precision & Recall (Kynkaanniemi et al., 2019) via k-NN manifold estimation. `evaluate_model()` — end-to-end: generate samples, extract features, compute all 3 metrics against MNIST test set. |
| **8. Sanity Check** | `sanity_check()` — instantiates the model, checks forward pass shapes, runs one training step, and runs Euler sampling on dummy data. |
| **9. CLI Entry Point** | `argparse`-based CLI with 3 modes: `sanity`, `train`, `evaluate`. |

---

## Phase 2 — MCL Ensemble (`src/mcl.py`)

### What is inside `src/mcl.py`

The file imports shared components from `src/baseline.py` and is organised into 7 sections:

| Section | Contents |
|---------|----------|
| **1. MCL Ensemble Module** | `MCLDiffusion(nn.Module)` — wraps `nn.ModuleList` of K `SmallUNet` experts. Forward pass returns stacked scores of shape `(K, B, 1, 28, 28)`. |
| **2. sMCL Loss** | `smcl_loss(ensemble, x0)` — full Stochastic MCL forward pass: noises data, runs all K experts, computes per-example per-expert errors `(K, B)`, selects winner via `argmin` per example, gathers winning losses via `torch.gather`, returns scalar loss + winner indices. |
| **3. MCL Training Loop** | `train_mcl(K, epochs, ...)` — single AdamW optimiser over all K experts, per-expert EMA, cosine annealing LR, gradient clipping. Logs per-expert win rates each epoch. |
| **4. MCL Sampling** | `sample_euler_mcl(ensemble, noise, expert_idx)` and `sample_heun_mcl(...)` — delegate to baseline ODE samplers for a specific expert. |
| **5. MCL Evaluation** | `evaluate_mcl(ensemble, ema_list, ...)` — generates samples from each expert separately, computes per-expert FID/Precision/Recall, then pools all samples for aggregate metrics. |
| **6. Sanity Check** | `sanity_check()` — gradient routing verification: runs one sMCL step and programmatically asserts that losing experts have zero gradients and winning experts have non-zero gradients. |
| **7. CLI Entry Point** | `argparse`-based CLI with 3 modes: `sanity`, `train`, `evaluate`. |

### Gradient Routing — How It Works

The gradient routing is the most critical part of the sMCL implementation. Here is exactly what happens:

```
1.  Forward all K experts on x_t           → scores:  (K, B, 1, 28, 28)
2.  Compute noise predictions x^k          → preds:   (K, B, 1, 28, 28)
3.  Compute per-example squared errors     → ell_k:   (K, B)
4.  argmin over K dimension (detached)     → k_star:  (B,)
5.  torch.gather(ell_k, dim=0, k_star)    → winners: (B,)
6.  winners.mean()                         → loss:    scalar
7.  loss.backward()                        → gradients flow ONLY through gathered entries
```

**Why `torch.gather` is sufficient:** The `(K, B)` tensor `ell_k` contains B*K entries, each connected to a single expert's computation graph. `torch.gather` selects exactly B entries (one per example). The resulting scalar loss only touches the winning experts' subgraphs. Autograd assigns zero gradients to all parameters not in those subgraphs.

**Verified empirically** (sanity check output):

```
Expert 0: won 0 examples → gradient norm = 0.000000e+00 (correctly zero)
Expert 1: won 7 examples → gradient norm = 6.078086e+08 (correctly non-zero)
Expert 2: won 1 examples → gradient norm = 7.637630e+02 (correctly non-zero)
```

### MCL Training Details

- **Single optimiser**: One `AdamW` covers all K experts' parameters. The WTA gradient routing naturally ensures only winners are updated.
- **Per-expert EMA**: Each expert has its own `EMA` wrapper tracking its shadow parameters independently.
- **Win rate logging**: Each epoch logs the fraction of training examples won by each expert (e.g., `E0:23.1%  E1:18.5%  E2:20.0%  E3:19.2%  E4:19.2%`). This helps detect expert collapse (one expert winning everything).

---

## Phase 3 — Routing Strategies (`src/routing.py`)

### What is inside `src/routing.py`

The file imports from `src/baseline.py` and `src/mcl.py` and is organised into 6 sections:

| Section | Contents |
|---------|----------|
| **1. Single Expert** | `sample_single_expert(ensemble, noise, expert_idx)` — picks one expert at the start, uses it for every ODE step. |
| **2. Heuristic Routing** | `sample_heuristic_routing(ensemble, noise, routing_rule)` — selects the expert at each step via a prescribed time-based rule. Default: partition $[0,1]$ into K equal bands (Expert 0 for high noise, Expert K-1 for fine detail). The rule is a swappable callable `(t, K) → k`. |
| **3a. GatingNetwork** | `GatingNetwork(nn.Module)` — small time-conditioned CNN (~44K params) that outputs K logits from $(x_t, t)$. Architecture: 3 conv layers (1→16→32→64) + adaptive pool + sinusoidal time embed + 2-layer MLP head. |
| **3b. Gating Training** | `train_gating_network(ensemble, K)` — freezes all expert weights, generates WTA labels $k^\star = \arg\min_k \ell_k$ on the fly, trains the gating network with cross-entropy loss. Logs accuracy each epoch. |
| **3c. Gated Sampling** | `sample_gated_routing(ensemble, gating, noise)` — at each Euler step, the gating network predicts which expert to use per example (argmax). Examples are grouped by expert for memory-efficient batched scoring. |
| **4. Sanity Check** | Runs the same initial noise through all 3 strategies, verifies output shapes, and confirms determinism (same noise → same output). Also tests one gating training step. |

### The Three Routing Strategies

All three share the same deterministic Euler ODE — they differ only in how $s_\theta$ is selected at each step:

| Strategy | Expert Selection | Learned? | Per-example? |
|----------|-----------------|----------|--------------|
| **Single Expert** | Fixed $k$ for all steps | No | No |
| **Heuristic** | $k = f(t, K)$ prescribed rule | No | No (same for batch) |
| **Gated** | $k = \arg\max$ GatingNetwork$(x_t, t)$ | Yes | Yes (per example) |

### GatingNetwork Architecture

```
GatingNetwork  (~44,355 parameters)
│
├── Spatial Feature Extractor
│   ├── Conv2d 1→16  stride=2        28→14
│   ├── Conv2d 16→32 stride=2        14→7
│   ├── Conv2d 32→64 stride=2         7→4
│   └── AdaptiveAvgPool2d              4→1×1 → (B, 64)
│
├── Time Embedding
│   └── Sinusoidal(64) → Linear → SiLU → (B, 64)
│
└── Classification Head
    └── Cat(spatial, time) → Linear(128→128) → SiLU → Linear(128→K)
```

---

## Phase 4 — Evaluation & Analysis (`src/evaluation.py`)

### What is inside `src/evaluation.py`

| Section | Contents |
|---------|----------|
| **1. Quantitative Metrics** | `compute_all_metrics(ensemble, gating)` — generates images for every routing strategy (each single expert, heuristic, gated, pooled ensemble), extracts InceptionV3 features, computes FID / Precision / Recall against MNIST test set, and prints a formatted comparison table. |
| **2. Intra-class Diversity** | `plot_trajectories(ensemble, gating)` — fixes one initial noise $x_N$, denoises it through Expert 0, Expert 1, and gated routing. Captures intermediate $x_t$ states at evenly-spaced ODE steps. Outputs a grid figure (rows = strategies, columns = time steps). Saves `fig_trajectories.png`. |
| **3. Temporal Specialisation** | `record_gating_decisions()` records which expert the gating network selects at each ODE step. `plot_temporal_specialisation()` renders a stacked area chart of expert usage frequency across the time axis. Saves `fig_temporal_specialisation.png`. |
| **4. Inter-class Diversity** | `plot_interclass_diversity(ensemble, classifier)` — generates 1000 images per expert, classifies them with a pre-trained `LeNetClassifier`, and plots per-expert digit distribution histograms showing mode specialisation. Saves `fig_interclass_diversity.png`. |
| **5. Sanity Check** | Verifies all metrics and all 3 plot functions on dummy data without requiring trained models or MNIST. |
| **6. CLI Entry Point** | `argparse`-based CLI with 2 modes: `sanity` and `full`. |

### Output Figures

| Figure | File | Description |
|--------|------|-------------|
| Intra-class diversity | `fig_trajectories.png` | Same noise, different routing → different outputs. Grid of intermediate denoising states. |
| Temporal specialisation | `fig_temporal_specialisation.png` | Stacked area chart showing which expert the gating network prefers at each noise level (high $t$ → low $t$). |
| Inter-class diversity | `fig_interclass_diversity.png` | Per-expert histograms of generated digit classes (0-9), revealing mode specialisation. |

### LeNet Classifier

A simple LeNet-5 classifier is provided for the inter-class diversity analysis:

```
LeNetClassifier
├── Conv2d 1→6 (5×5) → ReLU → MaxPool2d     28→14
├── Conv2d 6→16 (5×5) → ReLU → MaxPool2d     14→5
└── Flatten → Linear(400→120) → Linear(120→84) → Linear(84→10)
```

It is trained on real MNIST (5 epochs, ~99% accuracy) and used to classify generated images into digit classes 0-9. The trained weights are cached to `lenet_classifier.pt`.

---

## Architecture Details

```
SmallUNet  (~1,494,561 parameters)
│
├── Time Embedding
│   └── Sinusoidal positional encoding → Linear → SiLU → Linear  (dim=128)
│
├── Encoder
│   ├── Conv2d 1→32                          28×28
│   ├── ResBlock(32)  + time conditioning    28×28
│   ├── Conv2d 32→64  stride=2               14×14
│   ├── ResBlock(64)  + time conditioning    14×14
│   └── Conv2d 64→128 stride=2                7×7
│
├── Bottleneck
│   ├── ResBlock(128) + time conditioning     7×7
│   └── ResBlock(128) + time conditioning     7×7
│
├── Decoder (with skip connections)
│   ├── ConvTranspose2d 128→64               14×14
│   ├── Cat(skip) → ResBlock(128) → 1×1 Conv → 64   14×14
│   ├── ConvTranspose2d 64→32                28×28
│   └── Cat(skip) → ResBlock(64)  → 1×1 Conv → 32   28×28
│
└── Output
    └── GroupNorm → SiLU → Conv2d 32→1       28×28
```

Each `ResBlock` uses:
- GroupNorm (8 groups) + SiLU activation
- 3x3 convolutions (no change in spatial size)
- Time-conditioned affine modulation: `scale` and `shift` projected from the time embedding

---

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- torchvision
- numpy
- scipy (for FID computation)
- matplotlib (for figure generation)
- pillow

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

All commands are run from the project root via `run.py`:

```
python3 run.py <phase> <mode> [options]
```

### Sanity Checks (run all, no data needed)

```bash
python3 run.py baseline sanity    # model shapes, forward pass, Euler sampling
python3 run.py mcl sanity         # sMCL gradient routing verification
python3 run.py routing sanity     # all 3 routing strategies + determinism check
python3 run.py eval sanity        # metrics + all 3 plot functions on dummy data
```

### Phase 1 — Baseline

```bash
python3 run.py baseline train --epochs 20 --batch_size 128 --lr 2e-4
python3 run.py baseline evaluate --checkpoint baseline_diffusion.pt --sampler euler
```

### Phase 2 — MCL Ensemble

```bash
python3 run.py mcl train --K 5 --epochs 20 --batch_size 128 --lr 2e-4
python3 run.py mcl evaluate --checkpoint mcl_diffusion.pt --sampler euler
```

### Phase 3 — Routing & Gating Network

```bash
python3 run.py routing train_gating --ensemble_checkpoint mcl_diffusion.pt --epochs 10
```

### Phase 4 — Evaluation & Figures

```bash
python3 run.py eval full \
    --ensemble_checkpoint mcl_diffusion.pt \
    --gating_checkpoint gating_network.pt \
    --output_dir figures
```

This produces:
1. A printed comparison table of FID / Precision / Recall for every routing strategy
2. `figures/fig_trajectories.png` — intra-class diversity visualisation
3. `figures/fig_temporal_specialisation.png` — expert usage across time
4. `figures/fig_interclass_diversity.png` — per-expert digit distributions

### CLI Reference

**Shared flags** (available across phases):

| Flag | Default | Description |
|------|---------|-------------|
| `--sigma_min` | 0.01 | Minimum noise level |
| `--sigma_max` | 80.0 | Maximum noise level |

**Training flags** (baseline, mcl):

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 128 | Training batch size |
| `--lr` | 2e-4 | Initial learning rate (AdamW) |
| `--ema_decay` | 0.999 | EMA decay rate |

**Sampling/evaluation flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--sampler` | `euler` | ODE solver: `euler` or `heun` |
| `--n_steps` | 100 | ODE discretisation steps |
| `--n_samples` | 10000 | Images to generate for metrics |
| `--K` | 5 | Number of expert networks |
| `--checkpoint` | varies | Model checkpoint path |

**Phase-specific flags:**

| Flag | Phase | Default | Description |
|------|-------|---------|-------------|
| `--K` | mcl | 5 | Number of experts |
| `--n_samples_per_expert` | mcl, eval | 2000 / 1000 | Samples per expert |
| `--ensemble_checkpoint` | routing, eval | `mcl_diffusion.pt` | Trained ensemble |
| `--gating_checkpoint` | routing, eval | `gating_network.pt` | Trained gating network |
| `--lenet_checkpoint` | eval | `lenet_classifier.pt` | LeNet (auto-trained if missing) |
| `--output_dir` | eval | `figures` | Output directory for `.png` plots |

---

## Complete Pipeline (End-to-End)

```bash
# Phase 1: Train baseline
python3 run.py baseline train --epochs 20

# Phase 2: Train MCL ensemble
python3 run.py mcl train --K 5 --epochs 20

# Phase 3: Train gating network
python3 run.py routing train_gating --ensemble_checkpoint mcl_diffusion.pt

# Phase 4: Generate all metrics and figures
python3 run.py eval full \
    --ensemble_checkpoint mcl_diffusion.pt \
    --gating_checkpoint gating_network.pt \
    --output_dir figures
```

---

## Design Decisions

- **Lightweight architecture**: The U-Net uses only 32/64/128 channels (~1.5M params) so that K parallel copies can fit in memory for the future mixture extension.
- **Deterministic ODE only**: No stochastic SDE sampling is implemented. This is intentional — deterministic trajectories are required for the "fix initial noise, vary routing" ablation analysis.
- **Heun fallback at sigma=0**: The last Heun step falls back to Euler because $t_i$ is undefined when $\sigma_i = 0$ (log(0) is undefined in the sigma-to-time inversion).
- **EMA for sampling**: The `evaluate_model` function automatically applies EMA weights before generating and restores original weights after, so training can resume if needed.
- **Evaluation uses InceptionV3**: MNIST images (1x28x28) are converted to 3-channel 299x299 before feature extraction, following standard FID protocol.
- **`torch.gather` for gradient routing**: No manual gradient masking or hooks needed. By selecting only winning losses before computing the scalar, autograd naturally zeros out losing experts. This is both simpler and more memory-efficient than alternatives.
- **Per-expert EMA**: Each of the K experts maintains its own EMA shadow, updated independently after every optimiser step, regardless of whether that expert won any examples in the current batch.
- **Win rate monitoring**: Per-expert win rates are logged every epoch to detect mode collapse (one expert dominating) or expert death (one expert never winning).
- **Sigma=0 sentinel skip**: All Euler/Heun loops skip the step where $\sigma_{i+1} = 0$ (the appended sentinel in `discrete_sigmas`). This avoids `log(0) = -inf` in the sigma-to-time inversion and the resulting NaN propagation through the model.
- **Gating network is tiny**: At ~44K parameters (vs ~1.5M per expert), the gating network adds negligible overhead. It only classifies — it does not denoise.
- **Per-example gated routing**: During gated sampling, different images in the same batch can be routed to different experts at the same ODE step. Examples are grouped by expert for efficient batched inference.
- **Modular heuristic rule**: The heuristic routing function accepts any callable `(t, K) → k`, making it easy to experiment with different prescribed policies without modifying the ODE loop.
- **Matplotlib Agg backend**: The evaluation script uses the non-interactive `Agg` backend so figures are saved directly to `.png` without requiring a display server.
