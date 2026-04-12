# Mixture Score-Based Diffusion Models with Multiple Choice Learning

**IASD DataLab Project — Retake Exam**

A PyTorch implementation of score-based generative models (diffusion models) on MNIST, extended with a **Multiple Choice Learning (MCL)** strategy that trains $K$ expert score networks using a competitive Winner-Takes-All rule.  The project investigates expert specialisation through three inference-time routing policies and provides a full quantitative and qualitative evaluation pipeline.

## Abstract

Score-based diffusion models learn to reverse a continuous noising process via a neural score network $s_\theta(x_t, t)$.  We first train a **baseline** single-network model using the Tweedie denoising objective and deterministic Probability Flow ODE sampling.  We then replace the single network with an **ensemble of $K$ experts** trained under Stochastic Multiple Choice Learning (sMCL): for each training example, only the best-performing expert receives gradient updates.  At inference time, we compare three **routing strategies** — fixed single expert, time-based heuristic, and a learned gating network — and evaluate the results using FID, Precision, Recall, and three additional qualitative analyses (intra-class diversity, temporal specialisation, and inter-class mode coverage).

---

## 1. Environment Setup

**Requirements:** Python 3.9+, PyTorch >= 2.0, CUDA optional.

```bash
# Clone the repository
git clone https://github.com/ayouba83/Diffusion.git
cd Diffusion

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` contains:

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Core framework |
| `torchvision` | MNIST dataset, InceptionV3 for FID |
| `numpy` | Array operations |
| `scipy` | Matrix square root for FID |
| `matplotlib` | Figure generation |
| `pillow` | Image I/O |

MNIST is downloaded automatically on the first training run.

---

## 2. Repository Structure

```
Diffusion/
│
├── run.py                      # Unified CLI — single entry point for all phases
├── requirements.txt            # pip dependencies
├── README.md                   # This file
│
├── src/                        # Source package
│   ├── __init__.py             #   Re-exports key classes and functions
│   ├── baseline.py             #   Phase 1: SmallUNet, Tweedie loss, Euler/Heun ODE
│   ├── mcl.py                  #   Phase 2: MCLDiffusion ensemble, sMCL WTA training
│   ├── routing.py              #   Phase 3: 3 routing strategies + GatingNetwork
│   └── evaluation.py           #   Phase 4: FID/Precision/Recall + 3 analysis figures
│
├── figures/                    # Output directory for .png plots
│   └── .gitkeep
│
└── data/                       # MNIST data (auto-downloaded)
    └── .gitkeep
```

### Module Dependency Graph

```
src/baseline.py          (no local imports — foundation module)
      │
      ▼
src/mcl.py               (imports SmallUNet, EMA, sigma_schedule, samplers)
      │
      ▼
src/routing.py            (imports from baseline + mcl)
      │
      ▼
src/evaluation.py         (imports from baseline + mcl + routing)
```

---

## 3. Reproducing the Training

All commands use the unified `run.py` entry point.  Checkpoints are saved to the project root.

### 3.1 Verify the Installation (Sanity Checks)

These run on CPU with dummy tensors and require no data:

```bash
python3 run.py baseline sanity     # Forward pass, loss, Euler sampling
python3 run.py mcl sanity          # sMCL gradient routing correctness
python3 run.py routing sanity      # All 3 routing strategies + determinism
python3 run.py eval sanity         # Metrics + all plot functions
```

### 3.2 Train the Baseline Model (Phase 1)

Single score network trained with the Tweedie denoising loss:

```bash
python3 run.py baseline train \
    --epochs 20 \
    --batch_size 128 \
    --lr 2e-4 \
    --ema_decay 0.999
```

Saves `baseline_diffusion.pt`.

### 3.3 Train the MCL Ensemble (Phase 2)

$K = 5$ expert networks trained with Winner-Takes-All:

```bash
python3 run.py mcl train \
    --K 5 \
    --epochs 20 \
    --batch_size 128 \
    --lr 2e-4 \
    --ema_decay 0.999
```

Saves `mcl_diffusion.pt`.  Per-expert win rates are logged every epoch to monitor specialisation.

### 3.4 Train the Gating Network (Phase 3)

Auxiliary classifier that learns to predict the WTA winner from $(x_t, t)$:

```bash
python3 run.py routing train_gating \
    --ensemble_checkpoint mcl_diffusion.pt \
    --epochs 10 \
    --batch_size 256 \
    --lr 1e-3
```

Saves `gating_network.pt`.  Expert weights are frozen; only the gating network is trained with cross-entropy loss.

---

## 4. Reproducing the Inference

Image generation uses deterministic Probability Flow ODE sampling (Euler discretisation).  The only source of randomness is the initial noise $x_N \sim \mathcal{N}(0, \sigma_{\max}^2 I_d)$.

### 4.1 Strategy 1 — Single Expert

One fixed expert $k$ is used for every time step of the trajectory:

```python
from src import MCLDiffusion, sample_single_expert

ensemble = ...  # load trained MCLDiffusion
z = torch.randn(64, 1, 28, 28) * 80.0       # initial noise
images = sample_single_expert(ensemble, z, expert_idx=0, N=100)
```

### 4.2 Strategy 2 — Heuristic Routing

A prescribed time-based rule selects the expert at each ODE step (no learned parameters).  The default rule partitions $[0, 1]$ into $K$ equal bands:

```python
from src import sample_heuristic_routing

images = sample_heuristic_routing(ensemble, z, N=100)
```

The rule is a swappable callable `(t: float, K: int) -> int`.

### 4.3 Strategy 3 — Learned Gating

A trained `GatingNetwork` predicts the best expert per example at each step:

```python
from src import GatingNetwork, sample_gated_routing

gating = ...  # load trained GatingNetwork
images = sample_gated_routing(ensemble, gating, z, N=100)
```

Different examples in the same batch can be routed to different experts at the same step.

---

## 5. Reproducing the Evaluation

### 5.1 Quantitative Metrics (FID, Precision, Recall)

The full evaluation generates samples for every routing strategy, extracts InceptionV3 features, and computes all three metrics against the MNIST test set:

```bash
python3 run.py eval full \
    --ensemble_checkpoint mcl_diffusion.pt \
    --gating_checkpoint gating_network.pt \
    --n_samples 10000 \
    --output_dir figures
```

This prints a comparison table:

```
Strategy             FID   Precision    Recall
------------------------------------------------------------
expert_0            XX.XX     0.XXXX    0.XXXX
expert_1            XX.XX     0.XXXX    0.XXXX
...
heuristic           XX.XX     0.XXXX    0.XXXX
gated               XX.XX     0.XXXX    0.XXXX
pooled_ensemble     XX.XX     0.XXXX    0.XXXX
```

### 5.2 Qualitative Analysis 1 — Intra-class Diversity (Trajectory Visualisation)

Fixes a single initial noise tensor and denoises it through Expert 0, Expert 1, and the gated router.  Intermediate states are captured at evenly-spaced ODE steps.

**Output:** `figures/fig_trajectories.png` — grid where each row is a routing strategy and each column is a denoising step from $t = T$ (noise) to $t = 0$ (clean).

This visually proves that different routing policies produce different images from the same initial noise.

### 5.3 Qualitative Analysis 2 — Temporal Specialisation

Records which expert the gating network selects at each ODE step across a batch of 256 samples.

**Output:** `figures/fig_temporal_specialisation.png` — stacked area chart of expert usage frequency across the time axis (high noise on the left, clean on the right).

This reveals whether experts specialise in different noise regimes (e.g., one expert for coarse structure, another for fine details).

### 5.4 Qualitative Analysis 3 — Inter-class Diversity (Mode Coverage)

Generates 1000 images per expert using single-expert routing, classifies them with a pre-trained LeNet-5 classifier (~99% accuracy on MNIST), and plots the digit distribution.

**Output:** `figures/fig_interclass_diversity.png` — $K$ histograms showing the fraction of each digit class (0-9) generated by each expert.

This reveals whether experts have specialised in specific data modes (e.g., Expert 1 generates mostly 0s and 8s).

---

## 6. Complete Pipeline — One-Command Reproduction

```bash
# Step 1 — Train baseline (single network)
python3 run.py baseline train --epochs 20

# Step 2 — Train MCL ensemble (K=5 experts, WTA)
python3 run.py mcl train --K 5 --epochs 20

# Step 3 — Train gating network (learned router)
python3 run.py routing train_gating --ensemble_checkpoint mcl_diffusion.pt

# Step 4 — Generate all metrics and figures
python3 run.py eval full \
    --ensemble_checkpoint mcl_diffusion.pt \
    --gating_checkpoint gating_network.pt \
    --output_dir figures
```

After Step 4, the `figures/` directory contains all three analysis plots ready for the report.

---

## 7. Method Summary

### 7.1 Score Network Architecture

A lightweight time-conditioned U-Net (~1.5M parameters) designed to fit $K$ copies in memory:

```
SmallUNet
├── Encoder:    28×28 → 14×14 → 7×7    (channels: 32 → 64 → 128)
├── Bottleneck: 2 × ResBlock(128) with time conditioning
├── Decoder:    7×7 → 14×14 → 28×28    (skip connections at each level)
└── Output:     GroupNorm → SiLU → Conv2d → 1×28×28
```

Each `ResBlock` uses GroupNorm + SiLU activation and receives time conditioning via affine modulation (scale & shift) projected from a sinusoidal time embedding.

### 7.2 Training Objective (Tweedie Loss)

$$\mathcal{L} = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\left\|\sigma(t)^2\, s_\theta(x_0 + \epsilon,\, t) - \epsilon\right\|^2\right]$$

where $\epsilon \sim \mathcal{N}(0, \sigma(t)^2 I_d)$ and $\sigma(t) = \sigma_{\min}^{1-t} \cdot \sigma_{\max}^{t}$.

### 7.3 Stochastic MCL (Winner-Takes-All)

For each example $i$ in a batch, all $K$ experts predict the noise.  Only the winner receives gradients:

$$k^\star(i) = \arg\min_{k} \|\sigma(t)^2 s_{\theta_k}(x_t^{(i)}, t) - \epsilon^{(i)}\|^2$$

Gradient routing is implemented via `torch.gather` on the $(K, B)$ loss tensor — losing experts are automatically disconnected from the computation graph.

### 7.4 Deterministic ODE Sampling

**Euler (1st order):**

$$x_i = x_{i+1} + (\sigma_{i+1} - \sigma_i)\,\sigma_{i+1}\, s_\theta(x_{i+1},\, t_{i+1})$$

**Heun (2nd order):**

$$x'_i = x_{i+1} + \Delta\,\sigma_{i+1}\, s_\theta(x_{i+1},\, t_{i+1})$$

$$x_i = x_{i+1} + \frac{\Delta\,\sigma_{i+1}}{2}\left[s_\theta(x_{i+1},\, t_{i+1}) + s_\theta(x'_i,\, t_i)\right]$$

where $\Delta = \sigma_{i+1} - \sigma_i$.

### 7.5 Gating Network

A small time-conditioned CNN (~44K parameters) trained to predict $k^\star$ from $(x_t, t)$ using cross-entropy loss with frozen expert weights.

---

## 8. CLI Reference

```
python3 run.py <phase> <mode> [options]
```

| Phase | Modes | Description |
|-------|-------|-------------|
| `baseline` | `sanity`, `train`, `evaluate` | Single score network |
| `mcl` | `sanity`, `train`, `evaluate` | K-expert ensemble |
| `routing` | `sanity`, `train_gating` | Routing strategies |
| `eval` | `sanity`, `full` | Metrics & figures |

<details>
<summary><strong>Full flag reference</strong></summary>

| Flag | Phases | Default | Description |
|------|--------|---------|-------------|
| `--epochs` | baseline, mcl, routing | 20 / 20 / 10 | Training epochs |
| `--batch_size` | baseline, mcl, routing | 128 / 128 / 256 | Batch size |
| `--lr` | baseline, mcl, routing | 2e-4 / 2e-4 / 1e-3 | Learning rate |
| `--ema_decay` | baseline, mcl | 0.999 | EMA decay rate |
| `--K` | mcl | 5 | Number of experts |
| `--sigma_min` | all | 0.01 | Minimum noise level |
| `--sigma_max` | all | 80.0 | Maximum noise level |
| `--sampler` | baseline, mcl | `euler` | ODE solver (`euler` / `heun`) |
| `--n_steps` | baseline, mcl, eval | 100 | ODE discretisation steps |
| `--n_samples` | baseline, eval | 10000 | Images for FID computation |
| `--n_samples_per_expert` | mcl, eval | 2000 / 1000 | Images per expert |
| `--checkpoint` | baseline, mcl | `*.pt` | Model checkpoint path |
| `--ensemble_checkpoint` | routing, eval | `mcl_diffusion.pt` | Trained MCL ensemble |
| `--gating_checkpoint` | routing, eval | `gating_network.pt` | Trained gating network |
| `--lenet_checkpoint` | eval | `lenet_classifier.pt` | Digit classifier |
| `--output_dir` | eval | `figures` | Directory for `.png` figures |

</details>
