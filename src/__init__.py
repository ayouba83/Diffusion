"""
src — Score-Based Diffusion Models with Multiple Choice Learning
================================================================
Re-exports the key classes and functions from all four modules so that
users can write ``from src import SmallUNet, MCLDiffusion, ...``.
"""

# Phase 1 — Baseline
from src.baseline import (
    SmallUNet,
    SinusoidalTimeEmbedding,
    ResBlock,
    EMA,
    sigma_schedule,
    discrete_sigmas,
    tweedie_loss,
    sample_euler,
    sample_heun,
    train,
    get_inception_features,
    compute_fid,
    compute_precision_recall,
    evaluate_model,
)

# Phase 2 — MCL Ensemble
from src.mcl import (
    MCLDiffusion,
    smcl_loss,
    train_mcl,
    sample_euler_mcl,
    sample_heun_mcl,
    evaluate_mcl,
)

# Phase 3 — Routing Strategies
from src.routing import (
    GatingNetwork,
    sample_single_expert,
    sample_heuristic_routing,
    sample_gated_routing,
    default_heuristic_rule,
    train_gating_network,
)

# Phase 4 — Evaluation & Analysis
from src.evaluation import (
    compute_all_metrics,
    plot_trajectories,
    record_gating_decisions,
    plot_temporal_specialisation,
    LeNetClassifier,
    train_lenet_classifier,
    plot_interclass_diversity,
)
