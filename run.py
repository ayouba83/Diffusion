#!/usr/bin/env python3
"""
Unified CLI entry point for the MCL Diffusion project.
=======================================================

Usage
-----
    python3 run.py <phase> <mode> [options]

Phases
------
    baseline   Phase 1 — single score network
    mcl        Phase 2 — MCL ensemble (sMCL training)
    routing    Phase 3 — routing strategies & gating network
    eval       Phase 4 — evaluation, metrics & figures

Examples
--------
    python3 run.py baseline sanity
    python3 run.py baseline train --epochs 20
    python3 run.py mcl train --K 5 --epochs 20
    python3 run.py routing train_gating --ensemble_checkpoint mcl_diffusion.pt
    python3 run.py eval full --output_dir figures
"""

import argparse
import os
import sys

import torch


# ===================================================================
# Shared argument helpers
# ===================================================================

def _add_common_args(parser: argparse.ArgumentParser):
    """Arguments shared across multiple phases."""
    parser.add_argument("--sigma_min", type=float, default=0.01)
    parser.add_argument("--sigma_max", type=float, default=80.0)


def _add_training_args(parser: argparse.ArgumentParser):
    """Arguments for training phases."""
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)


# ===================================================================
# Phase dispatchers
# ===================================================================

def run_baseline(args):
    """Phase 1 — Baseline diffusion model."""
    from src.baseline import (
        SmallUNet, EMA, sanity_check, train, evaluate_model,
    )

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


def run_mcl(args):
    """Phase 2 — MCL ensemble training."""
    from src.mcl import (
        MCLDiffusion, EMA, sanity_check, train_mcl, evaluate_mcl,
    )

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


def run_routing(args):
    """Phase 3 — Routing strategies & gating network."""
    from src.routing import sanity_check, train_gating_network, GatingNetwork
    from src.mcl import MCLDiffusion, EMA

    if args.mode == "sanity":
        sanity_check()

    elif args.mode == "train_gating":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.ensemble_checkpoint, map_location=device)
        K = ckpt["K"]
        ensemble = MCLDiffusion(K=K).to(device)
        ensemble.load_state_dict(ckpt["ensemble_state_dict"])

        # Apply EMA weights for label generation
        ema_list = [EMA(expert, decay=0.999) for expert in ensemble.experts]
        for k in range(K):
            ema_list[k].shadow = ckpt["ema_shadows"][k]
            ema_list[k].apply(ensemble.experts[k])

        gating = train_gating_network(
            ensemble, K=K,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            device=device,
        )
        torch.save({
            "K": K,
            "gating_state_dict": gating.state_dict(),
        }, args.gating_checkpoint)
        print(f"Gating network saved to {args.gating_checkpoint}")


def run_eval(args):
    """Phase 4 — Evaluation & analysis."""
    from src.evaluation import (
        sanity_check,
        compute_all_metrics,
        plot_trajectories,
        record_gating_decisions,
        plot_temporal_specialisation,
        LeNetClassifier,
        train_lenet_classifier,
        plot_interclass_diversity,
    )
    from src.mcl import MCLDiffusion, EMA
    from src.routing import GatingNetwork

    if args.mode == "sanity":
        sanity_check()

    elif args.mode == "full":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(args.output_dir, exist_ok=True)

        # Load ensemble
        ckpt = torch.load(args.ensemble_checkpoint, map_location=device)
        K = ckpt["K"]
        ensemble = MCLDiffusion(K=K).to(device)
        ensemble.load_state_dict(ckpt["ensemble_state_dict"])
        ema_list = [EMA(expert, decay=0.999) for expert in ensemble.experts]
        for k in range(K):
            ema_list[k].shadow = ckpt["ema_shadows"][k]
            ema_list[k].apply(ensemble.experts[k])
        ensemble.eval()
        print(f"Loaded ensemble (K={K}) from {args.ensemble_checkpoint}")

        # Load gating network
        gating = None
        if os.path.isfile(args.gating_checkpoint):
            gating_ckpt = torch.load(args.gating_checkpoint, map_location=device)
            gating = GatingNetwork(K=K).to(device)
            gating.load_state_dict(gating_ckpt["gating_state_dict"])
            gating.eval()
            print(f"Loaded gating network from {args.gating_checkpoint}")
        else:
            print(f"[WARN] Gating checkpoint not found; skipping gated routing")

        # Train or load LeNet
        if os.path.isfile(args.lenet_checkpoint):
            classifier = LeNetClassifier().to(device)
            classifier.load_state_dict(
                torch.load(args.lenet_checkpoint, map_location=device)
            )
            classifier.eval()
        else:
            print("Training LeNet classifier on MNIST...")
            classifier = train_lenet_classifier(epochs=5, device=device)
            torch.save(classifier.state_dict(), args.lenet_checkpoint)

        # 1. Metrics
        print("\n" + "=" * 60)
        print("1. QUANTITATIVE METRICS")
        print("=" * 60)
        compute_all_metrics(
            ensemble, gating,
            N_steps=args.n_steps, n_samples=args.n_samples,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            device=device,
        )

        # 2. Trajectories
        print("\n" + "=" * 60)
        print("2. INTRA-CLASS DIVERSITY")
        print("=" * 60)
        plot_trajectories(
            ensemble, gating=gating, N=args.n_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            device=device,
            save_path=os.path.join(args.output_dir, "fig_trajectories.png"),
        )

        # 3. Temporal specialisation
        if gating is not None:
            print("\n" + "=" * 60)
            print("3. TEMPORAL SPECIALISATION")
            print("=" * 60)
            routing_log = record_gating_decisions(
                ensemble, gating, n_samples=256, N=args.n_steps,
                sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                device=device,
            )
            plot_temporal_specialisation(
                routing_log, K=K,
                save_path=os.path.join(args.output_dir, "fig_temporal_specialisation.png"),
            )

        # 4. Inter-class diversity
        print("\n" + "=" * 60)
        print("4. INTER-CLASS DIVERSITY")
        print("=" * 60)
        plot_interclass_diversity(
            ensemble, classifier,
            n_samples_per_expert=args.n_samples_per_expert,
            N_steps=args.n_steps,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            device=device,
            save_path=os.path.join(args.output_dir, "fig_interclass_diversity.png"),
        )

        print(f"\nAll figures saved to {args.output_dir}/")


# ===================================================================
# CLI parser
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MCL Diffusion — Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="phase", required=True,
                                        help="Project phase to run")

    # --- baseline ---
    p_base = subparsers.add_parser("baseline", help="Phase 1: single score network")
    p_base.add_argument("mode", choices=["sanity", "train", "evaluate"])
    _add_common_args(p_base)
    _add_training_args(p_base)
    p_base.add_argument("--checkpoint", default="baseline_diffusion.pt")
    p_base.add_argument("--sampler", choices=["euler", "heun"], default="euler")
    p_base.add_argument("--n_steps", type=int, default=100)
    p_base.add_argument("--n_samples", type=int, default=10000)

    # --- mcl ---
    p_mcl = subparsers.add_parser("mcl", help="Phase 2: MCL ensemble")
    p_mcl.add_argument("mode", choices=["sanity", "train", "evaluate"])
    _add_common_args(p_mcl)
    _add_training_args(p_mcl)
    p_mcl.add_argument("--K", type=int, default=5)
    p_mcl.add_argument("--checkpoint", default="mcl_diffusion.pt")
    p_mcl.add_argument("--sampler", choices=["euler", "heun"], default="euler")
    p_mcl.add_argument("--n_steps", type=int, default=100)
    p_mcl.add_argument("--n_samples_per_expert", type=int, default=2000)

    # --- routing ---
    p_rout = subparsers.add_parser("routing", help="Phase 3: routing & gating")
    p_rout.add_argument("mode", choices=["sanity", "train_gating"])
    _add_common_args(p_rout)
    p_rout.add_argument("--epochs", type=int, default=10)
    p_rout.add_argument("--batch_size", type=int, default=256)
    p_rout.add_argument("--lr", type=float, default=1e-3)
    p_rout.add_argument("--ensemble_checkpoint", default="mcl_diffusion.pt")
    p_rout.add_argument("--gating_checkpoint", default="gating_network.pt")

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="Phase 4: evaluation & figures")
    p_eval.add_argument("mode", choices=["sanity", "full"])
    _add_common_args(p_eval)
    p_eval.add_argument("--n_steps", type=int, default=100)
    p_eval.add_argument("--n_samples", type=int, default=10000)
    p_eval.add_argument("--n_samples_per_expert", type=int, default=1000)
    p_eval.add_argument("--ensemble_checkpoint", default="mcl_diffusion.pt")
    p_eval.add_argument("--gating_checkpoint", default="gating_network.pt")
    p_eval.add_argument("--lenet_checkpoint", default="lenet_classifier.pt")
    p_eval.add_argument("--output_dir", default="figures")

    args = parser.parse_args()

    dispatch = {
        "baseline": run_baseline,
        "mcl": run_mcl,
        "routing": run_routing,
        "eval": run_eval,
    }
    dispatch[args.phase](args)


if __name__ == "__main__":
    main()
