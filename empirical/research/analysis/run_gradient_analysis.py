#!/usr/bin/env python3
"""
empirical/research/analysis/run_gradient_analysis.py
Refactored gradient distribution analysis with clean property pipeline architecture.

This script computes gradient analysis using a declarative property pipeline approach
and writes artifacts so visualization can be rerun without recomputing SVDs.
The core insight: gradient analysis is just a dependency graph of transformations
applied across model layers. By separating the "what" (property definitions) from
the "how" (execution), we achieve dramatically improved readability and maintainability.

Usage:
    # Compute artifacts (expensive)
    torchrun --standalone --nproc_per_node=8 -m empirical.research.analysis.run_gradient_analysis medium_full_svd_20251103 --mode compute --testing 20 1700

    # Render from artifacts (cheap; no torchrun required)
    python -m empirical.research.analysis.run_gradient_analysis medium_full_svd_20251103 --mode render --testing 20 1700
"""
import logging
import argparse
from datetime import datetime
import math
import os
import sys
import re
from pathlib import Path
import csv
import json
import typing as tp
from typing import Dict, Tuple, Any

# Memory optimization like training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch._inductor.config.coordinate_descent_tuning = False
import torch.distributed as dist
import numpy as np

from empirical.research.analysis.model_utilities import _extract_layer_info as _extract
from empirical.research.training.training_core import (
    setup_distributed_training, Hyperparameters,
)
from empirical.research.analysis.model_utilities import (
    get_weight_matrices, get_accumulated_gradient_matrices,
    combine_layer_properties, GPTLayerProperty
)
from empirical.research.analysis.property_pipeline import PropertySpec, PropertyPipeline
from empirical.research.analysis.core_math import (
    matrix_shape_beta,
    stable_rank_from_tensor,
    estimate_gradient_noise_sigma2,
    compute_echo_fit_diagnostics_from_aligned_svds,
    compute_reverb_fit_relative_diagnostics,
    get_raw_svds,
    get_mean_svd,
    compute_alignment_angles_deg,
    align_svds_greedy_to_mean,
)
from empirical.research.analysis.core_visualization import (
    make_gif_from_layer_property_time_series,
    compute_panel_xs,
    newton_schulz_quintic_function,
    create_reverb_fit_relative_residual_vs_echo_loglog_subplot,
    create_reverb_fit_stratified_relative_residual_by_numerator_subplot,
    create_reverb_fit_stratified_relative_residual_by_denominator_subplot,
)
from empirical.research.analysis.logging_utilities import deserialize_model_checkpoint, log_from_rank
from empirical.research.analysis.constants import (
    FIELD_NAMES,
    NUM_ACCUMULATION_STEPS,
    LOG_EVERY
)
from empirical.research.analysis.artifact_store import (
    save_rank_artifact_shard,
    load_step_artifacts,
    list_artifact_steps,
)

def gradients_stable_rank_from_singulars(rep_singulars: torch.Tensor) -> torch.Tensor:
    """Compute per-replica stable rank from singular values [R, K]."""
    with torch.no_grad():
        s2 = rep_singulars * rep_singulars
        num = torch.sum(s2, dim=1)
        den = torch.max(s2, dim=1).values.clamp_min(1e-20)
        return num / den


def _alignment_z_from_angle_deg(angle_deg: tp.Any, d: int) -> np.ndarray:
    """
    z := sqrt(d) * cos(theta), with theta provided in degrees.
    d is taken to be the number of singular values for the layer (min(m, n)).
    """
    if isinstance(angle_deg, torch.Tensor):
        theta = angle_deg.detach().to(dtype=torch.float32) * (math.pi / 180.0)
        z = math.sqrt(float(d)) * torch.cos(theta)
        return z.detach().cpu().numpy()
    theta = np.asarray(angle_deg, dtype=np.float32) * (math.pi / 180.0)
    return (math.sqrt(float(d)) * np.cos(theta)).astype(np.float32)


ANALYSIS_SPECS = [
    # Stable rank computations (weights)
    PropertySpec("weights_stable_rank", ["checkpoint_weights"], stable_rank_from_tensor),

    # Core gradient analysis
    PropertySpec("mean_gradient", ["per_replicate_gradient"], lambda grads: grads.mean(dim=0)),

    # Mean SVD (for alignment diagnostics only; not serialized)
    PropertySpec("mean_svd", ["mean_gradient"], get_mean_svd),  # (U,S,V)

    # Residuals (only for sigma^2; cheap and useful)
    PropertySpec(
        "gradient_residuals",
        ["per_replicate_gradient", "mean_gradient"],
        lambda grads, mean: grads - mean.unsqueeze(0),
    ),

    # Per-replicate SVDs (raw) + greedy permutation alignment to mean-gradient SVD
    PropertySpec("raw_svds", ["per_replicate_gradient"], get_raw_svds),  # (U,S,V) raw
    PropertySpec(
        "aligned_svds",
        ["raw_svds", "mean_svd"],
        align_svds_greedy_to_mean,  # permute + sign-fix, ignores degeneracy blocks
    ),
    PropertySpec(
        "aligned_replicate_singular_values",
        ["aligned_svds"],
        lambda aligned: aligned[1],  # [R, D]
    ),
    # Also expose as replicate singulars for downstream consumers/CSV
    PropertySpec("replicate_singular_values",
                 ["aligned_replicate_singular_values"],
                 lambda x: x),

    # Per-replicate Frobenius norms ||G_r||_F (needed for sv normalization in render)
    PropertySpec(
        "per_replicate_gradient_fro_norm",
        ["per_replicate_gradient"],
        lambda g: torch.linalg.vector_norm(
            g.reshape(g.shape[0], -1),
            dim=1,
        ).clamp_min(1e-20),
    ),

    # Frobenius-normalized singular values: s_{r,i} / ||G_r||_F  (guaranteed <= 1)
    PropertySpec(
        "replicate_singular_values_fro_normalized",
        ["replicate_singular_values", "per_replicate_gradient_fro_norm"],
        lambda s_rep, fro: s_rep / fro[:, None],
    ),

    # Alignment diagnostics: per-atom angles vs mean-gradient directions
    PropertySpec(
        "left_alignment_angles_deg",
        ["aligned_svds", "mean_svd"],
        lambda rep, mean: compute_alignment_angles_deg(rep, mean)[0],  # [R,D]
    ),
    PropertySpec(
        "right_alignment_angles_deg",
        ["aligned_svds", "mean_svd"],
        lambda rep, mean: compute_alignment_angles_deg(rep, mean)[1],  # [R,D]
    ),

    # Spectral echo + diagnostics (weighted Triple-OLS + residual checks)
    PropertySpec("echo_fit_diagnostics", ["aligned_svds"], compute_echo_fit_diagnostics_from_aligned_svds),
    PropertySpec("spectral_echo_replicates", ["echo_fit_diagnostics"], lambda d: d["spectral_echo_replicates"]),
    PropertySpec(
        "spectral_echo",
        ["spectral_echo_replicates"],
        lambda z: torch.median(z, dim=0).values.clamp(0.0, 1.0),  # [D]
    ),
    PropertySpec("echo_fit_weighted_mse", ["echo_fit_diagnostics"], lambda d: d["echo_fit_weighted_mse"]),
    PropertySpec("triple_mse_bins", ["echo_fit_diagnostics"], lambda d: d["triple_mse_bins"]),
    PropertySpec("triple_mse_by_denom_bin", ["echo_fit_diagnostics"], lambda d: d["triple_mse_by_denom_bin"]),
    PropertySpec("triple_mse_by_numer_bin", ["echo_fit_diagnostics"], lambda d: d["triple_mse_by_numer_bin"]),

    # Reverb-fit diagnostics (relative-error versions)
    PropertySpec(
        "reverb_fit_relative_diagnostics",
        ["aligned_svds", "spectral_echo_replicates"],
        compute_reverb_fit_relative_diagnostics,  # (rel_by_echo, denom_x, denom_y, numer_x, numer_y)
    ),
    PropertySpec("reverb_fit_rel_residual_by_echo", ["reverb_fit_relative_diagnostics"], lambda t: t[0]),
    PropertySpec("reverb_fit_denom_bin_centers", ["reverb_fit_relative_diagnostics"], lambda t: t[1]),
    PropertySpec("reverb_fit_denom_rel_residual", ["reverb_fit_relative_diagnostics"], lambda t: t[2]),
    PropertySpec("reverb_fit_numer_bin_centers", ["reverb_fit_relative_diagnostics"], lambda t: t[3]),
    PropertySpec("reverb_fit_numer_rel_residual", ["reverb_fit_relative_diagnostics"], lambda t: t[4]),

    # Per-replica gradient stable ranks from singulars (original SVDs)
    PropertySpec("gradients_stable_rank",
                 ["aligned_replicate_singular_values"],
                 gradients_stable_rank_from_singulars),

    # Misc annotations and noise/phase relationship
    PropertySpec("aspect_ratio_beta", ["checkpoint_weights"],
                 lambda w: float(matrix_shape_beta(w.shape[-2:] if hasattr(w, 'shape') else w))),
    PropertySpec("worker_count", ["per_replicate_gradient"], lambda g: int(g.shape[0])),
    PropertySpec("m_big", ["checkpoint_weights"], lambda w: int(max(w.shape[-2], w.shape[-1]))),

    # Noise σ² from original residuals
    PropertySpec("gradient_noise_sigma2",
                 ["per_replicate_gradient", "mean_gradient"],
                 estimate_gradient_noise_sigma2),
]



def build_compiled_model(device: torch.device):
    """Build and compile the model once per process."""
    from empirical.research.training.architecture import GPT
    import empirical.research.training.training_core as training_core
    args = Hyperparameters()
    model = GPT(args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    model = training_core.safe_torch_compile(model, dynamic=False)
    return model


def load_weights_into_model(checkpoint_file: str, model: torch.nn.Module, device: torch.device):
    """Load checkpoint weights into an existing compiled model and broadcast from rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    log_from_rank(f"Loading checkpoint {checkpoint_file}", rank)
    # Use the unified deserializer; schema expects 'model'
    checkpoint_data = deserialize_model_checkpoint(Path(checkpoint_file))
    state_dict = checkpoint_data['model']

    # If compiling wrapped the module, load into the original module
    target = model._orig_mod if hasattr(model, "_orig_mod") else model

    # Normalize checkpoints that may contain '_orig_mod.' prefix in keys
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        state_dict = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k: v for k, v in state_dict.items()}

    target.load_state_dict(state_dict)

    # Broadcast parameters so all ranks are in sync with rank 0
    if dist.is_initialized():
        for param in target.parameters():
            dist.broadcast(param.detach(), 0)
    # Build param-name -> (param_type, layer) key map
    name_to_key = {}
    for name, _p in target.named_parameters():
        k = _extract(name)
        if k is not None:
            name_to_key[name] = k

    # Pull hidden optimizer meta (momentum buffers and group momentum)
    if 'hidden_optimizer_meta' in checkpoint_data:
        opt_meta = checkpoint_data['hidden_optimizer_meta']
    else:
        opt_meta = {}
    buf_by_key: Dict[Tuple[str, int], torch.Tensor] = {}
    if 'momentum_buffers' in opt_meta:
        momentum_buffers = opt_meta['momentum_buffers']
    else:
        momentum_buffers = {}
    for pname, buf in momentum_buffers.items():
        if pname in name_to_key:
            k = name_to_key[pname]
        else:
            k = None
        if k is not None and buf.ndim >= 2:     # only matrices (hidden weights)
            # Keep on GPU so analysis compute stays GPU-resident.
            buf_by_key[k] = buf.contiguous().to(device, non_blocking=True)

    # Use a single γ; if multiple groups exist, pick the first (hidden has one group in your setup)
    if 'group_momentum' in opt_meta and opt_meta['group_momentum']:
        gamma = float(opt_meta['group_momentum'][0])
    else:
        gamma = 0.0

    return int(checkpoint_data['step']), {'accum_buffers_by_key': buf_by_key, 'gamma': gamma}



def shard_param_keys(all_keys: list[Tuple[str, int]], rank: int, world_size: int) -> set:
    """Shard parameter keys across ranks uniformly."""
    n = len(all_keys)
    if world_size <= 1:
        return set(all_keys)
    per = n // world_size
    start = rank * per
    end = start + per if rank < world_size - 1 else n
    return set(all_keys[start:end])


def compute_analysis_for_step(
    step: int,
    num_accumulation_steps: int,
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    initial_props: GPTLayerProperty | None = None,
    specs: list[PropertySpec] | None = None,
    run_id: str | None = None,
    opt_meta: Dict[str, Any] | None = None,
) -> None:
    """Core analysis function - clean and focused."""
    
    # 1. Build initial properties (either provided for mock, or computed from model)
    args = Hyperparameters()
    
    if initial_props is None:
        # Get all parameters for sharding
        all_weights = get_weight_matrices(model, only_hidden=True)
        all_param_keys = list(all_weights.keys())
        # Shard parameters across ranks (owners)
        my_param_keys = shard_param_keys(all_param_keys, rank, world_size)
        log_from_rank(f"Processing a shard of {len(my_param_keys)} parameters out of {len(all_param_keys)} total (owners)", rank)

        # Build owner map deterministically across ranks
        owner_map = {}
        for r in range(world_size):
            keys_r = shard_param_keys(all_param_keys, r, world_size)
            for k in keys_r:
                owner_map[k] = r

        # Accumulate and gather microgradients per-key to owners only; owners will have 8xA replicates
        gradients = get_accumulated_gradient_matrices(
            model, args, step,
            num_accumulation_steps=num_accumulation_steps,
            assigned_params=my_param_keys,
            owner_map=owner_map,
            accum_buffer_map=(
                {} if opt_meta is None else opt_meta['accum_buffers_by_key']
            ),
            accum_gamma=(
                0.0 if opt_meta is None else float(opt_meta['gamma'])
            ),
        )
        my_weights = {key: tensor for key, tensor in all_weights.items() if key in my_param_keys}
        initial_props = combine_layer_properties(
            lambda w, g: {"checkpoint_weights": w, "per_replicate_gradient": g},
            my_weights, gradients
        )
        # Removed noise_sigma injection from checkpoint; sigma^2 is computed per layer in-pipeline
    
    # 3. Execute analysis pipeline (5 LOC)
    pipeline = PropertyPipeline(specs or ANALYSIS_SPECS)
    
    def progress_callback(completed: int, total: int):
        if completed % LOG_EVERY == 0:
            log_from_rank(f"Analyzed {completed}/{total} layers", rank)
    
    # Persist shapes for render-only runs (artifacts do NOT store checkpoint weights).
    layer_shapes = {
        layer_key: tuple(int(x) for x in props["checkpoint_weights"].shape[-2:])
        for layer_key, props in initial_props.items()
    }
    local_results = pipeline.execute_for_all_layers(initial_props, progress_callback)
    for layer_key, props in local_results.items():
        props["shape"] = layer_shapes[layer_key]

    # Stream results to per-rank CSV to avoid large in-memory payloads
    stream_write_analysis_results(local_results, step, rank, run_id or "unknown_run")
    save_rank_artifact_shard(run_id or "unknown_run", step, rank, local_results)

    if dist.is_initialized():
        dist.barrier()
    log_from_rank(f"Step {step}: Analysis complete (streamed to CSV)", rank)
    return None


def to_np16(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float16).cpu().numpy()
    return np.asarray(x)

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _to_float_scalar(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().cpu())
    return float(x)


def open_layer_stats_writer(csv_path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    """Ensure CSV exists with matching header; return (file_handle, writer)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = True
    if csv_path.exists():
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
            if existing_header == fieldnames:
                write_header = False
            else:
                csv_path.unlink()
        except Exception:
            csv_path.unlink(missing_ok=True)
    f = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    return f, writer


def stream_write_analysis_results(layer_props: GPTLayerProperty, step: int, rank: int, run_id: str):
    base_dir = Path(f"research_logs/per_layer_statistics/{run_id}")
    if dist.is_initialized():
        (rank == 0) and base_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()
    else:
        base_dir.mkdir(parents=True, exist_ok=True)
    csv_path = base_dir / f"step_{step:06d}_rank{rank}.csv"

    # CSV columns must match the row keys we write below.
    csv_fieldnames = [
        "param_type",
        "layer_num",
        "weight_stable_rank",
        "per_replicate_gradient_singular_values",
        "per_replicate_gradient_stable_rank",
        "spectral_echo",
        "shape",
        "gradient_noise_sigma2",
    ]
    f, writer = open_layer_stats_writer(csv_path, fieldnames=csv_fieldnames)
    try:
        for (param_type, layer_num), props in layer_props.items():
            # Pre-compute scalar extras
            grad_sigma2_val = _to_float_scalar(props['gradient_noise_sigma2'])

            row = {
                'param_type': param_type,
                'layer_num': layer_num,
                'weight_stable_rank': _to_float_scalar(props['weights_stable_rank']),
                'per_replicate_gradient_singular_values': json.dumps(to_np16(props['replicate_singular_values']).tolist()),
                # 'gradient_singular_value_standard_deviations': json.dumps(to_np16(props['singular_value_std']).tolist()),
                'per_replicate_gradient_stable_rank': json.dumps(to_np16(props['gradients_stable_rank']).tolist()),
                'spectral_echo': json.dumps(to_np16(props['spectral_echo']).tolist()),
                'shape': json.dumps(list(props['shape'])),
                'gradient_noise_sigma2': grad_sigma2_val,
            }
            writer.writerow(row)
    finally:
        f.close()


def find_all_checkpoints(run_id: str) -> list[tuple[int, str]]:
    """Find all checkpoint files for the given run."""
    ckpt_dir = Path("research_logs/checkpoints") / run_id
    unique: Dict[int, str] = {}
    for p in ckpt_dir.glob("model_step_*.pt"):
        m = re.search(r'step_(\d+)', p.stem)
        if m:
            unique[int(m.group(1))] = str(p)
    return [(s, unique[s]) for s in sorted(unique)]

## removed legacy main; simplified main is defined below

def create_spectral_echo_vs_sv_semilog_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(param_type)
    ax.set_xlabel('Singular value s (log scale)')
    ax.set_ylabel('Spectral projection coefficient')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    # Actual spectral echo vs s per layer (scatter)
    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = d['sv']
        echo = d['spectral_echo']
        sv = np.asarray(sv, dtype=float)
        echo = np.clip(np.asarray(echo, dtype=float), 0.0, 1.0)
        if sv.size == 0 or echo.size == 0:
            continue
        m = min(sv.size, echo.size)
        sv = sv[:m]; echo = echo[:m]
        color = viridis(layer / denom)
        order = np.argsort(sv)
        ax.scatter(sv[order], echo[order], s=6, alpha=0.25, c=[color])
    # Common x-grid across panel
    xs = compute_panel_xs(panel)
    # Overlay NS quintic in black (normalized xs)
    if xs.size:
        y_ns = np.clip(newton_schulz_quintic_function(xs), 0.0, 1.0)
        ax.plot(xs, y_ns, color='black', lw=1.2, alpha=0.9)
    return []


def create_spectral_echo_vs_sv_semilog_normalized_subplot(ax, panel: GPTLayerProperty, param_type: str, viridis, max_layers: int):
    ax.set_title(f"{param_type} (Frobenius-normalized s)")
    ax.set_xlabel(r"Normalized singular value $s/\|G\|_F$ (log scale)")
    ax.set_ylabel('Spectral echo')
    ax.set_xscale('log'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = np.asarray(d['sv'], dtype=float)
        sv_fro = np.asarray(d['sv_fro'], dtype=float)
        echo = np.clip(np.asarray(d['spectral_echo'], dtype=float), 0.0, 1.0)
        if sv.size == 0 or sv_fro.size == 0 or echo.size == 0:
            continue
        m = min(sv.size, sv_fro.size, echo.size)
        sv = sv[:m]
        sv_fro = sv_fro[:m]
        echo = echo[:m]
        color = viridis(layer / denom)
        mask = np.isfinite(sv_fro) & (sv_fro > 0) & np.isfinite(echo) & np.isfinite(sv)
        if not np.any(mask):
            continue
        x = sv_fro[mask]
        y = echo[mask]
        order = np.argsort(x)
        ax.scatter(x[order], y[order], s=6, alpha=0.25, c=[color])
    xs = compute_panel_xs(panel, key="sv_fro")
    if xs.size:
        y_ns = np.clip(newton_schulz_quintic_function(xs), 0.0, 1.0)
        ax.plot(xs, y_ns, color='black', lw=1.2, alpha=0.9)

    # Frobenius-normalized singular values satisfy 0 < s/||G||_F <= 1 (up to FP noise).
    ax.set_xlim(1e-4, 1.0)
    return []

def create_singular_gap_vs_sv_loglog_subplot(
    ax,
    panel: GPTLayerProperty,
    param_type: str,
    viridis,
    max_layers: int,
):
    """
    Scatter plot: local spectral gap Δs_i vs singular value s_i, on a log-x scale.

    - x-axis: singular value s_i (log scale)
    - y-axis: Δs_i = s_i - s_{i+1}
    - color: layer depth (0..max_layers-1)
    """
    ax.set_title(param_type)
    ax.set_xlabel("Singular value s (log scale)")
    ax.set_ylabel("Local spectral gap Δsᵢ = sᵢ - sᵢ₊₁")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = np.asarray(d["sv"], dtype=float)
        gap = np.asarray(d["gap"], dtype=float)

        if sv.size == 0 or gap.size == 0:
            continue

        m = min(sv.size, gap.size)
        sv = sv[:m]
        gap = gap[:m]

        # Filter out non-positive singulars for log-x safety
        mask = np.isfinite(sv) & (sv > 0) & np.isfinite(gap)
        if not np.any(mask):
            continue

        sv = sv[mask]
        gap = gap[mask]

        color = viridis(layer / denom)
        ax.scatter(sv, gap, s=6, alpha=0.25, c=[color])

    return []

def _create_alignment_angle_vs_sv_semilog_subplot(
    ax,
    panel: GPTLayerProperty,
    param_type: str,
    viridis,
    max_layers: int,
    which: str,  # "left" | "right"
):
    title = f"{param_type} ({which})"
    ax.set_title(title)
    ax.set_xlabel("Singular value s (log scale)")
    ax.set_ylabel(r"$z=\sqrt{d}\cos(\theta)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 3e3)
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = d["sv"]
        ang = d["angles"]
        sv = np.asarray(sv, dtype=float)              # [D]
        ang = np.asarray(ang, dtype=float)            # [R,D]
        if sv.ndim != 1 or ang.ndim != 2:
            continue
        R = ang.shape[0]
        D = min(sv.size, ang.shape[1])
        if D <= 0:
            continue
        sv = sv[:D]
        ang = ang[:, :D]
        z = _alignment_z_from_angle_deg(ang, D)

        # flatten distribution: (R*D,)
        xs = np.repeat(sv, R)
        ys = np.asarray(z, dtype=float).reshape(-1)
        mask = np.isfinite(xs) & (xs > 0) & np.isfinite(ys) & (ys > 0)
        if not np.any(mask):
            continue
        xs = xs[mask]
        ys = ys[mask]

        color = viridis(layer / denom)
        ax.scatter(xs, ys, s=6, alpha=1.0 / 510, c=[color])

    return []


def create_left_alignment_angle_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers: int):
    return _create_alignment_angle_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers, which="left")


def create_right_alignment_angle_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers: int):
    return _create_alignment_angle_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers, which="right")


def _create_alignment_angle_deg_vs_sv_semilog_subplot(
    ax,
    panel: GPTLayerProperty,
    param_type: str,
    viridis,
    max_layers: int,
    which: str,  # "left" | "right"
):
    title = f"{param_type} ({which})"
    ax.set_title(title)
    ax.set_xlabel("Singular value s (log scale)")
    ax.set_ylabel(r"Alignment angle $\theta$ (deg)")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 180.0)
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        sv = d["sv"]
        ang = d["angles"]
        sv = np.asarray(sv, dtype=float)              # [D]
        ang = np.asarray(ang, dtype=float)            # [R,D] degrees
        if sv.ndim != 1 or ang.ndim != 2:
            continue
        R = ang.shape[0]
        D = min(sv.size, ang.shape[1])
        if D <= 0:
            continue
        sv = sv[:D]
        ang = ang[:, :D]

        xs = np.repeat(sv, R)
        ys = ang.reshape(-1)
        ys = np.clip(ys, 0.0, 180.0)
        mask = np.isfinite(xs) & (xs > 0) & np.isfinite(ys)
        if not np.any(mask):
            continue
        xs = xs[mask]
        ys = ys[mask]

        color = viridis(layer / denom)
        ax.scatter(xs, ys, s=6, alpha=1.0 / 510, c=[color])

    return []


def create_left_alignment_angle_deg_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers: int):
    return _create_alignment_angle_deg_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers, which="left")


def create_right_alignment_angle_deg_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers: int):
    return _create_alignment_angle_deg_vs_sv_semilog_subplot(ax, panel, param_type, viridis, max_layers, which="right")



def main():
    parser = argparse.ArgumentParser(description="Gradient analysis runner")
    parser.add_argument("run_id", type=str, help="Run identifier (subdir under research_logs/checkpoints)")
    parser.add_argument("--mode", choices=["compute", "render"], default="compute",
                        help="compute: run expensive pipeline and write artifacts; render: load artifacts and plot")
    parser.add_argument("--testing", nargs="+", type=int, help="Only process the specified checkpoint steps (e.g., --testing 10 20)")
    args_ns = parser.parse_args()

    # Lightweight logging setup
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    run_id = args_ns.run_id
    requested_steps = set(args_ns.testing or [])

    if args_ns.mode == "render":
        # Cheap path: no distributed, no model, no checkpoints.
        steps = list_artifact_steps(run_id)
        if requested_steps:
            steps = [s for s in steps if s in requested_steps]
        if not steps:
            raise SystemExit(f"No artifacts found for run_id={run_id}. Did you run --mode compute?")

        echo_singular_direct_ts: Dict[int, GPTLayerProperty] = {}
        sv_gap_ts: Dict[int, GPTLayerProperty] = {}
        left_align_panel_ts: Dict[int, GPTLayerProperty] = {}
        right_align_panel_ts: Dict[int, GPTLayerProperty] = {}
        reverb_relres_ts: Dict[int, GPTLayerProperty] = {}
        reverb_numer_ts: Dict[int, GPTLayerProperty] = {}
        reverb_denom_ts: Dict[int, GPTLayerProperty] = {}

        for step in steps:
            aggregated_payload = load_step_artifacts(run_id, step)
            echo_singular_direct_ts[step] = build_spectral_echo_vs_sv_panel(aggregated_payload)
            sv_gap_ts[step] = build_singular_gap_panel(aggregated_payload)
            left_align_panel_ts[step] = build_alignment_angle_vs_sv_panel(aggregated_payload, which="left")
            right_align_panel_ts[step] = build_alignment_angle_vs_sv_panel(aggregated_payload, which="right")
            reverb_relres_ts[step] = build_reverb_fit_relative_residual_panel(aggregated_payload)
            reverb_numer_ts[step] = build_reverb_fit_numerator_stratified_panel(aggregated_payload)
            reverb_denom_ts[step] = build_reverb_fit_denominator_stratified_panel(aggregated_payload)

        ts_run = datetime.now().strftime("%Y%m%d%H%M%S")
        out_dir = Path(f"research_logs/visualizations/{run_id}_generated_at_{ts_run}")
        out_dir.mkdir(parents=True, exist_ok=True)
        generate_gifs_for_run(
            out_dir,
            echo_singular_direct_ts,
            sv_gap_ts,
            left_align_panel_ts,
            right_align_panel_ts,
            reverb_relres_ts,
            reverb_numer_ts,
            reverb_denom_ts,
        )
        return 0

    # --- Expensive compute path (distributed) ---
    _, rank, world_size, device, _ = setup_distributed_training()
    model = build_compiled_model(device)
    checkpoints = find_all_checkpoints(run_id)
    checkpoints = [(s, p) for (s, p) in checkpoints if s > 0]
    log_from_rank(f"Filtered checkpoints (skip step 0): {len(checkpoints)} found", rank)

    # Testing mode: restrict to requested specific steps if provided
    if requested_steps:
        ckpt_map = {s: p for s, p in checkpoints}
        missing = sorted(int(s) for s in requested_steps if s not in ckpt_map)
        checkpoints = [(s, ckpt_map[s]) for s in sorted(requested_steps) if s in ckpt_map]
        log_from_rank(f"Testing mode: requested steps {sorted(requested_steps)}; processing {len(checkpoints)} present steps", rank)
        if missing and rank == 0:
            logging.warning(f"Missing requested checkpoint steps: {missing}")

    for step, ckpt in checkpoints:
        step_loaded, opt_meta = load_weights_into_model(ckpt, model, device)
        compute_analysis_for_step(
            step_loaded,
            num_accumulation_steps=NUM_ACCUMULATION_STEPS,
            rank=rank,
            world_size=world_size,
            model=model,
            run_id=run_id,
            opt_meta=opt_meta,
        )
        if dist.is_initialized():
            dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


def build_spectral_echo_vs_sv_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Store per-direction singulars (median across replicas) and matched per-direction echoes.
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        s_rep = props['replicate_singular_values']  # [R,D]
        s_dir = torch.median(s_rep, dim=0).values.detach().cpu().numpy()  # [D]

        s_rep_fro = props['replicate_singular_values_fro_normalized']  # [R,D]
        s_dir_fro = torch.median(s_rep_fro, dim=0).values.detach().cpu().numpy()  # [D]

        echo = props['spectral_echo'].detach().cpu().numpy()  # [D]
        n = min(s_dir.size, s_dir_fro.size, echo.size)
        if n:
            out[key] = {
                'sv': s_dir[:n],
                'sv_fro': s_dir_fro[:n],
                'spectral_echo': echo[:n],
                'shape': tuple(int(x) for x in props['shape']),
            }
            # optional shape guards
            assert out[key]['sv'].ndim == 1 and out[key]['spectral_echo'].ndim == 1
            assert out[key]['sv'].shape == out[key]['spectral_echo'].shape
    return out

def build_singular_gap_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Build per-layer panel for singular value *gaps* vs singular values.

    For each layer:
      - take median singular values across replicas (aligned_replicate_singular_values)
      - assume they are sorted in descending order (SVD convention)
      - define local gaps Δs_i = s_i - s_{i+1}
      - drop the last singular value so sv and gap have the same length
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        s_rep = props["replicate_singular_values"]          # [B, Kc]
        # Median over replicas, preserve ordering along the singular dimension
        s_dir = torch.median(s_rep, dim=0).values.detach().cpu().numpy()  # [Kc]

        if s_dir.size < 2:
            # Not enough singulars to define a gap
            continue

        # SVD should give them in descending order; gaps are local differences
        sv = s_dir.astype(float)
        gap = sv[:-1] - sv[1:]          # Δs_i = s_i - s_{i+1}
        n = gap.size
        if n <= 0:
            continue

        shape = tuple(int(x) for x in props["shape"])
        out[key] = {
            "sv": sv[:n],               # match gap length
            "gap": gap,
            "shape": shape,
        }

        # Shape sanity
        assert out[key]["sv"].ndim == 1 and out[key]["gap"].ndim == 1
        assert out[key]["sv"].shape == out[key]["gap"].shape

    return out

def build_alignment_angle_vs_sv_panel(
    aggregated_payload: GPTLayerProperty,
    which: str,  # "left" | "right"
) -> GPTLayerProperty:
    """
    Per-layer panel:
      - sv: median singular value per direction across replicas, shape [D]
      - angles: per-replica alignment angles (degrees), shape [R,D]
    """
    out: GPTLayerProperty = {}
    key_ang = "left_alignment_angles_deg" if which == "left" else "right_alignment_angles_deg"
    for key, props in aggregated_payload.items():
        s_rep = props["replicate_singular_values"]
        ang = props[key_ang]
        # s_rep: [R,D], ang: [R,D]
        if isinstance(s_rep, torch.Tensor):
            s_rep_t = s_rep
        else:
            s_rep_t = torch.as_tensor(s_rep)
        s_dir = torch.median(s_rep_t, dim=0).values.detach().cpu().numpy()
        ang_np = _to_numpy(ang)

        n = min(s_dir.size, ang_np.shape[1] if ang_np.ndim == 2 else 0)
        if n <= 0:
            continue
        shape = tuple(int(x) for x in props["shape"])
        out[key] = {
            "sv": s_dir[:n],
            "angles": ang_np[:, :n],
            "shape": shape,
        }
    return out


def build_reverb_fit_relative_residual_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Panel payload:
      - echo: per-direction fitted echo (median across replicas) [D]
      - rel_resid: per-direction relative residual (median across pivots) [D]
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        echo = props.get("spectral_echo", None)
        rr = props.get("reverb_fit_rel_residual_by_echo", None)
        if echo is None or rr is None:
            continue
        echo_np = _to_numpy(echo)
        rr_np = _to_numpy(rr)
        n = min(int(echo_np.size), int(rr_np.size))
        if n <= 0:
            continue
        out[key] = {
            "echo": np.clip(echo_np[:n].astype(float), 1e-12, 1.0),
            "rel_resid": np.clip(rr_np[:n].astype(float), 1e-30, np.inf),
            "shape": tuple(int(x) for x in props["shape"]),
        }
    return out


def build_reverb_fit_denominator_stratified_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Panel payload:
      - bin_x: denominator magnitude bin centers |Z_ab| [B]
      - bin_y: relative residual aggregated in-bin [B]
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        x = props.get("reverb_fit_denom_bin_centers", None)
        y = props.get("reverb_fit_denom_rel_residual", None)
        if x is None or y is None:
            continue
        x_np = _to_numpy(x)
        y_np = _to_numpy(y)
        n = min(int(x_np.size), int(y_np.size))
        if n <= 0:
            continue
        out[key] = {
            "bin_x": np.clip(x_np[:n].astype(float), 1e-12, 1.0),
            "bin_y": np.clip(y_np[:n].astype(float), 1e-30, np.inf),
            "shape": tuple(int(x) for x in props["shape"]),
        }
    return out


def build_reverb_fit_numerator_stratified_panel(aggregated_payload: GPTLayerProperty) -> GPTLayerProperty:
    """
    Panel payload:
      - bin_x: numerator magnitude bin centers |Z_ap Z_bp| [B]
      - bin_y: relative residual aggregated in-bin [B]
    """
    out: GPTLayerProperty = {}
    for key, props in aggregated_payload.items():
        x = props.get("reverb_fit_numer_bin_centers", None)
        y = props.get("reverb_fit_numer_rel_residual", None)
        if x is None or y is None:
            continue
        x_np = _to_numpy(x)
        y_np = _to_numpy(y)
        n = min(int(x_np.size), int(y_np.size))
        if n <= 0:
            continue
        out[key] = {
            "bin_x": np.clip(x_np[:n].astype(float), 1e-12, 1.0),
            "bin_y": np.clip(y_np[:n].astype(float), 1e-30, np.inf),
            "shape": tuple(int(x) for x in props["shape"]),
        }
    return out


def generate_gifs_for_run(
    out_dir: Path,
    echo_ts_direct: Dict[int, GPTLayerProperty],
    sv_gap_ts: Dict[int, GPTLayerProperty],
    left_align_ts: Dict[int, GPTLayerProperty],
    right_align_ts: Dict[int, GPTLayerProperty],
    reverb_relres_ts: Dict[int, GPTLayerProperty],
    reverb_numer_ts: Dict[int, GPTLayerProperty],
    reverb_denom_ts: Dict[int, GPTLayerProperty],
):
    make_gif_from_layer_property_time_series(echo_ts_direct, create_spectral_echo_vs_sv_semilog_subplot, title="spectral_echo_vs_singular_values_direct", output_dir=out_dir)
    # Normalized-x spectral-echo plot (using direct tau2 overlays)
    make_gif_from_layer_property_time_series(echo_ts_direct, create_spectral_echo_vs_sv_semilog_normalized_subplot, title="spectral_echo_vs_singular_values_normalized_direct", output_dir=out_dir)
    make_gif_from_layer_property_time_series(
        sv_gap_ts,
        create_singular_gap_vs_sv_loglog_subplot,
        title="singular_value_gap_vs_singular_value",
        output_dir=out_dir,
    )
    make_gif_from_layer_property_time_series(left_align_ts, create_left_alignment_angle_vs_sv_semilog_subplot, title="left_alignment_z_vs_singular_value", output_dir=out_dir)
    make_gif_from_layer_property_time_series(right_align_ts, create_right_alignment_angle_vs_sv_semilog_subplot, title="right_alignment_z_vs_singular_value", output_dir=out_dir)
    make_gif_from_layer_property_time_series(left_align_ts, create_left_alignment_angle_deg_vs_sv_semilog_subplot, title="left_alignment_angle_deg_vs_singular_value", output_dir=out_dir)
    make_gif_from_layer_property_time_series(right_align_ts, create_right_alignment_angle_deg_vs_sv_semilog_subplot, title="right_alignment_angle_deg_vs_singular_value", output_dir=out_dir)

    # Reverb-fit diagnostics (relative-error y-axes):
    make_gif_from_layer_property_time_series(
        reverb_relres_ts,
        create_reverb_fit_relative_residual_vs_echo_loglog_subplot,
        title="reverb_fit_relative_residual_vs_echo",
        output_dir=out_dir,
    )
    make_gif_from_layer_property_time_series(
        reverb_numer_ts,
        create_reverb_fit_stratified_relative_residual_by_numerator_subplot,
        title="reverb_fit_stratified_relative_residual_by_numerator",
        output_dir=out_dir,
    )
    make_gif_from_layer_property_time_series(
        reverb_denom_ts,
        create_reverb_fit_stratified_relative_residual_by_denominator_subplot,
        title="reverb_fit_stratified_relative_residual_by_denominator",
        output_dir=out_dir,
    )

if __name__ == "__main__":
    sys.exit(main())
