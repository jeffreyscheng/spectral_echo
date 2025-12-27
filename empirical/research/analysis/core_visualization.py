"""
empirical/research/analysis/core_visualization.py
Core visualization utilities for gradient analysis.

This module provides all the plotting and gif generation functionality needed
across different analysis scripts. It eliminates duplication by providing
a single, consistent interface for all visualization needs.
"""

import math
from pathlib import Path
from typing import Dict, Tuple, Any, List, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio.v2 as imageio
import json
from empirical.research.training.zeropower import NEWTON_SCHULZ_QUINTIC_COEFFICIENTS
from empirical.research.analysis.model_utilities import GPTLayerProperty


# Standard parameter types for consistent visualization
PARAM_TYPES = [
    'Attention Q', 'Attention K', 'Attention V', 
    'Attention O', 'MLP Input', 'MLP Output'
]


def create_subplot_grid(
    layer_property: GPTLayerProperty,
    plot_fn: Callable[[plt.Axes, GPTLayerProperty, str, mcolors.Colormap, int], List[Any]],
    title: str,
    figsize: Tuple[int, int] = (20, 10),
    layout: str = "constrained",
    num_layers: int = 16,
) -> plt.Figure:
    """
    Generic subplot grid creator for 6-panel visualizations.
    
    Args:
        TODO: better descriptions
        layer_property: GPTLayerProperty
        plot_fn: Callable,
        title: str,
        ...
    """
    if layout == "constrained":
        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    fig.suptitle(title, fontsize=14)
    axes = axes.flatten()
    viridis = plt.cm.viridis

    # Precompute global max layer index for consistent colorbar mapping
    props_by_type: Dict[str, Dict[Tuple[str, int], Any]] = {pt: {} for pt in PARAM_TYPES}
    for (p_type, layer_num), arr in layer_property.items():
        if p_type in props_by_type:
            props_by_type[p_type][(p_type, layer_num)] = arr

    # Fixed layer count for coloring (GPT‑medium has 16 blocks: 0..15)
    global_max_layers = max(1, int(num_layers))

    for i, param_type in enumerate(PARAM_TYPES):
        ax = axes[i]
        prop = props_by_type[param_type]
        plot_fn(ax, prop, param_type, viridis, global_max_layers)
    
    if layout == "tight":
        plt.tight_layout()

    sm = cm.ScalarMappable(cmap=viridis, norm=mcolors.Normalize(vmin=0, vmax=global_max_layers - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location='right', fraction=0.02, pad=0.02)
    cbar.set_label('Layer', rotation=270, labelpad=12)
    try:
        cbar.set_ticks(list(range(0, global_max_layers)))
        cbar.set_ticklabels([str(i) for i in range(global_max_layers)])
    except Exception:
        pass
    
    return fig

def newton_schulz_quintic_function(x):
    """Newton-Schulz quintic function for overlay."""
    out = x
    for a, b, c in NEWTON_SCHULZ_QUINTIC_COEFFICIENTS:
        out = a * out + b * (out **3) + c * (out ** 5)
    return out


def predict_spectral_echo_curve_np(s: np.ndarray, tau2: float) -> np.ndarray:
    """Predict E[spectral_echo(s)] = 1 / (1 + tau^2 / s^2).

    Args:
        s: singular values (>=0)
        tau2: fitted phase constant
    """
    s = np.asarray(s, dtype=float)
    eps = 1e-12
    s2 = np.maximum(s * s, eps)
    return 1.0 / (1.0 + (tau2 / s2))


def create_reverb_fit_relative_residual_vs_echo_loglog_subplot(
    ax,
    panel: GPTLayerProperty,
    param_type: str,
    viridis,
    max_layers: int,
):
    """
    Scatter: x = fitted echo (median across replicas), y = relative residual (dimensionless).
    Both axes are log-scaled.
    """
    ax.set_title(param_type)
    ax.set_xlabel(r"Fitted echo $\hat{\zeta}$ (log scale)")
    ax.set_ylabel(r"Relative squared residual (dimensionless)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        x = np.asarray(d.get("echo", []), dtype=float)
        y = np.asarray(d.get("rel_resid", []), dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        m = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        if not np.any(m):
            continue
        color = viridis(layer / denom)
        ax.scatter(x[m], y[m], s=6, alpha=0.25, c=[color])

    return []


def create_reverb_fit_stratified_relative_residual_by_denominator_subplot(
    ax,
    panel: GPTLayerProperty,
    param_type: str,
    viridis,
    max_layers: int,
):
    """
    Line: x = |Z_ab| bin center, y = in-bin mean relative residual.
    """
    ax.set_title(param_type)
    ax.set_xlabel(r"Denominator magnitude $|Z_{ab}|$ (log scale)")
    ax.set_ylabel(r"Mean relative residual (dimensionless)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        x = np.asarray(d.get("bin_x", []), dtype=float)
        y = np.asarray(d.get("bin_y", []), dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        m = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        if not np.any(m):
            continue
        color = viridis(layer / denom)
        ax.plot(x[m], y[m], lw=1.2, alpha=0.9, color=color)

    return []


def create_reverb_fit_stratified_relative_residual_by_numerator_subplot(
    ax,
    panel: GPTLayerProperty,
    param_type: str,
    viridis,
    max_layers: int,
):
    """
    Line: x = |Z_ap Z_bp| bin center, y = in-bin mean relative residual.
    """
    ax.set_title(param_type)
    ax.set_xlabel(r"Numerator magnitude $|Z_{ap} Z_{bp}|$ (log scale)")
    ax.set_ylabel(r"Mean relative residual (dimensionless)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    denom = max(1, max_layers - 1)
    for (pt, layer), d in sorted(panel.items(), key=lambda x: x[0][1]):
        if pt != param_type or not isinstance(d, dict):
            continue
        x = np.asarray(d.get("bin_x", []), dtype=float)
        y = np.asarray(d.get("bin_y", []), dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        m = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        if not np.any(m):
            continue
        color = viridis(layer / denom)
        ax.plot(x[m], y[m], lw=1.2, alpha=0.9, color=color)

    return []


def compute_panel_xs(panel: GPTLayerProperty, key: str = "sv", eps: float = 1e-8) -> np.ndarray:
    """Build a log-spaced x-grid from positive values across panel layers."""
    vals = []
    for (_pt, _layer), d in panel.items():
        x = d.get(key) if isinstance(d, dict) else None
        if x is None:
            continue
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size:
            vals.append(x)
    if not vals:
        return np.geomspace(1e-6, 1.0, 256)
    all_x = np.concatenate(vals)
    lo = max(float(all_x.min()), eps)
    hi = float(all_x.max())
    if hi <= lo:
        hi = lo * 10.0
    return np.geomspace(lo, hi, 256)

def make_gif_from_layer_property_time_series(
    layer_property_time_series: dict[int, GPTLayerProperty],
    plot_fn: Callable[[plt.Axes, GPTLayerProperty, str, mcolors.Colormap, int], List[Any]],
    title: np.str_,
    output_dir: Path | None = None,
):
    """
    layer_property_time_series: maps from checkpoint idx to GPTLayerProperty
    """
    # 1) Create figures for each checkpoint
    frames = {
        step: create_subplot_grid(layer_property=lp,
                                   plot_fn=plot_fn,
                                   title=f"{title}, Checkpoint {step}")
        for step, lp in layer_property_time_series.items()
    }

    # 2) Measure per-subplot bounds across all frames (first infer scales from first frame)
    for fig in frames.values():
        fig.canvas.draw()
    # Only apply global bounds to the 6 data subplots (exclude the colorbar axis).
    n_panels = len(PARAM_TYPES)
    xmins, xmaxs, ymins, ymaxs = get_global_axis_bounds(list(frames.values()), n_axes=n_panels)

    # 3) Apply global bounds to each frame and save PNGs
    out_dir = Path("research_logs/visualizations") if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_paths: List[str] = []
    for step, fig in frames.items():
        axes = fig.axes[:n_panels]
        for i, ax in enumerate(axes):
            # Guard: never set NaN/Inf limits (can happen if a panel has no data in all frames).
            if not (
                np.isfinite(xmins[i])
                and np.isfinite(xmaxs[i])
                and np.isfinite(ymins[i])
                and np.isfinite(ymaxs[i])
            ):
                continue
            ax.set_xlim(xmins[i], xmaxs[i])
            ax.set_ylim(ymins[i], ymaxs[i])
        png = out_dir / f"{str(title).replace(' ', '_')}_{step:06d}.png"
        fig.savefig(png, dpi=150, bbox_inches='tight')
        png_paths.append(str(png))

    # 4) Write GIF
    images = [imageio.imread(p) for p in png_paths]
    gif_path = out_dir / f"{str(title).replace(' ', '_')}.gif"
    imageio.mimsave(gif_path, images, fps=12, loop=0)
    for p in png_paths:
        Path(p).unlink(missing_ok=True)
    return gif_path


def get_global_axis_bounds(frames: List[plt.Figure], n_axes: int | None = None):
    """Aggregate per-subplot global (xmin,xmax,ymin,ymax) across frames.

    Infers log/linear scales from the first frame's axes and enforces
    log-safe minima for log-scaled axes.
    """
    assert frames, "No frames provided"
    first_axes = frames[0].axes[:n_axes] if n_axes is not None else frames[0].axes
    n = len(first_axes)

    x_is_log = [ax.get_xscale() == 'log' for ax in first_axes]
    y_is_log = [ax.get_yscale() == 'log' for ax in first_axes]

    eps = 1e-8
    # Use NaN sentinels so we can detect "no data anywhere" cleanly.
    xmins = [np.nan] * n
    xmaxs = [np.nan] * n
    ymins = [np.nan] * n
    ymaxs = [np.nan] * n

    def _acc_min(old: float, new: float) -> float:
        if not np.isfinite(new):
            return old
        if np.isnan(old):
            return float(new)
        return float(min(old, new))

    def _acc_max(old: float, new: float) -> float:
        if not np.isfinite(new):
            return old
        if np.isnan(old):
            return float(new)
        return float(max(old, new))

    for fig in frames:
        axes = fig.axes[:n]  # exclude colorbar axis (and any extras)
        for i, ax in enumerate(axes):
            bb = ax.dataLim
            # dataLim can be Inf/-Inf if the axis has no artists/data.
            xmins[i] = _acc_min(xmins[i], bb.xmin)
            xmaxs[i] = _acc_max(xmaxs[i], bb.xmax)
            ymins[i] = _acc_min(ymins[i], bb.ymin)
            ymaxs[i] = _acc_max(ymaxs[i], bb.ymax)

    # Fallback for panels that had no data across all frames: use first-frame current limits.
    for i, ax in enumerate(first_axes):
        if not (np.isfinite(xmins[i]) and np.isfinite(xmaxs[i])):
            lo, hi = ax.get_xlim()
            xmins[i] = float(lo)
            xmaxs[i] = float(hi)
        if not (np.isfinite(ymins[i]) and np.isfinite(ymaxs[i])):
            lo, hi = ax.get_ylim()
            ymins[i] = float(lo)
            ymaxs[i] = float(hi)

        # Enforce sane ordering / non-degenerate spans.
        if not (np.isfinite(xmins[i]) and np.isfinite(xmaxs[i])) or xmaxs[i] <= xmins[i]:
            xmins[i] = eps
            xmaxs[i] = 1.0
        if not (np.isfinite(ymins[i]) and np.isfinite(ymaxs[i])) or ymaxs[i] <= ymins[i]:
            ymins[i] = eps
            ymaxs[i] = 1.0

    for i in range(n):
        if x_is_log[i]:
            xmins[i] = max(xmins[i], eps)
        if y_is_log[i]:
            ymins[i] = max(ymins[i], eps)
    return xmins, xmaxs, ymins, ymaxs
