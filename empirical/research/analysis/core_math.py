"""
Core mathematical functions for gradient analysis.

This module contains all the fundamental mathematical operations needed across
the analysis pipeline. It provides both numpy and torch implementations where
needed, with consistent interfaces.
"""

from typing import Union, Tuple, Dict, List, Any
import numpy as np
try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
except Exception:
    _scipy_curve_fit = None
import torch

from empirical.research.analysis.model_utilities import GPTLayerProperty
from empirical.research.analysis.logging_utilities import log_from_rank

def _pav_blocks_monotone_decreasing(
    y: torch.Tensor,
    *,
    hard_boundaries: torch.Tensor | None = None,  # length d-1, True forbids merge across i|i+1
) -> list[tuple[int, int, float]]:
    """
    Pool-Adjacent-Violators for enforcing y[0] >= y[1] >= ... >= y[d-1],
    with optional forbidden merge boundaries.

    Returns list of blocks (start, end_exclusive, block_value).
    """
    # Work on CPU float64 for determinism/stability; d is ~1e3 so overhead is tiny.
    y_np = y.detach().to(torch.float64).cpu().numpy()
    hb = None
    if hard_boundaries is not None:
        hb = hard_boundaries.detach().cpu().numpy().astype(bool)

    # Each stack element: [start, end, weight, value]
    stack: list[list[float]] = []
    for i, yi in enumerate(y_np.tolist()):
        stack.append([float(i), float(i + 1), 1.0, float(yi)])
        while len(stack) >= 2:
            a = stack[-2]
            b = stack[-1]
            # boundary between a and b is at index (a.end-1)
            boundary_idx = int(a[1] - 1)
            if hb is not None and 0 <= boundary_idx < hb.shape[0] and hb[boundary_idx]:
                break
            # Monotone decreasing: require a.val >= b.val
            if a[3] >= b[3]:
                break
            # Merge
            w = a[2] + b[2]
            v = (a[3] * a[2] + b[3] * b[2]) / w
            stack[-2] = [a[0], b[1], w, v]
            stack.pop()

    out: list[tuple[int, int, float]] = []
    for s, e, _w, v in stack:
        out.append((int(s), int(e), float(v)))
    return out


def compute_stable_rank(singular_values: Union[np.ndarray, torch.Tensor], epsilon: float = 1e-8) -> float:
    """
    Compute stable rank from singular values.
    
    Stable rank = ||A||_F^2 / ||A||_2^2 = sum(s^2) / s_max^2
    where s are the singular values.
    
    Args:
        singular_values: Singular values (sorted descending)
        epsilon: Threshold for considering singular values as zero
        
    Returns:
        Stable rank as a float
    """
    # Convert to numpy if needed
    if isinstance(singular_values, torch.Tensor):
        sv = singular_values.detach().cpu().numpy()
    else:
        sv = np.asarray(singular_values)
    
    # Filter out small singular values
    sv_filtered = sv[sv > epsilon]
    
    if len(sv_filtered) == 0:
        return 0.0
    
    # Stable rank formula
    return float(np.sum(sv_filtered**2) / (sv_filtered[0]**2))


def matrix_shape_beta(shape: Union[Tuple[int, int], torch.Size]) -> float:
    """
    Compute beta = min(n,m)/max(n,m) from a (n,m) shape tuple.
    Useful as an aspect ratio parameter in rectangular matrix analyses.
    
    Args:
        shape: Matrix shape (height, width)
        
    Returns:
        Beta parameter as float
    """
    n, m = int(shape[0]), int(shape[1])
    a, b = (n, m) if n < m else (m, n)
    return float(a) / float(b)


# Backward-compatible alias used elsewhere in the codebase
## No alias needed; use matrix_shape_beta directly


## Removed MP-specific density helpers (mp_pdf_singular_*). Finite-size Wishart overlays
## now come from tabulated CDFs (see wishart_tables.py).


def safe_svd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast, robust SVD for [H,W] or [B,H,W].

    - Promotes bf16 -> f32 for numerical stability.
    - Ensures contiguity for linalg kernels.
    - JIT offloads CPU tensors to CUDA for compute (if available), then
      returns results on CPU to keep GPU memory footprint low across layers.
    """
    with torch.no_grad():
        x = tensor
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.contiguous()

        need_offload = (x.device.type != 'cuda') and torch.cuda.is_available()
        if need_offload:
            x = x.cuda(non_blocking=True)

        U, s, Vh = torch.linalg.svd(x, full_matrices=False)

        # Move back to CPU if we offloaded to keep memory steady over many layers
        if need_offload:
            U = U.cpu()
            s = s.cpu()
            Vh = Vh.cpu()
        return U, s, Vh


########################
# Removed legacy echo/alignment helpers
########################


########################
# Core convenience functions
########################


# Convenience functions for common operations
def stable_rank_from_tensor(tensor: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute stable rank directly from a matrix (computes SVD internally)."""
    if isinstance(tensor, torch.Tensor):
        with torch.no_grad():
            # Cast to float32 if needed for SVD compatibility
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            s = torch.linalg.svdvals(tensor)
            return compute_stable_rank(s)
    else:
        s = np.linalg.svd(tensor, compute_uv=False)
        return compute_stable_rank(s)


########################
# Noise estimation and tau^2 fitting
########################

def estimate_gradient_noise_sigma2(
    per_replicate_gradient: torch.Tensor,
    mean_gradient: torch.Tensor
) -> float:
    """Unbiased per-entry variance estimate from per-replica gradients.

    sigma^2_hat = (1/((R-1) m n)) * sum_i ||G_i - G_bar||_F^2
    """
    with torch.no_grad():
        B, m, n = per_replicate_gradient.shape
        diffs = per_replicate_gradient - mean_gradient.unsqueeze(0)
        # Frobenius norm squared per replicate, then sum over replicates
        frob2_per_rep = torch.sum(diffs * diffs, dim=(-2, -1))  # [B]
        total_frob2 = torch.sum(frob2_per_rep)
        return float(total_frob2 / max(1, (B - 1) * m * n))

def _get_rank0() -> int:
    try:
        import torch.distributed as dist
        return dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        return 0

def _logspace_weights_np(s: np.ndarray, nbins: int = 32) -> np.ndarray:
    log_s = np.log(s)
    edges = np.linspace(log_s.min(), log_s.max(), nbins + 1)
    idx = np.clip(np.digitize(log_s, edges) - 1, 0, nbins - 1)
    counts = np.bincount(idx, minlength=nbins).astype(np.float64)
    w = 1.0 / np.maximum(counts[idx], 1.0)
    return w * (len(w) / w.sum())


def _spectral_echo_model_np(s: np.ndarray, tau2: float) -> np.ndarray:
    return 1.0 / (1.0 + (tau2 / (s * s)))


def fit_empirical_phase_constant_tau2(
    minibatch_singular_values: torch.Tensor,
    spectral_echo: torch.Tensor,
    eps: float = 1e-12,
    nbins: int = 32,
) -> float:
    """Fit tau^2 in echo(s)=1/(1+tau^2/s^2) with REQUIRED SciPy nonlinear LS.

    - `minibatch_singular_values` (replica singulars): [R, K]
    - `spectral_echo`: [Kc] (per-direction, aggregated across replicas)
    We pair the first Kc singulars per replica with the Kc echoes, tile echoes across B,
    and fit a single tau^2.

    Raises:
        RuntimeError on any missing SciPy or fitting failure.
        ValueError on shape mismatch.
    """
    if _scipy_curve_fit is None:
        raise RuntimeError("SciPy is required for fit_empirical_phase_constant_tau2 but is not available.")

    with torch.no_grad():
        if minibatch_singular_values.ndim != 2:
            raise ValueError(f"minibatch_singular_values must be [B,K], got {tuple(minibatch_singular_values.shape)}")
        if spectral_echo.ndim != 1:
            raise ValueError(f"spectral_echo must be [Kc], got {tuple(spectral_echo.shape)}")

        B, K = minibatch_singular_values.shape
        Kc = int(spectral_echo.numel())
        if Kc > K:
            raise ValueError(f"spectral_echo length Kc={Kc} exceeds K={K}.")

        # Pair only the first Kc directions with the Kc echoes
        s_use = minibatch_singular_values[:, :Kc]             # [B, Kc]
        echo_use = spectral_echo.view(1, Kc).expand(B, Kc)    # [B, Kc]

        s = s_use.reshape(-1).detach().cpu().numpy().astype(np.float64)       # [B*Kc]
        echo = echo_use.reshape(-1).detach().cpu().numpy().astype(np.float64) # [B*Kc]
        if s.size == 0:
            raise RuntimeError("No samples to fit tau^2 (B*Kc == 0).")

        # Guard against zeros to keep the model well-defined
        s = np.clip(s, eps, None)
        echo = np.clip(echo, eps, 1.0 - eps)

        # Reweight in log-space to avoid over-emphasizing dense low-s regions
        w = _logspace_weights_np(s, nbins=nbins)             # length = B*Kc
        sigma = 1.0 / np.sqrt(np.maximum(w, eps))

        # Moment-based positive initialization
        tau2_init = float(np.mean((s * s) * (1.0 / echo - 1.0)))
        tau2_init = max(tau2_init, eps)

        (tau2_hat,), _ = _scipy_curve_fit(
            _spectral_echo_model_np,
            xdata=s,
            ydata=echo,
            p0=(tau2_init,),
            bounds=(eps, 1e40),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=20000,
        )

        tau2_hat = float(tau2_hat)
        if not np.isfinite(tau2_hat) or tau2_hat < 0.0:
            raise RuntimeError(f"Invalid tau^2 estimate: {tau2_hat}")

        return tau2_hat


def fit_empirical_noise_to_phase_slope_kappa(
    gradient_noise_sigma2: GPTLayerProperty,
    empirical_phase_constant_tau2: GPTLayerProperty
) -> float:
    """Fit kappa in tau^2 ≈ kappa * sigma^2 using per-layer points.

    Input dicts map (param_type, layer) -> scalar. We perform a
    through-origin least squares fit across available layers.
    """
    with torch.no_grad():
        # Intersect keys to align x,y
        keys = [k for k in empirical_phase_constant_tau2.keys() if k in gradient_noise_sigma2]
        if not keys:
            return float('nan')
        sigma2s = torch.tensor([float(gradient_noise_sigma2[k]) for k in keys], dtype=torch.float64).reshape(-1, 1)
        tau2s = torch.tensor([float(empirical_phase_constant_tau2[k]) for k in keys], dtype=torch.float64).reshape(-1, 1)
        sol = torch.linalg.lstsq(sigma2s, tau2s).solution  # [1,1]
        return float(sol.squeeze())

########################
# Spectral reverb solver (tuning-free debiased variant)
########################

def solve_for_spectral_echo_using_reverb(
    left_bases_U: torch.Tensor,   # (r, H, Kc)  aligned left singular bases per replica
    right_bases_V: torch.Tensor,  # (r, W, Kc)  aligned right singular bases per replica
    noise_quantile: float = 0.02, # UNUSED in this debiased, tuning-free variant
    tau_mult: float = 1.5,        # UNUSED in this debiased, tuning-free variant
    weight_power: float = 2.0,    # UNUSED in this debiased, tuning-free variant
) -> torch.Tensor:                # returns echoes with shape (r, Kc)
    """
    Debiased spectral-echo estimator (no tunable floors, no weights).

    Model: independent replicas of \hat{G} = G + E (isotropic), alignment already applied.
    We estimate the per-direction overlap statistics of Z_{abk} directly from data and
    apply a Fieller/delta-method bias correction to the triple ratio before the sqrt.

    Notation:
        Z_{abk} = <U_a[:,k], U_b[:,k]> * <V_a[:,k], V_b[:,k]>
        r_{a,b->p,k} = (Z_{apk} * Z_{pbk}) / Z_{abk}

    Steps per direction k:
        1) Compute all off-diagonal Z_{abk} across replicas (a != b).
        2) Estimate μ_Z(k) = mean(Z_{abk}) and v_Z(k) = var(Z_{abk}) from those pairs.
        3) For each pivot p, average r_{a,b->p,k} over valid (a,b) without any floor/weights:
               rbar_{p,k} = mean_{a!=b, a!=p, b!=p} r_{a,b->p,k}
        4) Bias-correct the mean ratio using Fieller (second order):
               rtilde_{p,k} = rbar_{p,k} / (1 + v_Z(k) / μ_Z(k)^2)
        5) Echo per-pivot: ζ̂_{p,k} = sqrt(max(rtilde_{p,k}, 0))

    Returns:
        echoes: (r, Kc) tensor with per-pivot echoes; upstream can take median over pivots.
    """
    with torch.no_grad():
        U = left_bases_U
        V = right_bases_V
        if U.dtype == torch.bfloat16: U = U.float()
        if V.dtype == torch.bfloat16: V = V.float()

        r, H, Kc = U.shape
        _, W, Kc2 = V.shape
        assert Kc == Kc2, "Left/Right Kc mismatch"

        # --- small-r guard: need r >= 3 for triples; else pairwise fallback
        if r < 3:
            ZU = torch.einsum('aik,bik->abk', U, U)     # (r,r,Kc)
            ZV = torch.einsum('aik,bik->abk', V, V)     # (r,r,Kc)
            Z  = ZU * ZV                                # (r,r,Kc)
            mask = ~torch.eye(r, dtype=torch.bool, device=Z.device)
            Z_off = Z[mask].reshape(r*(r-1), Kc).abs().clamp_min(0.0)
            echoes = torch.sqrt(Z_off.median(dim=0).values.clamp_min(0.0))  # (Kc,)
            return echoes.unsqueeze(0).expand(r, -1)

        # --- Build Z_{abk} across replicas and directions
        # Z[a,b,k] = <U_a[:,k],U_b[:,k]> * <V_a[:,k],V_b[:,k}>
        ZU = torch.einsum('aik,bik->abk', U, U)         # (r,r,Kc)
        ZV = torch.einsum('aik,bik->abk', V, V)         # (r,r,Kc)
        Z  = ZU * ZV                                    # (r,r,Kc)

        # --- Estimate μ_Z(k) and v_Z(k) from off-diagonals (a != b), per direction k
        diag_mask = torch.eye(r, dtype=torch.bool, device=Z.device).unsqueeze(-1)  # (r,r,1)
        Z_off = Z.masked_select(~diag_mask).view(r*(r-1), Kc)  # (r*(r-1), Kc)
        # Unbiased sample mean and variance (per direction)
        muZ = Z_off.mean(dim=0)                                       # (Kc,)
        # Use unbiased var; guard tiny negatives from numerical noise
        if Z_off.size(0) > 1:
            vZ  = Z_off.var(dim=0, unbiased=True).clamp_min(0.0)     # (Kc,)
        else:
            vZ  = torch.zeros_like(muZ)

        # Avoid division by zero in correction factor
        eps = torch.finfo(Z.dtype).eps
        muZ_safe = torch.where(muZ.abs() < eps, torch.sign(muZ) * eps + (muZ == 0).float() * eps, muZ)
        corr = 1.0 + (vZ / (muZ_safe * muZ_safe))                    # (Kc,)

        # --- Form all triple ratios r_{a,b->p,k} without any floors/weights
        # Use vectorized construction:
        #   numer[p,a,b,k] = Z[a,p,k] * Z[p,b,k]
        #   denom[a,b,k]   = Z[a,b,k]
        idx = torch.arange(r, device=Z.device)
        Z_ap = Z[:, idx, :].permute(1, 0, 2).unsqueeze(2)   # (p,a,1,Kc) = Z[a,p,k]
        Z_pb = Z[idx, :, :].unsqueeze(2)                    # (p,1,b,Kc) = Z[p,b,k]
        numer = Z_ap * Z_pb                                 # (p,a,b,Kc)
        denom = Z.unsqueeze(0)                              # (1,a,b,Kc)

        # Valid triple mask: a!=b, a!=p, b!=p
        a_ne_b = idx.view(1, -1, 1) != idx.view(1, 1, -1)   # (1,a,b)
        a_ne_p = idx.view(-1, 1, 1) != idx.view(1, -1, 1)   # (p,a,1)
        b_ne_p = idx.view(-1, 1, 1) != idx.view(1, 1, -1)   # (p,1,b)
        valid  = (a_ne_b & a_ne_p & b_ne_p).unsqueeze(-1)   # (p,a,b,1)

        # Purely numerical protection against exact zeros (machine-precision only; NOT a tunable floor)
        tiny = eps
        denom_safe = torch.where(denom.abs() < tiny,
                                 denom + torch.sign(denom) * tiny,
                                 denom)

        triples = torch.where(valid, numer / denom_safe, torch.zeros_like(numer))  # (p,a,b,Kc)

        # Average across valid (a,b) for each pivot p, per direction k
        valid_f = valid.float()
        cnt = valid_f.sum(dim=(1, 2)).clamp_min(1.0)       # (p,1)
        rbar = (triples.sum(dim=(1, 2)) / cnt).squeeze(1)  # (p,Kc)

        # --- Fieller/delta-method debiasing: divide by (1 + vZ / muZ^2)
        # Broadcast corr (Kc,) over pivots p
        rtilde = rbar / corr.unsqueeze(0)                  # (p,Kc)

        # --- Echo per pivot: sqrt of positive part
        echoes = torch.sqrt(rtilde.clamp_min(0.0))         # (p,Kc) == (r,Kc)

        return echoes

def get_spectral_echoes_from_aligned_svds(
    aligned_svds: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Compute per-replica spectral echoes from *aligned* SVD bases.

    Args:
        aligned_svds: (U, S, V) where
            U: [R, N, D]
            S: [R, D]
            V: [R, M, D]
        (R = num_replicas, D = min(N, M))

    Returns:
        echoes_zeta: [R, D] tensor of spectral echoes per replica, per direction.
    """
    with torch.no_grad():
        aligned_U, _, aligned_V = aligned_svds  # U:(R,N,D), V:(R,M,D)
        if aligned_U.dtype == torch.bfloat16:
            aligned_U = aligned_U.float()
        if aligned_V.dtype == torch.bfloat16:
            aligned_V = aligned_V.float()

        num_replicates_R, _, dim_D = aligned_U.shape

        # (D, R, N) and (D, R, M)
        U_stacked = aligned_U.permute(2, 0, 1)
        V_stacked = aligned_V.permute(2, 0, 1)

        # Gram over feature dimension for each direction
        gram_U = torch.einsum('dri,dsi->drs', U_stacked, U_stacked)  # (D,R,R)
        gram_V = torch.einsum('dri,dsi->drs', V_stacked, V_stacked)  # (D,R,R)
        reverb_tensor_Z = gram_U * gram_V                            # (D,R,R)

        eye = torch.eye(num_replicates_R, device=reverb_tensor_Z.device, dtype=reverb_tensor_Z.dtype)
        reverb_tensor_Z = reverb_tensor_Z * (1 - eye.unsqueeze(0))   # zero diag

        # Triple-ratio estimator (same math as before)
        zz = reverb_tensor_Z @ reverb_tensor_Z                       # (D,R,R)
        numerator = (zz * reverb_tensor_Z.transpose(-1, -2)).sum(dim=-1)  # (D,R)
        denominator = (reverb_tensor_Z * reverb_tensor_Z).sum(dim=(-2, -1))
        denominator = denominator.clamp_min(torch.finfo(reverb_tensor_Z.dtype).eps)  # (D,)

        echoes_sq = numerator / denominator.unsqueeze(-1)            # (D,R)
        echoes_zeta = torch.sqrt(echoes_sq.clamp_min(0.0)).transpose(0, 1)  # (R,D)
        return echoes_zeta

def _pav_isotonic_nonincreasing(
    y: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Weighted isotonic regression with constraint y[0] >= y[1] >= ... >= y[d-1].
    Returns (y_star, blocks) where blocks are (start, end) half-open index ranges.
    """
    with torch.no_grad():
        if y.ndim != 1 or w.ndim != 1 or y.numel() != w.numel():
            raise ValueError("PAV expects 1D y and w with the same length.")

        device = y.device
        dtype = y.dtype
        y_np = y.detach().cpu().double().tolist()
        w_np = w.detach().cpu().double().tolist()
        d = len(y_np)

        # stack entries: [start, end, w_sum, value]
        stack: list[list[float]] = []
        for i in range(d):
            wi = float(w_np[i])
            yi = float(y_np[i])
            stack.append([i, i + 1, wi, yi])
            # violation for non-increasing: previous block value < last block value
            while len(stack) >= 2 and stack[-2][3] < stack[-1][3]:
                b2 = stack.pop()
                b1 = stack.pop()
                wsum = b1[2] + b2[2]
                val = (b1[3] * b1[2] + b2[3] * b2[2]) / wsum
                stack.append([b1[0], b2[1], wsum, val])

        y_star = torch.empty(d, dtype=torch.float64)
        blocks: list[tuple[int, int]] = []
        for s, e, _ws, val in stack:
            s_i = int(s)
            e_i = int(e)
            y_star[s_i:e_i] = val
            blocks.append((s_i, e_i))

        return y_star.to(device=device, dtype=dtype), blocks


def compute_pav_blocks_from_singleton_echo_replicates(
    singleton_echo_replicates: torch.Tensor,  # [R, D]
    eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Build monotone blocks (fewest adjacent merges) by projecting the per-direction
    echo estimates onto a non-increasing sequence via PAV.

    Returns:
        {
          "rough_echo": [D] median over pivots,
          "pav_echo":   [D] isotonic (non-increasing) projection,
          "blocks":     list[(start,end)] half-open ranges
        }
    """
    with torch.no_grad():
        if singleton_echo_replicates.ndim != 2:
            raise ValueError("singleton_echo_replicates must be [R, D].")
        # point estimate + uncertainty from pivot-to-pivot variability
        rough = torch.median(singleton_echo_replicates, dim=0).values.clamp(0.0, 1.0)  # [D]
        var = singleton_echo_replicates.var(dim=0, unbiased=True).clamp_min(0.0)       # [D]
        w = (var + eps).reciprocal()                                                   # [D]

        pav_echo, blocks = _pav_isotonic_nonincreasing(rough, w)
        pav_echo = pav_echo.clamp(0.0, 1.0)
        return {"rough_echo": rough, "pav_echo": pav_echo, "blocks": blocks}


def compute_alignment_bundle_pav_pass0_pass1(
    empirical_gradients: torch.Tensor,  # [R, n, m]
    mean_gradient: torch.Tensor,  # [n, m]
) -> dict[str, Any]:
    """
    Pass 0:
      - SVD all replicas once
      - singleton (per-direction) sign alignment to mean SVD
      - estimate rough spectral echo across directions
      - compute monotone-adjacent blocks via PAV

    Pass 1:
      - align within each PAV block via Procrustes (subspace alignment)

    Returns a small dict so the pipeline can expose multiple nodes without
    re-running SVDs.
    """
    with torch.no_grad():
        rank = _get_rank0()
        R, n, m = empirical_gradients.shape
        d = min(n, m)

        # Reference SVD
        G_mean = mean_gradient
        U_mean, S_mean, Vh_mean = safe_svd(G_mean)
        V_mean = Vh_mean.T

        # Replica SVDs (computed once)
        U_rep = torch.empty(R, n, d, dtype=U_mean.dtype, device=U_mean.device)
        V_rep = torch.empty(R, m, d, dtype=U_mean.dtype, device=U_mean.device)
        S_rep = torch.empty(R, d, dtype=S_mean.dtype, device=S_mean.device)
        for a in range(R):
            U_a, S_a, Vh_a = safe_svd(empirical_gradients[a])
            U_rep[a] = U_a
            V_rep[a] = Vh_a.T
            S_rep[a] = S_a

        # --- Pass 0: singleton sign alignment (no clustering)
        for a in range(R):
            u_dot = torch.sum(U_rep[a] * U_mean, dim=0)  # [d]
            v_dot = torch.sum(V_rep[a] * V_mean, dim=0)  # [d]
            sgn = torch.sign(u_dot * v_dot)
            sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            U_rep[a] = U_rep[a] * sgn.view(1, -1)
            V_rep[a] = V_rep[a] * sgn.view(1, -1)

        # Rough echoes from singleton alignment
        singleton_echo_reps = get_spectral_echoes_from_aligned_svds((U_rep, S_rep, V_rep))  # [R, d]
        rough_echo = torch.median(singleton_echo_reps, dim=0).values.clamp(0.0, 1.0)  # [d]

        sigma2_hat = estimate_gradient_noise_sigma2(empirical_gradients, G_mean)

        # --- forbid merges across resolvable spectral gaps
        # noise op-norm estimate: ||E||_2 ~ sqrt(max(H,W) * sigma^2)
        H, W = int(G_mean.shape[0]), int(G_mean.shape[1])
        noise_op = float((max(H, W) * float(sigma2_hat)) ** 0.5)
        gaps = (S_mean[:-1] - S_mean[1:]).clamp_min(0.0)  # [d-1]
        hard_boundaries = gaps > noise_op  # [d-1] bool

        # --- PAV (monotone decreasing in index; i.e. increasing in singular value)
        pav_blocks = _pav_blocks_monotone_decreasing(
            rough_echo,
            hard_boundaries=hard_boundaries,
        )

        # --- logs: noise level, forbidden boundaries, and block sizes
        sizes = [e - s for (s, e, _v) in pav_blocks]
        sizes_sorted = sorted(sizes, reverse=True)
        max_blk = sizes_sorted[0] if sizes_sorted else 0
        med_blk = sizes_sorted[len(sizes_sorted) // 2] if sizes_sorted else 0
        n_forbid = int(hard_boundaries.sum().item()) if gaps.numel() else 0
        log_from_rank(
            f"PAV(pass1): d={int(S_mean.numel())}  "
            f"||E||_2≈{noise_op:.3g}  forbidden={n_forbid}/{int(gaps.numel())}  "
            f"blocks={len(pav_blocks)}  max_blk={max_blk}  med_blk={med_blk}  "
            f"top5={sizes_sorted[:5]}",
            rank,
        )

        pav_echo = torch.empty_like(rough_echo)
        for s, e, v in pav_blocks:
            pav_echo[s:e] = float(v)
        pav_echo = pav_echo.clamp(0.0, 1.0)

        # --- Pass 1: align within each PAV block (subspace Procrustes)
        for a in range(R):
            for (s, e, _v) in pav_blocks:
                k = e - s
                if k <= 1:
                    continue
                U_ref = U_mean[:, s:e]  # [n,k]
                V_ref = V_mean[:, s:e]  # [m,k]
                U_blk = U_rep[a, :, s:e]  # [n,k]
                V_blk = V_rep[a, :, s:e]  # [m,k]

                M_left = U_blk.T @ U_ref
                M_right = V_blk.T @ V_ref

                Omega_l, _, Psi_l = torch.linalg.svd(M_left, full_matrices=False)
                Omega_r, _, Psi_r = torch.linalg.svd(M_right, full_matrices=False)
                Q_left = Omega_l @ Psi_l.T
                Q_right = Omega_r @ Psi_r.T
                Q = 0.5 * (Q_left + Q_right)
                Uq, _, Vtq = torch.linalg.svd(Q, full_matrices=False)
                Q = Uq @ Vtq

                U_rep[a, :, s:e] = U_blk @ Q
                V_rep[a, :, s:e] = V_blk @ Q

        return {
            "singleton_echo_replicates": singleton_echo_reps,  # [R, d]
            "rough_echo_singleton": rough_echo,  # [d]
            "pav_echo_singleton": pav_echo,  # [d]
            "pav_blocks": pav_blocks,  # list[(s,e,val)]
            "aligned_svds": (U_rep, S_rep, V_rep),  # final (pass 1)
        }


def compute_kron_whitening_factors_from_residuals(
    residuals: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Estimate left/right whitening factors from residual gradients.

    residuals: [R, H, W], centered per-replica gradients:
        residuals[r] = G_r - G_bar

    We approximate Σ ≈ C_R ⊗ C_L with
        C_L = (1/(R-1)) Σ_r (G_r - Ḡ)(G_r - Ḡ)^T  ∈ R^{H×H}
        C_R = (1/(R-1)) Σ_r (G_r - Ḡ)^T(G_r - Ḡ)  ∈ R^{W×W}

    and return their inverse square roots:
        W_L = C_L^{-1/2}, W_R = C_R^{-1/2}.
    """
    with torch.no_grad():
        if residuals.dtype == torch.bfloat16:
            residuals = residuals.float()
        R, H, W = residuals.shape
        if R <= 1:
            # Not enough replicas to estimate covariance; identity whitening.
            W_L = torch.eye(H, dtype=residuals.dtype, device=residuals.device)
            W_R = torch.eye(W, dtype=residuals.dtype, device=residuals.device)
            return {"left_inv_sqrt": W_L, "right_inv_sqrt": W_R}

        # Unbiased denominator
        denom = float(max(R - 1, 1))

        # Compute C_L and C_R via batched einsums
        # C_L[i,j] = 1/(R-1) Σ_{r,w} residuals[r,i,w] * residuals[r,j,w]
        C_L = torch.einsum("rhw,rkw->hk", residuals, residuals) / denom  # [H,H]
        # C_R[p,q] = 1/(R-1) Σ_{r,h} residuals[r,h,p] * residuals[r,h,q]
        C_R = torch.einsum("rhp,rhq->pq", residuals, residuals) / denom  # [W,W]

        def inv_sqrt_psd(C: torch.Tensor) -> torch.Tensor:
            # Symmetric PSD inverse square root via eigh
            evals, evecs = torch.linalg.eigh(C)
            evals_clamped = torch.clamp(evals, min=eps)
            inv_sqrt = evals_clamped.rsqrt()
            # (evecs * inv_sqrt) @ evecs^T
            return (evecs * inv_sqrt.unsqueeze(0)) @ evecs.transpose(-2, -1)

        W_L = inv_sqrt_psd(C_L)
        W_R = inv_sqrt_psd(C_R)
        return {"left_inv_sqrt": W_L, "right_inv_sqrt": W_R}


def apply_kron_whitening(
    per_replicate_gradient: torch.Tensor,
    whitening_factors: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply Kronecker whitening to per-replica gradients.

    Args:
        per_replicate_gradient: [R, H, W]
        whitening_factors: dict with
            'left_inv_sqrt':  [H,H]
            'right_inv_sqrt': [W,W]

    Returns:
        whitened_gradients: [R, H, W] with
            G̃_r = W_L @ G_r @ W_R
    """
    with torch.no_grad():
        G = per_replicate_gradient
        if G.dtype == torch.bfloat16:
            G = G.float()
        R, H, W = G.shape

        W_L = whitening_factors["left_inv_sqrt"]
        W_R = whitening_factors["right_inv_sqrt"]

        # Sanity: broadcast shapes must match per-layer dims
        assert W_L.shape == (H, H), f"Left whitening shape {W_L.shape} != ({H},{H})"
        assert W_R.shape == (W, W), f"Right whitening shape {W_R.shape} != ({W},{W})"

        out = torch.empty_like(G, dtype=G.dtype, device=G.device)
        for r in range(R):
            out[r] = W_L @ G[r] @ W_R
        return out
