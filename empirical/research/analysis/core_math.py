"""
empirical/research/analysis/core_math.py
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


def safe_svd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast, robust SVD for [H,W] or [B,H,W].

    - Promotes bf16 -> f32 for numerical stability.
    - Ensures contiguity for linalg kernels.
    - Never moves results to CPU. (If input is CUDA, output stays CUDA.)
    """
    with torch.no_grad():
        x = tensor
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.contiguous()

        U, s, Vh = torch.linalg.svd(x, full_matrices=False)
        return U, s, Vh

# Convenience functions for common operations
def stable_rank_from_tensor(tensor: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute stable rank directly from a matrix (computes SVD internally).

    IMPORTANT: Avoid material CPU transfers in the analysis compute pipeline.
    For torch inputs, returns a 0-dim torch scalar (on the input device).
    """
    if isinstance(tensor, torch.Tensor):
        with torch.no_grad():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            s = torch.linalg.svdvals(tensor)
            s2 = s * s
            smax2 = s2.max().clamp_min(1e-20)
            # ignore tiny singular values to reduce numerical-noise sensitivity
            mask = (s > 1e-8).to(dtype=s.dtype)
            num = (s2 * mask).sum()
            return (num / smax2)
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
        denom = max(1, (B - 1) * m * n)
        return total_frob2 / denom


########################
# Spectral reverb solver (tuning-free debiased variant)
########################

def solve_for_spectral_echo_using_reverb(
    left_bases_U: torch.Tensor,   # (r, H, Kc)  aligned left singular bases per replica
    right_bases_V: torch.Tensor,  # (r, W, Kc)  aligned right singular bases per replica
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

def compute_reverb_fit_relative_diagnostics(
    aligned_svds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    spectral_echo_replicates: torch.Tensor,  # [R, D] (zeta per replica, per direction)
    *,
    num_bins: int = 32,
    max_directions_for_binning: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute *relative* fit diagnostics for the reverb->echo least-squares surrogate.

    Outputs are designed for lightweight serialization and render-only plotting:
      1) rel_residual_by_echo: [D]
         For each singular direction k, we compute a pivotwise relative residual
             rel(p,k) := Var_w(r_{ab->p,k}) / (mean_w(r_{ab->p,k})^2 + eps)
         where r_{ab->p,k} are the triple ratios and w = Z_ab^2 (as in the weighted LS).
         We then aggregate across pivots p by median.

      2) denom_bin_centers: [B], denom_rel_residual: [B]
         Stratified *pair-averaged* relative residual vs |Z_ab| (denominator magnitude).

      3) numer_bin_centers: [B], numer_rel_residual: [B]
         Stratified relative residual vs |Z_ap Z_bp| (numerator magnitude), aggregated over pivots.

    Notes:
    - This uses only aligned bases and the already-computed per-replica echoes.
    - The bin-stratified curves are computed over a subsample of directions (default 64)
      to keep compute overhead controlled.
    """
    with torch.no_grad():
        U_rep, _S_rep, V_rep = aligned_svds  # U:[R,H,D], V:[R,W,D]
        if U_rep.dtype == torch.bfloat16:
            U_rep = U_rep.float()
        if V_rep.dtype == torch.bfloat16:
            V_rep = V_rep.float()
        zeta = spectral_echo_replicates
        if zeta.dtype == torch.bfloat16:
            zeta = zeta.float()

        R, _H, D = U_rep.shape
        eps = torch.finfo(U_rep.dtype).eps

        # Build reverb tensor Z[d,a,b] = <u_a,u_b><v_a,v_b>, with diag zero.
        U_stacked = U_rep.permute(2, 0, 1)  # (D,R,H)
        V_stacked = V_rep.permute(2, 0, 1)  # (D,R,W)
        gram_U = torch.einsum('dri,dsi->drs', U_stacked, U_stacked)  # (D,R,R)
        gram_V = torch.einsum('dri,dsi->drs', V_stacked, V_stacked)  # (D,R,R)
        Z = gram_U * gram_V  # (D,R,R)
        eye = torch.eye(R, device=Z.device, dtype=torch.bool).unsqueeze(0)  # (1,R,R)
        Z = Z.masked_fill(eye, 0.0)

        # ---------- 1) Per-direction relative residual vs fitted echo ----------
        # s_hat[p,k] is the LS estimate of zeta^2 (same object fitted by triple-OLS)
        s_hat = (zeta * zeta).transpose(0, 1).contiguous()  # (D,R)

        Z2 = Z * Z  # (D,R,R)
        S0 = Z2.sum(dim=(1, 2)).clamp_min(eps)  # (D,)
        t = Z2.sum(dim=1)  # (D,R)  (sum over a)

        # Weighted second moment of r under w=Z_ab^2:
        #   sum_{a,b} w r^2 = sum_{a,b} Z_ap^2 Z_bp^2 = (sum_a Z_ap^2)^2 = t^2
        Ew_r2 = (t * t) / S0.unsqueeze(1)  # (D,R)
        var_w = (Ew_r2 - (s_hat * s_hat)).clamp_min(0.0)  # (D,R)
        rel = var_w / (s_hat * s_hat + eps)  # (D,R)
        rel_residual_by_echo = torch.median(rel, dim=1).values  # (D,)

        # ---------- 2/3) Bin-stratified relative residuals ----------
        K = min(int(max_directions_for_binning), int(D))
        if K <= 0:
            denom_bin_centers = torch.empty((0,), device=Z.device, dtype=Z.dtype)
            denom_rel_residual = torch.empty((0,), device=Z.device, dtype=Z.dtype)
            numer_bin_centers = torch.empty((0,), device=Z.device, dtype=Z.dtype)
            numer_rel_residual = torch.empty((0,), device=Z.device, dtype=Z.dtype)
            return rel_residual_by_echo, denom_bin_centers, denom_rel_residual, numer_bin_centers, numer_rel_residual

        dir_idx = torch.linspace(0, D - 1, steps=K, device=Z.device).round().long().unique()
        K = int(dir_idx.numel())

        def _offdiag_mask(R_: int, device) -> torch.Tensor:
            return ~torch.eye(R_, device=device, dtype=torch.bool)

        off = _offdiag_mask(R, Z.device)

        denom_mag = Z[dir_idx][:, off].abs()
        denom_mag = denom_mag[denom_mag > 0]
        denom_lo = float(torch.quantile(denom_mag, 0.01).clamp_min(1e-8).item()) if denom_mag.numel() else 1e-8
        denom_hi = float(torch.quantile(denom_mag, 0.99).clamp_min(denom_lo * 10.0).item()) if denom_mag.numel() else 1.0

        numer_lo, numer_hi = 1e-10, 1.0
        if K:
            Zd0 = Z[int(dir_idx[0].item())]
            B0 = Zd0.t().abs()  # (R,R) with B0[p,a] = |Z_ap|
            Nm0 = (B0.unsqueeze(2) * B0.unsqueeze(1))  # (p,a,b) = |Z_ap|*|Z_bp|
            m0 = Nm0.reshape(-1)
            m0 = m0[m0 > 0]
            if m0.numel():
                numer_lo = float(torch.quantile(m0, 0.01).clamp_min(1e-10).item())
                numer_hi = float(torch.quantile(m0, 0.99).clamp_min(numer_lo * 10.0).item())
                numer_hi = min(numer_hi, 1.0)

        def _make_edges(lo: float, hi: float, nb: int, device, dtype) -> torch.Tensor:
            lo = max(lo, 1e-12)
            hi = max(hi, lo * 10.0)
            return torch.logspace(np.log10(lo), np.log10(hi), steps=nb + 1, device=device, dtype=dtype)

        denom_edges = _make_edges(denom_lo, denom_hi, num_bins, Z.device, Z.dtype)
        numer_edges = _make_edges(numer_lo, numer_hi, num_bins, Z.device, Z.dtype)
        denom_centers = torch.sqrt(denom_edges[:-1] * denom_edges[1:])
        numer_centers = torch.sqrt(numer_edges[:-1] * numer_edges[1:])

        denom_sum = torch.zeros((num_bins,), device=Z.device, dtype=torch.float64)
        denom_cnt = torch.zeros((num_bins,), device=Z.device, dtype=torch.float64)
        numer_sum = torch.zeros((num_bins,), device=Z.device, dtype=torch.float64)
        numer_cnt = torch.zeros((num_bins,), device=Z.device, dtype=torch.float64)

        idx = torch.arange(R, device=Z.device)
        a_grid = idx.view(1, R, 1)
        b_grid = idx.view(1, 1, R)
        p_grid = idx.view(R, 1, 1)
        valid_triple_mask = (a_grid != b_grid) & (a_grid != p_grid) & (b_grid != p_grid)  # (p,a,b)

        for di in dir_idx.tolist():
            Zd = Z[int(di)]            # (R,R)
            sd = s_hat[int(di)]        # (R,)

            # ---- Denominator-stratified (pair-averaged over pivots) ----
            s2 = (sd * sd + eps)
            alpha = sd / s2            # (R,)
            beta = 1.0 / s2            # (R,)
            gamma = (sd * sd) / s2     # (R,)
            mean_gamma = gamma.mean()

            Q = Zd * Zd
            A2 = (Zd * alpha.unsqueeze(0)) @ Zd              # Z diag(alpha) Z
            A3 = (Q * beta.unsqueeze(0)) @ Q                 # (Z^2) diag(beta) (Z^2)
            Rf = float(R)
            resid_pair = (mean_gamma * Q) - (2.0 / Rf) * (Zd * A2) + (1.0 / Rf) * A3
            resid_pair = resid_pair.clamp_min(0.0)

            denom_x = Zd.abs()[off]
            denom_y = resid_pair[off]
            bin_idx = torch.bucketize(denom_x, denom_edges) - 1
            m = (bin_idx >= 0) & (bin_idx < num_bins) & torch.isfinite(denom_y)
            if m.any():
                bi = bin_idx[m].to(torch.long)
                denom_sum.index_add_(0, bi, denom_y[m].to(torch.float64))
                denom_cnt.index_add_(0, bi, torch.ones_like(denom_y[m], dtype=torch.float64))

            # ---- Numerator-stratified (triple-level; aggregated over pivots) ----
            B = Zd.t().contiguous()  # (R,R) with B[p,a] = Z_ap
            N = B.unsqueeze(2) * B.unsqueeze(1)  # (p,a,b)
            denom_mat = Zd.unsqueeze(0)          # (1,a,b)
            s_mat = sd.view(R, 1, 1)             # (p,1,1)
            resid_triple = (s_mat * denom_mat - N)
            resid_triple = (resid_triple * resid_triple) / (s_mat * s_mat + eps)
            resid_triple = resid_triple.clamp_min(0.0)
            numer_x = (B.abs().unsqueeze(2) * B.abs().unsqueeze(1))  # (p,a,b)

            mx = numer_x[valid_triple_mask]
            my = resid_triple[valid_triple_mask]
            bin_idx2 = torch.bucketize(mx, numer_edges) - 1
            m2 = (bin_idx2 >= 0) & (bin_idx2 < num_bins) & torch.isfinite(my)
            if m2.any():
                bi2 = bin_idx2[m2].to(torch.long)
                numer_sum.index_add_(0, bi2, my[m2].to(torch.float64))
                numer_cnt.index_add_(0, bi2, torch.ones_like(my[m2], dtype=torch.float64))

        denom_rel = (denom_sum / denom_cnt.clamp_min(1.0)).to(dtype=Z.dtype)
        numer_rel = (numer_sum / numer_cnt.clamp_min(1.0)).to(dtype=Z.dtype)

        return rel_residual_by_echo, denom_centers, denom_rel, numer_centers, numer_rel


def compute_echo_fit_diagnostics_from_aligned_svds(
    aligned_svds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    nbins: int = 24,
    diag_num_directions: int = 128,
    min_mag: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Diagnostics for whether the weighted Triple-OLS echo fit is behaving.

    Outputs:
      - spectral_echo_replicates: ζ̂_{p,k} (R,D)
      - echo_fit_weighted_mse:  E_w[(ŝ_{p,k} - r)^2] per (p,k) under w_ab = Z_ab^2 (R,D)
        where ŝ_{p,k} = ζ̂_{p,k}^2 and r_{ab->p,k} = (Z_ap Z_pb) / Z_ab (diag excluded).
      - triple_mse_by_denom_bin: binned (by |Z_ab|) conditional weighted MSE (nbins,)
      - triple_mse_by_numer_bin: binned (by |Z_ap Z_pb|) conditional weighted MSE (nbins,)
      - triple_mse_bins: geometric bin centers in [min_mag, 1] (nbins,)

    Notes:
      - echo_fit_weighted_mse uses a division-free identity for the *weighted* objective:
            w_ab (ŝ - r)^2 with w_ab = Z_ab^2
        and computes E_w[r^2] exactly (excluding diagonal) without forming r.
      - binned triple-respect curves are computed on a log-spaced subset of directions
        (to keep compute bounded) but include all pivots p and all replica pairs (a,b).
    """
    with torch.no_grad():
        U_rep, _S_rep, V_rep = aligned_svds  # U:[R,H,D], V:[R,W,D]
        if U_rep.dtype == torch.bfloat16:
            U_rep = U_rep.float()
        if V_rep.dtype == torch.bfloat16:
            V_rep = V_rep.float()

        R, _H, D = U_rep.shape
        # (D,R,H/W)
        U_stacked = U_rep.permute(2, 0, 1)
        V_stacked = V_rep.permute(2, 0, 1)

        gram_U = torch.einsum("dri,dsi->drs", U_stacked, U_stacked)  # (D,R,R)
        gram_V = torch.einsum("dri,dsi->drs", V_stacked, V_stacked)  # (D,R,R)
        Z = gram_U * gram_V  # (D,R,R)

        eye = torch.eye(R, device=Z.device, dtype=Z.dtype)
        offdiag = (1.0 - eye).unsqueeze(0)  # (1,R,R)
        Z = Z * offdiag  # zero diag

        eps = torch.finfo(Z.dtype).eps
        denom = (Z * Z).sum(dim=(-2, -1)).clamp_min(eps)  # (D,)

        # ŝ_{p,k} = ζ̂^2 from division-free weighted Triple-OLS (same as existing estimator)
        zz = Z @ Z  # (D,R,R)
        numerator = (zz * Z.transpose(-1, -2)).sum(dim=-1)  # (D,R)
        echo_sq = (numerator / denom.unsqueeze(-1)).clamp_min(0.0)  # (D,R)
        echo_zeta = torch.sqrt(echo_sq).transpose(0, 1)  # (R,D)

        # Per-(p,k) weighted MSE of the LS objective in ŝ-space:
        #   MSE = E_w[(ŝ - r)^2] with ŝ = E_w[r] => MSE = E_w[r^2] - ŝ^2
        # With w_ab = Z_ab^2 and r = (Z_ap Z_pb)/Z_ab (diag excluded),
        #   w_ab * r^2 = (Z_ap^2)(Z_pb^2) for offdiag entries (division cancels),
        # so E_w[r^2] can be computed without forming r.
        Z2 = Z * Z  # (D,R,R)
        col_sum2 = Z2.sum(dim=1)  # (D,R) = Σ_a Z_ap^2
        col_sum4 = (Z2 * Z2).sum(dim=1)  # (D,R) = Σ_a Z_ap^4
        sumN2_offdiag = (col_sum2 * col_sum2) - col_sum4  # (D,R) = Σ_{a!=b} Z_ap^2 Z_pb^2
        Er2 = sumN2_offdiag / denom.unsqueeze(-1)  # (D,R)
        mse_shat = (Er2 - (echo_sq * echo_sq)).clamp_min(0.0)  # (D,R)
        mse_shat = mse_shat.transpose(0, 1)  # (R,D)

        # --- triple-respect curves (binned conditional weighted MSE) ---
        nbins = int(max(4, nbins))
        min_mag = float(min_mag)
        # Bin edges in [min_mag, 1]
        edges = torch.logspace(
            np.log10(min_mag),
            0.0,
            steps=nbins + 1,
            device=Z.device,
            dtype=Z.dtype,
        )
        bin_centers = torch.sqrt(edges[:-1] * edges[1:])  # (nbins,)

        # Choose direction indices (log-spaced over [0, D-1]) for bin curves
        nd = int(min(max(4, diag_num_directions), D))
        t = torch.linspace(0.0, 1.0, steps=nd, device=Z.device, dtype=torch.float32)
        idx = torch.round(torch.exp(t * float(np.log(max(D, 2)))) - 1.0).to(torch.long)
        idx = torch.clamp(idx, 0, D - 1)
        idx = torch.unique(torch.cat([idx, torch.tensor([0, D - 1], device=Z.device, dtype=torch.long)]))
        Zs = Z.index_select(0, idx)  # (Ds,R,R)
        echo_sq_s = echo_sq.index_select(0, idx)  # (Ds,R)

        Ds = int(Zs.shape[0])
        offdiag2 = (1.0 - eye)  # (R,R)
        denom_mag = Zs.abs()  # (Ds,R,R)
        w = (Zs * Zs) * offdiag2.unsqueeze(0)  # (Ds,R,R)

        # denom bins: membership depends only on (d,a,b), not p
        denom_idx = torch.bucketize(denom_mag.reshape(-1), edges) - 1
        denom_idx = denom_idx.clamp(0, nbins - 1)
        w_flat = w.reshape(-1)
        denom_w_once = torch.bincount(denom_idx, weights=w_flat, minlength=nbins)  # (nbins,)
        denom_w_total = denom_w_once * float(R)
        denom_c_total = torch.zeros(nbins, device=Z.device, dtype=Z.dtype)

        numer_w_total = torch.zeros(nbins, device=Z.device, dtype=Z.dtype)
        numer_c_total = torch.zeros(nbins, device=Z.device, dtype=Z.dtype)

        # Loop pivots; per-pivot contribution is computed division-free (diag excluded)
        for p in range(R):
            s = echo_sq_s[:, p].view(Ds, 1, 1)  # (Ds,1,1) == ŝ
            u = Zs[:, :, p].unsqueeze(2)  # (Ds,R,1)
            v = Zs[:, p, :].unsqueeze(1)  # (Ds,1,R)
            N = u * v  # (Ds,R,R) == Z_ap Z_pb

            # contrib = w * (ŝ - r)^2, with w=Z_ab^2 and r=N/Z_ab, diag excluded:
            #   w(ŝ - N/Z)^2 = (Z^2)(ŝ^2) - 2(Z)(ŝ)(N) + (N^2), for offdiag entries.
            contrib = ((Zs * Zs) * (s * s) - 2.0 * Zs * s * N + (N * N)) * offdiag2.unsqueeze(0)  # (Ds,R,R)
            c_flat = contrib.reshape(-1)
            denom_c_total += torch.bincount(denom_idx, weights=c_flat, minlength=nbins)

            # numerator bins depend on p via N
            numer_mag = N.abs()
            numer_idx = torch.bucketize(numer_mag.reshape(-1), edges) - 1
            numer_idx = numer_idx.clamp(0, nbins - 1)
            numer_w_total += torch.bincount(numer_idx, weights=w_flat, minlength=nbins)
            numer_c_total += torch.bincount(numer_idx, weights=c_flat, minlength=nbins)

        denom_mse = denom_c_total / denom_w_total.clamp_min(eps)  # (nbins,)
        numer_mse = numer_c_total / numer_w_total.clamp_min(eps)  # (nbins,)

        return {
            "spectral_echo_replicates": echo_zeta,  # (R,D)
            "echo_fit_weighted_mse": mse_shat,  # (R,D)
            "triple_mse_bins": bin_centers,  # (nbins,)
            "triple_mse_by_denom_bin": denom_mse,  # (nbins,)
            "triple_mse_by_numer_bin": numer_mse,  # (nbins,)
        }

def get_mean_svd(mean_gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SVD of the mean gradient, returned as (U, S, V) with V (not Vh).
    Shapes:
        U: [H, D], S: [D], V: [W, D], D=min(H,W)
    """
    with torch.no_grad():
        U, S, Vh = safe_svd(mean_gradient)
        return U, S, Vh.T


def compute_alignment_angles_deg(
    replicate_svds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mean_svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-atom alignment angles (degrees) between each replicate's singular
    directions and the mean-gradient singular directions.

    Returns:
        left_angles_deg:  [R, D] where angle = arccos(|<u_rk, u_mean_k>|) in degrees
        right_angles_deg: [R, D] where angle = arccos(|<v_rk, v_mean_k>|) in degrees
    """
    with torch.no_grad():
        U_rep, _S_rep, V_rep = replicate_svds  # U:[R,H,D], V:[R,W,D]
        U_mean, _S_mean, V_mean = mean_svd  # U:[H,D],   V:[W,D]

        if U_rep.dtype == torch.bfloat16:
            U_rep = U_rep.float()
        if V_rep.dtype == torch.bfloat16:
            V_rep = V_rep.float()
        if U_mean.dtype == torch.bfloat16:
            U_mean = U_mean.float()
        if V_mean.dtype == torch.bfloat16:
            V_mean = V_mean.float()

        # dot per (r,k): sum_i U_rep[r,i,k] * U_mean[i,k]
        dot_left = (U_rep * U_mean.unsqueeze(0)).sum(dim=1).abs()
        dot_right = (V_rep * V_mean.unsqueeze(0)).sum(dim=1).abs()

        dot_left = dot_left.clamp(0.0, 1.0)
        dot_right = dot_right.clamp(0.0, 1.0)

        left_angles = torch.rad2deg(torch.acos(dot_left))
        right_angles = torch.rad2deg(torch.acos(dot_right))
        return left_angles, right_angles


def _greedy_perm_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Greedy one-to-one assignment done entirely in torch.

    scores: [D, D] where scores[i, j] = affinity between replica atom i and reference atom j.
    Returns:
        perm: LongTensor [D] mapping reference index j -> chosen replica index i.
    """
    with torch.no_grad():
        assert scores.ndim == 2 and scores.shape[0] == scores.shape[1]
        D = int(scores.shape[0])
        used = torch.zeros(D, dtype=torch.bool, device=scores.device)
        perm = torch.empty(D, dtype=torch.long, device=scores.device)
        neg_inf = torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype)
        for j in range(D):
            col = scores[:, j].masked_fill(used, neg_inf)
            i = torch.argmax(col)
            perm[j] = i
            used[i] = True
        return perm


def align_svds_greedy_to_mean(
    raw_svds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    mean_svd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align per-replica SVD atoms to the mean-gradient SVD as reference using a greedy
    permutation and sign-fix. Intentionally ignores degenerate subspaces / blocks.

    This function is written to avoid any `.cpu()` transfers during pipeline compute.
    """
    with torch.no_grad():
        U_rep, S_rep, V_rep = raw_svds
        U_mean, _S_mean, V_mean = mean_svd

        if U_rep.dtype == torch.bfloat16:
            U_rep = U_rep.float()
        if V_rep.dtype == torch.bfloat16:
            V_rep = V_rep.float()
        if U_mean.dtype == torch.bfloat16:
            U_mean = U_mean.float()
        if V_mean.dtype == torch.bfloat16:
            V_mean = V_mean.float()

        R, H, D = U_rep.shape
        _, W, Dv = V_rep.shape
        assert D == Dv, "U/V D mismatch"
        assert U_mean.shape == (H, D), f"U_mean shape {U_mean.shape} != ({H},{D})"
        assert V_mean.shape == (W, D), f"V_mean shape {V_mean.shape} != ({W},{D})"

        U_out = torch.empty_like(U_rep)
        S_out = torch.empty_like(S_rep)
        V_out = torch.empty_like(V_rep)

        for r in range(R):
            Ur = U_rep[r]  # [H,D]
            Vr = V_rep[r]  # [W,D]
            Sr = S_rep[r]  # [D]

            MU = (Ur.transpose(0, 1) @ U_mean).abs()  # [D,D]
            MV = (Vr.transpose(0, 1) @ V_mean).abs()  # [D,D]
            M = (MU * MV).to(dtype=torch.float32)

            perm = _greedy_perm_from_scores(M)  # [D]

            Urp = Ur.index_select(dim=1, index=perm)
            Vrp = Vr.index_select(dim=1, index=perm)
            Srp = Sr.index_select(dim=0, index=perm)

            dot_u = (Urp * U_mean).sum(dim=0)
            dot_v = (Vrp * V_mean).sum(dim=0)
            t = torch.sign(dot_u * dot_v)
            t = torch.where(t == 0, torch.ones_like(t), t)
            Urp = Urp * t.unsqueeze(0)
            Vrp = Vrp * t.unsqueeze(0)

            U_out[r] = Urp.to(dtype=U_out.dtype)
            V_out[r] = Vrp.to(dtype=V_out.dtype)
            S_out[r] = Srp.to(dtype=S_out.dtype)

        return U_out, S_out, V_out


def get_raw_svds(
    empirical_gradients: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Per-replica SVDs with no alignment / clustering / merging.

    Returns:
        U: (R, H, D)
        S: (R, D)
        V: (R, W, D)
    where D=min(H,W). Directions are the raw SVD directions of each replicate.
    """
    with torch.no_grad():
        R, H, W = empirical_gradients.shape
        U_list: list[torch.Tensor] = []
        S_list: list[torch.Tensor] = []
        V_list: list[torch.Tensor] = []
        for r in range(R):
            U, S, Vh = safe_svd(empirical_gradients[r])
            U_list.append(U)
            S_list.append(S)
            V_list.append(Vh.T)
        U_rep = torch.stack(U_list, dim=0)  # (R,H,D)
        S_rep = torch.stack(S_list, dim=0)  # (R,D)
        V_rep = torch.stack(V_list, dim=0)  # (R,W,D)
        return U_rep, S_rep, V_rep


# Backward-compat alias for older call sites.
get_aligned_svds = get_raw_svds
