# empirical/research/training/echo.py
import torch
import torch.distributed as dist
from torch.optim import Optimizer

from empirical.research.analysis.core_math import (
    get_aligned_svds,
    solve_for_spectral_echo_using_reverb,
)

class SpectralEcho(Optimizer):
    """
    Noise-aware spectral update via reverb OLS echoes.
    Interface matches Muon(params, lr, weight_decay, momentum, rank, world_size).
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.9, rank=0, world_size=1):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size
        self._m = {}
        self._local = {}

        # capture local (pre-allreduce) grads per param
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim >= 2:
                    p.register_hook(lambda grad, p=p: self._local.__setitem__(p, grad.detach().float().clone()))

    @torch.no_grad()
    def _gather(self, G_local: torch.Tensor) -> torch.Tensor:
        bufs = [torch.empty_like(G_local) for _ in range(self.world_size)]
        dist.all_gather(bufs, G_local.contiguous())
        return torch.stack(bufs, dim=0)  # [R,H,W]
    
    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            closure()

        for group in self.param_groups:
            lr = group["lr"]; wd = group["weight_decay"]; mu = group["momentum"]

            for p in group["params"]:
                if p.grad is None or p.ndim < 2:
                    continue

                # Decoupled WD
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                gbar = p.grad.detach().float()

                # Momentum buffer (same shape as p)
                m = self._m.get(p)
                if m is None:
                    m = torch.zeros_like(gbar, dtype=torch.float32)
                    self._m[p] = m
                m.mul_(mu).add_(gbar, alpha=(1.0 - mu))

                # Gather local pre-allreduce grads for reverb
                G_repl_full = self._gather(self._local[p].float())  # shape: [R, ...p.shape]

                if p.ndim == 2:
                    # --- 2D case: [H, W] ---
                    U_aligned, _, V_aligned = get_aligned_svds(G_repl_full)       # (R,H,D),(R,W,D)
                    echoes = solve_for_spectral_echo_using_reverb(U_aligned, V_aligned)  # (R,D)
                    zeta = echoes.median(dim=0).values                              # (D,)

                    U, S, Vh = torch.linalg.svd(m, full_matrices=False)             # m is [H,W]
                    V = Vh.transpose(-2, -1)
                    D = zeta.numel()
                    update = (U[:, :D] * zeta.unsqueeze(0)) @ V[:, :D].transpose(0, 1)
                    p.add_(update, alpha=-lr)

                elif p.ndim == 3:
                    # --- 3D slice-stacked case: [S, H, W] ---
                    Slices = p.shape[0]
                    upd = torch.zeros_like(p, dtype=torch.float32)

                    for s in range(Slices):
                        G_repl = G_repl_full[:, s, :, :]                            # [R,H,W]
                        U_aligned, _, V_aligned = get_aligned_svds(G_repl)          # (R,H,D),(R,W,D)
                        echoes = solve_for_spectral_echo_using_reverb(U_aligned, V_aligned)  # (R,D)
                        zeta = echoes.median(dim=0).values                          # (D,)

                        U, S, Vh = torch.linalg.svd(m[s], full_matrices=False)      # m[s]: [H,W]
                        V = Vh.transpose(-2, -1)
                        D = zeta.numel()
                        upd[s] = (U[:, :D] * zeta.unsqueeze(0)) @ V[:, :D].transpose(0, 1)

                    p.add_(upd, alpha=-lr)

                else:
                    # Not supported here (you only optimize 2D and [4,H,W] hidden params)
                    continue

        return None

