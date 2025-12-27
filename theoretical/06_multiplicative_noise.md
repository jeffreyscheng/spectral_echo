# Multiplicative noise model for minibatch gradients

## Setup

Fix a layer and view its per-token gradient as a matrix
$$
G \in \mathbb{R}^{m \times n},
$$
for example $m$ = input dimension, $n$ = output dimension. For a given training step we have $R$ independent minibatch replicas, giving sample gradients
$$
\widehat{G}^{(r)}, \quad r = 1,\dots,R.
$$

We write the SVD of the *mean* gradient (or some reference gradient, e.g. full-batch or large-batch) as
$$
G = U \Sigma V^\top
= \sum_{i=1}^{d} \sigma_i\, u_i v_i^\top, \quad d = \mathrm{rank}(G),
$$
with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ and orthonormal singular vectors $u_i \in \mathbb{R}^m$, $v_i \in \mathbb{R}^n$.

Empirically we observe:

- top singular values are well-defined and stable across replicas,
- but singular directions $(u_i, v_i)$ vary across replicas,
- and the “spectral echo” of the reverb operator saturates at a plateau strictly $< 1$ even for the top singular directions.

A purely additive noise model does not explain this.

## Additive noise baseline

The standard baseline is
$$
\widehat{G}^{(r)} = G + \Xi^{(r)}, \quad \mathbb{E}[\Xi^{(r)}] = 0.
$$

If $\Xi^{(r)}$ is “small” and roughly isotropic, first-order perturbation theory says:

- the *mean* gradient direction is close to $G$,
- the SVD of each $\widehat{G}^{(r)}$ is a small perturbation of the SVD of $G$,
- in particular, the top singular vectors $u_1^{(r)}, v_1^{(r)}$ stay very close to $u_1, v_1$ as long as $\|\Xi^{(r)}\| \ll \sigma_1 - \sigma_2$.

In such a model, if we build any “echo” by averaging across replicas, the top singular directions should approach perfect alignment as $R$ grows, and the spectral echo plateau at the top direction should tend to $1$.

Empirically, we see a stable plateau $< 1$ for the top direction, which is incompatible with a pure additive-noise picture (unless the additive noise is large enough to destroy the notion of a well-separated top singular direction, which is not what we see).

This motivates a *multiplicative* noise model.

## Left–right multiplicative noise model

We model each minibatch gradient as
$$
\widehat{G}^{(r)} = (I_m + A^{(r)})\, G\, (I_n + B^{(r)}) + E^{(r)},
$$
where

- $A^{(r)} \in \mathbb{R}^{m \times m}$ and $B^{(r)} \in \mathbb{R}^{n \times n}$ are random matrices, modeling multiplicative noise in the left and right subspaces,
- $E^{(r)}$ is a residual additive-noise term (which we will mostly ignore in the theory),
- we assume
  $$
  \mathbb{E}[A^{(r)}] = 0, \quad \mathbb{E}[B^{(r)}] = 0, \quad \mathbb{E}[A^{(r)} B^{(r)}] = 0,
  $$
  and that $A^{(r)}, B^{(r)}$ are “small” in operator norm.

Intuition: $A^{(r)}$ and $B^{(r)}$ capture random rescalings, random rotations, and other structured distortions in the row/column spaces induced by dropout, attention masking, stochastic layernorm behavior, data sampling, etc. They mainly act by slightly mixing singular modes of $G$ with each other.

## Working in the SVD basis

Write $G = U \Sigma V^\top$ and rotate into this basis. Define
$$
\tilde{A}^{(r)} = U^\top A^{(r)} U, \quad
\tilde{B}^{(r)} = V^\top B^{(r)} V,
$$
and
$$
\widetilde{G}^{(r)} := U^\top \widehat{G}^{(r)} V.
$$

Then
$$
\widetilde{G}^{(r)} 
= (I + \tilde{A}^{(r)}) \Sigma (I + \tilde{B}^{(r)}) + \widetilde{E}^{(r)},
$$
with $\widetilde{E}^{(r)} = U^\top E^{(r)} V$.

For “small” noise we expand to first order:
$$
\widetilde{G}^{(r)} \approx \Sigma + \Delta^{(r)}, \quad
\Delta^{(r)} := \tilde{A}^{(r)} \Sigma + \Sigma \tilde{B}^{(r)}.
$$

Componentwise,
$$
\Delta^{(r)}_{ij} \approx
\sum_k \tilde{A}^{(r)}_{ik} \Sigma_{kj}
\;+\;
\sum_k \Sigma_{ik} \tilde{B}^{(r)}_{kj}.
$$

Because $\Sigma$ is diagonal, this simplifies to
$$
\Delta^{(r)}_{ij} \approx
\tilde{A}^{(r)}_{ij} \sigma_j + \sigma_i \tilde{B}^{(r)}_{ij}.
$$

Key point: off-diagonal entries of $\tilde{A}^{(r)}$ and $\tilde{B}^{(r)}$ mix different singular modes.

## Effect on singular vectors (first-order perturbation theory)

Let $(u_i, \sigma_i, v_i)$ be the $i$-th singular triple of $G$, and $(u_i^{(r)}, \sigma_i^{(r)}, v_i^{(r)})$ the $i$-th singular triple of $\widehat{G}^{(r)}$.

Classical first-order SVD perturbation theory says that for a simple singular value (well separated from its neighbors),
$$
u_i^{(r)} \approx u_i + \sum_{j \ne i} \alpha_{ji}^{(r)} u_j,
\quad
v_i^{(r)} \approx v_i + \sum_{j \ne i} \beta_{ji}^{(r)} v_j,
$$
with coefficients satisfying
$$
\alpha_{ji}^{(r)} \propto \frac{\Delta^{(r)}_{ji} \sigma_i + \Delta^{(r)}_{ij} \sigma_j}{\sigma_i^2 - \sigma_j^2},
\quad
\beta_{ji}^{(r)} \propto \frac{\Delta^{(r)}_{ij} \sigma_i + \Delta^{(r)}_{ji} \sigma_j}{\sigma_i^2 - \sigma_j^2}.
$$

Under our multiplicative model,
$$
\Delta^{(r)}_{ij}
\approx
\tilde{A}^{(r)}_{ij} \sigma_j + \sigma_i \tilde{B}^{(r)}_{ij},
$$
so the mixing coefficients $\alpha_{ji}^{(r)}, \beta_{ji}^{(r)}$ are *linear* in the off-diagonal entries of $\tilde{A}^{(r)}, \tilde{B}^{(r)}$, and their variance is controlled by
$$
\mathrm{Var}(\tilde{A}^{(r)}_{ij}),\ \mathrm{Var}(\tilde{B}^{(r)}_{ij}),
\quad
\text{and spectral gaps}\quad
|\sigma_i^2 - \sigma_j^2|.
$$

Thus for each singular direction $i$ we can write, to leading order,
$$
\mathbb{E}\Big[|\langle u_i^{(r)}, u_i\rangle|^2\Big]
\approx
1 - \sum_{j \ne i} \frac{\mathbb{E}\big[|\Delta^{(r)}_{ji}|^2\big]}{(\sigma_i^2 - \sigma_j^2)^2},
$$
and similarly for the right singular vectors $v_i^{(r)}$.

Conclusion: multiplicative noise induces *direction-dependent* random rotations of the singular vectors, with larger rotations when:

- off-diagonal multiplicative noise is strong in the SVD basis,
- spectral gaps are small (near-degenerate singular values).

## Connection to spectral echo

Let $\theta_{L,i}^{(r)}$ be the angle between $u_i^{(r)}$ and a reference $u_i$ (for example from the mean gradient), and similarly $\theta_{R,i}^{(r)}$ between $v_i^{(r)}$ and $v_i$.

Define the squared overlaps
$$
c_{L,i}^{2} := \mathbb{E}_r\big[\cos^2 \theta_{L,i}^{(r)}\big]
= \mathbb{E}_r\big[|\langle u_i^{(r)}, u_i\rangle|^2\big],
$$
$$
c_{R,i}^{2} := \mathbb{E}_r\big[\cos^2 \theta_{R,i}^{(r)}\big]
= \mathbb{E}_r\big[|\langle v_i^{(r)}, v_i\rangle|^2\big].
$$

The reverb operator we are using is built from cross-replica covariances of the gradients. In the SVD basis of the mean gradient, its eigenvectors are approximately the singular directions, and the corresponding eigenvalues encode cross-replica alignment.

A simple approximation (assuming weak coupling between left and right fluctuations and stationarity across replicas) is:
$$
\zeta_i^2 \approx 
\mathbb{E}_r\big[|\langle u_i^{(r)}, u_i\rangle|^2\big]\,
\mathbb{E}_r\big[|\langle v_i^{(r)}, v_i\rangle|^2\big]
= c_{L,i}^2\, c_{R,i}^2,
$$
so that a convenient *predicted* echo plateau for singular direction $i$ is
$$
\zeta_i^{\mathrm{pred}} \approx \sqrt{c_{L,i}^2 c_{R,i}^2}.
$$

Empirically we focus on the top $k$ singular directions (e.g. $k = 8$). For those we estimate

- the *empirical* plateau (from the reverb spectrum)
  $$
  \zeta_{\mathrm{emp, top}} := \mathrm{median}_{i \le k} \,\zeta_i,
  $$
- and the *predicted* plateau (from left/right alignment stats)
  $$
  \zeta_{\mathrm{pred, top}} := \mathrm{median}_{i \le k}\,\sqrt{c_{L,i}^2 c_{R,i}^2}.
  $$

If the multiplicative-noise model is a good description, we should have
$$
\zeta_{\mathrm{emp, top}} \approx \zeta_{\mathrm{pred, top}},
$$
up to noise and modeling error.

The “noise strength” for the top directions can be summarized by
$$
\eta_{\mathrm{mult}} := 1 - \overline{c_{L,i}^2 c_{R,i}^2}
\quad\text{(average over top $k$ directions)},
$$
and the “echo gap” by
$$
\gamma_{\mathrm{echo}} := 1 - \zeta_{\mathrm{emp, top}}^2.
$$
The model qualitatively predicts a roughly linear relationship
$$
\gamma_{\mathrm{echo}} \propto \eta_{\mathrm{mult}},
$$
modulo higher-order effects and spectral-gap dependence.

## Reparameterization invariance

Consider an invertible change of basis in the layer:
$$
x' = Sx, \quad y' = Ty,
$$
with $S \in \mathbb{R}^{m \times m}$, $T \in \mathbb{R}^{n \times n}$ invertible. Under such a change, the gradient transforms as
$$
G' = S G T^\top,
$$
and the noise as
$$
\widehat{G}'^{(r)} = S \widehat{G}^{(r)} T^\top.
$$

If we write the multiplicative noise model in the new basis,
$$
\widehat{G}'^{(r)} = (I + A'^{(r)}) G' (I + B'^{(r)}) + E'^{(r)},
$$
then
$$
A'^{(r)} = S A^{(r)} S^{-1}, \quad
B'^{(r)} = (T^{-1})^\top B^{(r)} T^\top,
$$
and the structure “left multiplicative + right multiplicative” is preserved.

Crucially, the *relative* rotations of singular vectors across replicas, and hence the overlaps $c_{L,i}^2, c_{R,i}^2$ and the echo plateau $\zeta_i$, are invariant under such reparameterizations. Rescaling or rotating activations and weights changes $G$ and the raw singular values $\sigma_i$, but does not change the statistics of alignment angles between replicas.

This matches the empirical observation that the echo plateau is largely insensitive to reparameterizations that preserve the function computed by the network (e.g. scaling of weights compensated by inverse scaling in adjacent layers).

## Empirical observables

From the model, the key empirically accessible quantities are:

- Left alignment angles:
  $$
  \theta_{L,i}^{(r)} = \angle(u_i^{(r)}, u_i),
  \quad
  c_{L,i}^2 = \mathbb{E}_r[\cos^2 \theta_{L,i}^{(r)}].
  $$
- Right alignment angles:
  $$
  \theta_{R,i}^{(r)} = \angle(v_i^{(r)}, v_i),
  \quad
  c_{R,i}^2 = \mathbb{E}_r[\cos^2 \theta_{R,i}^{(r)}].
  $$
- Echo eigenvalues $\zeta_i$ from the reverb operator in the SVD basis of the mean gradient.

The multiplicative noise model predicts:

1. Nontrivial echo plateau:
   $$
   \zeta_i < 1 \text{ for top directions } i,
   $$
   with a plateau level controlled by the amount of multiplicative noise.

2. A quantitative relationship:
   $$
   \zeta_i^{\mathrm{emp}} \approx \sqrt{c_{L,i}^2 c_{R,i}^2}.
   $$
   When averaged over the top $k$ directions, this gives a scalar predicted plateau $\zeta_{\mathrm{pred, top}}$ that we can compare to the empirical plateau $\zeta_{\mathrm{emp, top}}$.

3. A link between echo gap and multiplicative noise strength:
   $$
   1 - \zeta_{\mathrm{emp, top}}^2 \quad\text{vs}\quad
   1 - \overline{c_{L,i}^2 c_{R,i}^2},
   $$
   which should be roughly linearly related if the first-order perturbation picture is accurate.

These are the relationships we will test with the new statistics and visualizations.
