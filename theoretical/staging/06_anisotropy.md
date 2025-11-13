# Theory Notes: Spectral Echo under Anisotropic Noise, Mahalanobis Geometry, and Whitening

## Setup and notation

- True gradient: $G \in \mathbb{R}^{m\times n}$ with thin SVD $G=USV^\top$ and target rank-1 atom for direction $i$: $T:=u_i v_i^\top$.
- Replica gradients: $\widehat G^{(a)} = G + E^{(a)}$, $a=1,\dots,k$.
- **General anisotropic noise model:** $\mathrm{vec}(E^{(a)}) \sim \mathcal{N}(0,\Sigma)$ with $\Sigma \succ 0$ arbitrary (no Kronecker assumption required).
- Rank-1 atoms: $\widehat T^{(a)} := \widehat u_i^{(a)} \widehat v_i^{(a)\top}$ and $T := u_i v_i^\top$.
- Inner products: Euclidean $\langle X,Y\rangle_F = \mathrm{tr}(X^\top Y)$; Mahalanobis $\langle X,Y\rangle_M := \mathrm{vec}(X)^\top M\,\mathrm{vec}(Y)$.

The **spectral echo** for one replica under metric $M \succ 0$ is the GLS scalar
$$
c_M^* \;:=\; \arg\min_{c\in\mathbb{R}} \|c\,\widehat T^{(a)} - T\|_M^2
\;=\; \frac{\langle \widehat T^{(a)},\,T\rangle_M}{\|\widehat T^{(a)}\|_M^2}.
$$

Special cases:
- Euclidean ($M=I$): $c_I^* = \langle \widehat u^{(a)}, u\rangle\,\langle \widehat v^{(a)}, v\rangle$.
- Mahalanobis with noise: $M=\Sigma^{-1}$ (natural metric for Gaussian noise).

---

## 1) How anisotropy breaks the existing (Euclidean) reverb theory

**Reverb method (Euclidean):** Build pairwise overlaps for aligned replicas
$$
Z_{ab} \;:=\; \langle \widehat u^{(a)}, \widehat u^{(b)}\rangle\;\langle \widehat v^{(a)}, \widehat v^{(b)}\rangle,\qquad a\neq b,
$$
then exploit the **rank-1 factorization**
$\mathbb{E}[Z_{ab}] \approx \zeta^{(a)}\zeta^{(b)}$ with $\zeta^{(a)} := c_I^*$ to recover $\zeta$ by OLS (e.g., the division-free triple contraction).

**Where it fails:** Under anisotropic noise, the first-order perturbation of singular vectors depends on **projected covariance in the tangent space**, not a scalar variance. If $\mathrm{vec}(E)\sim(0,\Sigma)$,
- tangent coefficients inherit direction-dependent variances and cross-covariances $\propto P_i \Sigma P_i^\top$ ($P_i$ projects onto the SVD tangent at $T$),
- then, for distinct replicas $a\neq b$,
  $$
  \mathbb{E}[Z_{ab}] \;=\; \zeta^{(a)}\zeta^{(b)} \;+\; \underbrace{\Delta_{ab}}_{\text{anisotropy terms}} \;+\; O(\|E\|^3),
  $$
  where $\Delta_{ab}$ contains off-diagonal covariance contributions that **do not cancel** in Euclidean space.

**Resulting estimation errors (Euclidean reverb):**
- **Bias:** $O(\| \Sigma \|)$ terms appear already at second order via cross-covariances; clustered spectra (small gaps) amplify this bias.
- **Variance inflation:** The variance of $Z_{ab}$ is dominated by anisotropic directions (large eigenvalues of $P_i\Sigma P_i^\top$), so OLS has suboptimal constants and slower finite-$k$ concentration.

Takeaway: With anisotropic noise, Euclidean reverb **no longer estimates** $c_I^*$ cleanly; the rank-1 factorization in expectation breaks.

---

## 2) Why the Mahalanobis metric with $M=\Sigma^{-1}$ gives the correct answer

Define the **GLS (Mahalanobis) echo**
$$
c_{\Sigma^{-1}}^* \;=\; \frac{\langle \widehat T,\,T\rangle_{\Sigma^{-1}}}{\|\widehat T\|_{\Sigma^{-1}}^2}.
$$

**Decision-theoretic justification (linearized model):**
- Linearize $\widehat T$ around $T$: $\mathrm{vec}(\widehat T - T) \approx J\,\mathrm{vec}(E)$, where $J$ maps perturbations to the tangent space.
- For Gaussian noise, minimizing $\mathbb{E}\|c\widehat T - T\|_{\Sigma^{-1}}^2$ is **generalized least squares** (GLS): this is the MLE/BLUE in the linearized problem (Gauss–Markov and Cramér–Rao).
- Any other metric $M\neq \Sigma^{-1}$ is a **misspecified likelihood**: larger risk; and with anisotropy, first-order bias terms in pairwise overlaps do not vanish.

**First-order factorization restored:** Using Mahalanobis overlaps
$$
Z_{ab}^{(\Sigma)} \;:=\; \frac{\langle \widehat T^{(a)}, \widehat T^{(b)}\rangle_{\Sigma^{-1}}}{\|\widehat T^{(a)}\|_{\Sigma^{-1}}\;\|\widehat T^{(b)}\|_{\Sigma^{-1}}},
$$
one gets (independent replicas)
$$
\mathbb{E}\big[ Z_{ab}^{(\Sigma)} \big] \;\approx\; \zeta^{(a)}_{\Sigma}\,\zeta^{(b)}_{\Sigma},
\qquad \zeta^{(a)}_{\Sigma} := c_{\Sigma^{-1}}^*(\widehat T^{(a)},T),
$$
to first order in the noise, **even when $\Sigma$ is anisotropic**. Hence the same OLS algebra consistently estimates $c_{\Sigma^{-1}}^*$.

---

## 3) Why Mahalanobis estimation equals Euclidean estimation after whitening

Let $W$ be any invertible linear operator on matrices (acting on the vectorization) such that
$$
W\,\Sigma\,W^\top \;=\; I \quad\Longleftrightarrow\quad W^\top W = \Sigma^{-1}.
$$
Then for any $X,Y$:
$$
\langle X,Y\rangle_{\Sigma^{-1}} \;=\; \langle WX, WY\rangle_F, 
\qquad \|X\|_{\Sigma^{-1}} \;=\; \|WX\|_F.
$$
Therefore the GLS echo equals the **Euclidean echo after whitening**:
$$
\boxed{ \quad c_{\Sigma^{-1}}^*(\widehat T, T) \;=\; c_I^*\big( W\widehat T,\; WT \big). \quad }
$$
Consequently, if we replace all Euclidean overlaps in reverb by overlaps **after whitening** the atoms, we estimate **the same scalar** $c_{\Sigma^{-1}}^*$:
- Build whitened atoms: $\widetilde T^{(a)} := W\,\widehat T^{(a)},\ \widetilde T := W\,T$.
- Compute Euclidean overlaps $\widetilde Z_{ab} := \frac{\langle \widetilde T^{(a)}, \widetilde T^{(b)}\rangle_F}{\|\widetilde T^{(a)}\|_F\,\|\widetilde T^{(b)}\|_F}$.
- Run the identical OLS/triple-ratio estimator on $\widetilde Z$; this recovers $c_{\Sigma^{-1}}^*$.

This equivalence is a **congruence isometry**: whitening does not change the quantity being estimated; it moves to coordinates where the noise is isotropic and the reverb rank-1 factorization holds.

---

## 4) How to whiten the replicate gradients

We need a linear map $W$ on $\mathrm{vec}(\cdot)$ such that $W\Sigma W^\top=I$. In practice we use structured approximations to keep it as matrix–matrix multiplications.

### (a) Fully general (conceptual)
- Compute (or approximate) $\Sigma^{-1/2}$ and apply
  $$
  \mathrm{vec}(\widetilde G^{(a)}) \;=\; W\,\mathrm{vec}(\widehat G^{(a)}),\qquad W=\Sigma^{-1/2}.
  $$
- Then proceed with SVD alignment and reverb on $\{\widetilde G^{(a)}\}$ in **Euclidean** geometry.

This is usually too expensive (size $mn\times mn$), so use structure.

### (b) Kronecker (left/right) approximation
If you adopt a Kron approximation $\Sigma \approx C_R \otimes C_L$ with $C_L\succ0,\ C_R\succ0$ (estimated from replica residuals),
$$
W \;\approx\; C_R^{-1/2} \otimes C_L^{-1/2}
\quad\Longleftrightarrow\quad
\boxed{\ \widetilde G^{(a)} \;=\; C_L^{-1/2}\ \widehat G^{(a)}\ C_R^{-1/2}\ }.
$$
Compute $C_L^{-1/2}$ and $C_R^{-1/2}$ via Newton–Schulz (few iterations) or eigendecompositions of the *much smaller* $m\times m$ and $n\times n$ matrices.

Replica covariances (unbiased):
$$
\widehat C_L = \frac{1}{k-1}\sum_a (\widehat G^{(a)}-\bar G)(\widehat G^{(a)}-\bar G)^\top,
\quad
\widehat C_R = \frac{1}{k-1}\sum_a (\widehat G^{(a)}-\bar G)^\top(\widehat G^{(a)}-\bar G),
$$
with $\bar G = \frac{1}{k}\sum_a \widehat G^{(a)}$.

### (c) Sum of Kroneckers (higher fidelity)
If $W \approx \sum_{p=1}^P B_p\otimes A_p$, then
$$
\boxed{\ \widetilde G^{(a)} \;=\; \sum_{p=1}^P A_p\ \widehat G^{(a)}\ B_p^\top\ }.
$$

### (d) Diagonal / per-entry variance (cheapest)
Estimate per-entry variances
$
\widehat \sigma_{ij}^2 = \frac{1}{k-1}\sum_a (\widehat G^{(a)}_{ij}-\bar G_{ij})^2
$
and rescale rows/columns:
$
\widetilde G^{(a)} = W_L \widehat G^{(a)} W_R
$
with
$
(W_L)_{ii} \propto \big(\sum_j \widehat \sigma_{ij}^2\big)^{-1/2},\ 
(W_R)_{jj} \propto \big(\sum_i \widehat \sigma_{ij}^2\big)^{-1/2}.
$
Not exact GLS, but reduces anisotropy substantially.

---

## Practical conclusions

- **Anisotropy** breaks Euclidean reverb: expect $O(\|\Sigma\|)$ bias (worse with small spectral gaps) and inflated variance.
- **Mahalanobis (GLS) echo** with $M=\Sigma^{-1}$ is the statistically correct target: first-order unbiased and minimum-variance in the linearized regime.
- **Whitening equivalence:** $c_{\Sigma^{-1}}^*(\widehat T,T) \equiv c_I^*(W\widehat T,WT)$ for any $W$ with $W\Sigma W^\top=I$. This legitimizes whitening as an *estimation technique for the same quantity*.
- **Implementation:** use Kron (or sum-of-Krons / diagonal) approximations to build $W$ so whitening is just one (or a few) left/right matmuls per replica before SVD alignment and reverb.
