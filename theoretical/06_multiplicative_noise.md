# A one-number predictor for the echo plateau (per layer)

We have $R$ replica gradients for the same parameter matrix at the same checkpoint:
$$
G^{(1)},\dots,G^{(R)} \in \mathbb{R}^{n\times m}.
$$
From these replicas we can cheaply compute two scalars:
$$
\bar G := \frac{1}{R}\sum_{r=1}^R G^{(r)}, \qquad
\hat T := \frac{1}{R}\sum_{r=1}^R \|G^{(r)}\|_F^2, \qquad
\hat U := \|\bar G\|_F^2.
$$
The goal is to predict, using only $(\hat T,\hat U,R)$, the **plateau height** that the layer’s spectral echo curve approaches in the “large singular value” regime.

The point is not to predict the full curve shape. It’s to pin down the single amplitude that a cheap spectral function should saturate to during training.

---

## A checkpoint-level model that only talks about second moments

Assume each replica gradient is a shared mean plus independent noise:
$$
G^{(r)} = \mu + \Xi^{(r)},
\qquad
\mathbb{E}[\Xi^{(r)}]=0,
\qquad
\Xi^{(r)} \perp \Xi^{(s)} \text{ for } r\neq s.
$$
Define the signal and noise powers in Frobenius space:
$$
S := \|\mu\|_F^2,
\qquad
N := \mathbb{E}\big[\|\Xi^{(r)}\|_F^2\big].
$$
Everything below is built only from these two numbers.

---

## Two observable second moments

First moment we can observe (via $\hat T$):
$$
T := \mathbb{E}\big[\|G^{(r)}\|_F^2\big].
$$
Expand:
$$
\begin{aligned}
T
&= \mathbb{E}\big[\|\mu+\Xi^{(r)}\|_F^2\big] \\
&= \|\mu\|_F^2 + 2\langle \mu,\mathbb{E}\Xi^{(r)}\rangle + \mathbb{E}\|\Xi^{(r)}\|_F^2 \\
&= S + N.
\end{aligned}
$$

Second moment we can observe (via $\hat U$):
$$
U := \mathbb{E}\big[\|\bar G\|_F^2\big],
\qquad
\bar G=\mu+\bar\Xi,\quad \bar\Xi:=\frac{1}{R}\sum_{r=1}^R \Xi^{(r)}.
$$
Then
$$
U
= \|\mu\|_F^2 + \mathbb{E}\|\bar\Xi\|_F^2
= S + \mathbb{E}\left\|\frac{1}{R}\sum_{r=1}^R \Xi^{(r)}\right\|_F^2.
$$
Replica-independence kills cross terms, so
$$
\mathbb{E}\|\bar\Xi\|_F^2
= \frac{1}{R^2}\sum_{r=1}^R \mathbb{E}\|\Xi^{(r)}\|_F^2
= \frac{N}{R}.
$$
Therefore:
$$
T = S+N,
\qquad
U = S+\frac{N}{R}.
$$

This is the whole trick: we can estimate both $T$ and $U$ cheaply, and they give us two equations for $(S,N)$.

---

## Solve the 2×2 system, then read off the plateau coefficient

Solve for $S$ from
$$
T = S+N,
\qquad
U = S+\frac{N}{R}.
$$
Eliminate $N$ using $N=T-S$:
$$
U = S + \frac{T-S}{R}
= S\left(1-\frac{1}{R}\right) + \frac{T}{R}.
$$
So
$$
S = \frac{R\,U - T}{R-1},
\qquad
N = T-S = \frac{R}{R-1}(T-U).
$$

Now define the Frobenius SNR-like fraction
$$
\rho := \frac{S}{S+N} = \frac{S}{T}.
$$
Substitute $S$:
$$
\rho = \frac{R\,U - T}{(R-1)\,T}.
$$

In practice we replace $(T,U)$ by their sample estimates $(\hat T,\hat U)$:
$$
\hat\rho := \frac{R\,\hat U - \hat T}{(R-1)\,\hat T},
\qquad
\hat\rho_{\mathrm{clip}} := \min\bigl(1,\max(0,\hat\rho)\bigr).
$$

This uses only $(\hat T,\hat U,R)$, i.e. two Frobenius norms and one known constant.

---

## Why an echo plateau should track $\rho$

An “echo” constructed from replica-to-replica overlaps is trying to measure a simple thing: **how much of the gradient energy is reproducible across replicas**.

Under the model, the reproducible part is exactly $\mu$.
- The total expected energy in a replica is $S+N$.
- The shared energy (the part that does not average away) is $S$.

So $\rho=S/(S+N)$ is the fraction of energy that lives in the shared component.

Any reasonable overlap-based diagnostic that (i) is invariant to orthogonal reparameterizations of the matrix space and (ii) only depends on second moments in the high-SNR regime has essentially one scalar it can converge to: a monotone function of that reproducible fraction. Taking the plateau height as $\rho$ is the simplest consistent choice, and it has the right limiting behavior:
$$
\rho \to 1 \text{ when } N\ll S,
\qquad
\rho \to 0 \text{ when } S\ll N.
$$

So the training-time plateau predictor for a layer is:
$$
\boxed{\text{echo\_plateau\_pred} := \hat\rho_{\mathrm{clip}}.}
$$

---

## How this plugs into a spectral function

If we want a spectral function that saturates at the predicted plateau, we can factor “shape” from “amplitude”:
$$
f_\rho(\sigma) = \rho \, g\!\left(\frac{\sigma}{\sigma_\ast}\right),
$$
where
- $g$ is a fixed odd saturating shape (chosen once),
- $\sigma_\ast$ is a cheap scale (also chosen once per layer or globally),
- and $\rho$ is the only training-time amplitude, estimated by $\hat\rho_{\mathrm{clip}}$.

The only piece that needs to be measured during training is $\hat\rho_{\mathrm{clip}}$, which is just Frobenius norms of replicas and their mean.
