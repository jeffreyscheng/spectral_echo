# Success

Success for this project means any of the following:
- pushing the edge of our mathematical understanding towards a more complete or parsimonious description of how transformers learn good representations from self-supervised pretraining.
- improving speed of state of the art optimizers for transformers for practitioners on real workloads.

For either of these, I need to at the very least demonstrate the correctness of my assumptions and derivations on benchmarks like nanoGPT-medium.  The ultimate test is simply the wall-clock time to loss 2.92 on an 8xH100 using my algorithm compared to the state-of-the-art implementation.  However, some assumptions can be checked on a faster iteration loop.  I have enumerated them here.

# Assumptions

## Correct spectral descent is the desiderata.

This is justified by "A Spectral Condition for Feature Learning" (Yang, Simon, Bernstein): https://arxiv.org/abs/2310.17813 in that steepest descent under the spectral norm yields the µP regime.  My understanding is that µP is desirable for two reasons:
1. it does not have bad generalization properties like the NTK regime.
2. optimal hyperparameters discovered in the µP regime at one scale translate to higher scales, which reduces capex / iteration time in early model development.
However, if a different regime is more optimal than µP, then I think the entire theoretical motivation disappears.

## Msign of the true gradient is correct way to do spectral descent.

We have proofs that the msign of the gradient is the steepest direction under unit spectral norm updates.

## The gradient of a replicate is a reasonable starting point

We currently use 64*1024 tokens in a single minibatch.  This feels like a large number?  And since SGD gives an unbiased estimate of the true gradient, each replicate's gradient (we get 8 replicate gradients at a time in an 8xH100) should be an ok starting point.  This is important because msign and similar functions amplify directions with small singular values, which could blow up noise if the noise dominates signal in some direction.  Steepest descent methods could be a bad idea if:
1. turns out 64*1024 is many OOMs away from a good gradient estimate.
2. the kernel/hardware landscape shifts to accelerators with less memory.
3. the dominant modalities in pretraining reduce the number of tokens usable in a given minibatch.

# Our estimates of the spectral echo via the reverb are good.

1. Maybe 64 replicates is not enough.
2. Maybe the least-squares do not fit measurements across the spectrum well due to noise in certain parts of the spectrum dominating signal in otehr parts.
2. Maybe alignment is bad and prevents good spectral echo fitting.  I think the least squares fitting assumes that we have correctly associated singular atoms across replicates well, but if we're performing least squares on singular atoms that do not represent the same direction, the answers are bunk.  I think this can happen either due to bad permutation or due to degenerate subspaces.

## We can find a spectral function that acts as a reasonable surrogate for the true, intractable spectral echo.

We can imagine one world in which this is possible:
- the spectral echo is always monotonic in the corresponding true singular value, which is approximated well by the minibatch gradient's singular value.
- exact shape of the curve is determined by a small number of constants -- like O(1) in the dimensionality of the weights.
- these constants are either:
    - cheaply estimated at runtime on a per-minibatch basis
    - expensive to compute but stable throughout runtime

If we ever observe that the spectral echo seems to be non-monotonic in the singular values and it doesn't seem to be estimation error, then I think we should throw in the towel.

If we find that the number of knobs we need to turn in order fit the spectral echo curve shape well scales with $nxm$, the shape of the parameter matrix, then I think we should throw in the towel.

If we find that the shape changes quickly and unpredictably, then we should throw in the towel.