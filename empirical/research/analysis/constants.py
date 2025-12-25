"""
empirical/research/analysis/constants.py
Constants for the gradient analysis pipeline.
"""

FIELD_NAMES = [
    # >>> keep this list exactly in the order you currently write <<<
    # Examples (replace with your real keys):
    "weights_stable_rank",
    "gradients_stable_rank",
    "replicate_singular_values",
    "replicate_singular_values_fro_normalized",
    "left_alignment_angles_deg",
    "right_alignment_angles_deg",
    "spectral_echo",
    "empirical_phase_constant_tau2",
    "gradient_noise_sigma2",
    "aspect_ratio_beta",
    "worker_count",
    "m_big",
]

NUM_ACCUMULATION_STEPS = 8
LOG_EVERY = 5
