#!/usr/bin/env python3
"""
empirical/research/analysis/logging_utilities.py
Unified logging utilities for gradient analysis.

Consolidates online and offline logging functionality including:
- Model checkpoint serialization/deserialization
- Logging step determination
- Singular value computation and logging
- GIF creation for visualization
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.distributed as dist


def is_logging_step_piecewise_log(step: int, total_steps: int) -> bool:
    """
    Determine if current step should be logged using piecewise logarithmic sampling.
    
    Samples more frequently early in training, less frequently later.
    """
    if step == 0:
        return True
    
    # Early training: every 10 steps for first 100 steps
    if step <= 100:
        return step % 10 == 0
    
    # Mid training: every 100 steps up to 1000
    if step <= 1000:
        return step % 100 == 0
        
    # Late training: logarithmic sampling
    log_step = int(np.log10(step))
    interval = 10 ** max(1, log_step - 1)
    return step % interval == 0


def serialize_model_checkpoint(
    model,
    hidden_optimizer,
    other_state: Dict[str, Any],
    run_name: str,
    checkpoint_dir: Path,
):
    """
    Callback-style model checkpoint serializer.

    Expected signature for use with training.run_loggers:
      serialize_model_checkpoint(model, optimizer, other_state, checkpoint_dir)

    - model: PyTorch model to serialize
    - optimizer: unused placeholder to match callback interface
    - other_state: dict containing at least {'run_name': str, 'step': int}
    - checkpoint_dir: base directory to save checkpoints (Path or str)

    Writes checkpoints as {checkpoint_dir}/{run_name}/step_{step:06d}.pt on rank 0.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return None

    # Expect strict schema for other_state
    step = int(other_state["step"])  # KeyError if missing
    model_args = other_state["model_args"] if "model_args" in other_state else {}

    run_dir = Path(checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = run_dir / f"model_step_{step:06d}.pt"

    # Map param Tensor -> name (for optimizer.state lookup)
    name_of = {p: n for n, p in model.named_parameters()}

    # Extract momentum buffers B from the hidden optimizer’s state
    momentum_buffers: Dict[str, torch.Tensor] = {}
    for p, st in hidden_optimizer.state.items():
        if "momentum_buffer" in st:
            name = name_of.get(p, None)
            if name is not None:
                momentum_buffers[name] = st["momentum_buffer"].detach().cpu()

    # Current momentum γ (store per param group; usually a single group)
    group_momentum = [float(g.get("momentum", 0.0)) for g in hidden_optimizer.param_groups]


    checkpoint = {
        'model': model.state_dict(),
        'step': step,
        'model_args': model_args,
        'timestamp': time.time(),
        'hidden_optimizer_meta': {
            'class': type(hidden_optimizer).__name__,
            'group_momentum': group_momentum,
            'momentum_buffers': momentum_buffers,
        },
    }
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def deserialize_model_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Load model checkpoint with metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary containing model state, step, and metadata
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # Backward compatibility: normalize legacy keys
    if 'model' not in checkpoint and 'model_state_dict' in checkpoint:
        checkpoint['model'] = checkpoint['model_state_dict']
    if 'model_args' not in checkpoint:
        checkpoint['model_args'] = {}
    if 'step' not in checkpoint:
        # Try to parse from filename step_XXXX
        try:
            import re
            m = re.search(r'step_(\d+)', str(checkpoint_path))
            if m:
                checkpoint['step'] = int(m.group(1))
        except Exception:
            pass
    # Validate checkpoint contents
    required_keys = ['model', 'step', 'model_args']
    for key in required_keys:
        if key not in checkpoint:
            raise KeyError(f"Missing required key '{key}' in checkpoint")
    return checkpoint


def log_from_rank(
    msg: str,
    rank: int,
):
    if rank == 0:
        logging.info(msg)
