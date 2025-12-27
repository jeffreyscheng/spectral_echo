#!/usr/bin/env python3
"""
empirical/research/analysis/model_utilities.py
Unified model parameter processing utilities.

Consolidates model parameter extraction, layer property management,
and distributed communication functions from the original map.py.
"""

from pathlib import Path
import io
import pickle
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, TypeAlias
import numpy as np
import torch
import torch.distributed as dist
import time
from torch.nn import Parameter, Module
from empirical.research.analysis.constants import FIELD_NAMES
from empirical.research.training.training_core import distributed_data_generator, get_window_size_blocks

# Type alias for layer properties
GPTLayerProperty: TypeAlias = dict[tuple[str, int], Parameter | np.ndarray | torch.Tensor]


def _split_attention_tensor(tensor: torch.Tensor, layer_num: int) -> Iterator[tuple[tuple[str, int], torch.Tensor]]:
    """Split qkvo_w tensor into Q, K, V, O components."""
    for i, component in enumerate(["Q", "K", "V", "O"]):
        yield (f"Attention {component}", layer_num), tensor[i]


def _extract_layer_info(name: str) -> tuple[str, int] | None:
    """Extract param type and layer number from parameter name."""
    # Handle compiled model names that have _orig_mod prefix
    if name.startswith("_orig_mod."):
        name = name[10:]  # Remove "_orig_mod." prefix
    
    if not name.startswith("blocks."):
        return None
    
    parts = name.split('.')
    layer_num = int(parts[1])
    
    param_type_map = {
        "attn.qkvo_w": "Attention",  # Will be split later
        "mlp.fc_w": "MLP Input", 
        "mlp.proj_w": "MLP Output"
    }
    
    for pattern, param_type in param_type_map.items():
        if pattern in name:
            return param_type, layer_num
    
    return None


def process_model_parameters(model, only_hidden: bool, tensor_extractor, result_processor) -> GPTLayerProperty:
    """Process model parameters using functional approach with generators."""
    
    def filtered_params():
        """Generator of filtered parameters."""
        for name, param in model.named_parameters():
            if only_hidden and not ("blocks." in name and param.ndim >= 2 and "embed" not in name):
                continue
            tensor = tensor_extractor(name, param)
            if tensor is not None:
                yield name, param, tensor
                
    
    def parameter_entries():
        """Generator of (key, tensor) pairs."""
        for name, param, tensor in filtered_params():
            layer_info = _extract_layer_info(name)
            if layer_info is None:
                continue
            
            param_type, layer_num = layer_info
            
            # Handle attention tensor splitting
            if param_type == "Attention" and tensor.ndim >= 3:
                yield from _split_attention_tensor(tensor, layer_num)
            else:
                yield (param_type, layer_num), tensor
    
    return result_processor(parameter_entries())


def get_weight_matrices(model, only_hidden: bool = True) -> GPTLayerProperty:
    """Extract weight matrices from model."""
    return process_model_parameters(
        model, 
        only_hidden,
        tensor_extractor=lambda name, param: param.data,
        result_processor=lambda entries: dict(entries)
    )


def combine_layer_properties(fn: Callable, *layer_properties: GPTLayerProperty) -> GPTLayerProperty:
    """
    Combine multiple layer properties using a function.
    
    Args:
        fn: Function that takes (*tensors) and returns a tensor
        *layer_properties: Variable number of layer property dicts
        
    Returns:
        Combined layer properties
    """
    if not layer_properties:
        return {}
        
    # Get common keys
    common_keys = set(layer_properties[0].keys())
    for props in layer_properties[1:]:
        common_keys &= set(props.keys())
    
    result = {}
    for key in common_keys:
        tensors = [props[key] for props in layer_properties]
        result[key] = fn(*tensors)
    
    return result


def get_accumulated_gradient_matrices(
    model,
    args,
    step: int,
    num_accumulation_steps: int,
    assigned_params: set = None,
    owner_map: Dict[Tuple[str, int], int] | None = None,
    accum_buffer_map: Dict[Tuple[str, int], torch.Tensor] | None = None,
    accum_gamma: float = 0.0
) -> GPTLayerProperty:
    """
    Compute accumulated gradient matrices for analysis.
    
    This function runs forward/backward passes to accumulate gradients
    across multiple independent replicates for gradient analysis (DDP workers × accumulation steps).
    
    Args:
        model: The model to compute gradients for
        args: Hyperparameters for data loading
        step: Current training step (for logging)
        num_accumulation_steps: Number of accumulation steps to build replicates (A)
        
    Returns:
        GPTLayerProperty containing per-replicate gradient tensors
    """
    # Get data generator
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    data_generator = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
    
    # Storage for per-replicate gradients (only for keys owned by this rank)
    per_replicate_grads: Dict[Tuple[str, int], list[torch.Tensor]] = {}
    
    # Original behavior: eval mode outside; enable grads only for forward/backward block
    model.eval()  # Set to eval mode for consistent analysis
    with torch.no_grad():
        for accum_idx in range(num_accumulation_steps):
            try:
                inputs, targets = next(data_generator)
            except StopIteration:
                # Re-initialize data generator if exhausted
                data_generator = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
                inputs, targets = next(data_generator)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass with grads enabled in a tight scope
            with torch.enable_grad():
                model.train()  # Briefly switch to train for grad computation
                window_size_blocks = get_window_size_blocks(step, args.num_iterations).to(inputs.device)
                loss = model(inputs.to(torch.int32), targets, window_size_blocks)
                loss.backward()
                model.eval()  # Back to eval mode
            # Log after each backward accumulation (A accumulations)
            if rank == 0:
                print(f"[rank 0] Backward complete for accumulation {accum_idx+1}/{num_accumulation_steps}")
            
            # For each parameter key in model order, gather this replicate's gradient to the owner
            for name, param in model.named_parameters():
                if param.grad is None or param.ndim < 2 or "embed" in name:
                    continue
                layer_info = _extract_layer_info(name)
                if layer_info is None:
                    continue
                param_type, layer_num = layer_info

                def handle_key(key: Tuple[str, int], grad_tensor: torch.Tensor):
                    nonlocal per_replicate_grads
                    if owner_map is None:
                        # Default to assigned_params: owner is self if in assigned_params else skip
                        owner = rank if (assigned_params is None or key in assigned_params) else -1
                    else:
                        owner = int(owner_map[key])

                    g = grad_tensor.detach()
                    if dist.is_initialized():
                        if owner == rank:
                            # Gather tensors from all ranks for this key
                            gather_list = [torch.empty_like(g) for _ in range(world_size)]
                            dist.gather(g, gather_list=gather_list, dst=rank)
                            # Append all replicas for this accumulation
                            if key not in per_replicate_grads:
                                per_replicate_grads[key] = []
                            # Keep on GPU; analysis pipeline expects GPU-resident gradients.
                            per_replicate_grads[key].extend([t.detach() for t in gather_list])
                        else:
                            # Send to owner
                            dist.gather(g, dst=owner)
                    else:
                        # Single-process fallback: just append local tensor
                        if key not in per_replicate_grads:
                            per_replicate_grads[key] = []
                        per_replicate_grads[key].append(g)

                # Attention split or single matrix
                if param_type == "Attention" and param.grad.ndim >= 3:
                    for i, component in enumerate(["Q", "K", "V", "O"]):
                        key = (f"Attention {component}", layer_num)
                        handle_key(key, param.grad[i])
                else:
                    key = (param_type, layer_num)
                    handle_key(key, param.grad)

            # Ensure all ranks have finished sharding for this accumulation
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                print(f"[rank 0] Sharding complete for accumulation {accum_idx+1}/{num_accumulation_steps}")
    
    # Stack gradients into batched tensors (owners only)
    result: Dict[Tuple[str, int], torch.Tensor] = {}
    for key, grad_list in per_replicate_grads.items():
        if not grad_list:
            continue
        # Expect exactly (world_size * num_accumulation_steps) entries per key
        G = torch.stack(grad_list, dim=0)  # [R,H,W] (on GPU)
        if accum_buffer_map is not None and key in accum_buffer_map and accum_gamma != 0.0:
            B = accum_buffer_map[key]                    # [H,W]
            # Match attention split shapes too (if caller passed Q/K/V/O separately this is already [H,W])
            if B.ndim == G.ndim - 1:
                # broadcast across replicas
                A = (1.0 - accum_gamma) * G + accum_gamma * B.unsqueeze(0)
            else:
                # exact match [H,W]
                A = (1.0 - accum_gamma) * G + accum_gamma * B
            result[key] = A
        else:
            result[key] = G

    # Final synchronization and log once all sharing is complete across all minibatches
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(f"[rank 0] Completed distributed sharding of microgradients for all {num_accumulation_steps} accumulations")

    # (debug logs removed)

    return result
